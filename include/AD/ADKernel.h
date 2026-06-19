#pragma once

/**
 * @file ADKernel.h
 * @brief Clean, GPU-executable automatic differentiation kernel API.
 *
 * Usage:
 *   ADKernel1D kernel([](Var<int>& id) {
 *       auto W = buf_W[id];        // parameter buffer
 *       auto x = buf_x[id];
 *       auto y = W * x;
 *       auto loss = y * y;
 *       int iW = AD::Param(W);     // mark parameter, get index
 *       AD::Loss(loss);            // mark scalar loss
 *   }, N);
 *   kernel.Forward(groups);        // run forward pass on GPU
 *   kernel.Backward(groups);       // run forward+backward, write gradients
 *   auto grad_W = kernel.Gradient(iW);  // download gradients
 *
 * Gradient buffer sharing:
 *   Multiple parameters from the same source buffer (e.g. buf_W[0], buf_W[1])
 *   share a single gradient SSBO with an interleaved layout to stay within
 *   GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS (minimum 8 in OpenGL 4.3).
 */

#ifndef EASYGPU_AD_ADKERNEL_H
#define EASYGPU_AD_ADKERNEL_H

#include <AD/AdjointGenerator.h>
#include <AD/GradientTape.h>

#include <Backend/Backend.h>
#include <IR/Builder/Builder.h>
#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>
#include <Kernel/Kernel.h>
#include <Runtime/Context.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Windows SDK defines MemoryBarrier as a macro — undefine to use the method name
#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace GPU::AD {

// =============================================================================
// GLSLTypeName — C++ type → GLSL type string
// =============================================================================

template <typename T> inline std::string GLSLTypeName() {
	if constexpr (std::is_same_v<T, float>)
		return "float";
	else if constexpr (std::is_same_v<T, int>)
		return "int";
	else if constexpr (std::is_same_v<T, bool>)
		return "bool";
	else if constexpr (std::is_same_v<T, Math::Vec2>)
		return "vec2";
	else if constexpr (std::is_same_v<T, Math::Vec3>)
		return "vec3";
	else if constexpr (std::is_same_v<T, Math::Vec4>)
		return "vec4";
	else if constexpr (std::is_same_v<T, Math::IVec2>)
		return "ivec2";
	else if constexpr (std::is_same_v<T, Math::IVec3>)
		return "ivec3";
	else if constexpr (std::is_same_v<T, Math::IVec4>)
		return "ivec4";
	else
		return "float";
}

// =============================================================================
// Free functions: Param / Loss
// =============================================================================

/**
 * Mark a Var<T> as a trainable parameter.
 * Returns the parameter index for later gradient retrieval.
 * Must be called inside a kernel lambda during ADKernel construction.
 */
template <typename T> inline int Param(const IR::Value::Var<T> &var) {
	auto *tape = IR::Builder::Builder::Get().GetGradientTape();
	if (tape) {
		int idx = static_cast<int>(tape->Parameters().size());
		tape->RegisterParameter(var.VarName(), GLSLTypeName<T>());
		return idx;
	}
	return -1;
}

/**
 * Mark a whole buffer as a trainable parameter tensor.
 * The backward shader records one local adjoint array for the buffer and
 * writes all element gradients with a compact loop.
 */
template <typename T> inline int ParamBuffer(const IR::Value::BufferRef<T> &buffer, size_t elementCount) {
	auto *tape = IR::Builder::Builder::Get().GetGradientTape();
	if (tape) {
		int idx = static_cast<int>(tape->Parameters().size() + tape->BufferParameters().size());
		tape->RegisterBufferParameter(buffer.GetBufferName(), GLSLTypeName<T>(), elementCount);
		return idx;
	}
	return -1;
}

/**
 * Mark a Var<T> as the scalar loss.
 * The loss receives a seed adjoint of 1.0 at the start of the backward pass.
 */
template <typename T> inline void Loss(const IR::Value::Var<T> &var) {
	auto *tape = IR::Builder::Builder::Get().GetGradientTape();
	if (tape) {
		tape->MarkLoss(var.VarName(), GLSLTypeName<T>());
	}
}

// =============================================================================
// GradBufGroup — describes a single gradient SSBO that may hold multiple params
// =============================================================================

struct GradBufGroup {
	std::string baseName; // sanitized base buffer name (e.g. "buf2")
	int			binding = 0;
	int			stride	= 1; // number of interleaved params in this buffer
};

// =============================================================================
// MergeForwardBackward — assemble combined forward+backward GLSL
// =============================================================================

/**
 * Insert adjoint declarations, body, and gradient writebacks into the forward
 * shader's main() function.  The forward intermediate values stay in scope,
 * so no separate storage/reload is needed.
 *
 * Gradient buffers use an interleaved layout: for a group with stride S,
 * thread i writes its S parameter gradients at positions [i*S .. i*S+S-1].
 */
inline std::string MergeForwardBackward(const std::string &forwardCode, const AdjointBody &body, int workSizeX,
										int workSizeY, int workSizeZ,
										const std::vector<GradBufGroup> &gradBufGroups, int adjPoolBinding) {
	auto mainPos = forwardCode.find("void main()");
	if (mainPos == std::string::npos)
		throw std::runtime_error("MergeForwardBackward: 'void main()' not found");

	auto bracePos = forwardCode.find('{', mainPos);
	if (bracePos == std::string::npos)
		throw std::runtime_error("MergeForwardBackward: main() body opening brace not found");

	// Extract base name and constant index from "buf2[0]" → {"buf2", 0}
	auto parseBufAccess = [](const std::string &name) -> std::pair<std::string, int> {
		auto bpos = name.find('[');
		if (bpos == std::string::npos)
			return {name, -1};
		auto		cpos = name.find(']', bpos);
		std::string base = name.substr(0, bpos);
		if (cpos != std::string::npos && cpos > bpos + 1) {
			std::string idx = name.substr(bpos + 1, cpos - bpos - 1);
			if (idx.find('(') == std::string::npos && idx.find('v') == std::string::npos) {
				try {
					return {base, std::stoi(idx)};
				} catch (...) {
					return {base, -1};
				}
			}
		}
		return {base, -1};
	};

	// Build base-name → {binding, stride} lookup
	std::unordered_map<std::string, std::pair<int, int>> groupMap;
	for (const auto &g : gradBufGroups) {
		groupMap[g.baseName] = {g.binding, g.stride};
	}

	// Gradient buffer declarations (one per group, not per param)
	std::string gradBufDecls;
	for (const auto &g : gradBufGroups) {
		gradBufDecls +=
			std::format("layout(std430, binding = {}) buffer _ad_gradbuf_{} {{ float _ad_grad_{}_data[]; }};\n",
						g.binding, g.baseName, g.baseName);
	}

	auto adjBaseName = [](const std::string &name) {
		auto bpos = name.find('[');
		return bpos == std::string::npos ? name : name.substr(0, bpos);
	};

	std::unordered_map<std::string, std::pair<std::string, int>> paramAdjArrays;
	for (const auto &[paramName, adjName] : body.writebacks) {
		auto [baseName, index] = parseBufAccess(paramName);
		auto it				   = groupMap.find(baseName);
		if (it != groupMap.end()) {
			paramAdjArrays[adjBaseName(adjName)] = {baseName, it->second.second};
		}
	}
	for (const auto &wb : body.bufferWritebacks) {
		auto it = groupMap.find(wb.bufferName);
		if (it != groupMap.end()) {
			paramAdjArrays[adjBaseName(wb.adjName)] = {wb.bufferName, it->second.second};
		}
	}

	// Adjoint variable declarations.
	// Parameter adjoint arrays must stay local to each invocation; otherwise
	// multiple GPU threads race on the same grad_bufN[index] slot before the
	// per-thread gradient writeback happens. Large activation adjoints remain
	// in the shared SSBO pool because their indices already include thread ids.
	std::string				 adjDecls;
	std::vector<std::string> adjArrayNames;
	for (const auto &[adjName, glslType] : body.declarations) {
		auto bracketPos = glslType.find('[');
		if (bracketPos != std::string::npos) {
			if (!paramAdjArrays.count(adjName)) {
				adjArrayNames.push_back(adjName);
			}
		} else {
			adjDecls += std::format("    {} {} = {}(0);\n", glslType, adjName, glslType);
		}
	}
	if (!adjArrayNames.empty()) {
		gradBufDecls += std::format("layout(std430, binding = {}) buffer _adj_pool {{\n", adjPoolBinding);
		for (size_t ai = 0; ai < adjArrayNames.size(); ai++) {
			const std::string &adjNm = adjArrayNames[ai];
			// Extract array size from body.declarations (e.g. "float[1024]" → "[1024]")
			std::string		   arrSize;
			for (const auto &[dName, dType] : body.declarations) {
				if (dName == adjNm) {
					size_t ob = dType.find('[');
					size_t cb = dType.find(']', ob);
					if (ob != std::string::npos && cb != std::string::npos) {
						arrSize = dType.substr(ob, cb - ob + 1);
					}
					break;
				}
			}
			// Last member can be unsized; all others need explicit sizes
			bool isLast = (ai == adjArrayNames.size() - 1);
			if (isLast || arrSize.empty()) {
				gradBufDecls += std::format("    float {}[];\n", adjNm);
			} else {
				gradBufDecls += std::format("    float {}{};\n", adjNm, arrSize);
			}
		}
		gradBufDecls += "};\n";
	}

	// Adjoint body lines
	std::string adjBody;
	for (const auto &line : body.lines) {
		std::string rewritten = line;
		for (const auto &[adjBase, info] : paramAdjArrays) {
			const auto &[bufBase, stride] = info;
			std::string from			  = adjBase + "[";
			std::string to = std::format("_ad_grad_{}_data[int(gl_GlobalInvocationID.x) * {} + ", bufBase, stride);
			size_t		pos = 0;
			while ((pos = rewritten.find(from, pos)) != std::string::npos) {
				rewritten.replace(pos, from.size(), to);
				pos += to.size();
			}
		}
		adjBody += std::format("    {}\n", rewritten);
	}

	// Gradient writebacks with interleaved layout.
	// Use sequential group offset per base name (matching CPU gradOffset),
	// NOT the buffer element index from parseBufAccess.
	bool								 is1D = (workSizeY == 1 && workSizeZ == 1);
	std::string							 writebacks;
	std::unordered_map<std::string, int> baseWritebackCount;
	for (const auto &[paramName, adjName] : body.writebacks) {
		if (paramAdjArrays.count(adjBaseName(adjName)))
			continue;
		auto [baseName, index] = parseBufAccess(paramName);
		auto it				   = groupMap.find(baseName);
		if (it == groupMap.end())
			continue;
		int stride = it->second.second;
		int offset = baseWritebackCount[baseName]++;
		if (is1D) {
			writebacks += std::format("    _ad_grad_{}_data[int(gl_GlobalInvocationID.x) * {} + {}] = {};\n",
									  baseName, stride, offset, adjName);
		} else {
			writebacks += std::format("    _ad_grad_{}_data[(gl_GlobalInvocationID.y * gl_NumWorkGroups.x * "
									  "gl_WorkGroupSize.x + gl_GlobalInvocationID.x) * {} + {}] = {};\n",
									  baseName, stride, offset, adjName);
		}
	}
	for (const auto &wb : body.bufferWritebacks) {
		if (paramAdjArrays.count(adjBaseName(wb.adjName)))
			continue;
		auto it = groupMap.find(wb.bufferName);
		if (it == groupMap.end())
			continue;
		int stride = it->second.second;
		if (is1D) {
			writebacks += std::format("    for (uint _ad_bp = 0u; _ad_bp < {}u; ++_ad_bp) "
									  "_ad_grad_{}_data[int(gl_GlobalInvocationID.x) * {} + int(_ad_bp)] = {}[_ad_bp];\n",
									  wb.elementCount, wb.bufferName, stride, wb.adjName);
		} else {
			writebacks += std::format("    for (uint _ad_bp = 0u; _ad_bp < {}u; ++_ad_bp) "
									  "_ad_grad_{}_data[(gl_GlobalInvocationID.y * gl_NumWorkGroups.x * "
									  "gl_WorkGroupSize.x + gl_GlobalInvocationID.x) * {} + _ad_bp] = {}[_ad_bp];\n",
									  wb.elementCount, wb.bufferName, stride, wb.adjName);
		}
	}

	// Find the closing brace of main() (last '}' in the code)
	auto closePos = forwardCode.rfind('}');
	if (closePos == std::string::npos || closePos <= bracePos)
		throw std::runtime_error("MergeForwardBackward: main() closing brace not found");

	std::string forwardBody = forwardCode.substr(bracePos + 1, closePos - bracePos - 1);

	// Hoist forward temporary variable declarations (v1, v2, ...) to
	// function scope so the backward code can reference them even when
	// they were originally declared inside for/if blocks.
	std::string hoistedDecls;
	std::string strippedBody;
	strippedBody.reserve(forwardBody.size());
	hoistedDecls.reserve(forwardBody.size() / 4);

	// Pass 1: hoist uninitialized declarations (int v10; without =)
	{
		size_t lineStart = 0;
		while (lineStart < forwardBody.size()) {
			size_t lineEnd = forwardBody.find('\n', lineStart);
			if (lineEnd == std::string::npos)
				lineEnd = forwardBody.size();
			size_t			 lineLen = lineEnd - lineStart;
			std::string_view line(forwardBody.data() + lineStart, lineLen);

			bool			 isHoistable = false;
			size_t			 pos		 = 0;
			while (pos < lineLen && (line[pos] == ' ' || line[pos] == '\t'))
				pos++;
			auto checkType = [&](const char *typeName, size_t len) {
				if (pos + len < lineLen && line.compare(pos, len, typeName) == 0 && line[pos + len] == ' ') {
					size_t nameStart = pos + len + 1;
					if (nameStart < lineLen && line[nameStart] == 'v') {
						size_t j = nameStart + 1;
						while (j < lineLen && line[j] >= '0' && line[j] <= '9')
							j++;
						if (j < lineLen && line[j] == ';') {
							bool hasEq = false;
							for (size_t k = nameStart; k < j; k++) {
								if (line[k] == '=') {
									hasEq = true;
									break;
								}
							}
							if (!hasEq)
								isHoistable = true;
						}
					}
				}
			};
			static const char *types[] = {"float", "int",	"bool",	 "uint",  "vec2",  "vec3", "vec4", "ivec2",
										  "ivec3", "ivec4", "bvec2", "bvec3", "bvec4", "mat4", "mat3", "mat2"};
			for (const char *t : types) {
				if (isHoistable)
					break;
				checkType(t, std::char_traits<char>::length(t));
			}

			if (isHoistable) {
				hoistedDecls += line;
				hoistedDecls += '\n';
			} else {
				strippedBody += line;
				strippedBody += '\n';
			}
			lineStart = lineEnd + 1;
		}
	}

	// Pass 2: hoist for-loop variable declarations (including comma-separated).
	// Transform "for (type vNN = ..." -> "type vNN;" + "for (vNN = ..."
	{
		const char *forTypes[] = {"int", "float", "bool", "uint"};
		for (const char *ft : forTypes) {
			size_t		searchPos = 0;
			std::string pattern	  = std::string("for (") + ft + " v";
			while ((searchPos = strippedBody.find(pattern, searchPos)) != std::string::npos) {
				// Find the first variable name
				size_t vStart = searchPos + pattern.size() - 1;
				size_t vEnd	  = vStart + 1;
				while (vEnd < strippedBody.size() && strippedBody[vEnd] >= '0' && strippedBody[vEnd] <= '9')
					vEnd++;
				std::string				 varName = strippedBody.substr(vStart, vEnd - vStart);
				// Collect ALL comma-separated variable names
				std::vector<std::string> allVars;
				allVars.push_back(varName);
				size_t cursor = vEnd;
				while (cursor < strippedBody.size() && strippedBody[cursor] != ';') {
					if (strippedBody[cursor] == ',') {
						cursor++;
						while (cursor < strippedBody.size() &&
							   (strippedBody[cursor] == ' ' || strippedBody[cursor] == '\t'))
							cursor++;
						if (cursor < strippedBody.size() && strippedBody[cursor] == 'v') {
							size_t cvEnd = cursor + 1;
							while (cvEnd < strippedBody.size() && strippedBody[cvEnd] >= '0' &&
								   strippedBody[cvEnd] <= '9')
								cvEnd++;
							allVars.push_back(strippedBody.substr(cursor, cvEnd - cursor));
							cursor = cvEnd;
						}
					}
					cursor++;
				}
				// Hoist all collected variables
				for (const auto &v : allVars) {
					hoistedDecls += std::string(ft) + " " + v + ";\n";
				}
				// Remove "type " from the for-init
				size_t typeStart = searchPos + 5; // skip "for ("
				size_t typeLen	 = strlen(ft);
				strippedBody.erase(typeStart, typeLen + 1); // +1 for space after type
				searchPos++;								// move past
			}
		}
	}

	std::string result;
	result.reserve(forwardCode.size() + gradBufDecls.size() + adjDecls.size() + adjBody.size() + writebacks.size() +
				   hoistedDecls.size() + 200);

	result += forwardCode.substr(0, mainPos);
	result += gradBufDecls;
	if (!gradBufDecls.empty())
		result += "\n";
	result += forwardCode.substr(mainPos, bracePos - mainPos + 1);
	result += "\n";
	if (!hoistedDecls.empty())
		result += hoistedDecls;
	result += adjDecls;
	result += strippedBody;

	if (!adjBody.empty()) {
		result += "\n    // === Backward pass (auto-generated) ===\n";
		result += adjBody;
	}
	if (!writebacks.empty()) {
		result += "\n    // --- Gradient writebacks ---\n";
		result += writebacks;
	}
	result += forwardCode.substr(closePos);
	return result;
}

// =============================================================================
// ADKernel1D — 1D GPU-executable AD kernel
// =============================================================================

/**
 * A 1D GPU kernel with automatic differentiation.
 *
 * Builds a Kernel1D during construction while recording the gradient tape,
 * then generates a combined forward+backward GLSL shader.  Forward() dispatches
 * the user's computation.  Backward() dispatches the combined shader which
 * computes both the loss and its gradients in a single pass.
 *
 * Parameters from the same source buffer share a single interleaved gradient
 * SSBO to minimise the number of shader storage blocks used.
 */
class ADKernel1D {
public:
	/**
	 * Construct the AD kernel.
	 *
	 * @param func         The computation lambda: void(Var<int>& id)
	 * @param elementCount Total number of GPU threads (determines gradient buffer size)
	 * @param groupSize    Work group size (default 256)
	 */
	template <typename Func>
	ADKernel1D(Func &&func, size_t elementCount, int groupSize = 256)
		: _elementCount(elementCount), _workSizeX(groupSize) {

		// Phase 1: Build forward kernel while recording gradient tape
		{
			auto									&builder = IR::Builder::Builder::Get();
			IR::Builder::Builder::ScopedGradientTape tapeGuard(builder, &_tape);

			_forwardKernel = std::make_unique<Kernel::Kernel1D>(std::forward<Func>(func), groupSize);

			// Keep tape active during GetCode() so callable body operations
			// are recorded to sub-tapes via the sub-tape stack.
			_forwardCode   = _forwardKernel->GetCode();
		}

		_nextGradBinding = static_cast<int>(_forwardKernel->GetContext().GetNextBinding());

		// Phase 2: Record parameter → buffer mapping, group by source buffer
		for (const auto &[paramName, paramType] : _tape.Parameters()) {
			ParamMeta pm;
			pm.varName	  = paramName;
			pm.glslType	  = paramType;
			pm.count	  = elementCount;
			pm.elementCount = 1;
			pm.gradHandle = Backend::INVALID_BUFFER_HANDLE;
			_params.push_back(pm);
		}
		for (const auto &param : _tape.BufferParameters()) {
			ParamMeta pm;
			pm.varName		 = param.bufferName;
			pm.glslType		 = param.elementType;
			pm.count		 = elementCount;
			pm.elementCount = param.elementCount;
			pm.gradOffset	 = 0;
			pm.gradStride	 = static_cast<int>(param.elementCount);
			pm.isBufferParam = true;
			pm.gradHandle	 = Backend::INVALID_BUFFER_HANDLE;
			_params.push_back(pm);
		}

		// Group params by source buffer base name, assign shared bindings
		{
			std::map<std::string, std::vector<int>> groups; // baseName → param indices (deterministic order)
			for (int i = 0; i < (int)_params.size(); i++) {
				if (_params[i].isBufferParam)
					continue;
				std::string base = ExtractBaseName(_params[i].varName);
				groups[base].push_back(i);
			}

			for (auto &[baseName, indices] : groups) {
				int binding = _nextGradBinding++;
				if (binding >= static_cast<int>(Backend::MAX_BUFFER_BINDINGS)) {
					throw std::runtime_error("ADKernel1D: gradient buffers exceed backend buffer binding limit");
				}
				int stride	= (int)indices.size();
				for (int offset = 0; offset < stride; offset++) {
					int pi					= indices[offset];
					_params[pi].gradBinding = binding;
					_params[pi].gradOffset	= offset;
					_params[pi].gradStride	= stride;
				}
			}
			for (auto &pm : _params) {
				if (!pm.isBufferParam)
					continue;
				int binding = _nextGradBinding++;
				if (binding >= static_cast<int>(Backend::MAX_BUFFER_BINDINGS)) {
					throw std::runtime_error("ADKernel1D: gradient buffers exceed backend buffer binding limit");
				}
				pm.gradBinding = binding;
			}
		}

		// Phase 3: Generate adjoint body
		AdjointGenerator gen;
		_body = gen.GenerateBody(_tape, true);

		// Phase 3.5: Fix adjoint declarations for buffers that are used as
		// arrays but declared as scalars (because they lack registered params).
		{
			// Collect adjoint names used with array indexing in backward body
			std::unordered_map<std::string, size_t> adjMaxIdx;
			for (const auto &line : _body.lines) {
				size_t pos = 0;
				while (pos < line.size()) {
					auto bstart = line.find('[', pos);
					if (bstart == std::string::npos)
						break;
					size_t idEnd = bstart;
					while (idEnd > 0 && line[idEnd - 1] == ' ')
						idEnd--;
					size_t idStart = idEnd;
					while (idStart > 0 &&
						   (std::isalnum(static_cast<unsigned char>(line[idStart - 1])) || line[idStart - 1] == '_'))
						idStart--;
					std::string arrName = line.substr(idStart, idEnd - idStart);
					auto		bend	= line.find(']', bstart);
					if (bend != std::string::npos) {
						std::string idxStr = line.substr(bstart + 1, bend - bstart - 1);
						try {
							size_t idx = std::stoull(idxStr);
							if (idx + 1 > adjMaxIdx[arrName])
								adjMaxIdx[arrName] = idx + 1;
						} catch (...) {
							// Non-constant index: estimate from int(N) literals
							// and per-thread stride for gl_GlobalInvocationID.x
							size_t maxConst = 0;
							size_t cp		= 0;
							while (cp < idxStr.size()) {
								auto ip = idxStr.find("int(", cp);
								if (ip == std::string::npos)
									break;
								ip		+= 4;
								auto ie	 = idxStr.find(')', ip);
								if (ie != std::string::npos) {
									std::string ns = idxStr.substr(ip, ie - ip);
									try {
										size_t val = std::stoull(ns);
										if (val > maxConst)
											maxConst = val;
									} catch (...) {
									}
									cp = ie + 1;
								} else
									break;
							}
							size_t maxStride = 0;
							auto   gidPos	 = idxStr.find("gl_GlobalInvocationID.x");
							if (gidPos != std::string::npos) {
								// Extract the first int(N) after thread ID as per-thread stride.
								// Skip nested [...] sections.
								size_t sp	 = gidPos;
								int	   depth = 0;
								while (sp < idxStr.size() && maxStride == 0) {
									char c = idxStr[sp];
									if (c == '[')
										depth++;
									else if (c == ']') {
										if (depth > 0)
											depth--;
									} else if (depth == 0 && c == '(' && sp + 4 < idxStr.size() &&
											   idxStr.substr(sp, 5) == "(int(") {
										sp		+= 5;
										auto me	 = idxStr.find(')', sp);
										if (me != std::string::npos) {
											try {
												maxStride = std::stoull(idxStr.substr(sp, me - sp));
											} catch (...) {
											}
											sp = me;
										}
									}
									sp++;
								}
							}
							// Conservative estimate: base + threads*stride + maxConst padding
							size_t est =
								maxConst + (_elementCount > 0 ? _elementCount : 64) * maxStride + maxConst + 4096;
							// Cap at 500K elements (2 MB per array) to prevent GPU OOM
							if (est > 500000)
								est = 500000;
							if (est > adjMaxIdx[arrName])
								adjMaxIdx[arrName] = est;
						}
					}
					pos = bend + 1;
				}
			}
			// Fix scalar declarations to array types where needed
			for (auto &[adjName, glslType] : _body.declarations) {
				if (glslType.find('[') == std::string::npos && adjMaxIdx.count(adjName)) {
					size_t arrSize = adjMaxIdx[adjName];
					if (arrSize <= 1)
						arrSize = _elementCount > 0 ? _elementCount : 1024;
					glslType = std::format("{}[{}]", glslType, arrSize);
				}
			}
			// Track adjoint arrays for combined pool allocation (offsets, not bindings)
			{
				size_t runningOffset = 0;
				for (const auto &[adjName2, glslType2] : _body.declarations) {
					auto bracketPos2 = glslType2.find('[');
					if (bracketPos2 != std::string::npos) {
						std::string sizeStr = glslType2.substr(bracketPos2 + 1, glslType2.size() - bracketPos2 - 2);
						try {
							AdjArrayMeta am;
							am.adjName	   = adjName2;
							am.offset	   = runningOffset;
							am.size		   = std::stoull(sizeStr);
							runningOffset += am.size;
							_adjArrays.push_back(am);
						} catch (...) {
						}
					}
				}
				_adjPoolSize = runningOffset;
			}
		}

		if (_adjPoolSize > 0) {
			_adjPoolBinding = _nextGradBinding++;
			if (_adjPoolBinding >= static_cast<int>(Backend::MAX_BUFFER_BINDINGS)) {
				throw std::runtime_error("ADKernel1D: adjoint pool exceeds backend buffer binding limit");
			}
		}

		// Phase 4: Build grouped gradient buffer list for merge
		std::vector<GradBufGroup> gradBufGroups;
		{
			std::unordered_set<int> seenBindings;
			for (const auto &pm : _params) {
				if (seenBindings.insert(pm.gradBinding).second) {
					GradBufGroup gb;
					gb.baseName = ExtractBaseName(pm.varName);
					gb.binding	= pm.gradBinding;
					gb.stride	= pm.gradStride;
					gradBufGroups.push_back(gb);
				}
			}
		}

		// Phase 5: Merge forward + backward into combined GLSL
		if (!_body.lines.empty() || !_body.declarations.empty()) {
			_combinedCode = MergeForwardBackward(_forwardCode, _body, groupSize, 1, 1, gradBufGroups, _adjPoolBinding);

			// Insert thread ID bounds guard so threads beyond elementCount
			// do not read/write past undersized input/adjoint buffers.
			{
				auto mainPos = _combinedCode.find("void main()");
				if (mainPos != std::string::npos) {
					auto bracePos = _combinedCode.find("{", mainPos);
					if (bracePos != std::string::npos) {
						std::string guard =
							"\n    if (gl_GlobalInvocationID.x >= " + std::to_string(_elementCount) + ") return;\n";
						_combinedCode.insert(bracePos + 1, guard);
					}
				}
			}
		}
	}

	~ADKernel1D() {
		ReleaseGradientBuffers();
	}

	// ---- Execution -------------------------------------------------------

	/** Dispatch forward pass (just the user's computation). */
	void Forward(int groupCount, bool sync = false) {
		_forwardKernel->Dispatch(groupCount, sync);
	}

	/**
	 * Dispatch the combined forward+backward pass.
	 * Computes loss and writes gradients to internal gradient buffers.
	 */
	void Backward(int groupCount, bool sync = false) {
		if (_combinedCode.empty())
			return;
		EnsureGradientBuffers();
		// Parameter adjoints use gradient output buffers as per-thread
		// accumulators, so clear them on GPU before every backward pass.
		{
			std::unordered_set<Backend::BufferHandle> cleared;
			size_t dispThreads = ((_elementCount + _workSizeX - 1) / _workSizeX) * _workSizeX;
			for (const auto &pm : _params) {
				if (pm.gradHandle == Backend::INVALID_BUFFER_HANDLE)
					continue;
				if (!cleared.insert(pm.gradHandle).second)
					continue;
				ClearBufferGPU(pm.gradHandle, dispThreads * pm.gradStride);
			}
		}
		// Zero adjoint pool to prevent accumulation between dispatches.
		if (_adjPoolHandle != Backend::INVALID_BUFFER_HANDLE && _adjPoolSize > 0) {
			ClearBufferGPU(_adjPoolHandle, _adjPoolSize);
		}
		ExecuteCombinedDispatch(groupCount, sync);
	}

	/**
	 * Download the gradient for a parameter by index.
	 * Returns a vector of floats with elementCount entries.
	 * For grouped buffers, extracts the correct interleaved slice.
	 */
	std::vector<float> Gradient(int paramIndex) const {
		if (paramIndex < 0 || paramIndex >= (int)_params.size())
			return {};
		const auto &pm = _params[paramIndex];
		if (pm.gradHandle == Backend::INVALID_BUFFER_HANDLE)
			return {};

		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			return {};

		if (pm.isBufferParam) {
			size_t			   totalFloats = pm.count * pm.elementCount;
			std::vector<float> data(totalFloats);
			backend->DownloadBuffer(pm.gradHandle, 0, totalFloats * sizeof(float), data.data());
			return data;
		}

		if (pm.gradStride == 1) {
			// Fast path: param has its own buffer
			std::vector<float> data(pm.count);
			backend->DownloadBuffer(pm.gradHandle, 0, pm.count * sizeof(float), data.data());
			return data;
		}

		// Shared buffer: download full group, extract interleaved slice
		size_t			   totalFloats = pm.count * pm.gradStride;
		std::vector<float> full(totalFloats);
		backend->DownloadBuffer(pm.gradHandle, 0, totalFloats * sizeof(float), full.data());

		std::vector<float> data(pm.count);
		for (size_t i = 0; i < pm.count; i++) {
			data[i] = full[i * pm.gradStride + pm.gradOffset];
		}
		return data;
	}

	/**
	 * Download all parameter gradients in one efficient batch.
	 * Shared gradient buffers are downloaded only once and cached,
	 * avoiding redundant transfers for interleaved groups.
	 */
	std::vector<std::vector<float>> DownloadAllGradients() const {
		std::vector<std::vector<float>> result(_params.size());
		auto						   *backend = Runtime::Context::GetBackend();
		if (!backend)
			return result;

		std::unordered_map<Backend::BufferHandle, std::vector<float>> cache;

		for (size_t i = 0; i < _params.size(); i++) {
			const auto &pm = _params[i];
			if (pm.gradHandle == Backend::INVALID_BUFFER_HANDLE)
				continue;

			if (pm.isBufferParam) {
				result[i].resize(pm.count * pm.elementCount);
				backend->DownloadBuffer(pm.gradHandle, 0, result[i].size() * sizeof(float), result[i].data());
			} else if (pm.gradStride == 1) {
				result[i].resize(pm.count);
				backend->DownloadBuffer(pm.gradHandle, 0, pm.count * sizeof(float), result[i].data());
			} else {
				auto &cached = cache[pm.gradHandle];
				if (cached.empty()) {
					size_t totalFloats = pm.count * pm.gradStride;
					cached.resize(totalFloats);
					backend->DownloadBuffer(pm.gradHandle, 0, totalFloats * sizeof(float), cached.data());
				}
				result[i].resize(pm.count);
				for (size_t j = 0; j < pm.count; j++)
					result[i][j] = cached[j * pm.gradStride + pm.gradOffset];
			}
		}
		return result;
	}

	/**
	 * Download the gradient for a parameter by variable name.
	 */
	std::vector<float> Gradient(const std::string &paramVarName) const {
		for (int i = 0; i < (int)_params.size(); ++i) {
			if (_params[i].varName == paramVarName)
				return Gradient(i);
		}
		return {};
	}

	// ---- Debugging -------------------------------------------------------

	std::string ForwardCode() const {
		return _forwardCode;
	}
	std::string CombinedCode() const {
		return _combinedCode;
	}
	const GradientTape &Tape() const {
		return _tape;
	}
	size_t ParameterCount() const {
		size_t count = 0;
		for (const auto &pm : _params)
			count += pm.elementCount;
		return count;
	}
	const auto &Params() const {
		return _params;
	}

	struct GradientParamInfo {
		std::string			  varName;
		size_t				  sampleCount = 0;
		int					  gradOffset  = 0;
		int					  gradStride  = 1;
		Backend::BufferHandle gradHandle  = Backend::INVALID_BUFFER_HANDLE;
	};

	std::vector<GradientParamInfo> GradientParams() const {
		std::vector<GradientParamInfo> out;
		out.reserve(ParameterCount());
		for (const auto &pm : _params) {
			for (size_t elem = 0; elem < pm.elementCount; elem++) {
				GradientParamInfo info;
				info.varName	 = pm.isBufferParam ? std::format("{}[{}]", pm.varName, elem) : pm.varName;
				info.sampleCount = pm.count;
				info.gradOffset	 = pm.gradOffset + static_cast<int>(elem);
				info.gradStride	 = pm.gradStride;
				info.gradHandle	 = pm.gradHandle;
				out.push_back(std::move(info));
			}
		}
		return out;
	}

private:
	struct ParamMeta {
		std::string			  varName;
		std::string			  glslType;
		size_t				  count		  = 0;
		size_t				  elementCount = 1;
		int					  gradBinding = 0;
		int					  gradOffset  = 0; // offset within the interleaved group
		int					  gradStride  = 1; // number of params in the group
		bool				  isBufferParam = false;
		Backend::BufferHandle gradHandle  = 0;
	};

	struct AdjArrayMeta {
		std::string adjName;
		size_t		offset = 0; // offset (in floats) into the combined adjoint pool
		size_t		size   = 0;
	};

	struct ClearPipeline {
		Backend::ShaderHandle	shader	 = Backend::INVALID_SHADER_HANDLE;
		Backend::PipelineHandle pipeline = Backend::INVALID_PIPELINE_HANDLE;
	};

	/** Extract the base buffer name from a var name like "buf2[0]" → "buf2". */
	static std::string ExtractBaseName(const std::string &varName) {
		auto bpos = varName.find('[');
		if (bpos == std::string::npos)
			return varName;
		return varName.substr(0, bpos);
	}

	void EnsureGradientBuffers() {
		Runtime::Context::GetInstance().MakeCurrent();
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			return;

		// Allocate combined adjoint pool SSBO
		if (_adjPoolHandle == Backend::INVALID_BUFFER_HANDLE && _adjPoolSize > 0) {
			Backend::BufferDesc desc;
			desc.sizeInBytes = _adjPoolSize * sizeof(float);
			desc.mode		 = Backend::BufferMode::ReadWrite;
			desc.initialData = nullptr;
			_adjPoolHandle	 = backend->CreateBuffer(desc);
		}

		// Track which bindings already have a buffer
		std::unordered_set<int> created;
		for (auto &pm : _params) {
			if (pm.gradHandle != Backend::INVALID_BUFFER_HANDLE)
				continue;
			if (!created.insert(pm.gradBinding).second) {
				// Buffer already exists for this group — reuse handle
				for (const auto &other : _params) {
					if (other.gradBinding == pm.gradBinding && other.gradHandle != Backend::INVALID_BUFFER_HANDLE) {
						pm.gradHandle = other.gradHandle;
						break;
					}
				}
				continue;
			}

			Backend::BufferDesc desc;
			// Size for ALL dispatched threads (elementCount is rounded up to workgroup size)
			size_t				dispThreads = ((_elementCount + _workSizeX - 1) / _workSizeX) * _workSizeX;
			desc.sizeInBytes				= dispThreads * pm.gradStride * sizeof(float);
			desc.mode						= Backend::BufferMode::ReadWrite;
			desc.initialData				= nullptr;

			Backend::BufferHandle handle	= backend->CreateBuffer(desc);
			// Assign handle to all params in this group
			for (auto &pm2 : _params) {
				if (pm2.gradBinding == pm.gradBinding) {
					pm2.gradHandle = handle;
				}
			}
		}
	}

	void ReleaseGradientBuffers() {
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			return;
		std::unordered_set<Backend::BufferHandle> released;
		for (auto &pm : _params) {
			if (pm.gradHandle != Backend::INVALID_BUFFER_HANDLE && released.insert(pm.gradHandle).second) {
				backend->DestroyBuffer(pm.gradHandle);
				pm.gradHandle = Backend::INVALID_BUFFER_HANDLE;
			}
		}
		// Clear all handles (others in same group still have the stale value)
		for (auto &pm : _params) {
			pm.gradHandle = Backend::INVALID_BUFFER_HANDLE;
		}
		// Release combined adjoint pool SSBO
		if (_adjPoolHandle != Backend::INVALID_BUFFER_HANDLE) {
			backend->DestroyBuffer(_adjPoolHandle);
			_adjPoolHandle = Backend::INVALID_BUFFER_HANDLE;
		}
		if (_clearCountBuffer != Backend::INVALID_BUFFER_HANDLE) {
			backend->DestroyBuffer(_clearCountBuffer);
			_clearCountBuffer = Backend::INVALID_BUFFER_HANDLE;
		}
		if (_clearPipeline.pipeline != Backend::INVALID_PIPELINE_HANDLE) {
			backend->DestroyPipeline(_clearPipeline.pipeline);
			_clearPipeline.pipeline = Backend::INVALID_PIPELINE_HANDLE;
		}
		if (_clearPipeline.shader != Backend::INVALID_SHADER_HANDLE) {
			backend->DestroyShader(_clearPipeline.shader);
			_clearPipeline.shader = Backend::INVALID_SHADER_HANDLE;
		}
	}

	void ClearBufferGPU(Backend::BufferHandle handle, size_t floatCount) {
		if (handle == Backend::INVALID_BUFFER_HANDLE || floatCount == 0)
			return;

		Runtime::AutoInitContext();
		Runtime::Context::GetInstance().MakeCurrent();
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			throw std::runtime_error("Backend not available");

		auto &cp = _clearPipeline;
		if (cp.pipeline == Backend::INVALID_PIPELINE_HANDLE) {
			Backend::ShaderDesc shaderDesc;
			shaderDesc.type		  = Backend::ShaderType::Compute;
			shaderDesc.entryPoint = "main";
			shaderDesc.sourceCode = R"GLSL(#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer ClearBuf { float data[]; };
layout(std430, binding = 1) buffer ClearCountBuf { uint clearCount; };
void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= clearCount) return;
	data[i] = 0.0;
}
)GLSL";
			cp.shader			  = backend->CreateShader(shaderDesc);

			Backend::PipelineDesc pipeDesc;
			pipeDesc.computeShader	= cp.shader;
			pipeDesc.workGroupSizeX = 256;
			Backend::ResourceLayoutEntry entry;
			entry.binding  = 0;
			entry.type	   = Backend::BindingType::Buffer;
			entry.readOnly = false;
			pipeDesc.resources.push_back(entry);
			Backend::ResourceLayoutEntry countEntry;
			countEntry.binding  = 1;
			countEntry.type	   = Backend::BindingType::Buffer;
			countEntry.readOnly = true;
			pipeDesc.resources.push_back(countEntry);
			cp.pipeline = backend->CreatePipeline(pipeDesc);
			if (cp.pipeline == Backend::INVALID_PIPELINE_HANDLE)
				throw std::runtime_error("ADKernel1D: failed to create GPU clear pipeline");
		}

		if (_clearCountBuffer == Backend::INVALID_BUFFER_HANDLE) {
			uint32_t			initial = 0;
			Backend::BufferDesc desc;
			desc.sizeInBytes = sizeof(uint32_t);
			desc.mode		 = Backend::BufferMode::Read;
			desc.initialData = &initial;
			_clearCountBuffer = backend->CreateBuffer(desc);
		}
		uint32_t count32 = static_cast<uint32_t>(floatCount);
		backend->UploadBuffer(_clearCountBuffer, 0, sizeof(uint32_t), &count32);

		backend->BindPipeline(cp.pipeline);

		Backend::ResourceBinding bindings[2];
		bindings[0].binding  = 0;
		bindings[0].type	   = Backend::BindingType::Buffer;
		bindings[0].buffer   = handle;
		bindings[0].readOnly = false;
		bindings[1].binding  = 1;
		bindings[1].type	   = Backend::BindingType::Buffer;
		bindings[1].buffer   = _clearCountBuffer;
		bindings[1].readOnly = true;
		backend->BindResources(bindings, 2);
		backend->Dispatch(static_cast<uint32_t>((floatCount + 255) / 256), 1, 1);
		backend->MemoryBarrier(Backend::BarrierType::Buffer);
	}

	void ExecuteCombinedDispatch(int groupCount, bool sync) {
		Runtime::AutoInitContext();
		Runtime::Context::GetInstance().MakeCurrent();
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			throw std::runtime_error("Backend not available");

		// Compile combined pipeline (cached)
		if (_combinedPipeline == Backend::INVALID_PIPELINE_HANDLE) {
			_combinedPipeline = CompileCombinedPipeline(backend);
		}

		backend->BindPipeline(_combinedPipeline);

		// Collect all buffer bindings: forward buffers + gradient buffers
		std::vector<Backend::ResourceBinding> bindings;

		// Forward buffer bindings (from the original kernel context)
		const auto							 &runtimeBufs = _forwardKernel->GetContext().GetRuntimeBufferBindings();
		for (const auto &[binding, handle] : runtimeBufs) {
			Backend::ResourceBinding rb;
			rb.binding = binding;
			rb.type	   = Backend::BindingType::Buffer;
			rb.buffer  = static_cast<Backend::BufferHandle>(handle);
			bindings.push_back(rb);
		}

		// Gradient buffer bindings (deduplicated by binding)
		std::unordered_set<int> bound;
		for (const auto &pm : _params) {
			if (pm.gradHandle != Backend::INVALID_BUFFER_HANDLE && bound.insert(pm.gradBinding).second) {
				Backend::ResourceBinding rb;
				rb.binding = static_cast<uint32_t>(pm.gradBinding);
				rb.type	   = Backend::BindingType::Buffer;
				rb.buffer  = pm.gradHandle;
				bindings.push_back(rb);
			}
		}

		// Combined adjoint pool SSBO binding
		if (_adjPoolHandle != Backend::INVALID_BUFFER_HANDLE) {
			Backend::ResourceBinding rb;
			rb.binding = static_cast<uint32_t>(_adjPoolBinding);
			rb.type	   = Backend::BindingType::Buffer;
			rb.buffer  = _adjPoolHandle;
			bindings.push_back(rb);
		}

		if (!bindings.empty()) {
			backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
		}

		backend->Dispatch(groupCount, 1, 1);

		backend->MemoryBarrier(Backend::BarrierType::All);
		if (sync)
			backend->Finish();
	}

	Backend::PipelineHandle CompileCombinedPipeline(Backend::Backend *backend) {
		Backend::ShaderDesc shaderDesc;
		shaderDesc.type				 = Backend::ShaderType::Compute;
		shaderDesc.sourceCode		 = _combinedCode;
		shaderDesc.entryPoint		 = "main";

		Backend::ShaderHandle shader = backend->CreateShader(shaderDesc);
		if (shader == Backend::INVALID_SHADER_HANDLE)
			throw std::runtime_error("ADKernel1D: failed to compile combined shader");

		Backend::PipelineDesc pipeDesc;
		pipeDesc.computeShader	  = shader;
		pipeDesc.workGroupSizeX	  = static_cast<uint32_t>(_workSizeX);
		pipeDesc.workGroupSizeY	  = 1;
		pipeDesc.workGroupSizeZ	  = 1;
		pipeDesc.pushConstantSize = 0;

		// Forward buffer bindings from the original kernel context
		const auto &bufInfos	  = _forwardKernel->GetContext().GetBufferInfos();
		for (const auto &bi : bufInfos) {
			Backend::ResourceLayoutEntry entry;
			entry.binding  = bi.binding;
			entry.type	   = Backend::BindingType::Buffer;
			entry.readOnly = (bi.mode == Backend::BUFFER_MODE_READ_ONLY);
			pipeDesc.resources.push_back(entry);
		}

		// Gradient buffer bindings (deduplicated by binding)
		std::unordered_set<int> added;
		for (const auto &pm : _params) {
			if (added.insert(pm.gradBinding).second) {
				Backend::ResourceLayoutEntry entry;
				entry.binding  = static_cast<uint32_t>(pm.gradBinding);
				entry.type	   = Backend::BindingType::Buffer;
				entry.readOnly = false;
				pipeDesc.resources.push_back(entry);
			}
		}

		// Combined adjoint pool SSBO
		if (!_adjArrays.empty()) {
			Backend::ResourceLayoutEntry entry;
			entry.binding  = static_cast<uint32_t>(_adjPoolBinding);
			entry.type	   = Backend::BindingType::Buffer;
			entry.readOnly = false;
			pipeDesc.resources.push_back(entry);
		}

		Backend::PipelineHandle pipeline = backend->CreatePipeline(pipeDesc);
		backend->DestroyShader(shader);

		if (pipeline == Backend::INVALID_PIPELINE_HANDLE)
			throw std::runtime_error("ADKernel1D: failed to create combined pipeline");

		return pipeline;
	}

	// ---- Data members ----------------------------------------------------
	size_t									  _elementCount;
	int										  _workSizeX;

	std::unique_ptr<Kernel::Kernel1D>		  _forwardKernel;
	std::string								  _forwardCode;
	std::string								  _combinedCode;

	GradientTape							  _tape;
	AdjointBody								  _body;

	std::vector<ParamMeta>					  _params;
	int										  _nextGradBinding = 0;
	std::vector<AdjArrayMeta>				  _adjArrays;
	size_t									  _adjPoolSize		= 0;
	int										  _adjPoolBinding	= -1;
	Backend::BufferHandle					  _adjPoolHandle	= Backend::INVALID_BUFFER_HANDLE;

	Backend::PipelineHandle					  _combinedPipeline = Backend::INVALID_PIPELINE_HANDLE;
	ClearPipeline							  _clearPipeline;
	Backend::BufferHandle					  _clearCountBuffer = Backend::INVALID_BUFFER_HANDLE;
};

} // namespace GPU::AD

#endif // EASYGPU_AD_ADKERNEL_H
