#pragma once

/**
 * @file AdjointKernel.h
 * @brief GPU-executable AD kernels that merge forward and backward passes.
 *
 * AdKernel1D / AdKernel2D / AdKernel3D build on the InspectorKernel infrastructure
 * and add automatic differentiation. During construction the forward kernel is
 * built (recording the gradient tape), the backward pass is generated, and a
 * combined forward+backward shader is produced.
 *
 * Usage:
 *   AdjointKernel1D kernel([](Var<int>& id, AdjointContext& ctx) {
 *       Var<float> w; w = 2.0f;
 *       Var<float> x; x = 3.0f;
 *       Var<float> y = w * x;
 *       ctx.RegisterParameter(w);
 *       ctx.MarkLoss(y);
 *   });
 *   std::cout << kernel.GetCombinedCode();  // full forward+backward GLSL
 */

#ifndef EASYGPU_AD_ADJOINTKERNEL_H
#define EASYGPU_AD_ADJOINTKERNEL_H

#include <AD/AdjointGenerator.h>
#include <AD/AdjointTable.h>
#include <AD/GradientTape.h>
#include <AD/AdjointInspector.h>

#include <Backend/Backend.h>
#include <IR/Builder/Builder.h>
#include <Kernel/Kernel.h>

#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace GPU::AD {

// =============================================================================
// GradBuffer — tracks a gradient buffer for one parameter
// =============================================================================

/**
 * Tracks a gradient buffer associated with a registered parameter.
 * The buffer is allocated lazily and written to by the combined shader.
 */
struct GradBuffer {
	std::string				paramName;
	std::string				glslType;
	uint32_t				binding = 0;
	uint32_t				count	= 0;
	Backend::BufferHandle	handle = 0;
	bool					allocated = false;
};

// =============================================================================
// MergeForwardBackward — combine forward GLSL with adjoint code
// =============================================================================

/**
 * Insert adjoint declarations, body lines, and gradient writebacks into a
 * forward-pass GLSL shader. Returns the combined forward+backward source.
 *
 * The combined shader layout:
 *   - #version + buffer declarations (from forward)
 *   - gradient buffer declarations (added)
 *   - void main() {
 *   -   forward body (original)
 *   -   adjoint declarations (added, zero-initialized)
 *   -   adjoint body (added)
 *   -   gradient writebacks (added)
 *   - }
 *
 * @param forwardCode  Complete forward GLSL source
 * @param body         Generated adjoint body parts
 * @param gradBuffers  Gradient buffers to declare and write back to
 * @param workSizeX/Y/Z Work group sizes
 */
inline std::string
MergeForwardBackward(const std::string						&forwardCode,
					 const AdjointBody						&body,
					 const std::vector<GradBuffer>			&gradBuffers,
					 int workSizeX = 256, int workSizeY = 1, int workSizeZ = 1) {

	auto mainPos = forwardCode.find("void main()");
	if (mainPos == std::string::npos) {
		throw std::runtime_error("MergeForwardBackward: could not find 'void main()'");
	}

	auto bracePos = forwardCode.find('{', mainPos);
	if (bracePos == std::string::npos) {
		throw std::runtime_error("MergeForwardBackward: could not find main() body opening brace");
	}

	// Gradient buffer declarations
	std::string gradBufDecls;
	for (const auto &gb : gradBuffers) {
		gradBufDecls += std::format(
			"layout(std430, binding = {}) buffer grad_{} {{ {} _grad_{}_data[]; }};\n",
			gb.binding, gb.paramName, gb.glslType, gb.paramName);
	}

	// Adjoint variable declarations
	std::string adjDecls;
	for (const auto &[adjName, glslType] : body.declarations) {
		adjDecls += std::format("    {} {} = {}(0);\n", glslType, adjName, glslType);
	}

	// Adjoint body lines
	std::string adjBody;
	for (const auto &line : body.lines) {
		adjBody += std::format("    {}\n", line);
	}

	// Gradient writebacks
	bool is1D = (workSizeY == 1 && workSizeZ == 1);
	std::string writebacks;
	for (const auto &[paramName, adjName] : body.writebacks) {
		for (const auto &gb : gradBuffers) {
			if (gb.paramName == paramName) {
				if (is1D) {
					writebacks += std::format(
						"    _grad_{}_data[gl_GlobalInvocationID.x] = {};\n",
						gb.paramName, adjName);
				} else {
					writebacks += std::format(
						"    _grad_{}_data[gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x + gl_GlobalInvocationID.x] = {};\n",
						gb.paramName, adjName);
				}
				break;
			}
		}
	}

	// Merge: prefix + gradBufs + main through opening brace + adjoint stuff + rest
	std::string result;
	result.reserve(forwardCode.size() + gradBufDecls.size() + adjDecls.size() +
				   adjBody.size() + writebacks.size() + 200);

	result += forwardCode.substr(0, mainPos);
	result += gradBufDecls;
	if (!gradBufDecls.empty()) result += "\n";
	result += forwardCode.substr(mainPos, bracePos - mainPos + 1);
	result += "\n";
	result += adjDecls;
	if (!adjDecls.empty()) result += "\n    // === Backward pass (auto-generated) ===\n";
	result += adjBody;
	if (!writebacks.empty()) {
		result += "\n    // --- Gradient writebacks ---\n";
		result += writebacks;
	}
	result += forwardCode.substr(bracePos + 1);

	return result;
}

// =============================================================================
// AdjointKernel1D — 1D GPU-executable AD kernel
// =============================================================================

/**
 * A 1D GPU kernel with automatic differentiation.
 *
 * Builds the forward-pass kernel, records the gradient tape, and generates a
 * combined forward+backward shader. The combined shader computes both the loss
 * and its gradients in a single GPU dispatch.
 *
 * The kernel function signature is:
 *   void(Var<int>& threadId, AdjointContext& ctx)
 */
template <typename Func>
class AdjointKernel1D {
public:
	/**
	 * Construct the AD kernel.
	 * @param kernelFunc The DSL kernel function
	 * @param workSizeX Work group size in X (default 256)
	 */
	AdjointKernel1D(Func &&kernelFunc, int workSizeX = 256)
		: _workSizeX(workSizeX) {

		// Phase 1: Build forward kernel while recording gradient tape
		IR::Builder::Builder::Get().SetGradientTape(&_tape);

		_forwardKernel = std::make_unique<Kernel::InspectorKernel1D>(
			[this, func = std::forward<Func>(kernelFunc)](IR::Value::Var<int> &id) {
				AdjointContext ctx(_tape);
				func(id, ctx);
			},
			workSizeX);

		IR::Builder::Builder::Get().SetGradientTape(nullptr);

		_forwardCode = _forwardKernel->GetCode();

		// Phase 2: Generate adjoint body
		AdjointGenerator gen;
		_body = gen.GenerateBody(_tape, true);

		// Phase 3: Build gradient buffer info for registered parameters
		_nextBinding = 10; // Start gradient bindings at a safe offset
		for (const auto &[paramName, glslType] : _tape.Parameters()) {
			GradBuffer gb;
			gb.paramName = paramName;
			gb.glslType = glslType;
			gb.binding = _nextBinding++;
			_gradBuffers.push_back(gb);
		}

		// Phase 4: Merge forward + backward into combined shader
		if (!_body.lines.empty() || !_body.declarations.empty()) {
			_combinedCode = MergeForwardBackward(_forwardCode, _body, _gradBuffers,
												 workSizeX, 1, 1);
		}
	}

	/** Get the forward-only GLSL code. */
	std::string GetForwardCode() const { return _forwardCode; }

	/** Get the combined forward+backward GLSL code. */
	std::string GetCombinedCode() const { return _combinedCode; }

	/** Get the backward body GLSL (adjoint declarations + lines). */
	std::string GetBackwardBodyCode() const {
		std::string s;
		for (const auto &[name, type] : _body.declarations) {
			s += std::format("{} {};\n", type, name);
		}
		for (const auto &line : _body.lines) {
			s += line + "\n";
		}
		return s;
	}

	/** Access the gradient tape (for debugging). */
	const GradientTape &Tape() const { return _tape; }

	/** Get the adjoint body parts. */
	const AdjointBody &Body() const { return _body; }

	/** Get the gradient buffer list. */
	const std::vector<GradBuffer> &GradBuffers() const { return _gradBuffers; }

	/** Check if a combined shader was generated. */
	bool HasCombinedCode() const { return !_combinedCode.empty(); }

private:
	int _workSizeX;
	std::unique_ptr<Kernel::InspectorKernel1D> _forwardKernel;
	std::string _forwardCode;
	std::string _combinedCode;
	GradientTape _tape;
	AdjointBody _body;
	std::vector<GradBuffer> _gradBuffers;
	uint32_t _nextBinding = 0;
};

// =============================================================================
// AdjointKernel2D — 2D GPU-executable AD kernel
// =============================================================================

/**
 * A 2D GPU kernel with automatic differentiation.
 * The kernel function signature is:
 *   void(Var<int>& idX, Var<int>& idY, AdjointContext& ctx)
 */
template <typename Func>
class AdjointKernel2D {
public:
	AdjointKernel2D(Func &&kernelFunc, int workSizeX = 16, int workSizeY = 16)
		: _workSizeX(workSizeX), _workSizeY(workSizeY) {

		IR::Builder::Builder::Get().SetGradientTape(&_tape);

		_forwardKernel = std::make_unique<Kernel::InspectorKernel2D>(
			[this, func = std::forward<Func>(kernelFunc)](IR::Value::Var<int> &idX,
														  IR::Value::Var<int> &idY) {
				AdjointContext ctx(_tape);
				func(idX, idY, ctx);
			},
			workSizeX, workSizeY);

		IR::Builder::Builder::Get().SetGradientTape(nullptr);

		_forwardCode = _forwardKernel->GetCode();

		AdjointGenerator gen;
		_body = gen.GenerateBody(_tape, true);

		_nextBinding = 10;
		for (const auto &[paramName, glslType] : _tape.Parameters()) {
			GradBuffer gb;
			gb.paramName = paramName;
			gb.glslType = glslType;
			gb.binding = _nextBinding++;
			_gradBuffers.push_back(gb);
		}

		if (!_body.lines.empty() || !_body.declarations.empty()) {
			_combinedCode = MergeForwardBackward(_forwardCode, _body, _gradBuffers,
												 workSizeX, workSizeY, 1);
		}
	}

	std::string GetForwardCode() const { return _forwardCode; }
	std::string GetCombinedCode() const { return _combinedCode; }
	const GradientTape &Tape() const { return _tape; }
	bool HasCombinedCode() const { return !_combinedCode.empty(); }

private:
	int _workSizeX, _workSizeY;
	std::unique_ptr<Kernel::InspectorKernel2D> _forwardKernel;
	std::string _forwardCode;
	std::string _combinedCode;
	GradientTape _tape;
	AdjointBody _body;
	std::vector<GradBuffer> _gradBuffers;
	uint32_t _nextBinding = 0;
};

// =============================================================================
// AdjointKernel3D — 3D GPU-executable AD kernel
// =============================================================================

/**
 * A 3D GPU kernel with automatic differentiation.
 * The kernel function signature is:
 *   void(Var<int>& idX, Var<int>& idY, Var<int>& idZ, AdjointContext& ctx)
 */
template <typename Func>
class AdjointKernel3D {
public:
	AdjointKernel3D(Func &&kernelFunc, int workSizeX = 8, int workSizeY = 8, int workSizeZ = 4)
		: _workSizeX(workSizeX), _workSizeY(workSizeY), _workSizeZ(workSizeZ) {

		IR::Builder::Builder::Get().SetGradientTape(&_tape);

		_forwardKernel = std::make_unique<Kernel::InspectorKernel3D>(
			[this, func = std::forward<Func>(kernelFunc)](IR::Value::Var<int> &idX,
														  IR::Value::Var<int> &idY,
														  IR::Value::Var<int> &idZ) {
				AdjointContext ctx(_tape);
				func(idX, idY, idZ, ctx);
			},
			workSizeX, workSizeY, workSizeZ);

		IR::Builder::Builder::Get().SetGradientTape(nullptr);

		_forwardCode = _forwardKernel->GetCode();

		AdjointGenerator gen;
		_body = gen.GenerateBody(_tape, true);

		_nextBinding = 10;
		for (const auto &[paramName, glslType] : _tape.Parameters()) {
			GradBuffer gb;
			gb.paramName = paramName;
			gb.glslType = glslType;
			gb.binding = _nextBinding++;
			_gradBuffers.push_back(gb);
		}

		if (!_body.lines.empty() || !_body.declarations.empty()) {
			_combinedCode = MergeForwardBackward(_forwardCode, _body, _gradBuffers,
												 workSizeX, workSizeY, workSizeZ);
		}
	}

	std::string GetForwardCode() const { return _forwardCode; }
	std::string GetCombinedCode() const { return _combinedCode; }
	const GradientTape &Tape() const { return _tape; }
	bool HasCombinedCode() const { return !_combinedCode.empty(); }

private:
	int _workSizeX, _workSizeY, _workSizeZ;
	std::unique_ptr<Kernel::InspectorKernel3D> _forwardKernel;
	std::string _forwardCode;
	std::string _combinedCode;
	GradientTape _tape;
	AdjointBody _body;
	std::vector<GradBuffer> _gradBuffers;
	uint32_t _nextBinding = 0;
};

} // namespace GPU::AD

#endif // EASYGPU_AD_ADJOINTKERNEL_H
