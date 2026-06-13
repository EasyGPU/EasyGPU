#pragma once

/**
 * @file ActiveCompaction.h
 * @brief GPU active-list compaction for sparse workloads.
 */

#ifndef EASYGPU_UTILITY_ACTIVE_COMPACTION_H
#define EASYGPU_UTILITY_ACTIVE_COMPACTION_H

#include <Backend/Backend.h>
#include <Runtime/Buffer.h>
#include <Runtime/Context.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace GPU::Utility {

/**
 * @brief Compacts an integer activity mask into a dense active index list.
 *
 * The compaction pass runs entirely on the GPU:
 *
 *   activeCount = 0
 *   if (mask[i] != 0) activeIndices[atomicAdd(activeCount, 1)] = i
 *
 * Follow-up kernels can process activeIndices instead of running expensive work
 * for every pixel/ray/sample. Without indirect dispatch support, callers can
 * either dispatch a conservative upper bound and read activeCount in-shader, or
 * download CountBuffer() occasionally to choose a tighter dispatch size.
 */
class ActiveCompaction {
public:
	explicit ActiveCompaction(size_t maxElements)
		: maxElements_(maxElements), count_(std::vector<int>{0}, Runtime::BufferMode::ReadWrite),
		  indices_(maxElements, Runtime::BufferMode::ReadWrite) {
		if (maxElements == 0)
			throw std::invalid_argument("ActiveCompaction requires maxElements > 0");
	}

	~ActiveCompaction() {
		Release();
	}

	ActiveCompaction(const ActiveCompaction &)				  = delete;
	ActiveCompaction	 &operator=(const ActiveCompaction &) = delete;

	Runtime::Buffer<int> &CountBuffer() {
		return count_;
	}
	Runtime::Buffer<int> &IndicesBuffer() {
		return indices_;
	}
	const Runtime::Buffer<int> &CountBuffer() const {
		return count_;
	}
	const Runtime::Buffer<int> &IndicesBuffer() const {
		return indices_;
	}

	size_t MaxElements() const {
		return maxElements_;
	}

	/**
	 * @brief Compact a 0/1 mask into the internal index list.
	 * @param activeMask Integer mask buffer with at least elementCount entries.
	 * @param elementCount Number of source elements to scan.
	 * @param sync If true, wait for completion before returning.
	 */
	void Compact(Runtime::Buffer<int> &activeMask, size_t elementCount, bool sync = false) {
		if (elementCount > maxElements_)
			throw std::out_of_range("ActiveCompaction::Compact elementCount exceeds maxElements");
		EnsureClearPipeline();
		EnsureCompactPipeline(elementCount);

		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			throw std::runtime_error("ActiveCompaction backend not available");

		backend->BindPipeline(clearPipeline_);
		Backend::ResourceBinding clearBinding;
		clearBinding.binding  = 0;
		clearBinding.type	  = Backend::BindingType::Buffer;
		clearBinding.buffer	  = count_.GetHandle();
		clearBinding.readOnly = false;
		backend->BindResources(&clearBinding, 1);
		backend->Dispatch(1, 1, 1);
		backend->MemoryBarrier(Backend::BarrierType::Buffer);

		Backend::ResourceBinding bindings[3] = {};
		bindings[0].binding					 = 0;
		bindings[0].type					 = Backend::BindingType::Buffer;
		bindings[0].buffer					 = activeMask.GetHandle();
		bindings[0].readOnly				 = true;
		bindings[1].binding					 = 1;
		bindings[1].type					 = Backend::BindingType::Buffer;
		bindings[1].buffer					 = indices_.GetHandle();
		bindings[1].readOnly				 = false;
		bindings[2].binding					 = 2;
		bindings[2].type					 = Backend::BindingType::Buffer;
		bindings[2].buffer					 = count_.GetHandle();
		bindings[2].readOnly				 = false;

		backend->BindPipeline(compactPipeline_);
		backend->BindResources(bindings, 3);
		backend->Dispatch(static_cast<uint32_t>((elementCount + 255) / 256), 1, 1);
		backend->MemoryBarrier(Backend::BarrierType::Buffer);
		if (sync)
			backend->Finish();
	}

	int DownloadCount() {
		std::vector<int> out(1, 0);
		count_.Download(out);
		return out[0];
	}

	std::vector<int> DownloadIndices(size_t count) {
		count = std::min(count, maxElements_);
		std::vector<int> out(count, 0);
		if (count > 0)
			indices_.Download(out.data(), count);
		return out;
	}

private:
	void EnsureClearPipeline() {
		if (clearPipeline_ != Backend::INVALID_PIPELINE_HANDLE)
			return;

		Runtime::Context::GetInstance().MakeCurrent();
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			throw std::runtime_error("ActiveCompaction backend not available");

		Backend::ShaderDesc shaderDesc;
		shaderDesc.type		  = Backend::ShaderType::Compute;
		shaderDesc.sourceCode = R"GLSL(#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer CountBuf { int activeCount[]; };
void main() { activeCount[0] = 0; }
)GLSL";
		clearShader_		  = backend->CreateShader(shaderDesc);

		Backend::PipelineDesc pipelineDesc;
		pipelineDesc.computeShader	= clearShader_;
		pipelineDesc.workGroupSizeX = 1;
		pipelineDesc.resources.push_back({0, Backend::BindingType::Buffer, Backend::PixelFormat::RGBA8, false});
		clearPipeline_ = backend->CreatePipeline(pipelineDesc);
		if (clearPipeline_ == Backend::INVALID_PIPELINE_HANDLE)
			throw std::runtime_error("ActiveCompaction failed to create clear pipeline");
	}

	void EnsureCompactPipeline(size_t elementCount) {
		if (compactPipeline_ != Backend::INVALID_PIPELINE_HANDLE && compiledElements_ == elementCount)
			return;

		Runtime::Context::GetInstance().MakeCurrent();
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			throw std::runtime_error("ActiveCompaction backend not available");

		if (compactPipeline_ != Backend::INVALID_PIPELINE_HANDLE)
			backend->DestroyPipeline(compactPipeline_);
		if (compactShader_ != Backend::INVALID_SHADER_HANDLE)
			backend->DestroyShader(compactShader_);

		Backend::ShaderDesc shaderDesc;
		shaderDesc.type			= Backend::ShaderType::Compute;
		shaderDesc.sourceCode	= std::string(R"GLSL(#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer MaskBuf { int mask[]; };
layout(std430, binding = 1) buffer IndicesBuf { int activeIndices[]; };
layout(std430, binding = 2) buffer CountBuf { int activeCount[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= ELEMENT_COUNTu) return;
    if (mask[i] != 0) {
        int dst = atomicAdd(activeCount[0], 1);
        activeIndices[dst] = int(i);
    }
}
)GLSL");
		const std::string token = "ELEMENT_COUNT";
		const auto		  pos	= shaderDesc.sourceCode.find(token);
		shaderDesc.sourceCode.replace(pos, token.size(), std::to_string(elementCount));
		compactShader_ = backend->CreateShader(shaderDesc);

		Backend::PipelineDesc pipelineDesc;
		pipelineDesc.computeShader	= compactShader_;
		pipelineDesc.workGroupSizeX = 256;
		pipelineDesc.resources.push_back({0, Backend::BindingType::Buffer, Backend::PixelFormat::RGBA8, true});
		pipelineDesc.resources.push_back({1, Backend::BindingType::Buffer, Backend::PixelFormat::RGBA8, false});
		pipelineDesc.resources.push_back({2, Backend::BindingType::Buffer, Backend::PixelFormat::RGBA8, false});
		compactPipeline_ = backend->CreatePipeline(pipelineDesc);
		if (compactPipeline_ == Backend::INVALID_PIPELINE_HANDLE)
			throw std::runtime_error("ActiveCompaction failed to create compact pipeline");

		compiledElements_ = elementCount;
	}

	void Release() {
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			return;
		if (clearPipeline_ != Backend::INVALID_PIPELINE_HANDLE) {
			backend->DestroyPipeline(clearPipeline_);
			clearPipeline_ = Backend::INVALID_PIPELINE_HANDLE;
		}
		if (clearShader_ != Backend::INVALID_SHADER_HANDLE) {
			backend->DestroyShader(clearShader_);
			clearShader_ = Backend::INVALID_SHADER_HANDLE;
		}
		if (compactPipeline_ != Backend::INVALID_PIPELINE_HANDLE) {
			backend->DestroyPipeline(compactPipeline_);
			compactPipeline_ = Backend::INVALID_PIPELINE_HANDLE;
		}
		if (compactShader_ != Backend::INVALID_SHADER_HANDLE) {
			backend->DestroyShader(compactShader_);
			compactShader_ = Backend::INVALID_SHADER_HANDLE;
		}
	}

	size_t					maxElements_	  = 0;
	size_t					compiledElements_ = 0;
	Runtime::Buffer<int>	count_;
	Runtime::Buffer<int>	indices_;
	Backend::ShaderHandle	clearShader_	 = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle clearPipeline_	 = Backend::INVALID_PIPELINE_HANDLE;
	Backend::ShaderHandle	compactShader_	 = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle compactPipeline_ = Backend::INVALID_PIPELINE_HANDLE;
};

} // namespace GPU::Utility

#endif // EASYGPU_UTILITY_ACTIVE_COMPACTION_H
