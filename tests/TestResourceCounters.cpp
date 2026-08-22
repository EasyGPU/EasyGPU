#include <Backend/Backend.h>
#include <Runtime/Context.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
	GPU::Runtime::AutoInitContext();
	auto *backend = GPU::Runtime::Context::GetBackend();
	assert(backend != nullptr);

	const auto baseline = backend->GetResourceCounters();
	if (GPU::Runtime::Context::GetInstance().GetBackendType() != GPU::Backend::BackendType::Vulkan) {
		assert(!baseline.trackingSupported);
		std::cout << "Native resource counters are Vulkan-only; skipped\n";
		return 0;
	}
	assert(baseline.trackingSupported);

	GPU::Backend::BufferDesc bufferDesc{};
	bufferDesc.sizeInBytes			 = 64;
	bufferDesc.mode					 = GPU::Backend::BufferMode::ReadWrite;
	const auto				  buffer = backend->CreateBuffer(bufferDesc);

	GPU::Backend::TextureDesc textureDesc{};
	textureDesc.width	 = 4;
	textureDesc.height	 = 4;
	textureDesc.format	 = GPU::Backend::PixelFormat::RGBA8;
	const auto texture	 = backend->CreateTexture(textureDesc);

	const auto allocated = backend->GetResourceCounters();
	assert(allocated.liveBufferHandles == baseline.liveBufferHandles + 1);
	assert(allocated.liveTextureHandles == baseline.liveTextureHandles + 1);
	const auto bufferAllocation = backend->GetBufferAllocationInfo(buffer);
	const auto textureAllocation = backend->GetTextureAllocationInfo(texture);
	assert(bufferAllocation.available);
	assert(bufferAllocation.physicalBytes >= bufferDesc.sizeInBytes);
	assert(bufferAllocation.allocationGroup != 0);
	assert(textureAllocation.available);
	assert(textureAllocation.physicalBytes >= 4u * 4u * 4u);
	assert(textureAllocation.allocationGroup != 0);
	assert(textureAllocation.allocationGroup != bufferAllocation.allocationGroup);

	const auto submission = backend->Submit();
	const auto submitted  = backend->GetResourceCounters();
	assert(submitted.liveSubmissionHandles == baseline.liveSubmissionHandles + 1);
	assert(backend->WaitForSubmission(submission, std::numeric_limits<uint64_t>::max()));
	backend->ReleaseSubmission(submission);

	backend->DestroyTexture(texture);
	backend->DestroyBuffer(buffer);
	const auto released = backend->GetResourceCounters();
	assert(released.liveBufferHandles == baseline.liveBufferHandles);
	assert(released.liveTextureHandles == baseline.liveTextureHandles);
	assert(released.liveSubmissionHandles == baseline.liveSubmissionHandles);
	bool rejectedDestroyedBuffer = false;
	try {
		(void)backend->GetBufferAllocationInfo(buffer);
	} catch (const std::runtime_error &) {
		rejectedDestroyedBuffer = true;
	}
	assert(rejectedDestroyedBuffer);
	bool rejectedDestroyedTexture = false;
	try {
		(void)backend->GetTextureAllocationInfo(texture);
	} catch (const std::runtime_error &) {
		rejectedDestroyedTexture = true;
	}
	assert(rejectedDestroyedTexture);

	if (backend->GetCaps().supportsGraphics) {
		const char *vertexSource =
			"#version 450\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
		const char *fragmentSource =
			"#version 450\nlayout(location = 0) out vec4 color;\n"
			"void main() { color = vec4(1.0); }\n";
		GPU::Backend::ShaderDesc vertexDesc{GPU::Backend::ShaderType::Vertex, vertexSource, "main"};
		GPU::Backend::ShaderDesc fragmentDesc{GPU::Backend::ShaderType::Fragment, fragmentSource, "main"};
		const auto vertexShader = backend->CreateShader(vertexDesc);
		const auto fragmentShader = backend->CreateShader(fragmentDesc);
		GPU::Backend::GraphicsPipelineDesc pipelineDesc{};
		pipelineDesc.vertexShader = vertexShader;
		pipelineDesc.fragmentShader = fragmentShader;
		pipelineDesc.colorAttachmentFormat = GPU::Backend::PixelFormat::RGBA8;
		const auto pipeline = backend->CreateGraphicsPipeline(pipelineDesc);

		GPU::Backend::TextureDesc targetDesc{};
		targetDesc.width = 4;
		targetDesc.height = 4;
		targetDesc.format = GPU::Backend::PixelFormat::RGBA8;
		const auto target = backend->CreateTexture(targetDesc);
		GPU::Backend::RenderPassBeginDesc renderPass{};
		renderPass.colorAttachment = target;
		renderPass.clearColorFlag = true;
		backend->BeginRendering(renderPass);
		backend->SetViewport(0, 0, targetDesc.width, targetDesc.height);
		backend->SetScissor(0, 0, targetDesc.width, targetDesc.height);
		backend->BindPipeline(pipeline);
		backend->Draw(3, 1, 0, 0);
		backend->EndRendering();
		const auto graphicsSubmission = backend->Submit();

		const auto beforeDestroy = backend->GetOperationCounters();
		backend->DestroyPipeline(pipeline);
		backend->DestroyShader(vertexShader);
		backend->DestroyShader(fragmentShader);
		const auto afterDestroy = backend->GetOperationCounters();
		assert(afterDestroy.globalDrainCalls == beforeDestroy.globalDrainCalls);
		assert(afterDestroy.blockingSubmissionWaitCalls == beforeDestroy.blockingSubmissionWaitCalls);
		const auto deferred = backend->GetResourceCounters();
		assert(deferred.livePipelineHandles == baseline.livePipelineHandles + 1);
		assert(deferred.liveShaderHandles == baseline.liveShaderHandles);
		bool rejectedDestroyedPipeline = false;
		try {
			backend->BindPipeline(pipeline);
		} catch (const std::runtime_error &) {
			rejectedDestroyedPipeline = true;
		}
		assert(rejectedDestroyedPipeline);

		assert(backend->WaitForSubmission(graphicsSubmission, std::numeric_limits<uint64_t>::max()));
		const auto completed = backend->GetResourceCounters();
		assert(completed.livePipelineHandles == baseline.livePipelineHandles);
		backend->ReleaseSubmission(graphicsSubmission);
		backend->DestroyTexture(target);
	}

	std::cout << "Native resource counters track allocation and release\n";
	return 0;
}
