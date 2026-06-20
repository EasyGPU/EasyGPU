/**
 * @file GraphicsPipeline.cpp
 * @brief Graphics pipeline implementation (non-template methods only).
 */

#include <Kernel/GraphicsPipeline.h>

#include <Runtime/Context.h>
#include <iostream>

#include <stdexcept>
#include <utility>

namespace GPU::Kernel {

GraphicsPipeline::~GraphicsPipeline() {
	auto *backend = Runtime::Context::GetBackend();
	if (!backend) {
		return;
	}
	if (_pipeline != Backend::INVALID_PIPELINE_HANDLE) {
		backend->DestroyPipeline(_pipeline);
		_pipeline = Backend::INVALID_PIPELINE_HANDLE;
	}
	if (_fsHandle != Backend::INVALID_SHADER_HANDLE) {
		backend->DestroyShader(_fsHandle);
		_fsHandle = Backend::INVALID_SHADER_HANDLE;
	}
	if (_vsHandle != Backend::INVALID_SHADER_HANDLE) {
		backend->DestroyShader(_vsHandle);
		_vsHandle = Backend::INVALID_SHADER_HANDLE;
	}
}

void GraphicsPipeline::EnsureCompiled() {
	if (_compiled)
		return;

	auto *backend = Runtime::Context::GetBackend();
	if (!backend || !backend->IsInitialized()) {
		throw std::runtime_error("Backend not initialized");
	}

	// Compile vertex shader
	Backend::ShaderDesc vsDesc;
	vsDesc.type		  = Backend::ShaderType::Vertex;
	vsDesc.sourceCode = _vsCode;
	vsDesc.entryPoint = "main";
	_vsHandle		  = backend->CreateShader(vsDesc);

	// Compile fragment shader
	Backend::ShaderDesc fsDesc;
	fsDesc.type		  = Backend::ShaderType::Fragment;
	fsDesc.sourceCode = _fsCode;
	fsDesc.entryPoint = "main";
	_fsHandle		  = backend->CreateShader(fsDesc);

	// Create graphics pipeline
	Backend::GraphicsPipelineDesc pipeDesc;
	pipeDesc.vertexShader		   = _vsHandle;
	pipeDesc.fragmentShader		   = _fsHandle;
	pipeDesc.topology			   = Backend::PrimitiveTopology::TriangleList;
	pipeDesc.colorAttachmentFormat =
		_colorAttachmentFormats.empty() ? Backend::PixelFormat::RGBA8 : _colorAttachmentFormats.front();
	pipeDesc.colorAttachmentFormats = _colorAttachmentFormats;
	if (pipeDesc.colorAttachmentFormats.empty()) {
		pipeDesc.colorAttachmentFormats.assign(_colorOutputCount, Backend::PixelFormat::RGBA8);
	}
	pipeDesc.depthTestEnable	   = true;
	pipeDesc.depthWriteEnable	   = true;

	if (!_vertexLayout.empty()) {
		pipeDesc.vertexLayout = _vertexLayout;
	}

	if (_context) {
		for (const auto &buf : _context->GetBufferInfos()) {
			Backend::ResourceLayoutEntry res;
			res.binding	 = buf.binding;
			res.type	 = Backend::BindingType::Buffer;
			res.readOnly = (buf.mode == Backend::BUFFER_MODE_READ_ONLY);
			pipeDesc.resources.push_back(res);
		}
		for (const auto &tex : _context->GetTextureInfos()) {
			Backend::ResourceLayoutEntry res;
			res.binding = tex.binding;
			res.type	= tex.sampled ? Backend::BindingType::Sampler : Backend::BindingType::Texture;
			res.format	= static_cast<Backend::PixelFormat>(tex.format);
			pipeDesc.resources.push_back(res);
		}
	}
	pipeDesc.pushConstantSize = _context->GetPushConstantSize();

	_pipeline				  = backend->CreateGraphicsPipeline(pipeDesc);
	_compiled				  = true;
}

std::pair<uint32_t, uint32_t>
GraphicsPipeline::ValidateRenderTargets(const std::vector<RenderTargetAttachment> &renderTargets) {
	if (renderTargets.empty()) {
		throw std::runtime_error("GraphicsPipeline::Draw requires at least one render target");
	}
	if (renderTargets.size() > Backend::MAX_COLOR_ATTACHMENTS) {
		throw std::runtime_error("GraphicsPipeline::Draw render target count exceeds MAX_COLOR_ATTACHMENTS");
	}

	const uint32_t width  = renderTargets.front().width;
	const uint32_t height = renderTargets.front().height;
	if (width == 0 || height == 0) {
		throw std::runtime_error("GraphicsPipeline::Draw requires non-zero render target dimensions");
	}

	for (const auto &target : renderTargets) {
		if (target.handle == Backend::INVALID_TEXTURE_HANDLE) {
			throw std::runtime_error("GraphicsPipeline::Draw received an invalid render target");
		}
		if (target.width != width || target.height != height) {
			throw std::runtime_error("GraphicsPipeline::Draw MRT render targets must have identical dimensions");
		}
	}

	return {width, height};
}

std::vector<IR::Value::Var<GPU::Math::Vec4>> GraphicsPipeline::MakeFragmentColorOutputs(uint32_t colorOutputCount) {
	if (colorOutputCount == 0 || colorOutputCount > Backend::MAX_COLOR_ATTACHMENTS) {
		throw std::runtime_error("GraphicsPipeline MRT color output count must be between 1 and MAX_COLOR_ATTACHMENTS");
	}

	std::vector<IR::Value::Var<GPU::Math::Vec4>> outputs;
	outputs.reserve(colorOutputCount);
	for (uint32_t i = 0; i < colorOutputCount; ++i) {
		outputs.emplace_back(std::format("fragColor{}", i), true);
	}
	return outputs;
}

void GraphicsPipeline::DrawImpl(const std::vector<RenderTargetAttachment> &renderTargets,
								Backend::TextureHandle depthHandle, uint32_t width, uint32_t height,
								uint32_t vertexCount, uint32_t indexCount, bool indexed, bool sync) {
	auto *backend = Runtime::Context::GetBackend();
	if (!backend)
		throw std::runtime_error("Backend not available");
	if (renderTargets.size() != _colorOutputCount) {
		throw std::runtime_error("GraphicsPipeline::Draw render target count must match fragment output count");
	}
	std::vector<Backend::PixelFormat> drawFormats;
	drawFormats.reserve(renderTargets.size());
	for (const auto &target : renderTargets) {
		drawFormats.push_back(target.format);
	}
	if (_compiled) {
		if (drawFormats != _colorAttachmentFormats) {
			throw std::runtime_error("GraphicsPipeline::Draw render target formats differ from compiled pipeline");
		}
	} else {
		_colorAttachmentFormats = std::move(drawFormats);
	}
	EnsureCompiled();
	if (!_simpleVertexInput && vertexCount > 0 && !_hasVertexBuffer) {
		throw std::runtime_error("GraphicsPipeline::Draw requires a vertex buffer for this pipeline");
	}
	if (indexed && !_hasIndexBuffer) {
		throw std::runtime_error("GraphicsPipeline::DrawIndexed requires an index buffer");
	}

	Backend::RenderPassBeginDesc rpDesc;
	rpDesc.colorAttachment = renderTargets.front().handle;
	rpDesc.colorAttachments.reserve(renderTargets.size());
	for (const auto &target : renderTargets) {
		rpDesc.colorAttachments.push_back(target.handle);
	}
	rpDesc.clearColorFlag  = true;
	rpDesc.clearColor[0]   = 0.0f; // blue for debug
	rpDesc.clearColor[1]   = 0.0f;
	rpDesc.clearColor[2]   = 0.0f;
	rpDesc.clearColor[3]   = 1.0f;
	rpDesc.depthAttachment = depthHandle;
	rpDesc.clearDepthFlag  = (depthHandle != Backend::INVALID_TEXTURE_HANDLE);
	rpDesc.clearDepth	   = 1.0f;

	backend->BeginRendering(rpDesc);
	backend->SetViewport(0, 0, width, height);
	backend->SetScissor(0, 0, width, height);
	backend->BindPipeline(_pipeline);
	if (_hasVertexBuffer) {
		backend->BindVertexBuffer(_vertexBufferHandle, _vertexStride);
	}
	if (indexed) {
		backend->BindIndexBuffer(_indexBufferHandle);
	}

	// Bind resources
	std::vector<Backend::ResourceBinding> bindings;
	if (_context) {
		for (const auto &[binding, bufHandle] : _context->GetRuntimeBufferBindings()) {
			Backend::ResourceBinding rb;
			rb.binding = binding;
			rb.type	   = Backend::BindingType::Buffer;
			rb.buffer  = bufHandle;
			bindings.push_back(rb);
		}
		for (const auto &[binding, texHandle] : _context->GetRuntimeTextureBindings()) {
			Backend::ResourceBinding rb;
			rb.binding = binding;
			rb.type	   = Backend::BindingType::Sampler;
			rb.texture = texHandle;
			bindings.push_back(rb);
		}
	}
	if (!bindings.empty()) {
		backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
	}

	// Upload push constants
	if (_context && _context->GetPushConstantSize() > 0) {
		_context->UploadUniformValues(_pipeline);
	}

	if (indexed) {
		backend->DrawIndexed(indexCount, 1, 0, 0, 0);
	} else {
		backend->Draw(vertexCount, 1, 0, 0);
	}

	backend->EndRendering();

	if (sync)
		backend->Finish();
}

} // namespace GPU::Kernel
