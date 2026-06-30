/**
 * @file FragmentShader.cpp
 * @brief Fullscreen fragment shader implementation.
 */

#include <Kernel/FragmentShader.h>

#include <Runtime/Context.h>

#include <stdexcept>

namespace GPU::Kernel {

FragmentShader::FragmentShader(const FragmentFunc &func, uint32_t width, uint32_t height)
	: _name("FragmentShader"), _width(width), _height(height) {
	BuildShaders(func);
}

FragmentShader::FragmentShader(const std::string &name, const FragmentFunc &func, uint32_t width, uint32_t height)
	: _name(name), _width(width), _height(height) {
	BuildShaders(func);
}

FragmentShader::~FragmentShader() {
	auto *backend = Runtime::Context::GetBackend();
	if (!backend) {
		return;
	}
	if (_pipelineHandle != Backend::INVALID_PIPELINE_HANDLE) {
		backend->DestroyPipeline(_pipelineHandle);
		_pipelineHandle = Backend::INVALID_PIPELINE_HANDLE;
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

void FragmentShader::SetName(const std::string &name) {
	_name = name;
}
std::string FragmentShader::GetName() const {
	return _name;
}

void FragmentShader::SetResolution(uint32_t width, uint32_t height) {
	if (_width != width || _height != height) {
		_width	  = width;
		_height	  = height;
		_compiled = false; // Re-compile needed only if resolution is baked into shader
	}
}

std::string FragmentShader::GetShaderSource() {
	if (_context)
		return _context->GetCompleteCode();
	return "";
}

void FragmentShader::BuildShaders(const FragmentFunc &func) {
	_context  = std::make_unique<GraphicsBuildContext>();

	// --- Vertex Shader (hardcoded fullscreen triangle) ---
	_vsSource = std::string("#version 450 core\n\n"
							"const vec2 _verts[3] = vec2[](\n"
							"    vec2(-1.0, -1.0),\n"
							"    vec2( 3.0, -1.0),\n"
							"    vec2(-1.0,  3.0)\n"
							");\n\n"
							"layout(location=0) out vec2 v_uv;\n\n"
							"void main() {\n"
							"    vec2 pos = _verts[gl_VertexIndex];\n"
							"    v_uv = pos * 0.5 + 0.5;\n"
							"    gl_Position = vec4(pos, 0.0, 1.0);\n"
							"}\n");

	// --- Fragment Shader ---
	{
		IR::Builder::Builder			&builder = IR::Builder::Builder::Get();
		IR::Builder::Builder::ScopedBind bindGuard(builder, *_context);

		// Register the one varying we use
		_context->RegisterVarying("v_uv", "vec2");

		// Create DSL variables for the user
		IR::Value::Var<GPU::Math::Vec2> fragCoord("v_uv", true); // fragCoord = v_uv * resolution
		IR::Value::Var<GPU::Math::Vec4> fragColor("fragColor", true);

		func(fragCoord, fragColor);
	}
	_fsSource = _context->GetFragmentShaderCode();
}

void FragmentShader::EnsureCompiled() {
	if (_compiled)
		return;

	auto *backend = Runtime::Context::GetBackend();
	if (!backend || !backend->IsInitialized()) {
		throw std::runtime_error("Backend not initialized");
	}

	// Compile vertex shader
	Backend::ShaderDesc vsDesc;
	vsDesc.type		  = Backend::ShaderType::Vertex;
	vsDesc.sourceCode = _vsSource;
	vsDesc.entryPoint = "main";
	_vsHandle		  = backend->CreateShader(vsDesc);

	// Compile fragment shader
	Backend::ShaderDesc fsDesc;
	fsDesc.type		  = Backend::ShaderType::Fragment;
	fsDesc.sourceCode = _fsSource;
	fsDesc.entryPoint = "main";
	_fsHandle		  = backend->CreateShader(fsDesc);

	// Create graphics pipeline
	Backend::GraphicsPipelineDesc pipeDesc;
	pipeDesc.vertexShader		   = _vsHandle;
	pipeDesc.fragmentShader		   = _fsHandle;
	pipeDesc.topology			   = Backend::PrimitiveTopology::TriangleList;
	pipeDesc.colorAttachmentFormat = Backend::PixelFormat::RGBA8;

	// Copy resources from build context
	if (_context) {
		for (const auto &buf : _context->GetBufferInfos()) {
			Backend::ResourceLayoutEntry res;
			res.binding = buf.binding;
			res.type	= Backend::BindingType::Buffer;
			res.readOnly = (buf.mode == Backend::BUFFER_MODE_READ_ONLY);
			res.stageFlags = Backend::ResourceStageVertex | Backend::ResourceStageFragment;
			pipeDesc.resources.push_back(res);
		}
		for (const auto &tex : _context->GetTextureInfos()) {
			Backend::ResourceLayoutEntry res;
			res.binding = tex.binding;
			res.type	= tex.sampled ? Backend::BindingType::Sampler : Backend::BindingType::Texture;
			res.format	= static_cast<Backend::PixelFormat>(tex.format);
			res.readOnly = tex.sampled;
			res.stageFlags = Backend::ResourceStageFragment;
			pipeDesc.resources.push_back(res);
		}
		pipeDesc.pushConstantSize = _context->GetPushConstantSize();
	}

	_pipelineHandle = backend->CreateGraphicsPipeline(pipeDesc);
	_compiled		= true;
}

void FragmentShader::RenderToTexture(Backend::TextureHandle handle, uint32_t w, uint32_t h, bool sync) {
	auto *backend = Runtime::Context::GetBackend();
	if (!backend)
		throw std::runtime_error("Backend not available");

	Backend::RenderPassBeginDesc rpDesc;
	rpDesc.colorAttachment = handle;
	rpDesc.clearColorFlag  = true;
	rpDesc.clearColor[0]   = 0.0f;
	rpDesc.clearColor[1]   = 0.0f;
	rpDesc.clearColor[2]   = 0.0f;
	rpDesc.clearColor[3]   = 1.0f;

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

	backend->BindPipeline(_pipelineHandle);

	if (!bindings.empty()) {
		backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
	}

	// Upload push constants
	if (_context && _context->GetPushConstantSize() > 0) {
		_context->UploadUniformValues(_pipelineHandle);
	}

	backend->BeginRendering(rpDesc);
	backend->SetViewport(0, 0, w, h);
	backend->SetScissor(0, 0, w, h);
	if (!bindings.empty()) {
		backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
	}

	backend->Draw(3, 1, 0, 0); // Fullscreen triangle = 3 vertices
	backend->EndRendering();

	if (sync)
		backend->Finish();
}

} // namespace GPU::Kernel
