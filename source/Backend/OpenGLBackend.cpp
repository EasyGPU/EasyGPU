/**
 * @file OpenGLBackend.cpp
 * @brief OpenGL backend implementation.
 */

#include <Backend/OpenGLBackend.h>

#include <GLAD/glad.h>
#include <cstring>
#include <iostream>

// Platform-specific includes
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
// Linux: Use system X11/GLX headers directly
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

namespace GPU::Backend {

OpenGLBackend::OpenGLBackend() {
	std::fill(_boundBuffers.begin(), _boundBuffers.end(), 0);
	std::fill(_boundTextures.begin(), _boundTextures.end(), 0);
	std::fill(_boundTextureFormats.begin(), _boundTextureFormats.end(), 0);
}

OpenGLBackend::~OpenGLBackend() {
	if (_initialized) {
		Shutdown();
	}
}

void OpenGLBackend::Initialize() {
	if (_initialized) {
		return;
	}

	try {
		InitializePlatform();
		_initialized = true;
	} catch (const std::exception &e) {
		CleanupPlatform();
		throw std::runtime_error(std::string("Failed to initialize OpenGL backend: ") + e.what());
	}
}

void OpenGLBackend::Shutdown() {
	if (!_initialized) {
		return;
	}

	for (auto &[handle, info] : _pipelines) {
		if (info.glProgram != 0) {
			glDeleteProgram(info.glProgram);
		}
	}
	_pipelines.clear();

	for (auto &[handle, info] : _shaders) {
		if (info.glHandle != 0) {
			glDeleteShader(info.glHandle);
		}
	}
	_shaders.clear();

	for (auto &[handle, info] : _textures) {
		if (info.glHandle != 0) {
			glDeleteTextures(1, &info.glHandle);
		}
	}
	_textures.clear();

	for (auto &[handle, info] : _buffers) {
		if (info.glHandle != 0) {
			glDeleteBuffers(1, &info.glHandle);
		}
	}
	_buffers.clear();

	for (auto &query : _queries) {
		if (query.glQuery != 0) {
			glDeleteQueries(1, &query.glQuery);
		}
	}
	_queries.clear();

	CleanupPlatform();
	_initialized = false;
}

bool OpenGLBackend::IsInitialized() const {
	return _initialized;
}

void OpenGLBackend::MakeCurrent() {
	if (!_initialized) {
		throw std::runtime_error("OpenGL backend not initialized");
	}

#ifdef _WIN32
	if (!wglMakeCurrent(_hdc, _hglrc)) {
		throw std::runtime_error("Failed to make OpenGL context current");
	}
#else
	if (!glXMakeCurrent(static_cast<Display *>(_display), static_cast<Window>(_window),
						static_cast<GLXContext>(_glxContext))) {
		throw std::runtime_error("Failed to make OpenGL context current");
	}
#endif

	InvalidateCache();
}

void OpenGLBackend::MakeNoneCurrent() {
#ifdef _WIN32
	wglMakeCurrent(nullptr, nullptr);
#else
	if (_display) {
		glXMakeCurrent(static_cast<Display *>(_display), None, nullptr);
	}
#endif
}

BackendCaps OpenGLBackend::GetCaps() const {
	BackendCaps caps;

	if (!_initialized) {
		return caps;
	}

	const GLubyte *version = glGetString(GL_VERSION);
	caps.versionString	   = version ? reinterpret_cast<const char *>(version) : "Unknown";

	GLint major = 0, minor = 0;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);

	caps.supportsComputeShaders = (major > 4 || (major == 4 && minor >= 3));

	GLint workGroupSize[3];
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &workGroupSize[0]);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &workGroupSize[1]);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &workGroupSize[2]);
	caps.maxWorkGroupSizeX = workGroupSize[0];
	caps.maxWorkGroupSizeY = workGroupSize[1];
	caps.maxWorkGroupSizeZ = workGroupSize[2];

	GLint maxBindings	   = 0;
	glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &maxBindings);
	caps.maxBufferBindings = static_cast<uint32_t>(maxBindings);

	glGetIntegerv(GL_MAX_IMAGE_UNITS, &maxBindings);
	caps.maxTextureBindings	   = static_cast<uint32_t>(maxBindings);

	caps.supportsAsyncTransfer = caps.supportsComputeShaders;
	caps.supportsMultiQueue	   = false;

	return caps;
}

BufferHandle OpenGLBackend::CreateBuffer(const BufferDesc &desc) {
	if (!_initialized) {
		throw std::runtime_error("OpenGL backend not initialized");
	}

	uint32_t glHandle = 0;
	glGenBuffers(1, &glHandle);
	if (glHandle == 0) {
		throw std::runtime_error("Failed to create OpenGL buffer");
	}

	uint32_t glUsage = GetGLBufferUsage(desc.mode);
	uint32_t glMode	 = GetGLBufferMode(desc.mode);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, glHandle);
	glBufferData(GL_SHADER_STORAGE_BUFFER, desc.sizeInBytes, desc.initialData, glUsage);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	BufferHandle handle = _nextBufferHandle++;
	BufferInfo	 info;
	info.glHandle	 = glHandle;
	info.size		 = desc.sizeInBytes;
	info.mode		 = glMode;
	_buffers[handle] = info;

	return handle;
}

void OpenGLBackend::DestroyBuffer(BufferHandle buffer) {
	auto it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		return;
	}

	if (it->second.glHandle != 0) {
		glDeleteBuffers(1, &it->second.glHandle);
	}
	_buffers.erase(it);
}

void OpenGLBackend::UploadBuffer(BufferHandle buffer, size_t offset, size_t size, const void *data) {
	auto it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		throw std::runtime_error("Invalid buffer handle");
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, it->second.glHandle);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, size, data);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void OpenGLBackend::DownloadBuffer(BufferHandle buffer, size_t offset, size_t size, void *outData) {
	auto it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		throw std::runtime_error("Invalid buffer handle");
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, it->second.glHandle);
	void *mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, offset, size, GL_MAP_READ_BIT);
	if (mapped) {
		std::memcpy(outData, mapped, size);
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void *OpenGLBackend::MapBuffer(BufferHandle buffer, bool read, bool write) {
	auto it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		return nullptr;
	}

	GLbitfield access = 0;
	if (read)
		access |= GL_MAP_READ_BIT;
	if (write)
		access |= GL_MAP_WRITE_BIT;

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, it->second.glHandle);
	void *ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, it->second.size, access);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	return ptr;
}

void OpenGLBackend::UnmapBuffer(BufferHandle buffer) {
	auto it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		return;
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, it->second.glHandle);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

TextureHandle OpenGLBackend::CreateTexture(const TextureDesc &desc) {
	if (!_initialized) {
		throw std::runtime_error("OpenGL backend not initialized");
	}

	auto [internalFormat, format, type] = GetGLPixelFormat(desc.format);

	uint32_t glHandle					= 0;
	glGenTextures(1, &glHandle);
	if (glHandle == 0) {
		throw std::runtime_error("Failed to create OpenGL texture");
	}

	bool is3D = desc.depth > 1;
	if (is3D) {
		glBindTexture(GL_TEXTURE_3D, glHandle);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, desc.width, desc.height, desc.depth, 0, format, type,
					 desc.initialData);
		glBindTexture(GL_TEXTURE_3D, 0);
	} else {
		glBindTexture(GL_TEXTURE_2D, glHandle);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, desc.width, desc.height, 0, format, type, desc.initialData);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	TextureHandle handle = _nextTextureHandle++;
	TextureInfo	  info;
	info.glHandle		= glHandle;
	info.width			= desc.width;
	info.height			= desc.height;
	info.depth			= desc.depth;
	info.internalFormat = internalFormat;
	info.format			= format;
	info.type			= type;
	_textures[handle]	= info;

	return handle;
}

void OpenGLBackend::DestroyTexture(TextureHandle texture) {
	auto it = _textures.find(texture);
	if (it == _textures.end()) {
		return;
	}

	if (it->second.glHandle != 0) {
		glDeleteTextures(1, &it->second.glHandle);
	}
	_textures.erase(it);
}

void OpenGLBackend::UploadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
								  const void *data) {
	auto it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}

	glBindTexture(GL_TEXTURE_2D, it->second.glHandle);
	glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, width, height, it->second.format, it->second.type, data);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void OpenGLBackend::UploadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
									uint32_t height, uint32_t depth, const void *data) {
	auto it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}

	glBindTexture(GL_TEXTURE_3D, it->second.glHandle);
	glTexSubImage3D(GL_TEXTURE_3D, 0, x, y, z, width, height, depth, it->second.format, it->second.type, data);
	glBindTexture(GL_TEXTURE_3D, 0);
}

void OpenGLBackend::DownloadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
									void *outData) {
	auto it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}
	if (!outData) {
		throw std::runtime_error("DownloadTexture: outData is null");
	}

	// NOTE: glGetTexImage always downloads the full texture mip-level.
	// The (x, y, width, height) parameters are reserved for future sub-region
	// implementation (e.g. via FBO + glReadPixels). Callers must ensure outData
	// is large enough to hold the entire texture.
	(void)x;
	(void)y;
	(void)width;
	(void)height;

	glBindTexture(GL_TEXTURE_2D, it->second.glHandle);
	glGetTexImage(GL_TEXTURE_2D, 0, it->second.format, it->second.type, outData);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void OpenGLBackend::DownloadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
									  uint32_t height, uint32_t depth, void *outData) {
	auto it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}

	(void)x;
	(void)y;
	(void)z;
	(void)width;
	(void)height;
	(void)depth;

	glBindTexture(GL_TEXTURE_3D, it->second.glHandle);
	glGetTexImage(GL_TEXTURE_3D, 0, it->second.format, it->second.type, outData);
	glBindTexture(GL_TEXTURE_3D, 0);
}

ShaderHandle OpenGLBackend::CreateShader(const ShaderDesc &desc) {
	if (!_initialized) {
		throw std::runtime_error("OpenGL backend not initialized");
	}

	GLenum shaderType;
	switch (desc.type) {
	case ShaderType::Compute:
		shaderType = GL_COMPUTE_SHADER;
		break;
	case ShaderType::Vertex:
		shaderType = GL_VERTEX_SHADER;
		break;
	case ShaderType::Fragment:
		shaderType = GL_FRAGMENT_SHADER;
		break;
	default:
		throw std::runtime_error("Unknown shader type");
	}

	uint32_t shader = glCreateShader(shaderType);
	if (shader == 0) {
		throw std::runtime_error("Failed to create shader");
	}

	const char *source = desc.sourceCode.c_str();
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);

	GLint compiled;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
	if (!compiled) {
		GLint logLength;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
		std::vector<char> log(logLength);
		glGetShaderInfoLog(shader, logLength, nullptr, log.data());
		glDeleteShader(shader);
		throw std::runtime_error(std::string("Shader compilation failed: ") + log.data());
	}

	ShaderHandle handle = _nextShaderHandle++;
	ShaderInfo	 info;
	info.glHandle	 = shader;
	info.type		 = desc.type;
	_shaders[handle] = info;

	return handle;
}

void OpenGLBackend::DestroyShader(ShaderHandle shader) {
	auto it = _shaders.find(shader);
	if (it == _shaders.end()) {
		return;
	}

	if (it->second.glHandle != 0) {
		glDeleteShader(it->second.glHandle);
	}
	_shaders.erase(it);
}

PipelineHandle OpenGLBackend::CreatePipeline(const PipelineDesc &desc) {
	if (!_initialized) {
		throw std::runtime_error("OpenGL backend not initialized");
	}

	auto shaderIt = _shaders.find(desc.computeShader);
	if (shaderIt == _shaders.end()) {
		throw std::runtime_error("Invalid shader handle");
	}

	uint32_t program = glCreateProgram();
	if (program == 0) {
		throw std::runtime_error("Failed to create shader program");
	}

	glAttachShader(program, shaderIt->second.glHandle);
	glLinkProgram(program);

	GLint linked;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (!linked) {
		GLint logLength;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
		std::vector<char> log(logLength);
		glGetProgramInfoLog(program, logLength, nullptr, log.data());
		glDeleteProgram(program);
		throw std::runtime_error(std::string("Program linking failed: ") + log.data());
	}

	PipelineHandle handle = _nextPipelineHandle++;
	PipelineInfo   info;
	info.glProgram		= program;
	info.computeShader	= shaderIt->second.glHandle;
	info.workGroupSizeX = desc.workGroupSizeX;
	info.workGroupSizeY = desc.workGroupSizeY;
	info.workGroupSizeZ = desc.workGroupSizeZ;
	_pipelines[handle]	= info;

	return handle;
}

void OpenGLBackend::DestroyPipeline(PipelineHandle pipeline) {
	auto it = _pipelines.find(pipeline);
	if (it == _pipelines.end()) {
		return;
	}

	if (it->second.glProgram != 0) {
		glDeleteProgram(it->second.glProgram);
	}
	_pipelines.erase(it);
}

void OpenGLBackend::BindPipeline(PipelineHandle pipeline) {
	auto it = _pipelines.find(pipeline);
	if (it == _pipelines.end()) {
		throw std::runtime_error("Invalid pipeline handle");
	}

	BindProgram(it->second.glProgram);
}

void OpenGLBackend::BindResources(const ResourceBinding *bindings, uint32_t count) {
	for (uint32_t i = 0; i < count; ++i) {
		const auto &binding = bindings[i];

		if (binding.type == BindingType::Buffer) {
			auto it = _buffers.find(binding.buffer);
			if (it != _buffers.end()) {
				BindSSBO(binding.binding, it->second.glHandle);
			}
		} else if (binding.type == BindingType::Texture) {
			auto it = _textures.find(binding.texture);
			if (it != _textures.end()) {
				BindImageTexture(binding.binding, it->second.glHandle, GetGLImageFormat(binding.format),
								 binding.readOnly);
			}
		} else if (binding.type == BindingType::Sampler) {
			auto it = _textures.find(binding.texture);
			if (it != _textures.end()) {
				glActiveTexture(GL_TEXTURE0 + binding.binding);
				glBindTexture(GL_TEXTURE_2D, it->second.glHandle);
			}
		}
	}
}

void OpenGLBackend::SetUniform(PipelineHandle pipeline, const std::string &name, const std::string &type,
							   const void *data) {
	auto it = _pipelines.find(pipeline);
	if (it == _pipelines.end()) {
		return;
	}

	GLint location = glGetUniformLocation(it->second.glProgram, name.c_str());
	if (location == -1) {
		return;
	}

	if (type == "float") {
		glProgramUniform1f(it->second.glProgram, location, *static_cast<const float *>(data));
	} else if (type == "int" || type == "bool") {
		glProgramUniform1i(it->second.glProgram, location, *static_cast<const int *>(data));
	} else if (type == "vec2") {
		glProgramUniform2fv(it->second.glProgram, location, 1, static_cast<const float *>(data));
	} else if (type == "vec3") {
		glProgramUniform3fv(it->second.glProgram, location, 1, static_cast<const float *>(data));
	} else if (type == "vec4") {
		glProgramUniform4fv(it->second.glProgram, location, 1, static_cast<const float *>(data));
	} else if (type == "ivec2") {
		glProgramUniform2iv(it->second.glProgram, location, 1, static_cast<const int *>(data));
	} else if (type == "ivec3") {
		glProgramUniform3iv(it->second.glProgram, location, 1, static_cast<const int *>(data));
	} else if (type == "ivec4") {
		glProgramUniform4iv(it->second.glProgram, location, 1, static_cast<const int *>(data));
	} else if (type == "mat2") {
		glProgramUniformMatrix2fv(it->second.glProgram, location, 1, GL_FALSE, static_cast<const float *>(data));
	} else if (type == "mat3") {
		glProgramUniformMatrix3fv(it->second.glProgram, location, 1, GL_FALSE, static_cast<const float *>(data));
	} else if (type == "mat4") {
		glProgramUniformMatrix4fv(it->second.glProgram, location, 1, GL_FALSE, static_cast<const float *>(data));
	} else {
		// Unsupported uniform type - this is a development error
		throw std::runtime_error(std::string("Unsupported uniform type: ") + type);
	}
}

void OpenGLBackend::Dispatch(uint32_t groupX, uint32_t groupY, uint32_t groupZ) {
	glDispatchCompute(groupX, groupY, groupZ);
}

void OpenGLBackend::MemoryBarrier(BarrierType barrierType) {
	GLbitfield barriers = 0;

	if (HasFlag(barrierType, BarrierType::Buffer)) {
		barriers |= GL_SHADER_STORAGE_BARRIER_BIT;
	}
	if (HasFlag(barrierType, BarrierType::Texture)) {
		barriers |= GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
	}

	if (barriers != 0) {
		glMemoryBarrier(barriers);
	}
}

void OpenGLBackend::Finish() {
	glFinish();
}

uint32_t OpenGLBackend::BeginQuery() {
	for (uint32_t i = 0; i < _queries.size(); ++i) {
		if (!_queries[i].active) {
			if (_queries[i].glQuery == 0) {
				glGenQueries(1, &_queries[i].glQuery);
			}
			glBeginQuery(GL_TIME_ELAPSED, _queries[i].glQuery);
			_queries[i].active = true;
			return i;
		}
	}

	QueryInfo query;
	glGenQueries(1, &query.glQuery);
	glBeginQuery(GL_TIME_ELAPSED, query.glQuery);
	query.active   = true;
	uint32_t index = static_cast<uint32_t>(_queries.size());
	_queries.push_back(query);
	return index;
}

uint64_t OpenGLBackend::EndQuery(uint32_t query) {
	if (query >= _queries.size() || !_queries[query].active) {
		return 0;
	}

	glEndQuery(GL_TIME_ELAPSED);
	_queries[query].active = false;

	GLuint64 elapsed	   = 0;
	glGetQueryObjectui64v(_queries[query].glQuery, GL_QUERY_RESULT, &elapsed);
	return elapsed;
}

void OpenGLBackend::InvalidateCache() {
	_currentProgram = 0;
	std::fill(_boundBuffers.begin(), _boundBuffers.end(), 0);
	std::fill(_boundTextures.begin(), _boundTextures.end(), 0);
}

void OpenGLBackend::BindProgram(uint32_t program) {
	if (_currentProgram != program) {
		glUseProgram(program);
		_currentProgram = program;
	}
}

void OpenGLBackend::BindSSBO(uint32_t binding, uint32_t buffer) {
	if (binding < MAX_BUFFER_BINDINGS && _boundBuffers[binding] != buffer) {
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer);
		_boundBuffers[binding] = buffer;
	}
}

void OpenGLBackend::BindImageTexture(uint32_t binding, uint32_t texture, uint32_t format, bool readOnly) {
	if (binding < MAX_TEXTURE_BINDINGS) {
		GLenum access = readOnly ? GL_READ_ONLY : GL_READ_WRITE;
		if (_boundTextures[binding] != texture || _boundTextureFormats[binding] != format) {
			glBindImageTexture(binding, texture, 0, GL_FALSE, 0, access, format);
			_boundTextures[binding]		  = texture;
			_boundTextureFormats[binding] = format;
		}
	}
}

uint32_t OpenGLBackend::GetGLBufferMode(BufferMode mode) {
	switch (mode) {
	case BufferMode::Read:
		return GL_READ_ONLY;
	case BufferMode::Write:
		return GL_WRITE_ONLY;
	case BufferMode::ReadWrite:
		return GL_READ_WRITE;
	default:
		return GL_READ_WRITE;
	}
}

uint32_t OpenGLBackend::GetGLBufferUsage(BufferMode mode) {
	switch (mode) {
	case BufferMode::Read:
		return GL_STATIC_READ;
	case BufferMode::Write:
		return GL_STATIC_DRAW;
	case BufferMode::ReadWrite:
		return GL_DYNAMIC_COPY;
	default:
		return GL_DYNAMIC_COPY;
	}
}

std::tuple<uint32_t, uint32_t, uint32_t> OpenGLBackend::GetGLPixelFormat(PixelFormat format) {
	switch (format) {
	case PixelFormat::R8:
		return {GL_R8, GL_RED, GL_UNSIGNED_BYTE};
	case PixelFormat::RG8:
		return {GL_RG8, GL_RG, GL_UNSIGNED_BYTE};
	case PixelFormat::RGBA8:
		return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};
	case PixelFormat::R32F:
		return {GL_R32F, GL_RED, GL_FLOAT};
	case PixelFormat::RG32F:
		return {GL_RG32F, GL_RG, GL_FLOAT};
	case PixelFormat::RGBA32F:
		return {GL_RGBA32F, GL_RGBA, GL_FLOAT};
	case PixelFormat::R16F:
		return {GL_R16F, GL_RED, GL_HALF_FLOAT};
	case PixelFormat::RG16F:
		return {GL_RG16F, GL_RG, GL_HALF_FLOAT};
	case PixelFormat::RGBA16F:
		return {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT};
	case PixelFormat::R32I:
		return {GL_R32I, GL_RED_INTEGER, GL_INT};
	case PixelFormat::RG32I:
		return {GL_RG32I, GL_RG_INTEGER, GL_INT};
	case PixelFormat::RGBA32I:
		return {GL_RGBA32I, GL_RGBA_INTEGER, GL_INT};
	case PixelFormat::R32UI:
		return {GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT};
	case PixelFormat::RG32UI:
		return {GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT};
	case PixelFormat::RGBA32UI:
		return {GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT};
	default:
		return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};
	}
}

uint32_t OpenGLBackend::GetGLImageFormat(PixelFormat format) {
	switch (format) {
	case PixelFormat::R8:
		return GL_R8;
	case PixelFormat::RG8:
		return GL_RG8;
	case PixelFormat::RGBA8:
		return GL_RGBA8;
	case PixelFormat::R32F:
		return GL_R32F;
	case PixelFormat::RG32F:
		return GL_RG32F;
	case PixelFormat::RGBA32F:
		return GL_RGBA32F;
	case PixelFormat::R16F:
		return GL_R16F;
	case PixelFormat::RG16F:
		return GL_RG16F;
	case PixelFormat::RGBA16F:
		return GL_RGBA16F;
	case PixelFormat::R32I:
		return GL_R32I;
	case PixelFormat::RG32I:
		return GL_RG32I;
	case PixelFormat::RGBA32I:
		return GL_RGBA32I;
	case PixelFormat::R32UI:
		return GL_R32UI;
	case PixelFormat::RG32UI:
		return GL_RG32UI;
	case PixelFormat::RGBA32UI:
		return GL_RGBA32UI;
	default:
		return GL_RGBA8;
	}
}

// =============================================================================
// Platform-specific Implementation - Windows
// =============================================================================

#ifdef _WIN32

static const wchar_t *s_windowClassName = L"EasyGPUHiddenWindow";

void				  OpenGLBackend::InitializePlatform() {
	 _hInstance = GetModuleHandleW(nullptr);
	 if (!_hInstance) {
		 throw std::runtime_error("Failed to get module handle");
	 }

	 WNDCLASSEXW wcex	= {};
	 wcex.cbSize		= sizeof(WNDCLASSEXW);
	 wcex.lpfnWndProc	= DefWindowProcW;
	 wcex.hInstance		= _hInstance;
	 wcex.lpszClassName = s_windowClassName;

	 if (!GetClassInfoExW(_hInstance, s_windowClassName, &wcex)) {
		 if (!RegisterClassExW(&wcex)) {
			 throw std::runtime_error("Failed to register window class");
		 }
	 }

	 CreateHiddenWindow();
	 SetupGLContext();
	 LoadGLAD();
}

void OpenGLBackend::CreateHiddenWindow() {
	_hwnd = CreateWindowExW(0, s_windowClassName, L"EasyGPU Context", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
							1, 1, nullptr, nullptr, _hInstance, nullptr);

	if (!_hwnd) {
		throw std::runtime_error("Failed to create hidden window");
	}

	ShowWindow(_hwnd, SW_HIDE);

	_hdc = GetDC(_hwnd);
	if (!_hdc) {
		throw std::runtime_error("Failed to get device context");
	}
}

void OpenGLBackend::SetupGLContext() {
	PIXELFORMATDESCRIPTOR pfd = {};
	pfd.nSize				  = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion			  = 1;
	pfd.dwFlags				  = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType			  = PFD_TYPE_RGBA;
	pfd.cColorBits			  = 32;
	pfd.cDepthBits			  = 24;
	pfd.cStencilBits		  = 8;
	pfd.iLayerType			  = PFD_MAIN_PLANE;

	int pixelFormat			  = ChoosePixelFormat(_hdc, &pfd);
	if (pixelFormat == 0) {
		throw std::runtime_error("Failed to choose pixel format");
	}

	if (!SetPixelFormat(_hdc, pixelFormat, &pfd)) {
		throw std::runtime_error("Failed to set pixel format");
	}

	_hglrc = wglCreateContext(_hdc);
	if (!_hglrc) {
		throw std::runtime_error("Failed to create OpenGL context");
	}

	if (!wglMakeCurrent(_hdc, _hglrc)) {
		throw std::runtime_error("Failed to make OpenGL context current");
	}
}

void OpenGLBackend::LoadGLAD() {
	if (!gladLoadGL()) {
		throw std::runtime_error("Failed to initialize GLAD");
	}

	GLint major = 0, minor = 0;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);
}

void OpenGLBackend::DestroyHiddenWindow() {
	if (_hdc) {
		ReleaseDC(_hwnd, _hdc);
		_hdc = nullptr;
	}
	if (_hwnd) {
		DestroyWindow(_hwnd);
		_hwnd = nullptr;
	}
}

void OpenGLBackend::CleanupPlatform() {
	wglMakeCurrent(nullptr, nullptr);

	if (_hglrc) {
		wglDeleteContext(_hglrc);
		_hglrc = nullptr;
	}

	DestroyHiddenWindow();
}

#else // Linux

// =============================================================================
// Platform-specific Implementation - Linux
// =============================================================================

void OpenGLBackend::InitializePlatform() {
	CreateHiddenWindow();
	SetupGLContext();
	LoadGLAD();
}

void OpenGLBackend::CreateHiddenWindow() {
	Display *display = XOpenDisplay(nullptr);
	if (!display) {
		throw std::runtime_error("Failed to open X11 display");
	}
	_display		   = display;

	int			screen = DefaultScreen(display);
	Window		root   = RootWindow(display, screen);

	XVisualInfo visualInfo;
	if (!XMatchVisualInfo(display, screen, 24, TrueColor, &visualInfo)) {
		if (!XMatchVisualInfo(display, screen, 32, TrueColor, &visualInfo)) {
			visualInfo.visual = DefaultVisual(display, screen);
			visualInfo.depth  = DefaultDepth(display, screen);
		}
	}

	XSetWindowAttributes attrs;
	attrs.colormap	 = XCreateColormap(display, root, visualInfo.visual, AllocNone);
	attrs.event_mask = StructureNotifyMask;

	Window window	 = XCreateWindow(display, root, 0, 0, 1, 1, 0, visualInfo.depth, InputOutput, visualInfo.visual,
									 CWColormap | CWEventMask, &attrs);

	if (!window) {
		XCloseDisplay(display);
		_display = nullptr;
		throw std::runtime_error("Failed to create X11 window");
	}
	_window = window;

	XFlush(display);
}

void OpenGLBackend::SetupGLContext() {
	Display *display = static_cast<Display *>(_display);

	int		 glxMajor, glxMinor;
	if (!glXQueryVersion(display, &glxMajor, &glxMinor)) {
		throw std::runtime_error("GLX not available");
	}

	int			 visualAttribs[] = {GLX_X_RENDERABLE,
									True,
									GLX_DRAWABLE_TYPE,
									GLX_WINDOW_BIT,
									GLX_RENDER_TYPE,
									GLX_RGBA_BIT,
									GLX_X_VISUAL_TYPE,
									GLX_TRUE_COLOR,
									GLX_RED_SIZE,
									8,
									GLX_GREEN_SIZE,
									8,
									GLX_BLUE_SIZE,
									8,
									GLX_ALPHA_SIZE,
									8,
									GLX_DEPTH_SIZE,
									24,
									GLX_STENCIL_SIZE,
									8,
									None};

	int			 fbCount;
	GLXFBConfig *fbc = glXChooseFBConfig(display, DefaultScreen(display), visualAttribs, &fbCount);
	if (!fbc || fbCount == 0) {
		int minimalAttribs[] = {GLX_X_RENDERABLE, True, GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT, None};
		fbc					 = glXChooseFBConfig(display, DefaultScreen(display), minimalAttribs, &fbCount);
		if (!fbc || fbCount == 0) {
			throw std::runtime_error("Failed to choose GLX framebuffer config");
		}
	}

	GLXFBConfig bestFbc = fbc[0];
	XFree(fbc);

	XVisualInfo *vi = glXGetVisualFromFBConfig(display, bestFbc);
	if (!vi) {
		throw std::runtime_error("Failed to get visual from FB config");
	}

	using glXCreateContextAttribsARBProc = GLXContext (*)(Display *, GLXFBConfig, GLXContext, Bool, const int *);
	glXCreateContextAttribsARBProc glXCreateContextAttribsARB =
		(glXCreateContextAttribsARBProc)glXGetProcAddressARB((const GLubyte *)"glXCreateContextAttribsARB");

	GLXContext context = nullptr;
	if (glXCreateContextAttribsARB) {
		int contextAttribs[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB,	  4,   GLX_CONTEXT_MINOR_VERSION_ARB, 3, GLX_CONTEXT_PROFILE_MASK_ARB,
			GLX_CONTEXT_CORE_PROFILE_BIT_ARB, None};

		context = glXCreateContextAttribsARB(display, bestFbc, 0, True, contextAttribs);

		if (!context) {
			contextAttribs[1] = 3;
			contextAttribs[3] = 3;
			context			  = glXCreateContextAttribsARB(display, bestFbc, 0, True, contextAttribs);
		}
	}

	if (!context) {
		context = glXCreateContext(display, vi, nullptr, GL_TRUE);
	}

	XFree(vi);

	if (!context) {
		throw std::runtime_error("Failed to create GLX context");
	}
	_glxContext = context;

	if (!glXMakeCurrent(display, static_cast<Window>(_window), context)) {
		glXDestroyContext(display, context);
		_glxContext = nullptr;
		throw std::runtime_error("Failed to make GLX context current");
	}
}

void OpenGLBackend::LoadGLAD() {
	if (!gladLoadGL()) {
		throw std::runtime_error("Failed to initialize GLAD");
	}

	GLint major = 0, minor = 0;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);
}

void OpenGLBackend::DestroyHiddenWindow() {
	if (_display) {
		Display *display = static_cast<Display *>(_display);
		if (_window) {
			XDestroyWindow(display, static_cast<Window>(_window));
			_window = 0;
		}
		XCloseDisplay(display);
		_display = nullptr;
	}
}

void OpenGLBackend::CleanupPlatform() {
	if (_display) {
		Display *display = static_cast<Display *>(_display);
		glXMakeCurrent(display, None, nullptr);

		if (_glxContext) {
			glXDestroyContext(display, static_cast<GLXContext>(_glxContext));
			_glxContext = nullptr;
		}

		DestroyHiddenWindow();
	}
}

#endif

Backend *CreateOpenGLBackend() {
	return new OpenGLBackend();
}

// =============================================================================
// Binary Cache Support
// =============================================================================

PipelineHandle OpenGLBackend::CreatePipelineFromBinary(const PipelineDesc &desc, const void *binaryData,
													   size_t binarySize, uint32_t format) {
	if (!_initialized) {
		throw std::runtime_error("OpenGL backend not initialized");
	}

	// Create program from binary
	uint32_t program = glCreateProgram();
	if (program == 0) {
		return INVALID_PIPELINE_HANDLE;
	}

	// Load program binary
	glProgramBinary(program, format, binaryData, static_cast<GLsizei>(binarySize));

	// Check if loading succeeded
	GLint linked = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (!linked) {
		glDeleteProgram(program);
		return INVALID_PIPELINE_HANDLE;
	}

	PipelineHandle handle = _nextPipelineHandle++;
	PipelineInfo   info;
	info.glProgram		= program;
	info.computeShader	= 0; // Unknown from binary
	info.workGroupSizeX = desc.workGroupSizeX;
	info.workGroupSizeY = desc.workGroupSizeY;
	info.workGroupSizeZ = desc.workGroupSizeZ;
	_pipelines[handle]	= info;

	return handle;
}

std::vector<uint8_t> OpenGLBackend::GetPipelineBinary(PipelineHandle pipeline, uint32_t &format) {
	format	= 0;

	auto it = _pipelines.find(pipeline);
	if (it == _pipelines.end()) {
		return {};
	}

	GLuint program = it->second.glProgram;
	if (program == 0) {
		return {};
	}

	// Get binary length
	GLint length = 0;
	glGetProgramiv(program, GL_PROGRAM_BINARY_LENGTH, &length);
	if (length <= 0) {
		return {};
	}

	// Allocate buffer and retrieve binary
	std::vector<uint8_t> binary(length);
	GLenum				 binaryFormat	= 0;
	GLsizei				 returnedLength = 0;
	glGetProgramBinary(program, length, &returnedLength, &binaryFormat, binary.data());

	if (returnedLength <= 0) {
		return {};
	}

	binary.resize(returnedLength);
	format = static_cast<uint32_t>(binaryFormat);
	return binary;
}

bool OpenGLBackend::SupportsPipelineCache() const {
	if (!_initialized) {
		return false;
	}

	// Check if GL_ARB_get_program_binary is available (OpenGL 4.1+)
	GLint numFormats = 0;
	glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &numFormats);
	return numFormats > 0;
}

uint32_t OpenGLBackend::GetPipelineCacheFormat() const {
	if (!_initialized) {
		return 0;
	}

	// Create a test program to determine the preferred format
	GLuint testProgram = glCreateProgram();
	if (testProgram == 0) {
		return 0;
	}

	// Get available formats
	GLint numFormats = 0;
	glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &numFormats);
	if (numFormats <= 0) {
		glDeleteProgram(testProgram);
		return 0;
	}

	std::vector<GLint> formats(numFormats);
	glGetIntegerv(GL_PROGRAM_BINARY_FORMATS, formats.data());

	glDeleteProgram(testProgram);

	// Use the first available format as the identifier
	// In practice, most drivers only support one format
	return static_cast<uint32_t>(formats[0]);
}

} // namespace GPU::Backend
