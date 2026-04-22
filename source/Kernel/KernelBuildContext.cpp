/**
 * @file KernelBuildContext.cpp
 * @brief Kernel build context implementation with backend integration
 */

#include <Kernel/KernelBuildContext.h>
#include <Kernel/ShaderCache.h>

#include <IR/Builder/Builder.h>
#include <Runtime/BufferSlot.h>
#include <Runtime/Context.h>
#include <Runtime/TextureSlot.h>

#include <format>
#include <iostream>
#include <sstream>

namespace GPU::Kernel {

KernelDimensionOutOfRange::KernelDimensionOutOfRange() : std::out_of_range("Kernel dimension out of range!") {
}

KernelBuildContext::~KernelBuildContext() {
	// Destroy cached pipeline if any
	if (_cachedPipeline != Backend::INVALID_PIPELINE_HANDLE) {
		auto *backend = Runtime::Context::GetBackend();
		if (backend) {
			backend->DestroyPipeline(_cachedPipeline);
		}
		_cachedPipeline = Backend::INVALID_PIPELINE_HANDLE;
	}
}

KernelBuildContext::KernelBuildContext(int Dimension) : _variableIndex(0), _nextBinding(0), _dimension(Dimension) {
	if (_dimension == 1) {
		WorkSizeX = 256;
		WorkSizeY = 1;
		WorkSizeZ = 1;
	} else if (_dimension == 2) {
		WorkSizeX = 16;
		WorkSizeY = 16;
		WorkSizeZ = 1;
	} else if (_dimension == 3) {
		WorkSizeX = 8;
		WorkSizeY = 8;
		WorkSizeZ = 4;
	} else {
		throw KernelDimensionOutOfRange();
	}
}

void KernelBuildContext::PushTranslatedCode(std::string Code) {
	if (_inCallableBody) {
		_currentCallableBody.append(Code);
	} else {
		_code.append(Code);
	}
}

std::string KernelBuildContext::AssignVarName() {
	++_variableIndex;
	return std::format("v{}", _variableIndex);
}

std::string KernelBuildContext::GetCompleteCode() {
	std::ostringstream oss;

	// Version directive
#if defined(EASYGPU_BACKEND_VULKAN)
	oss << "#version 450 core\n\n";
#else
	oss << "#version 430 core\n\n";
#endif

	// Layout for compute shader
	if (_dimension == 1) {
		oss << std::format("layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;\n\n", WorkSizeX,
						   WorkSizeY, WorkSizeZ);
	}
	if (_dimension == 2) {
		oss << std::format("layout(local_size_x = {}, local_size_y = {}) in;\n\n", WorkSizeX, WorkSizeY);
	}
	if (_dimension == 3) {
		oss << std::format("layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;\n\n", WorkSizeX,
						   WorkSizeY, WorkSizeZ);
	}

	// ===================================================================
	// Phase 1: Execute all body generators to collect declarations AND generate bodies
	// ===================================================================
	{
		// Save current state
		std::string				savedCallableBody	= std::move(_currentCallableBody);
		bool					savedInCallableBody = _inCallableBody;
		std::stack<std::string> savedBodyStack		= std::move(_callableBodyStack);

		// Clear state for pre-execution
		_currentCallableBody.clear();
		_inCallableBody = false;
		while (!_callableBodyStack.empty()) {
			_callableBodyStack.pop();
		}

		auto &builder	  = IR::Builder::Builder::Get();
		auto *prevContext = builder.Context();
		builder.Bind(*this);

		// Execute all generators to collect declarations and generate bodies
		size_t processedCount = 0;
		while (processedCount < _callableBodyGenerators.size()) {
			size_t currentCount = _callableBodyGenerators.size();
			for (size_t i = processedCount; i < currentCount; ++i) {
				_callableBodyGenerators[i]();
			}
			processedCount = currentCount;
		}

		// Restore previous context
		if (prevContext) {
			builder.Bind(*prevContext);
		} else {
			builder.Unbind();
		}

		// Restore original state
		_currentCallableBody = std::move(savedCallableBody);
		_inCallableBody		 = savedInCallableBody;
		_callableBodyStack	 = std::move(savedBodyStack);
	}

	// ===================================================================
	// Output struct declarations AFTER Phase 1
	// ===================================================================
	for (const auto &structDef : GetStructDefinitions()) {
		oss << structDef;
	}
	if (!GetStructDefinitions().empty()) {
		oss << "\n";
	}

	// Output texture declarations (after struct definitions, before buffers)
	std::string textureDecls = GetTextureDeclarations();
	if (!textureDecls.empty()) {
		oss << textureDecls << "\n";
	}

	// Output buffer declarations (after texture declarations)
	std::string bufferDecls = GetBufferDeclarations();
	if (!bufferDecls.empty()) {
		oss << bufferDecls << "\n";
	}

	// Output uniform declarations (after buffer declarations, before callable declarations)
	std::string uniformDecls = GetUniformDeclarations();
	if (!uniformDecls.empty()) {
		oss << uniformDecls << "\n";
	}

	// Output shared memory declarations (after uniform declarations, before callable declarations)
	for (const auto &decl : _sharedMemoryDeclarations) {
		oss << decl << "\n";
	}
	if (!_sharedMemoryDeclarations.empty()) {
		oss << "\n";
	}

	// Output callable function forward declarations (before main)
	for (const auto &decl : _callableDeclarations) {
		oss << decl << ";\n";
	}
	if (!_callableDeclarations.empty()) {
		oss << "\n";
	}

	// Main function wrapper
	oss << "void main() {\n";

	// Add the kernel code with indentation
	std::istringstream codeStream(_code);
	std::string		   line;
	while (std::getline(codeStream, line)) {
		if (!line.empty()) {
			oss << "    " << line << "\n";
		} else {
			oss << "\n";
		}
	}

	oss << "}\n";

	// ===================================================================
	// Phase 2: Generate callable bodies and output definitions
	// ===================================================================
	{
		auto &builder	  = IR::Builder::Builder::Get();
		auto *prevContext = builder.Context();
		builder.Bind(*this);

		std::string callableDefs = GenerateCallableBodies();
		if (!callableDefs.empty()) {
			oss << "\n" << callableDefs;
		}

		// Restore previous context
		if (prevContext) {
			builder.Bind(*prevContext);
		} else {
			builder.Unbind();
		}
	}

	return oss.str();
}

bool KernelBuildContext::HasStructDefinition(const std::string &TypeName) const {
	return _definedStructs.count(TypeName) > 0;
}

void KernelBuildContext::AddStructDefinition(const std::string &TypeName, const std::string &Definition) {
	if (_definedStructs.insert(TypeName).second) {
		_structNames.push_back(TypeName);
		_structDefinitions.push_back(Definition);
	}
}

const std::vector<std::string> &KernelBuildContext::GetStructDefinitions() const {
	return _structDefinitions;
}

uint32_t KernelBuildContext::AllocateBindingSlot() {
	return _nextBinding++;
}

void KernelBuildContext::RegisterBuffer(uint32_t binding, const std::string &typeName, const std::string &bufferName,
										int mode) {
	_buffers.push_back({binding, typeName, bufferName, mode});
	_bufferBindings.push_back(binding);
}

std::string KernelBuildContext::GetBufferDeclarations() const {
	std::ostringstream oss;
	for (const auto &buf : _buffers) {
		std::string qualifier;
		if (buf.mode == GPU::Backend::BUFFER_MODE_READ_ONLY) {
			qualifier = "readonly ";
		} else if (buf.mode == GPU::Backend::BUFFER_MODE_WRITE_ONLY) {
			qualifier = "writeonly ";
		}
		// GL_READ_WRITE (0x88BA) has no qualifier

#if defined(EASYGPU_BACKEND_VULKAN)
		oss << std::format("layout(set=0, std430, binding={}) {}buffer {}_t {{\n", buf.binding, qualifier,
						   buf.bufferName);
#else
		oss << std::format("layout(std430, binding={}) {}buffer {}_t {{\n", buf.binding, qualifier, buf.bufferName);
#endif
		oss << std::format("    {} {}[];\n", buf.typeName, buf.bufferName);
		oss << "};\n";
	}
	return oss.str();
}

uint32_t KernelBuildContext::AllocateTextureBinding() {
	return _nextBinding++;
}

void KernelBuildContext::RegisterTexture(uint32_t binding, Runtime::PixelFormat format, const std::string &textureName,
										 uint32_t width, uint32_t height, bool sampled) {
	_textures.push_back({binding, format, textureName, width, height, 1, sampled});
	_textureBindings.push_back(binding);
}

void KernelBuildContext::RegisterTexture3D(uint32_t binding, Runtime::PixelFormat format,
										   const std::string &textureName, uint32_t width, uint32_t height,
										   uint32_t depth, bool sampled) {
	_textures.push_back({binding, format, textureName, width, height, depth, sampled});
	_textureBindings.push_back(binding);
}

/**
 * Get the GLSL image type name based on pixel format
 */
static std::string GetGLSLImageTypeName(Runtime::PixelFormat format, bool is3D = false) {
	using Runtime::PixelFormat;
	switch (format) {
	// Signed integer formats -> iimage2D
	case PixelFormat::R32I:
	case PixelFormat::RG32I:
	case PixelFormat::RGBA32I:
		return is3D ? "iimage3D" : "iimage2D";

	// Unsigned integer formats -> uimage2D
	case PixelFormat::R32UI:
	case PixelFormat::RG32UI:
	case PixelFormat::RGBA32UI:
		return is3D ? "uimage3D" : "uimage2D";

	// Float and normalized formats -> image2D
	default:
		return is3D ? "image3D" : "image2D";
	}
}

std::string KernelBuildContext::GetTextureDeclarations() const {
	std::ostringstream oss;
	for (const auto &tex : _textures) {
		if (tex.sampled) {
			std::string samplerType = GetGLSLSamplerType(tex.format);
			if (tex.depth > 1) {
				// Replace 2D with 3D in sampler type (e.g. sampler2D -> sampler3D)
				samplerType = samplerType.substr(0, samplerType.size() - 2) + "3D";
			}
#if defined(EASYGPU_BACKEND_VULKAN)
			oss << std::format("layout(set=0, binding={}) uniform {} {};\n", tex.binding, samplerType, tex.textureName);
#else
			oss << std::format("layout(binding={}) uniform {} {};\n", tex.binding, samplerType, tex.textureName);
#endif
			continue;
		}

		std::string formatQualifier = GetGLSLFormatQualifier(tex.format);
		std::string imageType		= GetGLSLImageTypeName(tex.format, tex.depth > 1);
#if defined(EASYGPU_BACKEND_VULKAN)
		oss << std::format("layout(set=0, {}, binding={}) uniform {} {};\n", formatQualifier, tex.binding, imageType,
						   tex.textureName);
#else
		oss << std::format("layout({}, binding={}) uniform {} {};\n", formatQualifier, tex.binding, imageType,
						   tex.textureName);
#endif
	}
	return oss.str();
}

const KernelBuildContext::TextureInfo *KernelBuildContext::FindTextureInfo(uint32_t binding) const {
	for (const auto &texture : _textures) {
		if (texture.binding == binding) {
			return &texture;
		}
	}
	return nullptr;
}

// ===================================================================
// Callable Function Support
// ===================================================================

void KernelBuildContext::AddCallableDeclaration(const std::string &declaration) {
	_callableDeclarations.push_back(declaration);
}

void KernelBuildContext::AddCallableBodyGenerator(std::function<void()> generator) {
	_callableBodyGenerators.push_back(std::move(generator));
}

void KernelBuildContext::PushCallableBody() {
	_callableBodyStack.push(_currentCallableBody);
	_currentCallableBody.clear();
	_inCallableBody = true;
}

void KernelBuildContext::PopCallableBody() {
	_callableBodies.push_back(std::move(_currentCallableBody));
	_currentCallableBody.clear();
	_inCallableBody = false;

	if (!_callableBodyStack.empty()) {
		_currentCallableBody = _callableBodyStack.top();
		_callableBodyStack.pop();
		_inCallableBody = true;
	}
}

std::vector<std::string> KernelBuildContext::GetCallableDeclarations() const {
	return _callableDeclarations;
}

std::string KernelBuildContext::GenerateCallableBodies() {
	std::ostringstream oss;

	size_t			   bodyCount = _callableBodies.size();
	for (size_t i = 0; i < _callableDeclarations.size(); ++i) {
		if (i < bodyCount && !_callableBodies[i].empty()) {
			oss << _callableDeclarations[i] << " {\n";

			std::istringstream bodyStream(_callableBodies[i]);
			std::string		   line;
			while (std::getline(bodyStream, line)) {
				if (!line.empty()) {
					oss << "    " << line << "\n";
				} else {
					oss << "\n";
				}
			}

			oss << "}\n\n";
		}
	}

	return oss.str();
}

// ===================================================================
// Uniform Support
// ===================================================================

std::string KernelBuildContext::RegisterUniform(
	const std::string &typeName, void *uniformPtr, size_t gpuSize, size_t gpuAlignment,
	std::function<void(uint32_t program, const std::string &name, void *ptr)> uploadFunc,
	std::function<void(void *dst, void *ptr)>								  packFunc) {
	for (const auto &entry : _uniforms) {
		if (entry.uniformPtr == uniformPtr) {
			return entry.name;
		}
	}

	std::string uniformName = std::format("u{}", _nextUniformIndex++);
	size_t		alignedOffset =
		 gpuAlignment == 0 ? _nextUniformOffset : ((_nextUniformOffset + gpuAlignment - 1) & ~(gpuAlignment - 1));
	_uniforms.push_back(
		{uniformName, typeName, uniformPtr, gpuSize, gpuAlignment, alignedOffset, uploadFunc, packFunc});
	_nextUniformOffset = alignedOffset + gpuSize;

	return uniformName;
}

std::string KernelBuildContext::GetUniformDeclarations() const {
	std::ostringstream oss;
#if defined(EASYGPU_BACKEND_VULKAN)
	if (_uniforms.empty()) {
		return {};
	}

	oss << "layout(push_constant) uniform EasyGPUUniformBlock {\n";
	for (const auto &entry : _uniforms) {
		oss << std::format("    {} {};\n", entry.typeName, entry.name);
	}
	oss << "};\n";
#else
	for (const auto &entry : _uniforms) {
		oss << std::format("uniform {} {};\n", entry.typeName, entry.name);
	}
#endif
	return oss.str();
}

void KernelBuildContext::UploadUniformValues(Backend::PipelineHandle pipeline) const {
#if defined(EASYGPU_BACKEND_VULKAN)
	auto *backend = Runtime::Context::GetBackend();
	if (!backend || _uniforms.empty()) {
		return;
	}

	std::vector<unsigned char> uniformData(GetPushConstantSize(), 0);
	for (const auto &entry : _uniforms) {
		if (entry.packFunc) {
			entry.packFunc(uniformData.data() + entry.gpuOffset, entry.uniformPtr);
		}
	}

	backend->SetUniformData(pipeline, uniformData.data(), uniformData.size());
#else
	for (const auto &entry : _uniforms) {
		if (entry.uploadFunc) {
			entry.uploadFunc(static_cast<uint32_t>(pipeline), entry.name, entry.uniformPtr);
		}
	}
#endif
}

uint32_t KernelBuildContext::GetPushConstantSize() const {
	return static_cast<uint32_t>(_nextUniformOffset);
}

// ===================================================================
// Buffer/Texture Slot Support
// ===================================================================

void KernelBuildContext::RegisterBufferSlot(Runtime::BufferSlotBase *slot) {
	uint32_t	binding	   = AllocateBindingSlot();
	std::string bufferName = std::format("buf_slot_{}", binding);

	// Default to READ_WRITE mode
	int			mode	   = 0x88BA; // GL_READ_WRITE equivalent

	RegisterBuffer(binding, slot->GetTypeName(), bufferName, mode);
	slot->SetBindingInfo(static_cast<int>(binding), bufferName);
	_bufferSlots.push_back(slot);
	_bufferSlotBindings[slot] = binding;
}

void KernelBuildContext::RegisterTextureSlot(Runtime::TextureSlotBase *slot) {
	uint32_t	binding		= AllocateTextureBinding();
	std::string textureName = std::format("tex_slot_{}", binding);

	uint32_t	width = 0, height = 0;
	slot->GetDimensions(width, height);
	uint32_t depth = slot->GetDepth();

	if (depth > 1) {
		RegisterTexture3D(binding, slot->GetFormat(), textureName, width, height, depth, slot->UsesSamplerBinding());
	} else {
		RegisterTexture(binding, slot->GetFormat(), textureName, width, height, slot->UsesSamplerBinding());
	}
	slot->SetBindingInfo(static_cast<int>(binding), textureName);
	_textureSlots.push_back(slot);
	_textureSlotBindings[slot] = binding;
}

uint32_t KernelBuildContext::GetBufferSlotBinding(Runtime::BufferSlotBase *slot) const {
	auto it = _bufferSlotBindings.find(slot);
	if (it != _bufferSlotBindings.end()) {
		return it->second;
	}
	return static_cast<uint32_t>(slot->GetBinding());
}

uint32_t KernelBuildContext::GetTextureSlotBinding(Runtime::TextureSlotBase *slot) const {
	auto it = _textureSlotBindings.find(slot);
	if (it != _textureSlotBindings.end()) {
		return it->second;
	}
	return static_cast<uint32_t>(slot->GetBinding());
}

// ===================================================================
// Shared Memory Support
// ===================================================================

void KernelBuildContext::PushSharedMemoryDeclaration(const std::string &declaration) {
	_sharedMemoryDeclarations.push_back(declaration);
}

std::vector<std::string> KernelBuildContext::GetSharedMemoryDeclarations() const {
	return _sharedMemoryDeclarations;
}

// ===================================================================
// Shader Cache Support
// ===================================================================

void KernelBuildContext::ComputeShaderHash() {
	std::string source = GetCompleteCode();
	_shaderHash		   = ShaderCache::ComputeShaderHash(source);
}

} // namespace GPU::Kernel
