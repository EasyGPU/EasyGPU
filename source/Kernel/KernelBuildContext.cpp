/**
 * @file KernelBuildContext.cpp
 * @brief Kernel build context implementation with backend integration.
 */

#include <Kernel/KernelBuildContext.h>
#include <Kernel/ShaderCache.h>

#include <AD/GradientTape.h>
#include <IR/Builder/Builder.h>
#include <Runtime/BufferSlot.h>
#include <Runtime/Context.h>
#include <Runtime/TextureSlot.h>
#include <Runtime/UniformBuffer.h>

#include <format>
#include <iostream>
#include <sstream>

namespace GPU::Kernel {

KernelDimensionOutOfRange::KernelDimensionOutOfRange() : std::out_of_range("Kernel dimension out of range!") {
}

KernelBuildContext::~KernelBuildContext() {
	InvalidateCachedPipeline();
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

void KernelBuildContext::InvalidateCachedPipeline() {
	if (_cachedPipeline != Backend::INVALID_PIPELINE_HANDLE) {
		auto *backend = Runtime::Context::GetBackend();
		if (backend) {
			backend->DestroyPipeline(_cachedPipeline);
		}
		_cachedPipeline = Backend::INVALID_PIPELINE_HANDLE;
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

	// Float atomic extensions for maximum GPU compatibility
	if (!_floatAtomicBuffers.empty()) {
		oss << "#extension GL_NV_shader_atomic_float : enable\n"
			<< "#extension GL_EXT_shader_atomic_float : enable\n\n";
	}

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
		std::stack<CallableBodyFrame> savedBodyStack	= std::move(_callableBodyStack);
		size_t					savedNextCallableBodyIndex = _nextCallableBodyIndex;
		auto				   &builder				= IR::Builder::Builder::Get();

		// Clear state for pre-execution
		_currentCallableBody.clear();
		_inCallableBody = false;
		_nextCallableBodyIndex = 0;
		IR::Builder::Builder::ScopedCallableBody callableBodyGuard(builder, false);
		while (!_callableBodyStack.empty()) {
			_callableBodyStack.pop();
		}

		IR::Builder::Builder::ScopedBind bindGuard(builder, *this);

		// Execute all generators to collect declarations and generate bodies
		size_t							 processedCount = 0;
		while (processedCount < _callableBodyGenerators.size()) {
			size_t currentCount = _callableBodyGenerators.size();
			for (size_t i = processedCount; i < currentCount; ++i) {
				_callableBodyGenerators[i]();
			}
			processedCount = currentCount;
		}

		// Restore original state
		_currentCallableBody = std::move(savedCallableBody);
		_inCallableBody		 = savedInCallableBody;
		_callableBodyStack	 = std::move(savedBodyStack);
		_nextCallableBodyIndex = savedNextCallableBodyIndex;
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
		auto							&builder = IR::Builder::Builder::Get();
		IR::Builder::Builder::ScopedBind bindGuard(builder, *this);

		std::string						 callableDefs = GenerateCallableBodies();
		if (!callableDefs.empty()) {
			oss << "\n" << callableDefs;
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
	InvalidateBindings();
	InvalidateBarrierType();
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

		// Declare int alias SSBO for float buffers used in atomic CAS-loop fallback
		if (_floatAtomicBuffers.count(buf.bufferName)) {
#if defined(EASYGPU_BACKEND_VULKAN)
			oss << std::format("layout(set=0, std430, binding={}) buffer {}_t_int {{\n", buf.binding, buf.bufferName);
#else
			oss << std::format("layout(std430, binding={}) buffer {}_t_int {{\n", buf.binding, buf.bufferName);
#endif
			oss << std::format("    int {}_int[];\n", buf.bufferName);
			oss << "};\n";
		}
	}
	return oss.str();
}

uint32_t KernelBuildContext::AllocateTextureBinding() {
	return _nextBinding++;
}

void KernelBuildContext::RegisterTexture(uint32_t binding, Runtime::PixelFormat format, const std::string &textureName,
										 uint32_t width, uint32_t height, bool sampled) {
	_textures.push_back({binding, format, textureName, width, height, 1, 2, sampled});
	_textureBindings.push_back(binding);
	InvalidateBindings();
	InvalidateBarrierType();
}

void KernelBuildContext::RegisterTexture3D(uint32_t binding, Runtime::PixelFormat format,
										   const std::string &textureName, uint32_t width, uint32_t height,
										   uint32_t depth, bool sampled) {
	_textures.push_back({binding, format, textureName, width, height, depth, 3, sampled});
	_textureBindings.push_back(binding);
	InvalidateBindings();
	InvalidateBarrierType();
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
		const auto is3D = tex.dimension == 3;
		if (tex.sampled) {
			std::string samplerType = GetGLSLSamplerType(tex.format);
			if (is3D) {
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
		std::string imageType		= GetGLSLImageTypeName(tex.format, is3D);
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
	auto callableName = std::string{};
	if (_nextCallableBodyIndex < _callableDeclarations.size()) {
		const auto &declaration = _callableDeclarations[_nextCallableBodyIndex];
		const auto paren = declaration.find('(');
		if (paren != std::string::npos) {
			const auto nameEnd = declaration.find_last_not_of(" \t\r\n", paren == 0 ? 0 : paren - 1);
			if (nameEnd != std::string::npos) {
				const auto nameStart = declaration.find_last_of(" \t\r\n", nameEnd);
				callableName = declaration.substr(
					nameStart == std::string::npos ? 0 : nameStart + 1,
					nameEnd - (nameStart == std::string::npos ? 0 : nameStart + 1) + 1);
			}
		}
	}
	_nextCallableBodyIndex++;
	PushCallableBody(callableName);
}

void KernelBuildContext::PushCallableBody(const std::string &callableName) {
	_callableBodyStack.push(CallableBodyFrame{std::move(_currentCallableBody), _inCallableBody});
	_currentCallableBody.clear();
	_inCallableBody = true;
	IR::Builder::Builder::Get().SetInCallableBody(true);
	// If gradient tape is active, push a sub-tape for callable body recording
	auto *tape = IR::Builder::Builder::Get().GetGradientTape();
	if (tape) {
		tape->PushSubTape(callableName);
	}
}

void KernelBuildContext::PopCallableBody() {
	_callableBodies.push_back(std::move(_currentCallableBody));
	_currentCallableBody.clear();
	_inCallableBody = false;
	IR::Builder::Builder::Get().SetInCallableBody(false);
	// Pop the sub-tape that was pushed in PushCallableBody
	auto *tape = IR::Builder::Builder::Get().GetGradientTape();
	if (tape) {
		tape->PopSubTape();
	}

	if (!_callableBodyStack.empty()) {
		auto frame = std::move(_callableBodyStack.top());
		_callableBodyStack.pop();
		_currentCallableBody = std::move(frame.body);
		_inCallableBody = frame.inCallableBody;
		IR::Builder::Builder::Get().SetInCallableBody(_inCallableBody);
		// Push new sub-tape for the restored callable body
		if (tape && _inCallableBody) {
			tape->PushSubTape();
		}
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
		if (uniformPtr != nullptr && entry.uniformPtr == uniformPtr) {
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
	auto existing = _bufferSlotBindings.find(slot);
	if (existing != _bufferSlotBindings.end()) {
		slot->SetBindingInfo(static_cast<int>(existing->second), std::format("buf_slot_{}", existing->second));
		return;
	}

	uint32_t	binding	   = AllocateBindingSlot();
	std::string bufferName = std::format("buf_slot_{}", binding);

	// Default to READ_WRITE mode
	int			mode	   = 0x88BA; // GL_READ_WRITE equivalent

	RegisterBuffer(binding, slot->GetTypeName(), bufferName, mode);
	slot->SetBindingInfo(static_cast<int>(binding), bufferName);
	_bufferSlots.push_back(slot);
	_bufferSlotBindings[slot] = binding;
	InvalidateBindings();
	InvalidateBarrierType();
}

void KernelBuildContext::RegisterTextureSlot(Runtime::TextureSlotBase *slot) {
	auto existing = _textureSlotBindings.find(slot);
	if (existing != _textureSlotBindings.end()) {
		slot->SetBindingInfo(static_cast<int>(existing->second), std::format("tex_slot_{}", existing->second));
		return;
	}

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
	InvalidateBindings();
	InvalidateBarrierType();
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

const std::vector<Backend::ResourceBinding> &KernelBuildContext::GetCachedBindings() {
	if (!_bindingsDirty && _bufferSlots.empty() && _textureSlots.empty()) {
		return _cachedBindings;
	}

	if (!_bindingsDirty) {
		if (_cachedBufferSlotHandles.size() != _bufferSlots.size() ||
			_cachedTextureSlotHandles.size() != _textureSlots.size()) {
			_bindingsDirty = true;
		}
	}

	if (!_bindingsDirty) {
		for (size_t i = 0; i < _bufferSlots.size(); ++i) {
			auto *slot = _bufferSlots[i];
			if (!slot->IsAttached() || slot->GetHandle() != _cachedBufferSlotHandles[i]) {
				_bindingsDirty = true;
				break;
			}
		}
	}

	if (!_bindingsDirty) {
		for (size_t i = 0; i < _textureSlots.size(); ++i) {
			auto *slot = _textureSlots[i];
			if (!slot->IsAttached() || slot->GetHandle() != _cachedTextureSlotHandles[i] ||
				slot->GetFormat() != _cachedTextureSlotFormats[i] ||
				slot->UsesSamplerBinding() != _cachedTextureSlotSampled[i]) {
				_bindingsDirty = true;
				break;
			}
		}
	}

	if (!_bindingsDirty) {
		return _cachedBindings;
	}

	_cachedBindings.clear();
	_cachedBindings.reserve(_runtimeBuffers.size() + _runtimeTextures.size() + _bufferSlots.size() +
							_textureSlots.size());
	_cachedBufferSlotHandles.clear();
	_cachedBufferSlotHandles.reserve(_bufferSlots.size());
	_cachedTextureSlotHandles.clear();
	_cachedTextureSlotHandles.reserve(_textureSlots.size());
	_cachedTextureSlotFormats.clear();
	_cachedTextureSlotFormats.reserve(_textureSlots.size());
	_cachedTextureSlotSampled.clear();
	_cachedTextureSlotSampled.reserve(_textureSlots.size());

	for (const auto &[binding, handle] : _runtimeBuffers) {
		Backend::ResourceBinding rb;
		rb.binding = binding;
		rb.type	   = Backend::BindingType::Buffer;
		rb.buffer  = static_cast<Backend::BufferHandle>(handle);
		_cachedBindings.push_back(rb);
	}

	for (const auto &[binding, handle] : _runtimeTextures) {
		const auto *textureInfo = FindTextureInfo(binding);
		if (!textureInfo) {
			throw std::runtime_error("Runtime texture binding missing shader-side texture metadata");
		}
		Backend::ResourceBinding rb;
		rb.binding	= binding;
		rb.type		= textureInfo->sampled ? Backend::BindingType::Sampler : Backend::BindingType::Texture;
		rb.texture	= static_cast<Backend::TextureHandle>(handle);
		rb.format	= Runtime::ToBackendPixelFormat(textureInfo->format);
		rb.readOnly = textureInfo->sampled;
		_cachedBindings.push_back(rb);
	}

	for (auto *slot : _bufferSlots) {
		if (!slot->IsAttached()) {
			throw std::runtime_error("BufferSlot not attached at dispatch time");
		}
		Backend::ResourceBinding rb;
		rb.binding = GetBufferSlotBinding(slot);
		rb.type	   = Backend::BindingType::Buffer;
		rb.buffer  = slot->GetHandle();
		_cachedBindings.push_back(rb);
		_cachedBufferSlotHandles.push_back(rb.buffer);
	}

	for (auto *slot : _textureSlots) {
		if (!slot->IsAttached()) {
			throw std::runtime_error("TextureSlot not attached at dispatch time");
		}
		uint32_t	slotBinding = GetTextureSlotBinding(slot);
		const auto *textureInfo = FindTextureInfo(slotBinding);
		if (!textureInfo) {
			throw std::runtime_error("TextureSlot missing shader-side texture metadata");
		}
		Backend::ResourceBinding rb;
		rb.binding	= slotBinding;
		rb.type		= textureInfo->sampled ? Backend::BindingType::Sampler : Backend::BindingType::Texture;
		rb.texture	= slot->GetHandle();
		rb.format	= Runtime::ToBackendPixelFormat(slot->GetFormat());
		rb.readOnly = textureInfo->sampled;
		_cachedBindings.push_back(rb);
		_cachedTextureSlotHandles.push_back(rb.texture);
		_cachedTextureSlotFormats.push_back(slot->GetFormat());
		_cachedTextureSlotSampled.push_back(slot->UsesSamplerBinding());
	}

	_bindingsDirty = false;
	return _cachedBindings;
}

Backend::BarrierType KernelBuildContext::GetRequiredBarrierType() {
	if (_barrierComputed) {
		return _requiredBarrierType;
	}

	_requiredBarrierType = Backend::BarrierType::None;

	for (const auto &bufferInfo : _buffers) {
		if (bufferInfo.mode != GPU::Backend::BUFFER_MODE_READ_ONLY) {
			_requiredBarrierType = _requiredBarrierType | Backend::BarrierType::Buffer;
			break;
		}
	}

	if (!HasFlag(_requiredBarrierType, Backend::BarrierType::Buffer)) {
		for (const auto *slot : _bufferSlots) {
			if (slot->GetMode() != GPU::Backend::BUFFER_MODE_READ_ONLY) {
				_requiredBarrierType = _requiredBarrierType | Backend::BarrierType::Buffer;
				break;
			}
		}
	}

	for (const auto &textureInfo : _textures) {
		if (!textureInfo.sampled) {
			_requiredBarrierType = _requiredBarrierType | Backend::BarrierType::Texture;
			break;
		}
	}

	for (const auto *slot : _textureSlots) {
		if (!slot->UsesSamplerBinding()) {
			_requiredBarrierType = _requiredBarrierType | Backend::BarrierType::Texture;
			break;
		}
	}

	_barrierComputed = true;
	return _requiredBarrierType;
}

// ===================================================================
// Shared Memory Support
// ===================================================================

void KernelBuildContext::PushSharedMemoryDeclaration(const std::string &declaration) {
	_sharedMemoryDeclarations.push_back(declaration);
}

void KernelBuildContext::RegisterFloatAtomicBuffer(const std::string &bufferName) {
	_floatAtomicBuffers.insert(bufferName);
}

std::vector<std::string> KernelBuildContext::GetSharedMemoryDeclarations() const {
	return _sharedMemoryDeclarations;
}

// ===================================================================
// Shader Cache Support
// ===================================================================

void KernelBuildContext::ComputeShaderHash() {
	std::string source = GetCompleteCode() + "\n// EasyGPUOptimizationLevel=" +
						 std::to_string(static_cast<int>(_optimizationLevel));
	_shaderHash		   = ShaderCache::ComputeShaderHash(source);
}

void KernelBuildContext::RegisterVarying(const std::string &name, const std::string &glslType) {
	(void)name;
	(void)glslType;
	// Base implementation: no-op. GraphicsBuildContext overrides this.
}

std::string KernelBuildContext::RegisterUniformBuffer(const std::string &typeName, void *ubo, size_t gpuSize) {
	(void)gpuSize;
	auto existing = _uniformBufferNames.find(ubo);
	if (existing != _uniformBufferNames.end()) {
		return existing->second + "[0]";
	}

	auto	   *uniformBuffer = static_cast<Runtime::UniformBufferBase *>(ubo);
	uint32_t	binding		  = AllocateBindingSlot();
	std::string name		  = std::format("ubo_{}", binding);

	RegisterBuffer(binding, typeName, name, Backend::BUFFER_MODE_READ_ONLY);
	BindRuntimeBuffer(binding, uniformBuffer->GetHandle());
	_uniformBufferNames.emplace(ubo, name);
	return name + "[0]";
}

} // namespace GPU::Kernel
