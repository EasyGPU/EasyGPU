#pragma once

/**
 * @file KernelBuildContext.h
 * @brief Kernel build context with backend integration
 */

#ifndef EASYGPU_KERNELBUILDCONTEXT_H
#define EASYGPU_KERNELBUILDCONTEXT_H

#include <Backend/Backend.h>
#include <IR/Builder/BuilderContext.h>
#include <Runtime/Buffer.h>
#include <Runtime/PixelFormat.h>

#include <functional>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace GPU::Kernel {

/**
 * The exception for dimension out of range
 */
class KernelDimensionOutOfRange : public std::out_of_range {
public:
	KernelDimensionOutOfRange();
};

/**
 * The build context for the kernel
 */
class KernelBuildContext : public IR::Builder::BuilderContext {
public:
	/**
	 * Buffer registration info
	 */
	struct BufferInfo {
		uint32_t	binding;
		std::string typeName;
		std::string bufferName;
		int			mode; // Backend::BufferMode
	};

	/**
	 * Texture registration info (2D)
	 */
	struct TextureInfo {
		uint32_t			 binding;
		Runtime::PixelFormat format;
		std::string			 textureName;
		uint32_t			 width;
		uint32_t			 height;
		uint32_t			 depth = 1;
		bool				 sampled = false;
	};

	struct UniformEntry;

	/**
	 * This constructor will construct the work size in default
	 * @param Dimension The dimension of the
	 * kernel
	 */
	KernelBuildContext(int Dimension);

	/**
	 * Destructor - cleans up cached pipeline if any
	 */
	~KernelBuildContext();

public:
	void				PushTranslatedCode(std::string Code) override;

	std::string			AssignVarName() override;

	/**
	 * Get the complete kernel code including struct definitions
	 * @return The complete GLSL code
	 */
	virtual std::string GetCompleteCode();

public:
	bool HasStructDefinition(const std::string &TypeName) const override;
	void AddStructDefinition(const std::string &TypeName, const std::string &Definition) override;
	const std::vector<std::string> &GetStructDefinitions() const override;

public:
	uint32_t	AllocateBindingSlot() override;
	void		RegisterBuffer(uint32_t binding, const std::string &typeName, const std::string &bufferName,
							   int mode) override;
	std::string GetBufferDeclarations() const override;
	const std::vector<uint32_t> &GetBufferBindings() const override {
		return _bufferBindings;
	}
	const std::vector<BufferInfo> &GetBufferInfos() const {
		return _buffers;
	}

	/**
	 * Bind a runtime GPU buffer to a binding slot
	 * This is called by Buffer::Bind() to associate the actual backend buffer with the binding
	 * @param binding The binding slot
	 * @param bufferHandle The backend buffer handle
	 */
	void BindRuntimeBuffer(uint32_t binding, Backend::BufferHandle bufferHandle) override {
		_runtimeBuffers[binding] = bufferHandle;
	}

	const std::unordered_map<uint32_t, uint32_t> &GetRuntimeBufferBindings() const override {
		return _runtimeBuffers;
	}

public:
	// ===================================================================
	// Texture Support (2D)
	// ===================================================================

	uint32_t AllocateTextureBinding() override;
	void RegisterTexture(uint32_t binding, Runtime::PixelFormat format, const std::string &textureName, uint32_t width,
						 uint32_t height, bool sampled = false) override;
	std::string					 GetTextureDeclarations() const override;
	const std::vector<uint32_t> &GetTextureBindings() const override {
		return _textureBindings;
	}
	const std::vector<TextureInfo> &GetTextureInfos() const {
		return _textures;
	}
	const TextureInfo *FindTextureInfo(uint32_t binding) const;

	/**
	 * Bind a runtime GPU texture to a binding slot
	 * This is called by Texture2D::Bind() to associate the actual backend texture with the binding
	 * @param binding The binding slot
	 * @param textureHandle The backend texture handle
	 */
	void			   BindRuntimeTexture(uint32_t binding, uint32_t textureHandle) override {
		  _runtimeTextures[binding] = textureHandle;
	}

	const std::unordered_map<uint32_t, uint32_t> &GetRuntimeTextureBindings() const override {
		return _runtimeTextures;
	}

public:
	// ===================================================================
	// Uniform Support
	// ===================================================================

	std::string RegisterUniform(const std::string &typeName, void *uniformPtr, size_t gpuSize, size_t gpuAlignment,
								std::function<void(uint32_t program, const std::string &name, void *ptr)> uploadFunc,
								std::function<void(void *dst, void *ptr)> packFunc) override;

	std::string GetUniformDeclarations() const override;

	/**
	 * Set uniform values using backend uniform functions
	 * This is called during dispatch to upload uniform values to GPU
	 * @param pipeline The backend pipeline handle
	 */
	void		UploadUniformValues(Backend::PipelineHandle pipeline) const;
	uint32_t	GetPushConstantSize() const;

	/**
	 * Get the uniform upload functions for dispatch
	 * @return Vector of uniform entries
	 */
	struct UniformEntry {
		std::string																  name;
		std::string																  typeName;
		void																	 *uniformPtr;
		size_t																	  gpuSize	   = 0;
		size_t																	  gpuAlignment = 0;
		size_t																	  gpuOffset	   = 0;
		std::function<void(uint32_t program, const std::string &name, void *ptr)> uploadFunc;
		std::function<void(void *dst, void *ptr)>								  packFunc;
	};

	const std::vector<UniformEntry> &GetUniformEntries() const {
		return _uniforms;
	}

public:
	// ===================================================================
	// Buffer/Texture Slot Support (Dynamic Resource Switching)
	// ===================================================================

	void										  RegisterBufferSlot(Runtime::BufferSlotBase *slot) override;
	void										  RegisterTextureSlot(Runtime::TextureSlotBase *slot) override;
	void RegisterTexture3D(uint32_t binding, Runtime::PixelFormat format, const std::string &textureName, uint32_t width,
															   uint32_t height, uint32_t depth, bool sampled = false) override;

	const std::vector<Runtime::BufferSlotBase *> &GetBufferSlots() const {
		return _bufferSlots;
	}

	const std::vector<Runtime::TextureSlotBase *> &GetTextureSlots() const {
		return _textureSlots;
	}

public:
	// ===================================================================
	// Callable Function Support
	// ===================================================================

	void					 AddCallableDeclaration(const std::string &declaration) override;
	void					 AddCallableBodyGenerator(std::function<void()> generator) override;
	void					 PushCallableBody() override;
	void					 PopCallableBody() override;
	std::vector<std::string> GetCallableDeclarations() const override;
	std::string				 GenerateCallableBodies() override;

public:
	int WorkSizeX;
	int WorkSizeY;
	int WorkSizeZ;

protected:
	// Callable support
	std::vector<std::string>				_callableDeclarations;
	std::vector<std::function<void()>>		_callableBodyGenerators;
	std::vector<std::string>				_callableBodies;
	std::stack<std::string>					_callableBodyStack;
	std::string								_currentCallableBody;
	bool									_inCallableBody = false;

	uint32_t								_nextBinding	= 0;
	std::vector<BufferInfo>					_buffers;
	std::vector<uint32_t>					_bufferBindings;
	std::unordered_map<uint32_t, uint32_t>	_runtimeBuffers; // binding -> backend buffer handle

	std::vector<TextureInfo>				_textures;
	std::vector<uint32_t>					_textureBindings;
	std::unordered_map<uint32_t, uint32_t>	_runtimeTextures; // binding -> backend texture handle

	std::vector<UniformEntry>				_uniforms;
	int										_nextUniformIndex  = 0;
	size_t									_nextUniformOffset = 0;

	// Slot support for dynamic resource switching
	std::vector<Runtime::BufferSlotBase *>	_bufferSlots;
	std::vector<Runtime::TextureSlotBase *> _textureSlots;

	// Shared memory declarations
	std::vector<std::string>				_sharedMemoryDeclarations;

public:
	// ===================================================================
	// Shared Memory Support
	// ===================================================================

	/**
	 * Push a shared memory declaration to the context
	 * @param declaration The shared memory declaration string
	 */
	void					 PushSharedMemoryDeclaration(const std::string &declaration) override;

	/**
	 * Get all shared memory declarations
	 * @return Vector of shared memory declarations
	 */
	std::vector<std::string> GetSharedMemoryDeclarations() const override;

public:
	// ===================================================================
	// Pipeline Cache (replaces program cache)
	// ===================================================================

	/**
	 * Get the cached backend pipeline handle
	 * @return The cached pipeline handle, or INVALID_PIPELINE_HANDLE if not cached
	 */
	Backend::PipelineHandle GetCachedPipeline() const {
		return _cachedPipeline;
	}

	/**
	 * Set the cached backend pipeline handle
	 * @param pipeline The pipeline handle to cache
	 */
	void SetCachedPipeline(Backend::PipelineHandle pipeline) {
		_cachedPipeline = pipeline;
	}

	/**
	 * Check if a pipeline is cached
	 * @return True if a pipeline is cached
	 */
	bool HasCachedPipeline() const {
		return _cachedPipeline != Backend::INVALID_PIPELINE_HANDLE;
	}

	/**
	 * Invalidate the cached pipeline (force recompilation)
	 */
	void InvalidateCachedPipeline() {
		_cachedPipeline = Backend::INVALID_PIPELINE_HANDLE;
	}

public:
	// ===================================================================
	// Shader Cache Support
	// ===================================================================

	/**
	 * Get the shader source hash for cache lookup
	 * @return The computed shader hash
	 */
	const std::string &GetShaderHash() const {
		return _shaderHash;
	}

	/**
	 * Compute and store the shader hash from source code
	 * This should be called after the complete shader code is generated
	 */
	void						ComputeShaderHash();

	/**
	 * Get the cached shader binary data
	 * @return Reference to cached binary data
	 */
	const std::vector<uint8_t> &GetCachedBinary() const {
		return _cachedBinary;
	}

	/**
	 * Set the cached shader binary data
	 * @param data Binary data to cache
	 */
	void SetCachedBinary(std::vector<uint8_t> data) {
		_cachedBinary = std::move(data);
	}

	/**
	 * Get the cached binary format identifier
	 * @return Format identifier from backend
	 */
	uint32_t GetCachedBinaryFormat() const {
		return _cachedBinaryFormat;
	}

	/**
	 * Set the cached binary format identifier
	 * @param format Format identifier
	 */
	void SetCachedBinaryFormat(uint32_t format) {
		_cachedBinaryFormat = format;
	}

protected:
	Backend::PipelineHandle			_cachedPipeline = Backend::INVALID_PIPELINE_HANDLE;

	// Shader cache support
	std::string						_shaderHash;
	std::vector<uint8_t>			_cachedBinary;
	uint32_t						_cachedBinaryFormat = 0;

	int								_variableIndex;
	int								_dimension;
	std::string						_code;
	std::unordered_set<std::string> _definedStructs;
	std::vector<std::string>		_structNames;
	std::vector<std::string>		_structDefinitions;

	friend class Kernel;
	friend class FragmentKernel2D;
};

} // namespace GPU::Kernel

#endif // EASYGPU_KERNELBUILDCONTEXT_H
