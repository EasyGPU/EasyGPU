#pragma once

/**
 * @file KernelBuildContext.h
 * @brief Kernel build context with backend integration.
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
#include <unordered_set>
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
 * @brief Build context for assembling GLSL compute shaders from IR nodes.
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
		uint32_t			 depth	 = 1;
		uint32_t			 dimension = 2;
		bool				 sampled = false;
	};

	struct UniformEntry;

	/**
	 * @brief Construct a kernel build context with the given work dimension.
	 * @param Dimension The dimension of the kernel (1, 2, or 3).
	 */
	KernelBuildContext(int Dimension);

	/**
	 * Destructor - cleans up cached pipeline if any
	 */
	~KernelBuildContext();

public:
	/**
	 * @brief Push a line of translated GLSL code into the kernel body.
	 * @param Code The GLSL code string to append.
	 */
	void				PushTranslatedCode(std::string Code) override;

	/**
	 * @brief Assign a unique variable name for IR lowering.
	 * @return A new unique variable name.
	 */
	std::string			AssignVarName() override;

	/**
	 * @brief Get the complete kernel code including struct definitions.
	 * @return The complete GLSL code.
	 */
	virtual std::string GetCompleteCode();

public:
	/**
	 * @brief Check if a struct type has been registered.
	 * @param TypeName The name of the struct type.
	 * @return true if the struct definition is known.
	 */
	bool HasStructDefinition(const std::string &TypeName) const override;

	/**
	 * @brief Register a user-defined struct for GLSL code generation.
	 * @param TypeName The name of the struct type.
	 * @param Definition The GLSL struct definition string.
	 */
	void AddStructDefinition(const std::string &TypeName, const std::string &Definition) override;

	/**
	 * @brief Get all registered struct definitions.
	 * @return Vector of GLSL struct definition strings.
	 */
	const std::vector<std::string> &GetStructDefinitions() const override;

public:
	/**
	 * @brief Allocate a new buffer binding slot.
	 * @return The allocated binding index.
	 */
	uint32_t AllocateBindingSlot() override;

	/**
	 * @brief Get the next available binding slot without allocating it.
	 */
	uint32_t GetNextBinding() const {
		return _nextBinding;
	}

	/**
	 * @brief Register a buffer with the given binding slot.
	 * @param binding The binding slot index.
	 * @param typeName The GLSL type name for the buffer element.
	 * @param bufferName The GLSL variable name for the buffer.
	 * @param mode The buffer access mode (read/write).
	 */
	void		RegisterBuffer(uint32_t binding, const std::string &typeName, const std::string &bufferName,
							   int mode) override;

	/**
	 * @brief Get all buffer declarations as GLSL source.
	 * @return GLSL buffer declaration string.
	 */
	std::string GetBufferDeclarations() const override;

	/**
	 * @brief Get the ordered list of buffer bindings.
	 * @return Vector of binding slot indices.
	 */
	const std::vector<uint32_t> &GetBufferBindings() const override {
		return _bufferBindings;
	}

	/**
	 * @brief Get the registered buffer metadata.
	 * @return Vector of BufferInfo records.
	 */
	const std::vector<BufferInfo> &GetBufferInfos() const {
		return _buffers;
	}

	/**
	 * @brief Bind a runtime GPU buffer to a binding slot.
	 *
	 * Called by Buffer::Bind() to associate the actual backend buffer with the binding.
	 * @param binding The binding slot.
	 * @param bufferHandle The backend buffer handle.
	 */
	void BindRuntimeBuffer(uint32_t binding, Backend::BufferHandle bufferHandle) override {
		auto it = _runtimeBuffers.find(binding);
		if (it == _runtimeBuffers.end() || it->second != bufferHandle) {
			_runtimeBuffers[binding] = bufferHandle;
			InvalidateBindings();
		}
	}

	/**
	 * @brief Get the mapping of binding slots to backend buffer handles.
	 * @return Map of binding index to backend buffer handle.
	 */
	const std::unordered_map<uint32_t, uint32_t> &GetRuntimeBufferBindings() const override {
		return _runtimeBuffers;
	}

public:
	// ===================================================================
	// Texture Support (2D)
	// ===================================================================

	/**
	 * @brief Allocate a new texture binding slot.
	 * @return The allocated binding index.
	 */
	uint32_t AllocateTextureBinding() override;

	/**
	 * @brief Register a 2D texture with the given binding slot.
	 * @param binding The binding slot index.
	 * @param format The pixel format of the texture.
	 * @param textureName The GLSL variable name for the texture.
	 * @param width Texture width in pixels.
	 * @param height Texture height in pixels.
	 * @param sampled Whether to use sampler2D instead of image2D.
	 */
	void RegisterTexture(uint32_t binding, Runtime::PixelFormat format, const std::string &textureName, uint32_t width,
						 uint32_t height, bool sampled = false) override;

	/**
	 * @brief Get all texture declarations as GLSL source.
	 * @return GLSL texture/image declaration string.
	 */
	std::string					 GetTextureDeclarations() const override;

	/**
	 * @brief Get the ordered list of texture bindings.
	 * @return Vector of texture binding indices.
	 */
	const std::vector<uint32_t> &GetTextureBindings() const override {
		return _textureBindings;
	}

	/**
	 * @brief Get the registered texture metadata.
	 * @return Vector of TextureInfo records.
	 */
	const std::vector<TextureInfo> &GetTextureInfos() const {
		return _textures;
	}

	/**
	 * @brief Find texture metadata by binding slot.
	 * @param binding The binding slot to look up.
	 * @return Pointer to TextureInfo, or nullptr if not found.
	 */
	const TextureInfo *FindTextureInfo(uint32_t binding) const;

	/**
	 * @brief Bind a runtime GPU texture to a binding slot.
	 *
	 * Called by Texture2D::Bind() to associate the actual backend texture with the binding.
	 * @param binding The binding slot.
	 * @param textureHandle The backend texture handle.
	 */
	void			   BindRuntimeTexture(uint32_t binding, uint32_t textureHandle) override {
		auto it = _runtimeTextures.find(binding);
		if (it == _runtimeTextures.end() || it->second != textureHandle) {
			_runtimeTextures[binding] = textureHandle;
			InvalidateBindings();
		}
	}

	/** @brief Override the sampler used by a sampled runtime texture binding. */
	void BindRuntimeTextureSampler(uint32_t binding, const Backend::SamplerDesc &sampler) override {
		_runtimeTextureSamplers[binding] = sampler;
		InvalidateBindings();
	}

	/**
	 * @brief Get the mapping of binding slots to backend texture handles.
	 * @return Map of binding index to backend texture handle.
	 */
	const std::unordered_map<uint32_t, uint32_t> &GetRuntimeTextureBindings() const override {
		return _runtimeTextures;
	}

	/** @brief Get sampler descriptor overrides keyed by texture binding slot. */
	const std::unordered_map<uint32_t, Backend::SamplerDesc> &GetRuntimeTextureSamplerBindings() const override {
		return _runtimeTextureSamplers;
	}

public:
	// ===================================================================
	// Uniform Support
	// ===================================================================

	/**
	 * @brief Register a uniform variable and return its GLSL declaration name.
	 * @param typeName The GLSL type name.
	 * @param uniformPtr Host-side pointer to uniform data.
	 * @param gpuSize Size of the uniform in GPU memory.
	 * @param gpuAlignment Required GPU alignment.
	 * @param uploadFunc Function to upload data to the GPU.
	 * @param packFunc Function to pack host data into the GPU buffer.
	 * @return The GLSL declaration string.
	 */
	std::string RegisterUniform(const std::string &typeName, void *uniformPtr, size_t gpuSize, size_t gpuAlignment,
								std::function<void(uint32_t program, const std::string &name, void *ptr)> uploadFunc,
								std::function<void(void *dst, void *ptr)> packFunc) override;

	/**
	 * @brief Get the GLSL uniform block declaration.
	 * @return GLSL uniform block source string.
	 */
	std::string GetUniformDeclarations() const override;

	/**
	 * @brief Upload uniform values to GPU during dispatch.
	 * @param pipeline The backend pipeline handle.
	 */
	void		UploadUniformValues(Backend::PipelineHandle pipeline) const;

	/**
	 * @brief Get the push constant size needed for uniforms.
	 * @return Push constant buffer size in bytes.
	 */
	uint32_t	GetPushConstantSize() const;

	/**
	 * @brief Metadata for a registered uniform variable.
	 */
	struct UniformEntry {
		std::string name;			  /**< @brief GLSL variable name. */
		std::string typeName;		  /**< @brief GLSL type name. */
		void	   *uniformPtr;		  /**< @brief Host-side pointer to uniform data. */
		size_t		gpuSize		 = 0; /**< @brief Size in GPU memory. */
		size_t		gpuAlignment = 0; /**< @brief Required GPU alignment. */
		size_t		gpuOffset	 = 0; /**< @brief Offset in the push constant buffer. */
		std::function<void(uint32_t program, const std::string &name, void *ptr)>
												  uploadFunc; /**< @brief Function to upload data to GPU. */
		std::function<void(void *dst, void *ptr)> packFunc;	  /**< @brief Function to pack host data into GPU buffer. */
	};

	/**
	 * @brief Get the registered uniform entries.
	 * @return Vector of UniformEntry records.
	 */
	const std::vector<UniformEntry> &GetUniformEntries() const {
		return _uniforms;
	}

public:
	// ===================================================================
	// Buffer/Texture Slot Support (Dynamic Resource Switching)
	// ===================================================================

	/**
	 * @brief Register a buffer slot for dynamic resource switching.
	 * @param slot The buffer slot to register.
	 */
	void RegisterBufferSlot(Runtime::BufferSlotBase *slot) override;

	/**
	 * @brief Register a texture slot for dynamic resource switching.
	 * @param slot The texture slot to register.
	 */
	void RegisterTextureSlot(Runtime::TextureSlotBase *slot) override;

	/**
	 * @brief Register a 3D texture with the given binding slot.
	 * @param binding The binding slot index.
	 * @param format The pixel format of the texture.
	 * @param textureName The GLSL variable name for the texture.
	 * @param width Texture width in pixels.
	 * @param height Texture height in pixels.
	 * @param depth Texture depth in pixels.
	 * @param sampled Whether to use sampler3D instead of image3D.
	 */
	void RegisterTexture3D(uint32_t binding, Runtime::PixelFormat format, const std::string &textureName,
						   uint32_t width, uint32_t height, uint32_t depth, bool sampled = false) override;

	/**
	 * @brief Get all registered buffer slots.
	 * @return Vector of buffer slot pointers.
	 */
	const std::vector<Runtime::BufferSlotBase *> &GetBufferSlots() const {
		return _bufferSlots;
	}

	/**
	 * @brief Get all registered texture slots.
	 * @return Vector of texture slot pointers.
	 */
	const std::vector<Runtime::TextureSlotBase *> &GetTextureSlots() const {
		return _textureSlots;
	}

	/**
	 * @brief Get the binding assigned to a buffer slot within this kernel context.
	 * @param slot The buffer slot.
	 * @return The binding index, or the slot's global binding if not found.
	 */
	uint32_t									 GetBufferSlotBinding(Runtime::BufferSlotBase *slot) const;

	/**
	 * @brief Get the binding assigned to a texture slot within this kernel context.
	 * @param slot The texture slot.
	 * @return The binding index, or the slot's global binding if not found.
	 */
	uint32_t									 GetTextureSlotBinding(Runtime::TextureSlotBase *slot) const;

	/**
	 * @brief Get resource bindings for dispatch, rebuilding only after resources change.
	 */
	const std::vector<Backend::ResourceBinding> &GetCachedBindings();

	/**
	 * @brief Mark cached resource bindings as dirty.
	 */
	void										 InvalidateBindings() {
		_bindingsDirty = true;
	}

	/**
	 * @brief Get the pre-computed barrier type for this kernel.
	 */
	Backend::BarrierType GetRequiredBarrierType();

	/**
	 * @brief Mark cached barrier requirements as dirty.
	 */
	void				 InvalidateBarrierType() {
		_barrierComputed = false;
	}

public:
	// ===================================================================
	// Callable Function Support
	// ===================================================================

	/**
	 * @brief Register a callable function declaration.
	 * @param declaration The GLSL function declaration.
	 */
	void					 AddCallableDeclaration(const std::string &declaration) override;

	/**
	 * @brief Register a generator for a callable function body.
	 * @param generator A callable that generates the function body.
	 */
	void					 AddCallableBodyGenerator(std::function<void()> generator) override;

	/**
	 * @brief Begin capturing a callable function body.
	 */
	void					 PushCallableBody() override;

	/**
	 * @brief Begin capturing a named callable function body.
	 * @param callableName Emitted GLSL function name for AD sub-tape identity.
	 */
	void					 PushCallableBody(const std::string &callableName) override;

	/**
	 * @brief Begin capturing a named callable function body with ordered parameter names.
	 * @param callableName Emitted GLSL function name for AD sub-tape identity.
	 * @param parameterNames GLSL parameter names in declaration order.
	 */
	void					 PushCallableBody(const std::string &callableName,
											  const std::vector<std::string> &parameterNames) override;

	/**
	 * @brief Begin capturing a named callable function body with ordered parameter names and types.
	 * @param callableName Emitted GLSL function name for AD sub-tape identity.
	 * @param parameterNames GLSL parameter names in declaration order.
	 * @param parameterTypes GLSL parameter types in declaration order.
	 */
	void					 PushCallableBody(const std::string &callableName,
											  const std::vector<std::string> &parameterNames,
											  const std::vector<std::string> &parameterTypes) override;

	/**
	 * @brief End capturing a callable function body.
	 */
	void					 PopCallableBody() override;

	/**
	 * @brief Get all registered callable declarations.
	 * @return Vector of GLSL function declaration strings.
	 */
	std::vector<std::string> GetCallableDeclarations() const override;

	/**
	 * @brief Generate all callable function bodies as GLSL source.
	 * @return Concatenated GLSL function body string.
	 */
	std::string				 GenerateCallableBodies() override;

public:
	int WorkSizeX; /**< @brief Local work group size in the X dimension. */
	int WorkSizeY; /**< @brief Local work group size in the Y dimension. */
	int WorkSizeZ; /**< @brief Local work group size in the Z dimension. */

protected:
	// Callable support
	struct CallableBodyFrame {
		std::string body;
		bool		inCallableBody = false;
	};

	std::vector<std::string>								 _callableDeclarations;
	std::vector<std::function<void()>>						 _callableBodyGenerators;
	std::vector<std::string>								 _callableBodies;
	std::stack<CallableBodyFrame>							 _callableBodyStack;
	std::string												 _currentCallableBody;
	bool													 _inCallableBody = false;
	size_t													 _nextCallableBodyIndex = 0;

	uint32_t												 _nextBinding	 = 0;
	std::vector<BufferInfo>									 _buffers;
	std::vector<uint32_t>									 _bufferBindings;
	std::unordered_map<uint32_t, uint32_t>					 _runtimeBuffers; // binding -> backend buffer handle

	std::vector<TextureInfo>								 _textures;
	std::vector<uint32_t>									 _textureBindings;
	std::unordered_map<uint32_t, uint32_t>					 _runtimeTextures; // binding -> backend texture handle
	std::unordered_map<uint32_t, Backend::SamplerDesc>	 _runtimeTextureSamplers;

	std::vector<UniformEntry>								 _uniforms;
	int														 _nextUniformIndex	= 0;
	size_t													 _nextUniformOffset = 0;

	// Slot support for dynamic resource switching
	std::vector<Runtime::BufferSlotBase *>					 _bufferSlots;
	std::vector<Runtime::TextureSlotBase *>					 _textureSlots;
	std::unordered_map<void *, std::string>					 _uniformBufferNames;

	// Per-context slot binding mappings (required because slots are shared across kernels)
	std::unordered_map<Runtime::BufferSlotBase *, uint32_t>	 _bufferSlotBindings;
	std::unordered_map<Runtime::TextureSlotBase *, uint32_t> _textureSlotBindings;

	// Shared memory declarations
	std::vector<std::string>								 _sharedMemoryDeclarations;

	// Float atomic buffer tracking (buffer names needing int alias for CAS-loop fallback)
	std::unordered_set<std::string>							 _floatAtomicBuffers;

public:
	// ===================================================================
	// Shared Memory Support
	// ===================================================================

	/**
	 * @brief Push a shared memory declaration to the context.
	 * @param declaration The shared memory declaration string.
	 */
	void					 PushSharedMemoryDeclaration(const std::string &declaration) override;

	/**
	 * @brief Get all shared memory declarations.
	 * @return Vector of shared memory declaration strings.
	 */
	std::vector<std::string> GetSharedMemoryDeclarations() const override;

public:
	// ===================================================================
	// Float Atomic Support
	// ===================================================================

	void		RegisterFloatAtomicBuffer(const std::string &bufferName) override;
	void		RegisterVarying(const std::string &name, const std::string &glslType) override;
	std::string RegisterUniformBuffer(const std::string &typeName, void *ubo, size_t gpuSize) override;
	bool		HasFloatAtomics() const {
		return !_floatAtomicBuffers.empty();
	}

public:
	// ===================================================================
	// Pipeline Cache (replaces program cache)
	// ===================================================================

	/**
	 * @brief Get the cached backend pipeline handle.
	 * @return The cached pipeline handle, or INVALID_PIPELINE_HANDLE if not cached.
	 */
	Backend::PipelineHandle GetCachedPipeline() const {
		return _cachedPipeline;
	}

	/**
	 * @brief Set the cached backend pipeline handle.
	 * @param pipeline The pipeline handle to cache.
	 */
	void SetCachedPipeline(Backend::PipelineHandle pipeline) {
		_cachedPipeline = pipeline;
	}

	/**
	 * @brief Check if a pipeline is currently cached.
	 * @return True if a pipeline is cached.
	 */
	bool HasCachedPipeline() const {
		return _cachedPipeline != Backend::INVALID_PIPELINE_HANDLE;
	}

	/**
	 * @brief Invalidate the cached pipeline (force recompilation).
	 */
	void InvalidateCachedPipeline();

	/**
	 * @brief Set SPIR-V optimization preset for backends that support it.
	 * @param level Optimization preset used during shader compilation.
	 */
	void SetOptimizationLevel(Backend::ShaderOptimizationLevel level) {
		if (_optimizationLevel != level) {
			InvalidateCachedPipeline();
			_shaderHash.clear();
			_cachedBinary.clear();
			_cachedBinaryFormat = 0;
			_optimizationLevel  = level;
		}
	}

	/**
	 * @brief Get SPIR-V optimization preset.
	 * @return Optimization preset used during shader compilation.
	 */
	Backend::ShaderOptimizationLevel GetOptimizationLevel() const {
		return _optimizationLevel;
	}

public:
	// ===================================================================
	// Shader Cache Support
	// ===================================================================

	/**
	 * @brief Get the shader source hash for cache lookup.
	 * @return The computed shader hash.
	 */
	const std::string &GetShaderHash() const {
		return _shaderHash;
	}

	/**
	 * @brief Compute and store the shader hash from source code.
	 *
	 * Should be called after the complete shader code is generated.
	 */
	void						ComputeShaderHash();

	/**
	 * @brief Get the cached shader binary data.
	 * @return Reference to cached binary data.
	 */
	const std::vector<uint8_t> &GetCachedBinary() const {
		return _cachedBinary;
	}

	/**
	 * @brief Set the cached shader binary data.
	 * @param data Binary data to cache.
	 */
	void SetCachedBinary(std::vector<uint8_t> data) {
		_cachedBinary = std::move(data);
	}

	/**
	 * @brief Get the cached binary format identifier.
	 * @return Format identifier from backend.
	 */
	uint32_t GetCachedBinaryFormat() const {
		return _cachedBinaryFormat;
	}

	/**
	 * @brief Set the cached binary format identifier.
	 * @param format Format identifier.
	 */
	void SetCachedBinaryFormat(uint32_t format) {
		_cachedBinaryFormat = format;
	}

protected:
	Backend::PipelineHandle				  _cachedPipeline = Backend::INVALID_PIPELINE_HANDLE;

	// Shader cache support
	std::string							  _shaderHash;
	std::vector<uint8_t>				  _cachedBinary;
	uint32_t							  _cachedBinaryFormat = 0;
	Backend::ShaderOptimizationLevel	  _optimizationLevel =
		Backend::ShaderOptimizationLevel::Aggressive;
	std::vector<Backend::ResourceBinding> _cachedBindings;
	std::vector<Backend::BufferHandle>	  _cachedBufferSlotHandles;
	std::vector<Backend::TextureHandle>	  _cachedTextureSlotHandles;
	std::vector<Runtime::PixelFormat>	  _cachedTextureSlotFormats;
	std::vector<bool>					  _cachedTextureSlotSampled;
	bool								  _bindingsDirty	   = true;
	Backend::BarrierType				  _requiredBarrierType = Backend::BarrierType::None;
	bool								  _barrierComputed	   = false;

	int									  _variableIndex;
	int									  _dimension;
	std::string							  _code;
	std::unordered_set<std::string>		  _definedStructs;
	std::vector<std::string>			  _structNames;
	std::vector<std::string>			  _structDefinitions;

	friend class Kernel;
	friend class FragmentKernel2D;
};

} // namespace GPU::Kernel

#endif // EASYGPU_KERNELBUILDCONTEXT_H
