#pragma once

/**
 * @file BuilderContext.h
 * @brief The context for the builder to bind.
 */

#ifndef EASYGPU_BUILDERCONTEXT_H
#define EASYGPU_BUILDERCONTEXT_H

#include <Backend/Backend.h>

#include <Runtime/PixelFormat.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declarations for Slot support
namespace GPU::Runtime {
class BufferSlotBase;
class TextureSlotBase;
} // namespace GPU::Runtime

namespace GPU::IR::Builder {
/**
 * Generation state for callable functions
 */
struct CallableGenState {
	bool declared = false; // Forward declaration generated
	bool defined  = false; // Function body generated
};

// Forward declaration
using PixelFormat = GPU::Runtime::PixelFormat;
/**
 * The context for the builder to bind, which provide a series of abstracted API
 * for the class to accomplish
 */
class BuilderContext {
public:
	virtual ~BuilderContext() = default;

public:
	/**
	 * Pushing the translated code to the context
	 * @param Code The coded that translated from the builder
	 */
	virtual void		PushTranslatedCode(std::string Code) = 0;

	/**
	 * Assigning the variable name
	 * @return The variable name assigned
	 */
	virtual std::string AssignVarName()						 = 0;

public:
	/**
	 * Check if a struct type is already defined
	 * @param TypeName The struct type name
	 * @return True if already defined
	 */
	virtual bool HasStructDefinition(const std::string &TypeName) const							 = 0;

	/**
	 * Add a struct type definition
	 * @param TypeName The struct type name
	 * @param Definition The GLSL struct definition code
	 */
	virtual void AddStructDefinition(const std::string &TypeName, const std::string &Definition) = 0;

	/**
	 * Get all struct definitions
	 * @return Vector of struct definitions in order of registration
	 */
	virtual const std::vector<std::string> &GetStructDefinitions() const						 = 0;

public:
	/**
	 * Allocate a binding slot for buffer/image
	 * @return The allocated binding slot index
	 */
	virtual uint32_t	AllocateBindingSlot()																	 = 0;

	/**
	 * Register a buffer for the kernel
	 * @param binding The binding slot
	 * @param typeName The element type name in GLSL
	 * @param bufferName The buffer variable name
	 * @param mode The buffer access mode
	 */
	virtual void		RegisterBuffer(uint32_t binding, const std::string &typeName, const std::string &bufferName,
									   int mode)																 = 0;

	/**
	 * Get the buffer declarations for GLSL
	 * @return The buffer declaration string
	 */
	virtual std::string GetBufferDeclarations() const															 = 0;

	/**
	 * Get all registered buffer bindings
	 * @return Vector of binding slots
	 */
	virtual const std::vector<uint32_t> &GetBufferBindings() const												 = 0;

	/**
	 * Bind a runtime GPU buffer to a binding slot
	 * This is called by Buffer::Bind() to associate the actual GL buffer with the binding
	 * @param binding The binding slot
	 * @param bufferHandle The OpenGL buffer handle
	 */
	virtual void						 BindRuntimeBuffer(uint32_t binding, Backend::BufferHandle bufferHandle) = 0;

	/**
	 * Get all runtime buffer bindings for dispatch
	 * @return Map of binding slot -> OpenGL buffer handle
	 */
	virtual const std::unordered_map<uint32_t, uint32_t> &GetRuntimeBufferBindings() const						 = 0;

public:
	// ===================================================================
	// Texture Support (2D)
	// ===================================================================

	/**
	 * Allocate a binding slot for texture/image
	 * @return The allocated binding slot index
	 */
	virtual uint32_t AllocateTextureBinding()							= 0;

	/**
	 * Register a 2D texture for the kernel
	 * @param binding The binding slot
	 * @param format The pixel format
	 * @param textureName The texture variable name in GLSL
	 * @param width Texture width
	 * @param height Texture height
	 */
	virtual void RegisterTexture(uint32_t binding, PixelFormat format, const std::string &textureName, uint32_t width,
								 uint32_t height, bool sampled = false) = 0;
	virtual void RegisterTexture3D(uint32_t binding, PixelFormat format, const std::string &textureName, uint32_t width,
								   uint32_t height, uint32_t depth, bool sampled = false) {
	}

	/**
	 * Get the texture declarations for GLSL
	 * @return The texture declaration string
	 */
	virtual std::string					 GetTextureDeclarations() const								  = 0;

	/**
	 * Get all registered texture bindings
	 * @return Vector of binding slots
	 */
	virtual const std::vector<uint32_t> &GetTextureBindings() const									  = 0;

	/**
	 * Bind a runtime GPU texture to a binding slot
	 * This is called by Texture2D::Bind() to associate the actual GL texture with the binding
	 * @param binding The binding slot
	 * @param textureHandle The OpenGL texture handle
	 */
	virtual void						 BindRuntimeTexture(uint32_t binding, uint32_t textureHandle) = 0;

	/**
	 * Get all runtime texture bindings for dispatch
	 * @return Map of binding slot -> OpenGL texture handle
	 */
	virtual const std::unordered_map<uint32_t, uint32_t> &GetRuntimeTextureBindings() const			  = 0;

public:
	// ===================================================================
	// Uniform Support
	// ===================================================================

	/**
	 * Register a uniform variable for the kernel
	 * @param typeName The GLSL type name
	 * @param uniformPtr Pointer to the Uniform object (as void* for type erasure)
	 * @param uploadFunc Function to upload the uniform value to GPU
	 * @return The assigned uniform variable name in GLSL
	 */
	virtual std::string RegisterUniform(
		const std::string &typeName, void *uniformPtr, size_t gpuSize, size_t gpuAlignment,
		std::function<void(uint32_t program, const std::string &name, void *ptr)> uploadFunc,
		std::function<void(void *dst, void *ptr)>								  packFunc) = 0;

	/**
	 * Get the uniform declarations for GLSL
	 * @return The uniform declaration string
	 */
	virtual std::string GetUniformDeclarations() const										= 0;

public:
	// ===================================================================
	// Shared Memory Support
	// ===================================================================

	/**
	 * Push a shared memory declaration to the context
	 * Shared memory is declared at global scope in GLSL (outside main)
	 * @param declaration The shared memory declaration string (e.g., "shared float data[256];")
	 */
	virtual void PushSharedMemoryDeclaration(const std::string &declaration) {
		// Default implementation does nothing - subclasses should override
	}

	/**
	 * Get all shared memory declarations
	 * @return Vector of shared memory declarations
	 */
	virtual std::vector<std::string> GetSharedMemoryDeclarations() const {
		return {};
	}

public:
	// ===================================================================
	// Callable Function Support
	// ===================================================================

	/**
	 * Get the generation state for a callable in this context
	 * @param callablePtr Pointer to the callable object (as void* to avoid template dependency)
	 * @return Reference to the generation state
	 */
	virtual CallableGenState &GetCallableState(const void *callablePtr) {
		return _callableStates[callablePtr];
	}

	/**
	 * Add a callable function declaration (forward declaration)
	 * @param declaration The function prototype string
	 */
	virtual void					 AddCallableDeclaration(const std::string &declaration)	   = 0;

	/**
	 * Register a callable body generator function
	 * This will be called later to generate the function body after main()
	 * @param generator The function that generates the callable body
	 */
	virtual void					 AddCallableBodyGenerator(std::function<void()> generator) = 0;

	/**
	 * Enter callable body generation mode
	 * Pushes a new code buffer for collecting callable body code
	 */
	virtual void					 PushCallableBody()										   = 0;

	/**
	 * Enter callable body generation mode for a known GLSL callable symbol.
	 * Implementations may use the name to associate AD sub-tapes with callable identity.
	 */
	virtual void					 PushCallableBody(const std::string &callableName) {
		(void)callableName;
		PushCallableBody();
	}

	/**
	 * Enter callable body generation mode for a known GLSL callable symbol and
	 * its ordered GLSL parameter names. AD uses the names to remap callable
	 * sub-tapes at call sites without guessing from expression leaves.
	 */
	virtual void					 PushCallableBody(const std::string &callableName,
													  const std::vector<std::string> &parameterNames) {
		(void)parameterNames;
		PushCallableBody(callableName);
	}

	/**
	 * Enter callable body generation mode with ordered parameter names and
	 * GLSL parameter types. AD uses the types to scatter gradients through
	 * callable parameter member/swizzle accesses.
	 */
	virtual void					 PushCallableBody(const std::string &callableName,
													  const std::vector<std::string> &parameterNames,
													  const std::vector<std::string> &parameterTypes) {
		(void)parameterTypes;
		PushCallableBody(callableName, parameterNames);
	}

	/**
	 * Exit callable body generation mode
	 * Pops the callable body code buffer and stores it for later output
	 */
	virtual void					 PopCallableBody()										   = 0;

	/**
	 * Get all callable function declarations
	 * @return Vector of function declarations
	 */
	virtual std::vector<std::string> GetCallableDeclarations() const						   = 0;

	/**
	 * Generate all callable function bodies
	 * This should be called after main() generation
	 * @return The complete callable function definitions string
	 */
	virtual std::string				 GenerateCallableBodies()								   = 0;

public:
	// ===================================================================
	// Slot Support (Dynamic Resource Switching)
	// ===================================================================

	/**
	 * Register a buffer slot for dynamic binding at dispatch time
	 * @param slot Pointer to the BufferSlotBase
	 */
	virtual void RegisterBufferSlot(Runtime::BufferSlotBase *slot) {
	}

	/**
	 * Register that a float buffer needs an int alias for atomic CAS-loop fallback.
	 * @param bufferName The GLSL array name (e.g. "buf_slot_0")
	 */
	virtual void RegisterFloatAtomicBuffer(const std::string &bufferName) {
	}

	/**
	 * Register a texture slot for dynamic binding at dispatch time
	 * @param slot Pointer to the TextureSlotBase
	 */
	virtual void RegisterTextureSlot(Runtime::TextureSlotBase *slot) {
	}

	/**
	 * @brief Register a varying variable for graphics pipeline (VS out -> FS in).
	 * @param name GLSL variable name.
	 * @param glslType GLSL type string.
	 */
	virtual void RegisterVarying(const std::string &name, const std::string &glslType) {
		(void)name;
		(void)glslType;
	}

	/**
	 * @brief Register a uniform buffer (UBO) with this context.
	 * @param typeName GLSL struct type name.
	 * @param ubo Pointer to UniformBufferBase.
	 * @param gpuSize Size of the UBO in GPU memory.
	 * @return The GLSL variable name for accessing the UBO.
	 */
	virtual std::string RegisterUniformBuffer(const std::string &typeName, void *ubo, size_t gpuSize) {
		(void)typeName;
		(void)ubo;
		(void)gpuSize;
		return "";
	}

protected:
	std::unordered_map<const void *, CallableGenState> _callableStates;
};
} // namespace GPU::IR::Builder

#endif // EASYGPU_BUILDERCONTEXT_H
