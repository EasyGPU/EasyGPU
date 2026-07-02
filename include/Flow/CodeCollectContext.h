#pragma once

/**
 * @file CodeCollectContext.h
 * @brief The context for collecting code during control flow lambda execution.
 */

#ifndef EASYGPU_CODECOLLECTCONTEXT_H
#define EASYGPU_CODECOLLECTCONTEXT_H

#include <IR/Builder/BuilderContext.h>

#include <Backend/Backend.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace GPU::Flow {
/**
 * @brief A temporary context for collecting code during lambda execution.
 *
 * Instead of pushing code to the main output, it stores lines in a vector.
 * Delegates variable naming and other operations to the parent context.
 */
class CodeCollectContext : public IR::Builder::BuilderContext {
public:
	/** @brief Default constructor. */
	CodeCollectContext();

public:
	/**
	 * @brief Collect code into internal vector instead of emitting it directly.
	 * @param Code The GLSL code line to collect.
	 */
	void		PushTranslatedCode(std::string Code) override;

	/**
	 * @brief Delegate variable name assignment to the parent context.
	 * @return A unique variable name from the parent context.
	 */
	std::string AssignVarName() override;

	/**
	 * @brief Check if a struct type has been defined.
	 * @param TypeName The struct type name to check.
	 * @return True if the struct type has been defined.
	 */
	bool		HasStructDefinition(const std::string &TypeName) const override;

	/**
	 * @brief Register a struct type definition.
	 * @param TypeName The struct type name.
	 * @param Definition The GLSL struct definition string.
	 */
	void		AddStructDefinition(const std::string &TypeName, const std::string &Definition) override;

	/**
	 * @brief Get all registered struct definitions.
	 * @return Reference to the vector of struct definition strings.
	 */
	const std::vector<std::string> &GetStructDefinitions() const override;

	/**
	 * @brief Allocate a buffer binding slot from the parent context.
	 * @return The allocated binding index.
	 */
	uint32_t						AllocateBindingSlot() override;

	/**
	 * @brief Register a buffer declaration.
	 * @param binding The binding slot index.
	 * @param typeName The buffer element GLSL type name.
	 * @param bufferName The buffer uniform name.
	 * @param mode The buffer access mode.
	 */
	void		RegisterBuffer(uint32_t binding, const std::string &typeName, const std::string &bufferName,
							   int mode) override;

	/**
	 * @brief Get buffer declarations as a GLSL string.
	 * @return The collected buffer declarations.
	 */
	std::string GetBufferDeclarations() const override;

	/**
	 * @brief Get all buffer binding slot indices.
	 * @return Vector of buffer binding indices.
	 */
	const std::vector<uint32_t>					 &GetBufferBindings() const override;

	/**
	 * @brief Bind a runtime buffer handle to a binding slot.
	 * @param binding The binding slot index.
	 * @param bufferHandle The runtime buffer handle.
	 */
	void										  BindRuntimeBuffer(uint32_t binding, uint32_t bufferHandle) override;

	/**
	 * @brief Get the runtime buffer binding map.
	 * @return Map of binding index to runtime buffer handle.
	 */
	const std::unordered_map<uint32_t, uint32_t> &GetRuntimeBufferBindings() const override;

	/**
	 * @brief Allocate a texture binding slot from the parent context.
	 * @return The allocated binding index.
	 */
	uint32_t									  AllocateTextureBinding() override;

	/**
	 * @brief Register a texture declaration.
	 * @param binding The binding slot index.
	 * @param format The pixel format of the texture.
	 * @param textureName The texture uniform name.
	 * @param width The texture width.
	 * @param height The texture height.
	 * @param sampled Whether the texture is a sampler2D (default false for image).
	 */
	void RegisterTexture(uint32_t binding, Runtime::PixelFormat format, const std::string &textureName, uint32_t width,
						 uint32_t height, bool sampled = false) override;

	/**
	 * @brief Get texture declarations as a GLSL string.
	 * @return The collected texture declarations.
	 */
	std::string									  GetTextureDeclarations() const override;

	/**
	 * @brief Get all texture binding slot indices.
	 * @return Vector of texture binding indices.
	 */
	const std::vector<uint32_t>					 &GetTextureBindings() const override;

	/**
	 * @brief Bind a runtime texture handle to a binding slot.
	 * @param binding The binding slot index.
	 * @param textureHandle The runtime texture handle.
	 */
	void										  BindRuntimeTexture(uint32_t binding, uint32_t textureHandle) override;

	/**
	 * @brief Get the runtime texture binding map.
	 * @return Map of binding index to runtime texture handle.
	 */
	const std::unordered_map<uint32_t, uint32_t> &GetRuntimeTextureBindings() const override;

	/**
	 * @brief Register a uniform value with the parent context.
	 * @param typeName The GLSL type name.
	 * @param uniformPtr Pointer to the CPU-side uniform data.
	 * @param gpuSize Size of the uniform on the GPU.
	 * @param gpuAlignment Alignment requirement on the GPU.
	 * @param uploadFunc Function to upload uniform data to the GPU.
	 * @param packFunc Function to pack the uniform data for buffer uploads.
	 * @return The generated uniform name string.
	 */
	std::string RegisterUniform(const std::string &typeName, void *uniformPtr, size_t gpuSize, size_t gpuAlignment,
								std::function<void(uint32_t program, const std::string &name, void *ptr)> uploadFunc,
								std::function<void(void *dst, void *ptr)> packFunc) override;

	/**
	 * @brief Get uniform declarations as a GLSL string.
	 * @return The collected uniform declarations.
	 */
	std::string GetUniformDeclarations() const override;

	/**
	 * @brief Register a callable function forward declaration.
	 * @param declaration The GLSL function prototype string.
	 */
	void		AddCallableDeclaration(const std::string &declaration) override;

	/**
	 * @brief Register a callable body generator for deferred emission.
	 * @param generator A callable that emits the function body GLSL when invoked.
	 */
	void		AddCallableBodyGenerator(std::function<void()> generator) override;

	/** @brief Enter callable body generation mode. */
	void		PushCallableBody() override;

	/** @brief Enter named callable body generation mode. */
	void		PushCallableBody(const std::string &callableName) override;

	/** @brief Enter named callable body generation mode with ordered parameter names. */
	void		PushCallableBody(const std::string &callableName,
								 const std::vector<std::string> &parameterNames) override;

	/** @brief Exit callable body generation mode and emit the function body. */
	void		PopCallableBody() override;

	/**
	 * @brief Get all registered callable function declarations.
	 * @return Vector of GLSL function prototype strings.
	 */
	std::vector<std::string> GetCallableDeclarations() const override;

	/**
	 * @brief Generate all deferred callable function bodies.
	 * @return The generated GLSL function body code.
	 */
	std::string				 GenerateCallableBodies() override;

public:
	/**
	 * @brief Set the parent context for delegation.
	 * @param parent The parent builder context to delegate to.
	 */
	void							SetParentContext(IR::Builder::BuilderContext *parent);

	/**
	 * @brief Get the collected code lines.
	 * @return Const reference to the vector of collected code lines.
	 */
	const std::vector<std::string> &GetCollectedCode() const;

	/**
	 * @brief Move out the collected code lines.
	 * @return Vector of collected code lines (moved).
	 */
	std::vector<std::string>		ReleaseCollectedCode();

	/** @brief Clear all collected code lines. */
	void							Clear();

	/**
	 * @brief Get the callable generation state, delegating to the parent context.
	 *
	 * This ensures Callable declarations are not duplicated when used inside control flow.
	 * @param callablePtr The callable pointer used as a key.
	 * @return Reference to the callable generation state.
	 */
	IR::Builder::CallableGenState  &GetCallableState(const void *callablePtr) override;

	void							RegisterFloatAtomicBuffer(const std::string &bufferName) override {
		if (_parentContext)
			_parentContext->RegisterFloatAtomicBuffer(bufferName);
	}

	void RegisterBufferSlot(Runtime::BufferSlotBase *slot) override {
		if (_parentContext)
			_parentContext->RegisterBufferSlot(slot);
	}

	void RegisterTextureSlot(Runtime::TextureSlotBase *slot) override {
		if (_parentContext)
			_parentContext->RegisterTextureSlot(slot);
	}

	void PushSharedMemoryDeclaration(const std::string &declaration) override {
		if (_parentContext)
			_parentContext->PushSharedMemoryDeclaration(declaration);
	}

	std::vector<std::string> GetSharedMemoryDeclarations() const override {
		if (_parentContext)
			return _parentContext->GetSharedMemoryDeclarations();
		return {};
	}

private:
	IR::Builder::BuilderContext *_parentContext;
	std::vector<std::string>	 _collectedCode;
};
} // namespace GPU::Flow

#endif // EASYGPU_CODECOLLECTCONTEXT_H
