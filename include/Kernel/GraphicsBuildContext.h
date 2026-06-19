#pragma once

/**
 * @file GraphicsBuildContext.h
 * @brief Build context for graphics pipelines — generates vertex + fragment shader pair.
 *
 * Inherits from KernelBuildContext and manages two-stage code generation:
 *  1. Vertex shader: collects varying outputs, vertex inputs, user VS DSL code
 *  2. Fragment shader: collects varying inputs, texture samplers, user FS DSL code
 */

#ifndef EASYGPU_GRAPHICS_BUILD_CONTEXT_H
#define EASYGPU_GRAPHICS_BUILD_CONTEXT_H

#include <Backend/Backend.h>
#include <Kernel/KernelBuildContext.h>

#include <string>
#include <stdexcept>
#include <vector>

namespace GPU::Kernel {

/**
 * @brief Build context for graphics pipeline code generation.
 *
 * Generates a vertex shader + fragment shader pair from C++ DSL lambdas.
 * Varying<T> instances registered during VS/FS construction are automatically
 * matched by name and assigned layout locations.
 */
class GraphicsBuildContext : public KernelBuildContext {
public:
	/** @brief Information about a registered varying variable. */
	struct VaryingInfo {
		std::string name;	  // GLSL variable name
		std::string glslType; // GLSL type (vec3, vec2, etc.)
		uint32_t	location; // Assigned layout location
	};

	/**
	 * @brief Construct a graphics build context.
	 */
	GraphicsBuildContext();

	~GraphicsBuildContext() override							  = default;

	// Disable copy
	GraphicsBuildContext(const GraphicsBuildContext &)			  = delete;
	GraphicsBuildContext &operator=(const GraphicsBuildContext &) = delete;
	GraphicsBuildContext(GraphicsBuildContext &&)				  = default;
	GraphicsBuildContext &operator=(GraphicsBuildContext &&)	  = default;

public:
	// ===================================================================
	// Varying Registration
	// ===================================================================

	/**
	 * @brief Register a varying variable.
	 * @param name GLSL variable name.
	 * @param glslType GLSL type string.
	 */
	void							RegisterVarying(const std::string &name, const std::string &glslType) override;

	/**
	 * @brief Get all registered varyings.
	 * @return Vector of VaryingInfo records.
	 */
	const std::vector<VaryingInfo> &GetVaryings() const {
		return _varyings;
	}

public:
	// ===================================================================
	// Vertex Format
	// ===================================================================

	/**
	 * @brief Set the vertex input layout for vertex shader generation.
	 * @param layout Vertex attribute layout entries.
	 */
	void SetVertexLayout(const std::vector<Backend::VertexLayoutEntry> &layout);

	/**
	 * @brief Set the GLSL used to construct the user vertex variable from attributes.
	 */
	void SetVertexInputSetupCode(std::string code) {
		_vertexInputSetupCode = std::move(code);
	}

	/**
	 * @brief Get the vertex layout entries.
	 */
	const std::vector<Backend::VertexLayoutEntry> &GetVertexLayout() const {
		return _vertexLayout;
	}

public:
	// ===================================================================
	// Code Generation
	// ===================================================================

	/**
	 * @brief Get the complete vertex shader GLSL source code.
	 * @return Vertex shader GLSL source.
	 */
	std::string GetVertexShaderCode();

	/**
	 * @brief Get the complete fragment shader GLSL source code.
	 * @return Fragment shader GLSL source.
	 */
	std::string GetFragmentShaderCode();

	/**
	 * @brief Get combined VS + FS source for debugging.
	 * @return Full shader source string.
	 */
	std::string GetCompleteCode() override;

	/**
	 * @brief Get texture declarations for fragment shader (uses sampler2D).
	 */
	std::string GetTextureDeclarations() const override;

	/**
	 * @brief Push stage-specific translated GLSL code.
	 */
	void PushTranslatedCode(std::string Code) override;

protected:
	/** @brief Generate common GLSL headers (#version, extensions, structs). */
	std::string GenerateCommonHeaders();

	/** @brief Generate vertex input declarations. */
	std::string GenerateVertexInputs();

	/** @brief Generate varying output declarations (for VS). */
	std::string GenerateVaryingOutputs();

	/** @brief Generate varying input declarations (for FS). */
	std::string GenerateVaryingInputs();

	/** @brief Generate the vertex shader main function. */
	std::string GenerateVertexShaderMain();

	/** @brief Generate the fragment shader main function. */
	std::string GenerateFragmentShaderMain();

public:
	// ===================================================================
	// Stage Management
	// ===================================================================

	/** @brief Call before executing VS lambda: captures current _code as VS body. */
	void EndVSStage() {
		if (_stage != ShaderStage::VS) {
			throw std::runtime_error("GraphicsBuildContext::EndVSStage called outside vertex stage");
		}
		_vsBodyCode = std::move(_code);
		_code.clear();
		_stage = ShaderStage::Idle;
	}

	/** @brief Call after executing FS lambda: captures current _code as FS body. */
	void EndFSStage() {
		if (_stage != ShaderStage::FS) {
			throw std::runtime_error("GraphicsBuildContext::EndFSStage called outside fragment stage");
		}
		_fsBodyCode = std::move(_code);
		_code.clear();
		_stage = ShaderStage::Done;
	}

	void BeginVSStage() {
		if (_stage != ShaderStage::Idle) {
			throw std::runtime_error("GraphicsBuildContext::BeginVSStage called in invalid state");
		}
		_code.clear();
		_stage = ShaderStage::VS;
	}

	void BeginFSStage() {
		if (_stage != ShaderStage::Idle) {
			throw std::runtime_error("GraphicsBuildContext::BeginFSStage called in invalid state");
		}
		_code.clear();
		_stage = ShaderStage::FS;
	}

protected:
	enum class ShaderStage { Idle, VS, FS, Done };

	std::vector<VaryingInfo>				_varyings;
	std::vector<Backend::VertexLayoutEntry> _vertexLayout;
	std::string								_vsBodyCode;
	std::string								_fsBodyCode;
	std::string								_vertexInputSetupCode;
	ShaderStage							_stage = ShaderStage::Idle;
};

} // namespace GPU::Kernel

#endif // EASYGPU_GRAPHICS_BUILD_CONTEXT_H
