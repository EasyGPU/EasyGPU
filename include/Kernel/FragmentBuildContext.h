#pragma once

/**
 * @file FragmentBuildContext.h
 * @brief Build context for FragmentKernel - generates vertex/fragment shader pair.
 */

#ifndef EASYGPU_FRAGMENT_BUILD_CONTEXT_H
#define EASYGPU_FRAGMENT_BUILD_CONTEXT_H

#include <Kernel/KernelBuildContext.h>

namespace GPU::Kernel {

/**
 * @brief Build context for fragment shader generation.
 *
 * Generates VS + FS pair for rasterization-based rendering.
 */
class FragmentBuildContext : public KernelBuildContext {
public:
	/**
	 * @brief Construct a fragment build context.
	 * @param width Rendering width in pixels.
	 * @param height Rendering height in pixels.
	 */
	FragmentBuildContext(uint32_t width, uint32_t height);

	~FragmentBuildContext() override							  = default;

	// Disable copy
	FragmentBuildContext(const FragmentBuildContext &)			  = delete;
	FragmentBuildContext &operator=(const FragmentBuildContext &) = delete;

	// Allow move
	FragmentBuildContext(FragmentBuildContext &&)				  = default;
	FragmentBuildContext &operator=(FragmentBuildContext &&)	  = default;

public:
	// ===================================================================
	// Overrides from KernelBuildContext
	// ===================================================================

	/**
	 * @brief Get complete shader program source (VS + FS).
	 *
	 * Overrides to generate vertex/fragment shader pair instead of compute shader.
	 * @return The full shader program source code.
	 */
	std::string GetCompleteCode() override;

	/**
	 * @brief Get vertex shader source only.
	 * @return The vertex shader GLSL source code.
	 */
	std::string GetVertexShaderSource();

	/**
	 * @brief Get fragment shader source only.
	 * @return The fragment shader GLSL source code.
	 */
	std::string GetFragmentShaderSource();

	/**
	 * @brief Get texture declarations for fragment shader.
	 *
	 * Uses sampler2D instead of image2D for rasterization pipeline.
	 * @return GLSL sampler declaration string.
	 */
	std::string GetTextureDeclarations() const override;

public:
	// ===================================================================
	// Fragment-specific Methods
	// ===================================================================

	/**
	 * @brief Get current resolution width.
	 * @return Width in pixels.
	 */
	uint32_t GetWidth() const {
		return _width;
	}

	/**
	 * @brief Get current resolution height.
	 * @return Height in pixels.
	 */
	uint32_t GetHeight() const {
		return _height;
	}

	/**
	 * @brief Set resolution (called on window resize).
	 * @param width New width in pixels.
	 * @param height New height in pixels.
	 */
	void SetResolution(uint32_t width, uint32_t height);

	/**
	 * @brief Mark that shader needs recompilation.
	 */
	void InvalidateShader() {
		InvalidateCachedPipeline();
	}

	/**
	 * @brief Check if shader is valid for current state.
	 * @return true if a valid pipeline is cached.
	 */
	bool IsShaderValid() const {
		return HasCachedPipeline();
	}

protected:
	/**
	 * Generate common headers (structs, uniforms, callables)
	 */
	void			   GenerateCommonHeaders(std::ostringstream &oss);

	/**
	 * Generate vertex shader source
	 * Creates a simple pass-through shader that generates a full-screen triangle
	 */
	std::string		   GenerateVertexShader();

	/**
	 * Generate fragment shader source
	 * Wraps user code with necessary declarations and output
	 */
	std::string		   GenerateFragmentShader();

	/**
	 * Generate version directive
	 */
	static std::string GenerateHeader();

protected:
	uint32_t _width;
	uint32_t _height;
};

} // namespace GPU::Kernel

#endif // EASYGPU_FRAGMENT_BUILD_CONTEXT_H
