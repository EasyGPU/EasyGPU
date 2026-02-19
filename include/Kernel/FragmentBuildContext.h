#pragma once

/**
 * FragmentBuildContext.h:
 *      @Descripiton    :   Build context for FragmentKernel - generates vertex/fragment shader pair
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/19/2026
 * 
 * Inherits from KernelBuildContext to reuse all resource management.
 * Overrides code generation to produce VS/FS pair instead of compute shader.
 */

#ifndef EASYGPU_FRAGMENT_BUILD_CONTEXT_H
#define EASYGPU_FRAGMENT_BUILD_CONTEXT_H

#include <Kernel/KernelBuildContext.h>

namespace GPU::Kernel {

    /**
     * Build context for fragment shader generation
     * Generates VS + FS pair for rasterization-based rendering
     */
    class FragmentBuildContext : public KernelBuildContext {
    public:
        /**
         * Construct a fragment build context
         * @param width Rendering width
         * @param height Rendering height
         */
        FragmentBuildContext(uint32_t width, uint32_t height);
        
        ~FragmentBuildContext() override = default;

        // Disable copy
        FragmentBuildContext(const FragmentBuildContext&) = delete;
        FragmentBuildContext& operator=(const FragmentBuildContext&) = delete;

        // Allow move
        FragmentBuildContext(FragmentBuildContext&&) = default;
        FragmentBuildContext& operator=(FragmentBuildContext&&) = default;

    public:
        // ===================================================================
        // Overrides from KernelBuildContext
        // ===================================================================
        
        /**
         * Get complete shader program source (VS + FS)
         * Overrides to generate vertex/fragment shader pair instead of compute shader
         */
        std::string GetCompleteCode() override;

        /**
         * Get vertex shader source only
         */
        std::string GetVertexShaderSource();

        /**
         * Get fragment shader source only
         */
        std::string GetFragmentShaderSource();

        /**
         * Get texture declarations for fragment shader
         * Uses sampler2D instead of image2D
         */
        std::string GetTextureDeclarations() const override;

        /**
         * Get 3D texture declarations for fragment shader
         * Uses sampler3D instead of image3D
         */
        std::string GetTexture3DDeclarations() const override;

    public:
        // ===================================================================
        // Fragment-specific Methods
        // ===================================================================
        
        /**
         * Get current resolution width
         */
        uint32_t GetWidth() const { return _width; }

        /**
         * Get current resolution height
         */
        uint32_t GetHeight() const { return _height; }

        /**
         * Set resolution (called on window resize)
         */
        void SetResolution(uint32_t width, uint32_t height);

        /**
         * Mark that shader needs recompilation
         */
        void InvalidateShader() { InvalidateCachedProgram(); }

        /**
         * Check if shader is valid for current state
         */
        bool IsShaderValid() const { return HasCachedProgram(); }

    protected:
        /**
         * Generate common headers (structs, uniforms, callables)
         */
        void GenerateCommonHeaders(std::ostringstream& oss);

        /**
         * Generate vertex shader source
         * Creates a simple pass-through shader that generates a full-screen triangle
         */
        std::string GenerateVertexShader();

        /**
         * Generate fragment shader source
         * Wraps user code with necessary declarations and output
         */
        std::string GenerateFragmentShader();

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
