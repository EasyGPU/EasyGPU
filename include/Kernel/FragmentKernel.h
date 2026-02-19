#pragma once

/**
 * FragmentKernel.h:
 *      @Descripiton    :   Fragment shader based kernel for high-performance pixel rendering
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/19/2026
 * 
 * Provides rasterization-based GPU execution using vertex/fragment shader pipeline.
 * Much faster than compute shaders for pixel-oriented workloads (no CPU readback).
 * 
 * Usage:
 *   FragmentKernel2D kernel("Effect", [&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
 *       Float2 uv = fragCoord / resolution;
 *       fragColor = MakeFloat4(uv.x(), uv.y(), 0.0f, 1.0f);
 *   }, 1920, 1080);
 *   
 *   kernel.Attach(hwnd);  // Bind to window
 *   while (running) {
 *       kernel.Flush();    // Render frame
 *   }
 */

#ifndef EASYGPU_FRAGMENT_KERNEL_H
#define EASYGPU_FRAGMENT_KERNEL_H

#ifdef _WIN32

#include <Kernel/FragmentBuildContext.h>
#include <Kernel/WindowAttachment.h>
#include <Kernel/KernelProfiler.h>

#include <IR/Value/Var.h>
#include <IR/Value/VarVector.h>
#include <IR/Builder/Builder.h>

#include <string>
#include <functional>

namespace GPU::Kernel {

    // Forward declaration
    class FragmentKernelBuilderGuard;

    /**
     * 2D Fragment Kernel for pixel-based GPU rendering
     * Uses traditional rasterization pipeline (VS + FS) instead of compute shaders
     */
    class FragmentKernel2D {
    public:
        /**
         * Construct a fragment kernel
         * @param name Kernel name for profiling
         * @param func User DSL function receiving (fragColor) output variable
         * @param width Initial rendering width
         * @param height Initial rendering height
         */
        FragmentKernel2D(const std::string& name,
                         const std::function<void(IR::Value::Var<GPU::Math::Vec2>& fragCoord,
                             IR::Value::Var<GPU::Math::Vec2>& resolution,
                             IR::Value::Var<GPU::Math::Vec4>& fragColor)>& func,
                         uint32_t width, uint32_t height);

        /**
         * Destructor - cleans up OpenGL resources
         */
        ~FragmentKernel2D();

        // Disable copy, allow move
        FragmentKernel2D(const FragmentKernel2D&) = delete;
        FragmentKernel2D& operator=(const FragmentKernel2D&) = delete;
        FragmentKernel2D(FragmentKernel2D&&) noexcept;
        FragmentKernel2D& operator=(FragmentKernel2D&&) noexcept;

    public:
        /**
         * Attach kernel to a window for rendering
         * Sets up OpenGL context on the window and installs resize hook
         * @param hwnd Target window handle (must be valid)
         * @return true if attachment succeeded
         */
        bool Attach(HWND hwnd);

        /**
         * Detach from current window
         */
        void Detach();

        /**
         * Check if attached to a window
         */
        bool IsAttached() const;

        /**
         * Get attached window handle
         */
        HWND GetWindow() const;

    public:
        /**
         * Execute rendering and present to screen
         * Must be called after Attach()
         * This is equivalent to Dispatch() in compute kernels
         */
        void Flush();

        /**
         * Set the kernel name for profiling
         */
        void SetName(const std::string& name);

        /**
         * Get the kernel name
         */
        std::string GetName() const;

        /**
         * Get the generated GLSL shader source
         */
        std::string GetShaderSource();

    public:
        /**
         * Get current rendering width
         */
        uint32_t GetWidth() const;

        /**
         * Get current rendering height
         */
        uint32_t GetHeight() const;

        /**
         * Set rendering resolution
         */
        void SetResolution(uint32_t width, uint32_t height);

    public:
        /**
         * Enable or disable profiling for this kernel
         * When enabled, each Flush() will record GPU execution time
         * Use KernelProfiler::PrintInfo() to view results
         */
        void SetProfilingEnabled(bool enabled);

        /**
         * Check if profiling is enabled
         */
        bool IsProfilingEnabled() const;

    private:
        /**
         * Initialize OpenGL resources (VAO, shader)
         */
        void InitializeResources();

        /**
         * Cleanup OpenGL resources
         */
        void CleanupResources();

        /**
         * Compile shader program if needed
         */
        void EnsureShaderCompiled();

        /**
         * Handle window resize
         */
        void OnResize(uint32_t width, uint32_t height);

        /**
         * Execute actual rendering
         */
        void ExecuteRender();

    private:
        std::string _name;
        std::unique_ptr<FragmentBuildContext> _context;
        std::unique_ptr<WindowAttachment> _windowAttachment;

        // OpenGL resources
        uint32_t _vao = 0;
        uint32_t _shaderProgram = 0;

        // State
        bool _resourcesInitialized = false;
        uint32_t _width;
        uint32_t _height;
        bool _profilingEnabled = false;
    };

} // namespace GPU::Kernel

#endif // _WIN32

#endif // EASYGPU_FRAGMENT_KERNEL_H
