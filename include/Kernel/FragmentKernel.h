#pragma once

/**
 * @file FragmentKernel.h
 * @brief Fragment shader based kernel for high-performance pixel rendering.
 */

#ifndef EASYGPU_FRAGMENT_KERNEL_H
#define EASYGPU_FRAGMENT_KERNEL_H

#ifdef _WIN32

#include <Kernel/FragmentBuildContext.h>
#include <Kernel/KernelProfiler.h>
#include <Kernel/WindowAttachment.h>

#include <IR/Builder/Builder.h>
#include <IR/Value/Var.h>

#include <functional>
#include <string>

namespace GPU::Kernel {

/**
 * @brief 2D Fragment Kernel for pixel-based GPU rendering.
 *
 * Uses traditional rasterization pipeline (VS + FS) instead of compute shaders.
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
	FragmentKernel2D(const std::string													   &name,
					 const std::function<void(IR::Value::Var<GPU::Math::Vec2> &fragCoord,
											  IR::Value::Var<GPU::Math::Vec2> &resolution,
											  IR::Value::Var<GPU::Math::Vec4> &fragColor)> &func,
					 uint32_t width, uint32_t height);

	/**
	 * Destructor - cleans up OpenGL resources
	 */
	~FragmentKernel2D();

	// Disable copy, allow move
	FragmentKernel2D(const FragmentKernel2D &)			  = delete;
	FragmentKernel2D &operator=(const FragmentKernel2D &) = delete;
	FragmentKernel2D(FragmentKernel2D &&) noexcept;
	FragmentKernel2D &operator=(FragmentKernel2D &&) noexcept;

public:
	/**
	 * @brief Attach kernel to a window for rendering.
	 *
	 * Sets up OpenGL context on the window and installs resize hook.
	 * @param hwnd Target window handle (must be valid).
	 * @return true if attachment succeeded.
	 */
	bool Attach(HWND hwnd);

	/**
	 * @brief Detach from current window.
	 */
	void Detach();

	/**
	 * @brief Check if attached to a window.
	 * @return true if currently attached.
	 */
	bool IsAttached() const;

	/**
	 * @brief Get attached window handle.
	 * @return The HWND, or nullptr if not attached.
	 */
	HWND GetWindow() const;

public:
	/**
	 * @brief Execute rendering and present to screen.
	 *
	 * Must be called after Attach(). Equivalent to Dispatch() in compute kernels.
	 */
	void		Flush();

	/**
	 * @brief Set the kernel name for profiling.
	 * @param name The kernel name.
	 */
	void		SetName(const std::string &name);

	/**
	 * @brief Get the kernel name.
	 * @return The kernel name string.
	 */
	std::string GetName() const;

	/**
	 * @brief Get the generated GLSL shader source.
	 * @return The full GLSL shader source code.
	 */
	std::string GetShaderSource();

public:
	/**
	 * @brief Get current rendering width.
	 * @return Width in pixels.
	 */
	uint32_t GetWidth() const;

	/**
	 * @brief Get current rendering height.
	 * @return Height in pixels.
	 */
	uint32_t GetHeight() const;

	/**
	 * @brief Set rendering resolution.
	 * @param width New width in pixels.
	 * @param height New height in pixels.
	 */
	void	 SetResolution(uint32_t width, uint32_t height);

public:
	/**
	 * @brief Enable or disable profiling for this kernel.
	 *
	 * When enabled, each Flush() will record GPU execution time.
	 * Use KernelProfiler::PrintInfo() to view results.
	 * @param enabled True to enable profiling.
	 */
	void SetProfilingEnabled(bool enabled);

	/**
	 * @brief Check if profiling is enabled.
	 * @return true if profiling is active.
	 */
	bool IsProfilingEnabled() const;

private:
	/**
	 * @brief Initialize OpenGL resources (VAO, shader).
	 */
	void InitializeResources();

	/**
	 * @brief Cleanup OpenGL resources.
	 */
	void CleanupResources();

	/**
	 * @brief Compile shader program if needed.
	 */
	void EnsureShaderCompiled();

	/**
	 * @brief Handle window resize.
	 */
	void OnResize(uint32_t width, uint32_t height);

	/**
	 * @brief Execute actual rendering.
	 */
	void ExecuteRender();

private:
	std::string							  _name;
	std::unique_ptr<FragmentBuildContext> _context;
	std::unique_ptr<WindowAttachment>	  _windowAttachment;

	// OpenGL resources
	uint32_t							  _vao					= 0;
	uint32_t							  _shaderProgram		= 0;

	// State
	bool								  _resourcesInitialized = false;
	uint32_t							  _width;
	uint32_t							  _height;
	bool								  _profilingEnabled = false;
};

} // namespace GPU::Kernel

#endif // _WIN32

#endif // EASYGPU_FRAGMENT_KERNEL_H
