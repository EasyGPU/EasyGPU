/**
 * @file Context.h
 * @brief GPU context management with backend abstraction
 *
 * This singleton manages the GPU backend and provides automatic initialization.
 * It creates a hidden window for off-screen compute shader execution.
 */

#pragma once

// X11 defines Bool as typedef int Bool, which conflicts with our Bool alias
// Must be done before any X11 headers are included
#ifdef Bool
#undef Bool
#endif

#include <Backend/Backend.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace GPU::Runtime {

using BackendPtr = Backend::Backend *;

/**
 * Singleton GPU context manager with automatic lazy initialization
 * Automatically creates a hidden window and initializes the backend on first use.
 * Users don't need to manually initialize anything - just use Kernel or Buffer.
 */
class Context {
public:
	~Context();

	// Non-copyable, non-movable singleton
	Context(const Context &)						= delete;
	Context &operator=(const Context &)				= delete;
	Context(Context &&)								= delete;
	Context					 &operator=(Context &&) = delete;

	/**
	 * Get the singleton instance, auto-initializing if needed
	 */
	static Context			 &GetInstance();

	/**
	 * Get the active backend instance
	 * @return Pointer to the active backend
	 */
	static Backend::Backend	 *GetBackend();

	/**
	 * Explicitly initialize the context (optional, called automatically)
	 * @throw std::runtime_error if initialization fails
	 */
	void					  Initialize();

	/**
	 * Check if context is already initialized
	 */
	[[nodiscard]] bool		  IsInitialized() const;

	/**
	 * Make the backend context current on this thread
	 */
	void					  MakeCurrent();

	/**
	 * Release the context from current thread
	 */
	void					  MakeNoneCurrent();

	/**
	 * Get backend version string
	 */
	[[nodiscard]] std::string GetVersionString() const;

	/**
	 * Check if compute shaders are supported
	 */
	[[nodiscard]] bool		  HasComputeShadersSupport() const;

	/**
	 * Get compute shader max work group size
	 */
	void					  GetMaxWorkGroupSize(int &x, int &y, int &z) const;

	/**
	 * Get the native OpenGL context handle (for FragmentKernel compatibility)
	 * @return The native GL context handle (HGLRC on Windows, GLXContext on Linux)
	 *         Returns nullptr if the current backend is not OpenGL
	 */
	[[nodiscard]] void		 *GetNativeGLContext() const;

	/**
	 * Get the current backend type
	 * @return The type of backend currently in use
	 */
	[[nodiscard]] Backend::BackendType GetBackendType() const;

private:
	Context() = default;

	void CreateBackend();
	void DestroyBackend();

private:
	bool							  _initialized = false;
	std::unique_ptr<Backend::Backend> _backend;

	// Reference count for automatic cleanup consideration
	static Context					 *_instance;
	static bool						  _destroyed;
};

/**
 * RAII guard for making context current on a scope
 */
class ContextGuard {
public:
	explicit ContextGuard(Context &ctx) : _ctx(ctx) {
		_ctx.MakeCurrent();
	}

	~ContextGuard() {
		_ctx.MakeNoneCurrent();
	}

	ContextGuard(const ContextGuard &)			  = delete;
	ContextGuard &operator=(const ContextGuard &) = delete;

private:
	Context &_ctx;
};

/**
 * Auto-initialization helper - call this in any GPU operation entry point
 */
inline void AutoInitContext() {
	Context::GetInstance().Initialize();
}

} // namespace GPU::Runtime
