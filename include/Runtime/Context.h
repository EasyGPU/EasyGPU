#pragma once

/**
 * @file Context.h
 * @brief GPU context management with backend abstraction.
 */

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

/** @brief Convenience typedef for a raw backend pointer */
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
	Context(const Context &)								 = delete;
	Context &operator=(const Context &)						 = delete;
	Context(Context &&)										 = delete;
	Context							  &operator=(Context &&) = delete;

	/**
	 * @brief Get the singleton instance, auto-initializing if needed
	 * @return Reference to the singleton Context instance
	 */
	static Context					  &GetInstance();

	/**
	 * Get the active backend instance
	 * @return Pointer to the active backend
	 */
	static Backend::Backend			  *GetBackend();

	template <typename T> static T &GetBackend() {
		auto *backend = dynamic_cast<T *>(GetBackend());
		if (!backend) {
			throw std::runtime_error("Requested backend type does not match the active backend");
		}
		return *backend;
	}

	/**
	 * Explicitly initialize the context (optional, called automatically)
	 * @throw std::runtime_error if initialization fails
	 */
	void							   Initialize();

	/**
	 * @brief Check if context is already initialized
	 * @return true if the context has been initialized, false otherwise
	 */
	[[nodiscard]] bool				   IsInitialized() const;

	/**
	 * Make the backend context current on this thread
	 */
	void							   MakeCurrent();

	/**
	 * Release the context from current thread
	 */
	void							   MakeNoneCurrent();

	/**
	 * @brief Get backend version string
	 * @return Human-readable backend version string (e.g., "OpenGL 4.6")
	 */
	[[nodiscard]] std::string		   GetVersionString() const;

	/**
	 * @brief Check if compute shaders are supported
	 * @return true if the backend supports compute shaders, false otherwise
	 */
	[[nodiscard]] bool				   HasComputeShadersSupport() const;

	/**
	 * @brief Get compute shader max work group size
	 * @param[out] x Maximum work group size in X dimension
	 * @param[out] y Maximum work group size in Y dimension
	 * @param[out] z Maximum work group size in Z dimension
	 */
	void							   GetMaxWorkGroupSize(int &x, int &y, int &z) const;

	/**
	 * Get the native OpenGL context handle (for FragmentKernel compatibility)
	 * @return The native GL context handle (HGLRC on Windows, GLXContext on Linux)
	 *         Returns nullptr if the current backend is not OpenGL
	 */
	[[nodiscard]] void				  *GetNativeGLContext() const;

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
};

/**
 * @brief RAII guard for making context current on a scope.
 *
 * Makes the context current upon construction and releases it upon destruction.
 */
class ContextGuard {
public:
	/**
	 * @brief Construct a guard and make the context current.
	 * @param ctx The Context to make current for the lifetime of this guard.
	 */
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
 * @brief Auto-initialization helper.
 *
 * Call this at the entry point of any GPU operation to ensure the context is initialized.
 */
inline void AutoInitContext() {
	Context::GetInstance().Initialize();
}

} // namespace GPU::Runtime
