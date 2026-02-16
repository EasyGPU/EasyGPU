/**
 * @file Context.h
 * @brief Non-intrusive OpenGL context management with automatic initialization
 *
 * This singleton automatically initializes OpenGL and GLAD on first use,
 * creating a hidden window for off-screen compute shader execution.
 */

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

// Platform-specific includes
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#endif

// GLAD
#include <GLAD/glad.h>

namespace GPU::Runtime {

    /**
     * Singleton OpenGL context manager with automatic lazy initialization
     * Automatically creates a hidden window and OpenGL context on first use.
     * Users don't need to manually initialize anything - just use Kernel or Buffer.
     */
    class Context {
    public:
        ~Context();

        // Non-copyable, non-movable singleton
        Context(const Context &) = delete;

        Context &operator=(const Context &) = delete;

        Context(Context &&) = delete;

        Context &operator=(Context &&) = delete;

        /**
         * Get the singleton instance, auto-initializing if needed
         */
        static Context &GetInstance();

        /**
         * Explicitly initialize the context (optional, called automatically)
         * @throw std::runtime_error if initialization fails
         */
        void Initialize();

        /**
         * Check if context is already initialized
         */
        [[nodiscard]] bool IsInitialized() const;

        /**
         * Make the OpenGL context current on this thread
         */
        void MakeCurrent();

        /**
         * Release the context from current thread
         */
        void MakeNoneCurrent();

        /**
         * Get OpenGL version string
         */
        [[nodiscard]] std::string GetVersionString() const;

        /**
         * Check if compute shaders are supported
         */
        [[nodiscard]] bool HasComputeShadersSupport() const;

        /**
         * Get compute shader max work group size
         */
        void GetMaxWorkGroupSize(int &x, int &y, int &z) const;

    private:
        Context() = default;

        void InitializePlatform();

        void CleanupPlatform();

        void CreateHiddenWindow();

        void DestroyHiddenWindow();

        void SetupGLContext();

        void LoadGLAD();

    private:
        bool _initialized = false;

#ifdef _WIN32
        HINSTANCE _hInstance = nullptr;
        HWND _hwnd = nullptr;
        HDC _hdc = nullptr;
        HGLRC _hglrc = nullptr;
#endif

        // Reference count for automatic cleanup consideration
        static Context *_instance;
        static bool _destroyed;
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

        ContextGuard(const ContextGuard &) = delete;

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
}