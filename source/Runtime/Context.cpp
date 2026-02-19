/**
 * @file Context.cpp
 * @brief Platform-specific OpenGL context implementation
 */

#include <Runtime/Context.h>
#include <Runtime/GLStateCache.h>

#include <iostream>
#include <sstream>

// GLAD
#include <GLAD/glad.h>

namespace GPU::Runtime {
// Static members
    Context *Context::_instance = nullptr;
    bool Context::_destroyed = false;

    Context &Context::GetInstance() {
        if (_instance == nullptr) {
            if (_destroyed) {
                throw std::runtime_error("Context was destroyed and cannot be recreated");
            }
            _instance = new Context();
        }
        // Auto-initialize on first access
        if (!_instance->_initialized) {
            _instance->Initialize();
        }
        return *_instance;
    }

    Context::~Context() {
        CleanupPlatform();
        _destroyed = true;
        _instance = nullptr;
    }

    void Context::Initialize() {
        if (_initialized) {
            return;
        }

        try {
            InitializePlatform();
            _initialized = true;
        } catch (const std::exception &e) {
            CleanupPlatform();
            throw std::runtime_error(std::string("Failed to initialize OpenGL context: ") + e.what());
        }
    }

    bool Context::IsInitialized() const {
        return _initialized;
    }

    void Context::MakeCurrent() {
        if (!_initialized) {
            throw std::runtime_error("Context not initialized");
        }
#ifdef _WIN32
        if (!wglMakeCurrent(_hdc, _hglrc)) {
            throw std::runtime_error("Failed to make OpenGL context current");
        }
#endif
        // Invalidate state cache when context becomes current
        // This ensures we re-bind state after any context switch
        GetStateCache().Invalidate();
    }

    void Context::MakeNoneCurrent() {
#ifdef _WIN32
        wglMakeCurrent(nullptr, nullptr);
#endif
    }

    std::string Context::GetVersionString() const {
        if (!_initialized) {
            return "Not initialized";
        }
        const GLubyte *version = glGetString(GL_VERSION);
        return version ? reinterpret_cast<const char *>(version) : "Unknown";
    }

    bool Context::HasComputeShadersSupport() const {
        if (!_initialized) {
            return false;
        }
        // Check OpenGL 4.3+ for compute shaders
        GLint major = 0, minor = 0;
        glGetIntegerv(GL_MAJOR_VERSION, &major);
        glGetIntegerv(GL_MINOR_VERSION, &minor);

        return (major > 4 || (major == 4 && minor >= 3));
    }

    void Context::GetMaxWorkGroupSize(int &x, int &y, int &z) const {
        if (!_initialized) {
            x = y = z = 0;
            return;
        }
        GLint workGroupSize[3];
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &workGroupSize[0]);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &workGroupSize[1]);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &workGroupSize[2]);
        x = workGroupSize[0];
        y = workGroupSize[1];
        z = workGroupSize[2];
    }

#ifdef _WIN32

// Window class name
    static const wchar_t *s_windowClassName = L"EasyGPUHiddenWindow";

    void Context::InitializePlatform() {
        // Get module handle
        _hInstance = GetModuleHandleW(nullptr);
        if (!_hInstance) {
            throw std::runtime_error("Failed to get module handle");
        }

        // Register window class (one-time)
        WNDCLASSEXW wcex = {};
        wcex.cbSize = sizeof(WNDCLASSEXW);
        wcex.lpfnWndProc = DefWindowProcW;
        wcex.hInstance = _hInstance;
        wcex.lpszClassName = s_windowClassName;

        // Try to register, ignore if already exists
        if (!GetClassInfoExW(_hInstance, s_windowClassName, &wcex)) {
            if (!RegisterClassExW(&wcex)) {
                throw std::runtime_error("Failed to register window class");
            }
        }

        CreateHiddenWindow();
        SetupGLContext();
        LoadGLAD();
    }

    void Context::CreateHiddenWindow() {
        // Create hidden window
        _hwnd = CreateWindowExW(
                0,                              // Extended style
                s_windowClassName,              // Class name
                L"EasyGPU Context",             // Window name
                WS_OVERLAPPEDWINDOW,            // Style
                CW_USEDEFAULT, CW_USEDEFAULT,   // Position
                1, 1,                           // Size (1x1, hidden)
                nullptr,                        // Parent
                nullptr,                        // Menu
                _hInstance,                     // Instance
                nullptr                         // Param
        );

        if (!_hwnd) {
            throw std::runtime_error("Failed to create hidden window");
        }

        // Hide the window completely
        ShowWindow(_hwnd, SW_HIDE);

        // Get device context
        _hdc = GetDC(_hwnd);
        if (!_hdc) {
            throw std::runtime_error("Failed to get device context");
        }
    }

    void Context::SetupGLContext() {
        // Choose pixel format
        PIXELFORMATDESCRIPTOR pfd = {};
        pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
        pfd.nVersion = 1;
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        pfd.iPixelType = PFD_TYPE_RGBA;
        pfd.cColorBits = 32;
        pfd.cDepthBits = 24;
        pfd.cStencilBits = 8;
        pfd.iLayerType = PFD_MAIN_PLANE;

        int pixelFormat = ChoosePixelFormat(_hdc, &pfd);
        if (pixelFormat == 0) {
            throw std::runtime_error("Failed to choose pixel format");
        }

        if (!SetPixelFormat(_hdc, pixelFormat, &pfd)) {
            throw std::runtime_error("Failed to set pixel format");
        }

        // Create legacy context first (required for loading extensions)
        _hglrc = wglCreateContext(_hdc);
        if (!_hglrc) {
            throw std::runtime_error("Failed to create OpenGL context");
        }

        // Make current temporarily to load extensions
        if (!wglMakeCurrent(_hdc, _hglrc)) {
            throw std::runtime_error("Failed to make OpenGL context current");
        }

        // Load GLAD for OpenGL functions
        LoadGLAD();

        // Note: For modern context creation with specific version,
        // we would need wglCreateContextAttribsARB from WGL extensions.
        // This requires loading WGL extensions via gladLoadWGL() which
        // needs a current context first. For simplicity, we rely on
        // the default context created by the driver (usually 4.x on modern systems).
    }

    void Context::LoadGLAD() {
        if (!gladLoadGL()) {
            throw std::runtime_error("Failed to initialize GLAD");
        }

        // Check version
        GLint major = 0, minor = 0;
        glGetIntegerv(GL_MAJOR_VERSION, &major);
        glGetIntegerv(GL_MINOR_VERSION, &minor);

        if (major < 4 || (major == 4 && minor < 3)) {
            // Warning but don't fail - user might not need compute shaders
            // Log to stderr for debugging
            std::cerr << "Warning: OpenGL " << major << "." << minor
                      << " detected. Compute shaders require 4.3+." << std::endl;
        }
    }

    void Context::DestroyHiddenWindow() {
        if (_hdc) {
            ReleaseDC(_hwnd, _hdc);
            _hdc = nullptr;
        }
        if (_hwnd) {
            DestroyWindow(_hwnd);
            _hwnd = nullptr;
        }
    }

    void Context::CleanupPlatform() {
        // Make sure context is not current
        wglMakeCurrent(nullptr, nullptr);

        // Delete context
        if (_hglrc) {
            wglDeleteContext(_hglrc);
            _hglrc = nullptr;
        }

        DestroyHiddenWindow();

        // Unregister window class (optional, on Windows it persists until process exit)
        // UnregisterClassW(s_windowClassName, _hInstance);
    }

#else
    // Linux/X11 implementation would go here
    void Context::InitializePlatform() {
        throw std::runtime_error("Non-Windows platforms not yet implemented");
    }
    void Context::CleanupPlatform() {}
    void Context::CreateHiddenWindow() {}
    void Context::DestroyHiddenWindow() {}
    void Context::SetupGLContext() {}
    void Context::LoadGLAD() {}
#endif

}