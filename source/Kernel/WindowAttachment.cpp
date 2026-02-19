/**
 * WindowAttachment.cpp:
 *      @Descripiton    :   Window management implementation for FragmentKernel
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/19/2026
 */

#ifdef _WIN32

#include <Kernel/WindowAttachment.h>

#include <glad/glad.h>

#include <stdexcept>

namespace GPU::Kernel {

    WindowAttachment::WindowAttachment() = default;

    WindowAttachment::~WindowAttachment() {
        Detach();
    }

    bool WindowAttachment::Attach(HWND hwnd, ResizeCallback resizeCallback) {
        if (_attached) {
            if (_hwnd == hwnd) {
                // Already attached to this window, just update callback
                _resizeCallback = resizeCallback;
                return true;
            }
            // Attached to different window, detach first
            Detach();
        }

        if (!hwnd || !IsWindow(hwnd)) {
            return false;
        }

        _hwnd = hwnd;
        _resizeCallback = resizeCallback;

        // Get window DC
        _hdc = GetDC(hwnd);
        if (!_hdc) {
            _hwnd = nullptr;
            return false;
        }

        // Set up pixel format
        if (!SetupPixelFormat()) {
            ReleaseDC(_hwnd, _hdc);
            _hdc = nullptr;
            _hwnd = nullptr;
            return false;
        }

        // Get initial window size
        RECT rect;
        GetClientRect(_hwnd, &rect);
        _width = static_cast<uint32_t>(rect.right - rect.left);
        _height = static_cast<uint32_t>(rect.bottom - rect.top);

        // Install window subclass hook
        // Store 'this' pointer in window property for static callback access
        SetPropW(_hwnd, s_propName, this);
        
        // Use SetWindowSubclass API (modern way to subclass windows)
        // But for maximum compatibility, we'll use SetWindowLongPtr method
        _originalWndProc = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(
            _hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(SubclassWndProc)));

        if (!_originalWndProc) {
            RemovePropW(_hwnd, s_propName);
            ReleaseDC(_hwnd, _hdc);
            _hdc = nullptr;
            _hwnd = nullptr;
            return false;
        }

        _attached = true;
        return true;
    }

    void WindowAttachment::Detach() {
        if (!_attached || !_hwnd) {
            return;
        }

        // Restore original window procedure
        if (_originalWndProc) {
            SetWindowLongPtrW(_hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(_originalWndProc));
            _originalWndProc = nullptr;
        }

        // Remove property
        RemovePropW(_hwnd, s_propName);

        // Release DC
        if (_hdc) {
            ReleaseDC(_hwnd, _hdc);
            _hdc = nullptr;
        }

        _hwnd = nullptr;
        _attached = false;
        _pixelFormatSet = false;
    }

    bool WindowAttachment::IsAttached() const {
        return _attached && _hwnd && IsWindow(_hwnd);
    }

    HWND WindowAttachment::GetHWND() const {
        return _hwnd;
    }

    HDC WindowAttachment::GetWindowDC() const {
        return _hdc;
    }

    uint32_t WindowAttachment::GetWidth() const {
        return _width;
    }

    uint32_t WindowAttachment::GetHeight() const {
        return _height;
    }

    void WindowAttachment::SwapBuffers() {
        if (_hdc) {
            ::SwapBuffers(_hdc);
        }
    }

    bool WindowAttachment::MakeCurrent(HGLRC hglrc) {
        if (!_hdc || !hglrc) {
            return false;
        }
        return wglMakeCurrent(_hdc, hglrc) == TRUE;
    }

    bool WindowAttachment::IsPixelFormatSet() const {
        return _pixelFormatSet;
    }

    bool WindowAttachment::SetupPixelFormat() {
        if (!_hdc) {
            return false;
        }

        // Check if pixel format is already set
        int existingFormat = GetPixelFormat(_hdc);
        if (existingFormat != 0) {
            // Already set, verify it's suitable for OpenGL
            PIXELFORMATDESCRIPTOR pfd;
            if (DescribePixelFormat(_hdc, existingFormat, sizeof(pfd), &pfd)) {
                if ((pfd.dwFlags & PFD_SUPPORT_OPENGL) && 
                    (pfd.dwFlags & PFD_DRAW_TO_WINDOW)) {
                    // Suitable format already set
                    _pixelFormatSet = false;  // Not set by us, but suitable
                    return true;
                }
            }
            // Existing format not suitable, can't change it (window already has context)
            // But we'll try anyway and hope for the best
        }

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
            return false;
        }

        if (!SetPixelFormat(_hdc, pixelFormat, &pfd)) {
            // Failed to set pixel format - might already be set by another context
            DWORD error = GetLastError();
            if (error == ERROR_INVALID_PIXEL_FORMAT) {
                // Already set, check if compatible
                return true;  // Hope for the best
            }
            return false;
        }

        _pixelFormatSet = true;
        return true;
    }

    LRESULT CALLBACK WindowAttachment::SubclassWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        // Get instance from window property
        WindowAttachment* instance = GetInstanceFromHWND(hwnd);
        if (!instance) {
            // No instance found, use DefWindowProc
            return DefWindowProcW(hwnd, msg, wParam, lParam);
        }

        // Handle messages
        switch (msg) {
            case WM_SIZE: {
                uint32_t width = LOWORD(lParam);
                uint32_t height = HIWORD(lParam);
                instance->OnResize(width, height);
                break;
            }
            
            case WM_DESTROY: {
                // Window being destroyed, clean up
                instance->Detach();
                break;
            }
        }

        // Call original window procedure
        if (instance->_originalWndProc) {
            return CallWindowProcW(instance->_originalWndProc, hwnd, msg, wParam, lParam);
        }

        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }

    WindowAttachment* WindowAttachment::GetInstanceFromHWND(HWND hwnd) {
        if (!hwnd) {
            return nullptr;
        }
        return reinterpret_cast<WindowAttachment*>(GetPropW(hwnd, s_propName));
    }

    void WindowAttachment::OnResize(uint32_t width, uint32_t height) {
        _width = width;
        _height = height;
        
        if (_resizeCallback) {
            _resizeCallback(width, height);
        }
    }

} // namespace GPU::Kernel

#endif // _WIN32
