#pragma once

/**
 * @file WindowAttachment.h
 * @brief Window management for FragmentKernel with dynamic HWND binding.
 */

#ifndef EASYGPU_WINDOW_ATTACHMENT_H
#define EASYGPU_WINDOW_ATTACHMENT_H

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <functional>
#include <memory>

namespace GPU::Kernel {

/**
 * Window attachment manager for FragmentKernel
 * Handles HWND binding, pixel format setup, and resize callbacks
 */
class WindowAttachment {
public:
	/**
	 * @brief Resize callback signature.
	 * @param width New client width.
	 * @param height New client height.
	 */
	using ResizeCallback = std::function<void(uint32_t width, uint32_t height)>;

public:
	/**
	 * @brief Construct a window attachment.
	 */
	WindowAttachment();

	/**
	 * @brief Destructor - detaches from window if still attached.
	 */
	~WindowAttachment();

	// Disable copy and move
	WindowAttachment(const WindowAttachment &)			  = delete;
	WindowAttachment &operator=(const WindowAttachment &) = delete;
	WindowAttachment(WindowAttachment &&)				  = delete;
	WindowAttachment &operator=(WindowAttachment &&)	  = delete;

public:
	/**
	 * @brief Attach to a HWND window.
	 *
	 * Sets up pixel format and installs window message hook.
	 * @param hwnd Target window handle.
	 * @param resizeCallback Callback invoked on window resize.
	 * @return true if attachment succeeded.
	 */
	bool	 Attach(HWND hwnd, ResizeCallback resizeCallback);

	/**
	 * @brief Detach from current window.
	 *
	 * Restores original window procedure.
	 */
	void	 Detach();

	/**
	 * @brief Check if currently attached to a window.
	 * @return true if attached.
	 */
	bool	 IsAttached() const;

	/**
	 * @brief Get the attached HWND.
	 * @return The window handle.
	 */
	HWND	 GetHWND() const;

	/**
	 * @brief Get the window device context.
	 * @return The window DC handle.
	 */
	HDC		 GetWindowDC() const;

	/**
	 * @brief Get current client width.
	 * @return Width in pixels.
	 */
	uint32_t GetWidth() const;

	/**
	 * @brief Get current client height.
	 * @return Height in pixels.
	 */
	uint32_t GetHeight() const;

	/**
	 * @brief Swap buffers (present rendered frame).
	 */
	void	 SwapBuffers();

	/**
	 * @brief Make the window DC current for OpenGL rendering.
	 * @param hglrc The OpenGL context to bind.
	 * @return true if successful.
	 */
	bool	 MakeCurrent(HGLRC hglrc);

	/**
	 * @brief Check if pixel format was set by this attachment.
	 * @return true if pixel format was configured.
	 */
	bool	 IsPixelFormatSet() const;

private:
	/**
	 * @brief Window procedure hook for intercepting WM_SIZE.
	 */
	static LRESULT CALLBACK	 SubclassWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

	/**
	 * @brief Get the WindowAttachment instance from HWND user data.
	 */
	static WindowAttachment *GetInstanceFromHWND(HWND hwnd);

	/**
	 * @brief Set up OpenGL pixel format for the window.
	 */
	bool					 SetupPixelFormat();

	/**
	 * @brief Handle WM_SIZE message.
	 */
	void					 OnResize(uint32_t width, uint32_t height);

private:
	HWND							_hwnd			 = nullptr;
	HDC								_hdc			 = nullptr;

	// Original window procedure for fallback
	WNDPROC							_originalWndProc = nullptr;

	// Window dimensions
	uint32_t						_width			 = 0;
	uint32_t						_height			 = 0;

	// State flags
	bool							_pixelFormatSet	 = false;
	bool							_attached		 = false;

	// Resize callback
	ResizeCallback					_resizeCallback;

	// User data slot for window instance mapping
	static constexpr const wchar_t *s_propName = L"EasyGPU_WindowAttachment";
};

} // namespace GPU::Kernel

#endif // _WIN32

#endif // EASYGPU_WINDOW_ATTACHMENT_H
