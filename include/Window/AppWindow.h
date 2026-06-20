#pragma once

/**
 * @file AppWindow.h
 * @brief Cross-platform window for interactive GPU compute visualization.
 */

#ifndef EASYGPU_WINDOW_H
#define EASYGPU_WINDOW_H

#include <Window/PixelBuffer.h>
#include <Window/WindowConfig.h>
#include <Window/WindowEvents.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <string>

#ifdef EASYGPU_BACKEND_VULKAN
struct VkCommandBuffer_T;
using VkCommandBuffer = VkCommandBuffer_T *;
#endif

namespace GPU::Window {

// Forward declaration for platform implementation
class IWindowPlatform;
class TexturePresenter;
class UIContext;

/**
 * @brief Cross-platform window for displaying compute results
 *
 * This class provides a lightweight, easy-to-use window abstraction
 * for GPU compute visualization. It is NOT a full GUI framework,
 * but rather a simple way to display pixel buffers and handle input.
 *
 * Usage:
 *   Window window({.width = 800, .height = 600, .title = "My App"});
 *   PixelBuffer buffer(800, 600);
 *
 *   while (window.IsOpen()) {
 *       window.PollEvents();
 *       // ... compute and fill buffer ...
 *       window.Present(buffer);
 *   }
 */
class AppWindow {
public:
	/**
	 * @brief Create a window with the specified configuration
	 * @param config Window configuration options
	 * @throws std::runtime_error if window creation fails
	 */
	explicit AppWindow(const WindowConfig &config = {});

	/**
	 * @brief Destructor - closes the window and cleans up resources
	 */
	~AppWindow();

	// Move operations
	AppWindow(AppWindow &&other) noexcept;
	AppWindow &operator=(AppWindow &&other) noexcept;

	// Disable copy
	AppWindow(const AppWindow &)			= delete;
	AppWindow &operator=(const AppWindow &) = delete;

public:
	/**
	 * @brief Check if the window is still open
	 * @return true if open and should continue rendering
	 */
	[[nodiscard]] bool	   IsOpen() const noexcept;

	/**
	 * @brief Request the window to close
	 *
	 * This will cause IsOpen() to return false on the next check.
	 * The window is not destroyed immediately to allow proper cleanup.
	 */
	void				   Close();

	/**
	 * @brief Get window width
	 */
	[[nodiscard]] uint32_t Width() const noexcept;

	/**
	 * @brief Get window height
	 */
	[[nodiscard]] uint32_t Height() const noexcept;

	/**
	 * @brief Get window aspect ratio (width / height)
	 */
	[[nodiscard]] float	   Aspect() const noexcept;

	/**
	 * @brief Set window title
	 */
	void				   SetTitle(const std::string &title);

	/**
	 * @brief Enable or disable vertical sync
	 */
	void				   SetVSync(bool enabled);

	/**
	 * @brief Enable or disable window resizing
	 */
	void				   SetResizable(bool enabled);

public:
	/**
	 * @brief Poll for window events (non-blocking)
	 *
	 * This updates the internal event queue. Call this once per frame
	 * before processing events with PollEvent().
	 */
	void PollEvents();

	/**
	 * @brief Poll for a single event (non-blocking)
	 * @param event Output event structure
	 * @return true if an event was retrieved
	 *
	 * Usage:
	 *   WindowEvent event;
	 *   while (window.PollEvent(event)) {
	 *       // process event...
	 *   }
	 */
	bool PollEvent(WindowEvent &event);

	/**
	 * @brief Wait for an event (blocking)
	 *
	 * This is useful for applications that don't need to render continuously.
	 * The function returns when an event is available or the window is closed.
	 */
	void WaitEvents();

public:
	/**
	 * @brief Check if a key is currently pressed
	 * @param key The key to check
	 * @return true if the key is pressed
	 */
	[[nodiscard]] bool						  IsKeyDown(Key key) const;

	/**
	 * @brief Check if a mouse button is currently pressed
	 * @param button The button to check
	 * @return true if the button is pressed
	 */
	[[nodiscard]] bool						  IsMouseDown(MouseButton button) const;

	/**
	 * @brief Get current mouse position in window coordinates
	 * @return Pair of (x, y) coordinates
	 */
	[[nodiscard]] std::pair<int32_t, int32_t> MousePosition() const noexcept;

	/**
	 * @brief Get current mouse scroll delta
	 * @return Pair of (dx, dy) scroll values
	 */
	[[nodiscard]] std::pair<float, float>	  MouseScroll() const noexcept;

public:
	/**
	 * @brief Present a pixel buffer to the window
	 * @param buffer The pixel buffer to display
	 * @throws std::invalid_argument if buffer dimensions don't match
	 *
	 * The buffer is expected to be in RGBA8 format.
	 * This function handles viewport scaling automatically.
	 */
	void Present(const PixelBuffer &buffer);

	/**
	 * @brief Present raw pixel data to the window
	 * @param pixels Pointer to RGBA8 pixel data
	 * @param width Width of the pixel data
	 * @param height Height of the pixel data
	 * @throws std::invalid_argument if dimensions are invalid
	 */
	void Present(const uint32_t *pixels, uint32_t width, uint32_t height);

public:
	/**
	 * @brief Set callback for window resize events
	 * @param callback Function to call when window is resized
	 */
	void						   SetResizeCallback(std::function<void(uint32_t, uint32_t)> callback);

	/**
	 * @brief Set callback for window close events
	 * @param callback Function to call when window close is requested
	 *                 Return false to prevent closing
	 */
	void						   SetCloseCallback(std::function<bool()> callback);

	/**
	 * @brief Set callback for focus events
	 * @param callback Function to call when window gains/loses focus
	 */
	void						   SetFocusCallback(std::function<void(bool)> callback);

#ifdef EASYGPU_BACKEND_VULKAN
	using VulkanOverlayCallback = std::function<void(VkCommandBuffer, uint32_t)>;
	void SetNextVulkanOverlay(VulkanOverlayCallback callback) {
		_nextVulkanOverlay = std::move(callback);
	}
	VulkanOverlayCallback TakeNextVulkanOverlay() {
		return std::move(_nextVulkanOverlay);
	}
#endif
private:
	[[nodiscard]] IWindowPlatform *Platform() noexcept {
		return _platform.get();
	}
	[[nodiscard]] const IWindowPlatform *Platform() const noexcept {
		return _platform.get();
	}

	// Platform-specific implementation
	std::unique_ptr<IWindowPlatform>		_platform;

	// Event queue (filled during PollEvents)
	std::queue<WindowEvent>					_eventQueue;

	// Callbacks
	std::function<void(uint32_t, uint32_t)> _resizeCallback;
	std::function<bool()>					_closeCallback;
	std::function<void(bool)>				_focusCallback;

	// Configuration
	WindowConfig							_config;

	// State
	bool									_isOpen = false;
	uint32_t								_width	= 0;
	uint32_t								_height = 0;

#ifdef EASYGPU_BACKEND_VULKAN
	VulkanOverlayCallback _nextVulkanOverlay;
#endif
	// Friend platform implementations for event injection
	friend class MiniFBWindowPlatform;
	friend class GLFWWindowPlatform;
	friend class TexturePresenter;
	friend class UIContext;
};

} // namespace GPU::Window

#endif // EASYGPU_WINDOW_H
