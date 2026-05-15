/**
 * @file MiniFBWindowPlatform.cpp
 * @brief Implementation of MiniFB-based window platform backend.
 */

// Platform-specific includes (needed for Sleep/usleep)
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "MiniFBWindowPlatform.h"

#include <MiniFB.h>
#include <MiniFB_internal.h>

#include <cstring>
#include <stdexcept>

namespace GPU::Window {

// Static instance pointer for callbacks (minifb doesn't support userdata in callbacks directly)
// We use mfb_set_user_data / mfb_get_user_data to store the platform instance
static MiniFBWindowPlatform *GetPlatformFromWindow(mfb_window *window) {
	return static_cast<MiniFBWindowPlatform *>(mfb_get_user_data(window));
}

// minifb callback trampolines (extern "C" linkage compatible)
extern "C" {
static void OnActiveCallback(mfb_window *window, bool isActive) {
	if (auto *platform = GetPlatformFromWindow(window)) {
		platform->HandleActive(isActive);
	}
}

static void OnResizeCallback(mfb_window *window, int width, int height) {
	if (auto *platform = GetPlatformFromWindow(window)) {
		platform->HandleResize(width, height);
	}
}

static bool OnCloseCallback(mfb_window *window) {
	if (auto *platform = GetPlatformFromWindow(window)) {
		return platform->HandleClose();
	}
	return true; // Allow close by default
}

static void OnKeyboardCallback(mfb_window *window, mfb_key key, mfb_key_mod mod, bool isPressed) {
	if (auto *platform = GetPlatformFromWindow(window)) {
		platform->HandleKeyboard(static_cast<int>(key), static_cast<int>(mod), isPressed);
	}
}

static void OnCharInputCallback(mfb_window *window, unsigned int code) {
	if (auto *platform = GetPlatformFromWindow(window)) {
		platform->HandleCharInput(code);
	}
}

static void OnMouseButtonCallback(mfb_window *window, mfb_mouse_button button, mfb_key_mod mod, bool isPressed) {
	if (auto *platform = GetPlatformFromWindow(window)) {
		platform->HandleMouseButton(static_cast<int>(button), static_cast<int>(mod), isPressed);
	}
}

static void OnMouseMoveCallback(mfb_window *window, int x, int y) {
	if (auto *platform = GetPlatformFromWindow(window)) {
		platform->HandleMouseMove(x, y);
	}
}

static void OnMouseScrollCallback(mfb_window *window, mfb_key_mod mod, float deltaX, float deltaY) {
	if (auto *platform = GetPlatformFromWindow(window)) {
		platform->HandleMouseScroll(static_cast<int>(mod), deltaX, deltaY);
	}
}
}

// Instance handlers
void MiniFBWindowPlatform::HandleActive(bool isActive) {
	_eventQueue.push(WindowFocusEvent{.focused = isActive});
	if (focusCallback) {
		focusCallback(isActive);
	}
}

void MiniFBWindowPlatform::HandleResize(int width, int height) {
	_width	= static_cast<uint32_t>(width);
	_height = static_cast<uint32_t>(height);
	_eventQueue.push(WindowResizeEvent{.width = _width, .height = _height});
	if (resizeCallback) {
		resizeCallback(_width, _height);
	}
}

bool MiniFBWindowPlatform::HandleClose() {
	if (closeCallback) {
		bool allowClose = closeCallback();
		if (allowClose) {
			_isOpen = false;
		}
		return allowClose;
	}
	_isOpen = false;
	return true;
}

void MiniFBWindowPlatform::HandleKeyboard(int key, int mod, bool isPressed) {
	_eventQueue.push(KeyEvent{.key = ConvertKey(key), .pressed = isPressed, .modifiers = ConvertMods(mod)});
}

void MiniFBWindowPlatform::HandleCharInput(unsigned int code) {
	_eventQueue.push(CharInputEvent{.codepoint = code});
}

void MiniFBWindowPlatform::HandleMouseButton(int button, int mod, bool isPressed) {
	_eventQueue.push(MouseButtonEvent{.button	 = ConvertMouseButton(button),
									  .pressed	 = isPressed,
									  .x		 = _mouseX,
									  .y		 = _mouseY,
									  .modifiers = ConvertMods(mod)});
}

void MiniFBWindowPlatform::HandleMouseMove(int x, int y) {
	int32_t newX = static_cast<int32_t>(x);
	int32_t newY = static_cast<int32_t>(y);
	int32_t dx	 = newX - _mouseX;
	int32_t dy	 = newY - _mouseY;
	_mouseX		 = newX;
	_mouseY		 = newY;

	_eventQueue.push(MouseMoveEvent{.x = _mouseX, .y = _mouseY, .dx = dx, .dy = dy});
}

void MiniFBWindowPlatform::HandleMouseScroll(int mod, float deltaX, float deltaY) {
	_scrollX = deltaX;
	_scrollY = deltaY;
	_eventQueue.push(MouseScrollEvent{.dx = deltaX, .dy = deltaY});
}

// Conversion helpers
Key MiniFBWindowPlatform::ConvertKey(int mfbKey) {
	// minifb keys match our Key enum values directly
	return static_cast<Key>(mfbKey);
}

ModifierFlags MiniFBWindowPlatform::ConvertMods(int mfbMods) {
	ModifierFlags result = ModifierFlags::None;
	if (mfbMods & KB_MOD_SHIFT)
		result = result | ModifierFlags::Shift;
	if (mfbMods & KB_MOD_CONTROL)
		result = result | ModifierFlags::Ctrl;
	if (mfbMods & KB_MOD_ALT)
		result = result | ModifierFlags::Alt;
	if (mfbMods & KB_MOD_SUPER)
		result = result | ModifierFlags::Super;
	if (mfbMods & KB_MOD_CAPS_LOCK)
		result = result | ModifierFlags::CapsLock;
	if (mfbMods & KB_MOD_NUM_LOCK)
		result = result | ModifierFlags::NumLock;
	return result;
}

MouseButton MiniFBWindowPlatform::ConvertMouseButton(int mfbButton) {
	// minifb: 0=None, 1=Left, 2=Right, 3=Middle, etc.
	// Ours: 0=Left, 1=Right, 2=Middle
	switch (mfbButton) {
	case MOUSE_LEFT:
		return MouseButton::Left;
	case MOUSE_RIGHT:
		return MouseButton::Right;
	case MOUSE_MIDDLE:
		return MouseButton::Middle;
	case MOUSE_BTN_4:
		return MouseButton::Button4;
	case MOUSE_BTN_5:
		return MouseButton::Button5;
	case MOUSE_BTN_6:
		return MouseButton::Button6;
	case MOUSE_BTN_7:
		return MouseButton::Button7;
	default:
		return MouseButton::Left;
	}
}

// Constructor / Destructor
MiniFBWindowPlatform::MiniFBWindowPlatform(const WindowConfig &config, std::queue<WindowEvent> &eventQueue)
	: _eventQueue(eventQueue), _width(config.width), _height(config.height) {

	unsigned flags = 0;
	if (config.resizable)
		flags |= WF_RESIZABLE;
	if (!config.visible)
		flags |= WF_BORDERLESS; // Best approximation

	_window = mfb_open_ex(config.title.c_str(), config.width, config.height, flags);
	if (!_window) {
		throw std::runtime_error("Failed to create window");
	}

	// Store this instance in the window's user data for callbacks
	mfb_set_user_data(_window, this);

	// Set up callbacks
	mfb_set_active_callback(_window, OnActiveCallback);
	mfb_set_resize_callback(_window, OnResizeCallback);
	mfb_set_close_callback(_window, OnCloseCallback);
	mfb_set_keyboard_callback(_window, OnKeyboardCallback);
	mfb_set_char_input_callback(_window, OnCharInputCallback);
	mfb_set_mouse_button_callback(_window, OnMouseButtonCallback);
	mfb_set_mouse_move_callback(_window, OnMouseMoveCallback);
	mfb_set_mouse_scroll_callback(_window, OnMouseScrollCallback);

	// Create timer for vsync
	_timer	= mfb_timer_create();

	// Set initial state
	_isOpen = true;
	_mouseX = mfb_get_mouse_x(_window);
	_mouseY = mfb_get_mouse_y(_window);
}

MiniFBWindowPlatform::~MiniFBWindowPlatform() {
	if (_timer) {
		mfb_timer_destroy(_timer);
	}
	if (_window) {
		mfb_close(_window);
	}
}

// IWindowPlatform implementation
bool MiniFBWindowPlatform::IsOpen() const {
	return _isOpen && _window != nullptr;
}

void MiniFBWindowPlatform::Close() {
	_isOpen = false;
	if (_window) {
		mfb_close(_window);
		_window = nullptr;
	}
}

uint32_t MiniFBWindowPlatform::Width() const {
	return _width;
}

uint32_t MiniFBWindowPlatform::Height() const {
	return _height;
}

void MiniFBWindowPlatform::SetTitle(const std::string &title) {
	// minifb doesn't support changing title after creation
	// This would need platform-specific code
	// For now, we silently ignore this
	(void)title;
}

void MiniFBWindowPlatform::Present(const uint32_t *pixels, uint32_t width, uint32_t height) {
	if (!_window || !_isOpen)
		return;

	mfb_update_state state =
		mfb_update_ex(_window, const_cast<void *>(static_cast<const void *>(pixels)), width, height);

	if (state == STATE_INVALID_WINDOW || state == STATE_INTERNAL_ERROR) {
		_isOpen = false;
	}
}

void MiniFBWindowPlatform::PollEvents() {
	if (!_window || !_isOpen)
		return;

	mfb_update_state state	   = mfb_update_events(_window);

	// Update mouse position (minifb tracks this internally)
	int				 newMouseX = mfb_get_mouse_x(_window);
	int				 newMouseY = mfb_get_mouse_y(_window);
	if (newMouseX != _mouseX || newMouseY != _mouseY) {
		HandleMouseMove(newMouseX, newMouseY);
	}

	// Update scroll
	float scrollX = mfb_get_mouse_scroll_x(_window);
	float scrollY = mfb_get_mouse_scroll_y(_window);
	if (scrollX != 0.0f || scrollY != 0.0f) {
		HandleMouseScroll(0, scrollX, scrollY);

		// Reset minifb's scroll values to prevent duplicate events
		// Scroll values are cumulative and don't auto-reset after reading
		auto *windowData = static_cast<SWindowData *>(mfb_get_user_data(_window));
		if (windowData) {
			windowData->mouse_wheel_x = 0.0f;
			windowData->mouse_wheel_y = 0.0f;
		}
	}

	if (state == STATE_INVALID_WINDOW || state == STATE_INTERNAL_ERROR) {
		_isOpen = false;
	}
}

void MiniFBWindowPlatform::WaitEvents() {
	if (!_window || !_isOpen)
		return;

	// minifb doesn't have a direct WaitEvents, so we poll and sleep
	// A proper implementation would use platform-specific event waiting
	PollEvents();

	if (_eventQueue.empty()) {
// Simple sleep to avoid busy-waiting
// In a production implementation, this should use proper OS primitives
#ifdef _WIN32
		Sleep(1);
#else
		usleep(1000);
#endif
	}
}

bool MiniFBWindowPlatform::PollEvent(WindowEvent &event) {
	if (_eventQueue.empty()) {
		return false;
	}

	event = _eventQueue.front();
	_eventQueue.pop();
	return true;
}

bool MiniFBWindowPlatform::IsKeyDown(int keyCode) const {
	if (!_window)
		return false;

	const uint8_t *keyBuffer = mfb_get_key_buffer(_window);
	if (keyBuffer && keyCode >= 0 && keyCode < 512) {
		return keyBuffer[keyCode] != 0;
	}
	return false;
}

bool MiniFBWindowPlatform::IsMouseDown(int button) const {
	if (!_window)
		return false;

	const uint8_t *mouseBuffer = mfb_get_mouse_button_buffer(_window);
	if (mouseBuffer && button >= 0 && button < 8) {
		return mouseBuffer[button] != 0;
	}
	return false;
}

std::pair<int32_t, int32_t> MiniFBWindowPlatform::MousePosition() const {
	return {_mouseX, _mouseY};
}

std::pair<float, float> MiniFBWindowPlatform::MouseScroll() const {
	return {_scrollX, _scrollY};
}

} // namespace GPU::Window
