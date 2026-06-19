/**
 * @file Window.cpp
 * @brief Implementation of window management for GPU rendering.
 */

// Include Window headers only - avoid GPU core headers to prevent template instantiation issues
#include "Platform/MiniFBWindowPlatform.h"
#include <Window/AppWindow.h>

namespace GPU::Window {

// Constructor
AppWindow::AppWindow(const WindowConfig &config) : _config(config) {
	_platform = std::make_unique<MiniFBWindowPlatform>(config, _eventQueue);
	_isOpen	  = _platform->IsOpen();
	_width	  = _platform->Width();
	_height	  = _platform->Height();
}

// Destructor
AppWindow::~AppWindow() = default;

// Move operations
AppWindow::AppWindow(AppWindow &&other) noexcept
	: _platform(std::move(other._platform)), _eventQueue(std::move(other._eventQueue)),
	  _resizeCallback(std::move(other._resizeCallback)), _closeCallback(std::move(other._closeCallback)),
	  _focusCallback(std::move(other._focusCallback)), _config(other._config), _isOpen(other._isOpen),
	  _width(other._width), _height(other._height) {
	other._isOpen = false;
	other._width  = 0;
	other._height = 0;
}

AppWindow &AppWindow::operator=(AppWindow &&other) noexcept {
	if (this != &other) {
		_platform		= std::move(other._platform);
		_eventQueue		= std::move(other._eventQueue);
		_resizeCallback = std::move(other._resizeCallback);
		_closeCallback	= std::move(other._closeCallback);
		_focusCallback	= std::move(other._focusCallback);
		_config			= other._config;
		_isOpen			= other._isOpen;
		_width			= other._width;
		_height			= other._height;

		other._isOpen	= false;
		other._width	= 0;
		other._height	= 0;
	}
	return *this;
}

// Window state
bool AppWindow::IsOpen() const noexcept {
	return _isOpen && _platform && _platform->IsOpen();
}

void AppWindow::Close() {
	_isOpen = false;
	if (_platform) {
		_platform->Close();
	}
}

// Window properties
uint32_t AppWindow::Width() const noexcept {
	return _width;
}

uint32_t AppWindow::Height() const noexcept {
	return _height;
}

float AppWindow::Aspect() const noexcept {
	if (_height == 0)
		return 1.0f;
	return static_cast<float>(_width) / static_cast<float>(_height);
}

void AppWindow::SetTitle(const std::string &title) {
	if (_platform) {
		_platform->SetTitle(title);
	}
}

void AppWindow::SetVSync(bool enabled) {
	// minifb doesn't expose vsync control directly
	// This would need platform-specific implementation
	// For now, we use mfb_wait_sync in Present
	_config.vsync = enabled;
}

void AppWindow::SetResizable(bool enabled) {
	// minifb doesn't support changing resizable state after creation
	_config.resizable = enabled;
}

// Events
void AppWindow::PollEvents() {
	if (!_platform)
		return;

	_platform->PollEvents();

	// Update cached dimensions from platform
	_width	= _platform->Width();
	_height = _platform->Height();

	// Check if window was closed
	if (!_platform->IsOpen()) {
		_isOpen = false;
	}
}

bool AppWindow::PollEvent(WindowEvent &event) {
	// First try our own queue (filled by callbacks)
	if (!_eventQueue.empty()) {
		event = _eventQueue.front();
		_eventQueue.pop();
		return true;
	}

	// Then try platform queue
	if (_platform) {
		return _platform->PollEvent(event);
	}

	return false;
}

void AppWindow::WaitEvents() {
	if (_platform) {
		_platform->WaitEvents();
	}
}

// Input state
bool AppWindow::IsKeyDown(Key key) const {
	if (!_platform)
		return false;
	return _platform->IsKeyDown(static_cast<int>(key));
}

bool AppWindow::IsMouseDown(MouseButton button) const {
	if (!_platform)
		return false;
	return _platform->IsMouseDown(static_cast<int>(button));
}

std::pair<int32_t, int32_t> AppWindow::MousePosition() const noexcept {
	if (!_platform)
		return {0, 0};
	return _platform->MousePosition();
}

std::pair<float, float> AppWindow::MouseScroll() const noexcept {
	if (!_platform)
		return {0.0f, 0.0f};
	return _platform->MouseScroll();
}

// Presentation
void AppWindow::Present(const PixelBuffer &buffer) {
	Present(buffer.Data(), buffer.Width(), buffer.Height());
}

void AppWindow::Present(const uint32_t *pixels, uint32_t width, uint32_t height) {
	if (!_platform || !_isOpen)
		return;

	_platform->Present(pixels, width, height);

	// Handle vsync
	if (_config.vsync) {
		_platform->WaitSync();
	}
}

// Callbacks
void AppWindow::SetResizeCallback(std::function<void(uint32_t, uint32_t)> callback) {
	_resizeCallback = callback;
	if (_platform) {
		_platform->resizeCallback = callback;
	}
}

void AppWindow::SetCloseCallback(std::function<bool()> callback) {
	_closeCallback = callback;
	if (_platform) {
		_platform->closeCallback = callback;
	}
}

void AppWindow::SetFocusCallback(std::function<void(bool)> callback) {
	_focusCallback = callback;
	if (_platform) {
		_platform->focusCallback = callback;
	}
}

} // namespace GPU::Window
