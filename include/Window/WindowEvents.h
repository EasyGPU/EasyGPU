#pragma once

/**
 * @file WindowEvents.h
 * @brief Window event types using std::variant.
 */

#ifndef EASYGPU_WINDOW_EVENTS_H
#define EASYGPU_WINDOW_EVENTS_H

#include <Window/Input.h>

#include <cstdint>
#include <variant>

namespace GPU::Window {

/**
 * @brief Window resize event
 */
struct WindowResizeEvent {
	uint32_t width;
	uint32_t height;
};

/**
 * @brief Window close event
 */
struct WindowCloseEvent {};

/**
 * @brief Keyboard key event
 */
struct KeyEvent {
	Key			  key;
	bool		  pressed;
	ModifierFlags modifiers;
};

/**
 * @brief Character input event (for text entry)
 */
struct CharInputEvent {
	uint32_t codepoint;
};

/**
 * @brief Mouse button event
 */
struct MouseButtonEvent {
	MouseButton	  button;
	bool		  pressed;
	int32_t		  x;
	int32_t		  y;
	ModifierFlags modifiers;
};

/**
 * @brief Mouse move event
 */
struct MouseMoveEvent {
	int32_t x;
	int32_t y;
	int32_t dx;
	int32_t dy;
};

/**
 * @brief Mouse scroll event
 */
struct MouseScrollEvent {
	float dx;
	float dy;
};

/**
 * @brief Window focus event
 */
struct WindowFocusEvent {
	bool focused;
};

/**
 * @brief Variant type for all window events
 *
 * Usage:
 *   WindowEvent event;
 *   while (window.PollEvent(event)) {
 *       if (std::holds_alternative<KeyEvent>(event)) {
 *           auto& key = std::get<KeyEvent>(event);
 *           // handle key...
 *       }
 *   }
 */
using WindowEvent = std::variant<WindowResizeEvent, WindowCloseEvent, KeyEvent, CharInputEvent, MouseButtonEvent,
								 MouseMoveEvent, MouseScrollEvent, WindowFocusEvent>;

} // namespace GPU::Window

#endif // EASYGPU_WINDOW_EVENTS_H
