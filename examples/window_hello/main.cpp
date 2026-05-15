/**
 * @file main.cpp
 * @brief Minimal window example - opens a window and handles events.
 */

#include <Window/AppWindow.h>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

int main() {
	using namespace GPU::Window;

	// Create a window with configuration
	AppWindow window({.width = 800, .height = 600, .title = "EasyGPU Window - Hello", .resizable = true});

	std::cout << "Window created: " << window.Width() << "x" << window.Height() << std::endl;
	std::cout << "Press ESC to close, or click the X button" << std::endl;

	// Main loop
	while (window.IsOpen()) {
		// Poll for window events (keyboard, mouse, resize, etc.)
		window.PollEvents();

		// Process events from the queue
		WindowEvent event;
		while (window.PollEvent(event)) {
			// Handle key events
			if (std::holds_alternative<KeyEvent>(event)) {
				auto &key = std::get<KeyEvent>(event);
				if (key.key == Key::Escape && key.pressed) {
					std::cout << "ESC pressed, closing..." << std::endl;
					window.Close();
				}
				if (key.key == Key::Space && key.pressed) {
					std::cout << "Space pressed" << std::endl;
				}
			}
			// Handle mouse button events
			else if (std::holds_alternative<MouseButtonEvent>(event)) {
				auto &mouse = std::get<MouseButtonEvent>(event);
				if (mouse.pressed) {
					std::cout << "Mouse button " << static_cast<int>(mouse.button) << " pressed at (" << mouse.x << ", "
							  << mouse.y << ")" << std::endl;
				}
			}
			// Handle window resize
			else if (std::holds_alternative<WindowResizeEvent>(event)) {
				auto &resize = std::get<WindowResizeEvent>(event);
				std::cout << "Window resized to " << resize.width << "x" << resize.height << std::endl;
			}
		}

// In a real application, you would render something here
// For this example, we just sleep a bit to avoid busy-waiting
#ifdef _WIN32
		Sleep(16); // ~60 FPS
#else
		usleep(16000);
#endif
	}

	std::cout << "Window closed" << std::endl;
	return 0;
}
