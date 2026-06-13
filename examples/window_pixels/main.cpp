/**
 * @file main.cpp
 * @brief Pixel buffer example - draws a colorful animated pattern.
 */

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <Window/AppWindow.h>
#include <Window/PixelBuffer.h>

#include <cmath>
#include <iostream>

int main() {
	using namespace GPU::Window;

	const uint32_t WIDTH  = 800;
	const uint32_t HEIGHT = 600;

	// Create window and pixel buffer
	AppWindow	   window(
		{.width = WIDTH, .height = HEIGHT, .title = "EasyGPU Window - Pixel Buffer", .resizable = true, .vsync = true});

	PixelBuffer pixels(WIDTH, HEIGHT);

	std::cout << "Pixel buffer example" << std::endl;
	std::cout << "Press ESC to close" << std::endl;
	std::cout << "Press R to clear to red, G for green, B for blue" << std::endl;

	float	 time		= 0.0f;
	uint32_t clearColor = 0xFF101020; // Dark blue-gray

	// Main loop
	while (window.IsOpen()) {
		// Poll events
		window.PollEvents();

		// Process input
		WindowEvent event;
		while (window.PollEvent(event)) {
			if (std::holds_alternative<KeyEvent>(event)) {
				auto &key = std::get<KeyEvent>(event);
				if (key.key == Key::Escape && key.pressed) {
					window.Close();
				}
				if (key.key == Key::R && key.pressed) {
					clearColor = 0xFFFF0000; // Red
				}
				if (key.key == Key::G && key.pressed) {
					clearColor = 0xFF00FF00; // Green
				}
				if (key.key == Key::B && key.pressed) {
					clearColor = 0xFF0000FF; // Blue
				}
			}
		}

		// Clear with background color
		pixels.Clear(clearColor);

		// Draw an animated sine wave pattern
		for (uint32_t x = 0; x < WIDTH; ++x) {
			float fx = static_cast<float>(x) / WIDTH * 10.0f + time;
			int	  y	 = static_cast<int>((HEIGHT / 2) + std::sin(fx) * (HEIGHT / 4));

			if (y >= 0 && y < static_cast<int>(HEIGHT)) {
				// Rainbow color based on x position
				uint8_t r = static_cast<uint8_t>(128 + 127 * std::sin(fx));
				uint8_t g = static_cast<uint8_t>(128 + 127 * std::sin(fx + 2.0f));
				uint8_t b = static_cast<uint8_t>(128 + 127 * std::sin(fx + 4.0f));
				pixels.SetPixelUnchecked(x, y, PackRGBA(r, g, b));

				// Draw thicker line
				if (y + 1 < static_cast<int>(HEIGHT)) {
					pixels.SetPixelUnchecked(x, y + 1, PackRGBA(r, g, b));
				}
			}
		}

		// Draw a moving circle
		float cx	 = WIDTH / 2.0f + std::cos(time * 0.5f) * (WIDTH / 3.0f);
		float cy	 = HEIGHT / 2.0f + std::sin(time * 0.7f) * (HEIGHT / 3.0f);
		float radius = 30.0f + 10.0f * std::sin(time * 2.0f);

		for (int y = static_cast<int>(cy - radius); y <= static_cast<int>(cy + radius); ++y) {
			for (int x = static_cast<int>(cx - radius); x <= static_cast<int>(cx + radius); ++x) {
				if (x >= 0 && x < static_cast<int>(WIDTH) && y >= 0 && y < static_cast<int>(HEIGHT)) {
					float dx   = x - cx;
					float dy   = y - cy;
					float dist = std::sqrt(dx * dx + dy * dy);
					if (dist <= radius) {
						uint8_t alpha = static_cast<uint8_t>(255 * (1.0f - dist / radius));
						pixels.SetPixelUnchecked(x, y, PackRGBA(255, 255, 0, alpha));
					}
				}
			}
		}

		// Draw checkerboard pattern in corners
		for (uint32_t y = 0; y < 100; ++y) {
			for (uint32_t x = 0; x < 100; ++x) {
				bool	 check = ((x / 10) ^ (y / 10)) & 1;
				uint32_t color = check ? 0xFFFFFFFF : 0xFF000000;
				pixels.SetPixelUnchecked(x, y, color);
			}
		}

		// Present to window
		window.Present(pixels);

		// Update time
		time += 0.05f;
	}

	std::cout << "Window closed" << std::endl;
	return 0;
}
