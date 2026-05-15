/**
 * @file main.cpp
 * @brief Real-time GPU compute visualization using TexturePresenter.
 */

#include <GPU.h>
#include <Window/AppWindow.h>
#include <Window/TexturePresenter.h>

#include <chrono>
#include <format>
#include <iostream>

// Platform-specific sleep
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

int main() {
	using namespace GPU;
	using namespace GPU::Window;
	using namespace GPU::Runtime;

	constexpr uint32_t WIDTH  = 1024;
	constexpr uint32_t HEIGHT = 768;

	std::cout << "EasyGPU Real-time Compute Visualization" << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << "Controls:" << std::endl;
	std::cout << "  ESC - Exit" << std::endl;
	std::cout << "  SPACE - Pause/Resume animation" << std::endl;
	std::cout << "  Mouse - The pattern follows your mouse" << std::endl;
	std::cout << std::endl;

	// Create window
	AppWindow window(
		{.width = WIDTH, .height = HEIGHT, .title = "EasyGPU Real-time Compute", .resizable = true, .vsync = true});

	// Create GPU texture for rendering
	Texture2D<PixelFormat::RGBA8> renderTarget(WIDTH, HEIGHT);

	// Create uniform buffer for time and mouse
	Uniform<float>				  timeUniform(0.0f);
	Uniform<Vec2>				  mouseUniform(Vec2(0.5f, 0.5f));

	// Create the rendering kernel - generates a colorful plasma effect
	// This kernel runs on the GPU
	Kernel2D					  plasmaKernel([&](Int px, Int py) {
		 auto	tex		   = renderTarget.Bind();

		 // Get uniform values
		 Float	time	   = timeUniform.Load();
		 Float2 mouse	   = mouseUniform.Load();

		 // Normalized coordinates
		 Float	u		   = ToFloat(px) / Float(WIDTH);
		 Float	v		   = ToFloat(py) / Float(HEIGHT);

		 // Distance from mouse (for interaction)
		 Float	dx		   = u - mouse.x();
		 Float	dy		   = v - mouse.y();
		 Float	dist	   = Sqrt(dx * dx + dy * dy);

		 // Animated plasma effect
		 Float	value1	   = Sin(u * 10.0f + time);
		 Float	value2	   = Sin(v * 10.0f + time * 0.5f);
		 Float	value3	   = Sin((u + v) * 5.0f + time * 0.3f);
		 Float	value4	   = Sin(dist * 20.0f - time * 2.0f);

		 // Combine waves
		 Float	finalValue = (value1 + value2 + value3 + value4) * 0.25f;

		 // Color mapping
		 Float	r		   = 0.5f + 0.5f * Sin(finalValue * 3.14159f + time);
		 Float	g		   = 0.5f + 0.5f * Sin(finalValue * 3.14159f + 2.0f + time * 0.5f);
		 Float	b		   = 0.5f + 0.5f * Sin(finalValue * 3.14159f + 4.0f + time * 0.3f);

		 // Add mouse glow
		 Float	glow	   = Exp(-dist * 5.0f) * 0.5f;
		 r				   = r + glow;
		 g				   = g + glow * 0.5f;

		 // Output color (RGBA)
		 tex.Write(px, py, MakeFloat4(r, g, b, 1.0f));
	 });

	// Create the presenter for displaying GPU texture
	TexturePresenter			  presenter(window);

	// Animation state
	float						  time		  = 0.0f;
	bool						  paused	  = false;
	int							  frameCount  = 0;
	auto						  lastFpsTime = std::chrono::steady_clock::now();
	float						  currentFps  = 0.0f;

	// Main loop
	while (window.IsOpen()) {
		// Poll window events
		window.PollEvents();

		// Process input events
		GPU::Window::WindowEvent event;
		while (window.PollEvent(event)) {
			if (std::holds_alternative<GPU::Window::KeyEvent>(event)) {
				auto &key = std::get<GPU::Window::KeyEvent>(event);
				if (key.key == Key::Escape && key.pressed) {
					window.Close();
				}
				if (key.key == Key::Space && key.pressed) {
					paused = !paused;
					std::cout << (paused ? "Paused" : "Resumed") << std::endl;
				}
			}
		}

		// Update mouse uniform
		auto [mouseX, mouseY] = window.MousePosition();
		float mx			  = static_cast<float>(mouseX) / window.Width();
		float my			  = 1.0f - static_cast<float>(mouseY) / window.Height(); // Flip Y
		mouseUniform		  = (Vec2(mx, my));

		// Update time
		if (!paused) {
			time += 0.016f; // ~60fps animation speed
		}
		timeUniform = (time);

		// Dispatch kernel to generate frame (GPU computation)
		plasmaKernel.Dispatch((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

		// Present GPU texture to window
		presenter.Present(renderTarget);

		// Calculate FPS
		frameCount++;
		auto now	 = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFpsTime).count();
		if (elapsed >= 1000) {
			currentFps		  = frameCount * 1000.0f / elapsed;
			frameCount		  = 0;
			lastFpsTime		  = now;

			// Update window title with FPS
			std::string title = std::format("EasyGPU Real-time Compute - {:.1f} FPS", currentFps);
			window.SetTitle(title);
		}
	}

	std::cout << "Exiting..." << std::endl;
	return 0;
}
