/**
 * @file main.cpp
 * @brief Interactive EasyGPU + Window + Dear ImGui demo.
 */

#include <GPU.h>
#include <Window/AppWindow.h>
#include <Window/TexturePresenter.h>
#include <Window/UIContext.h>

#include <imgui.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>

int main() {
	using namespace GPU;
	using namespace GPU::Flow;
	using namespace GPU::Math;
	using namespace GPU::Runtime;
	using namespace GPU::Window;

	constexpr uint32_t Width  = 1280;
	constexpr uint32_t Height = 720;
	constexpr float	  Pi	 = 3.14159265359f;

	AppWindow window({.width = Width,
					  .height = Height,
					  .title = "EasyGPU ImGui Lab",
					  .resizable = true,
					  .vsync = true});

	Texture2D<PixelFormat::RGBA8> renderTarget(Width, Height);
	TexturePresenter			   presenter(window);
	UIContext					   ui(window);

	Uniform<float>				   timeUniform(0.0f);
	Uniform<Vec2>				   mouseUniform(Vec2(0.5f, 0.5f));
	Uniform<int>				   modeUniform(0);
	Uniform<int>				   iterationsUniform(96);
	Uniform<float>				   zoomUniform(1.15f);
	Uniform<float>				   speedUniform(1.0f);
	Uniform<float>				   warpUniform(0.55f);
	Uniform<float>				   exposureUniform(1.0f);
	Uniform<float>				   mousePowerUniform(0.65f);
	Uniform<Vec3>				   tintUniform(Vec3(1.0f, 0.92f, 0.78f));
	Uniform<Vec3>				   accentUniform(Vec3(0.18f, 0.72f, 1.0f));

	Kernel2D					   labKernel([&](Int px, Int py) {
		auto   tex		  = renderTarget.Bind();
		Float  t		  = timeUniform.Load();
		Float2 mouse	  = mouseUniform.Load();
		Int	   mode		  = modeUniform.Load();
		Int	   iterations = iterationsUniform.Load();
		Float  zoom		  = zoomUniform.Load();
		Float  speed	  = speedUniform.Load();
		Float  warp		  = warpUniform.Load();
		Float  exposure	  = exposureUniform.Load();
		Float  mousePower = mousePowerUniform.Load();
		Float3 tint		  = tintUniform.Load();
		Float3 accent	  = accentUniform.Load();

		Float  u		  = (ToFloat(px) + 0.5f) / Float(Width);
		Float  v		  = (ToFloat(py) + 0.5f) / Float(Height);
		Float2 p		  = MakeFloat2((u - 0.5f) * Float(Width) / Float(Height), v - 0.5f) * zoom;
		Float2 m		  = MakeFloat2(mouse.x() - 0.5f, mouse.y() - 0.5f);
		Float  md		  = Length(p - m);
		Float  angle	  = Atan2(p.y(), p.x());
		Float  radius	  = Length(p);

		Float  value	  = Sin((p.x() * 8.0f + t * speed) + Sin(p.y() * 6.0f - t * 0.7f));
		value			  = value + Cos((p.y() * 9.0f - t * 0.5f) + Cos(p.x() * 5.0f));
		value			  = value + Sin(radius * (18.0f + warp * 24.0f) - t * (1.4f + speed));
		value			  = value + Exp(-md * (3.0f + mousePower * 10.0f)) * (2.2f + mousePower);

		Float fractal	  = MakeFloat(0.0f);
		Float zx		  = p.x() + Sin(t * 0.17f) * 0.12f;
		Float zy		  = p.y() + Cos(t * 0.13f) * 0.12f;
		Float cx		  = -0.72f + m.x() * 0.55f;
		Float cy		  = 0.18f + m.y() * 0.55f;
		For(0, 192, [&](Int &i) {
			If(i >= iterations, [&]() { Break(); });
			Float zx2 = zx * zx;
			Float zy2 = zy * zy;
			If(zx2 + zy2 > 4.0f, [&]() {
				fractal = ToFloat(i) / ToFloat(Max(iterations, 1));
				Break();
			});
			Float newZy = 2.0f * zx * zy + cy;
			zx			= zx2 - zy2 + cx;
			zy			= newZy;
		});

		Float rings = 0.5f + 0.5f * Sin(radius * 32.0f - t * 2.0f + warp * Sin(angle * 6.0f));
		Float field = 0.5f + 0.5f * Sin(value + fractal * 8.0f);
		Float chosen = field;
		If(mode == 1, [&]() { chosen = fractal; });
		If(mode == 2, [&]() { chosen = rings * 0.55f + field * 0.45f; });
		If(mode == 3, [&]() { chosen = Fract(field + fractal + angle / (2.0f * Pi)); });

		Float3 base = Mix(accent, tint, Clamp(chosen, 0.0f, 1.0f));
		Float3 wave = MakeFloat3(0.5f + 0.5f * Sin(chosen * 6.28318f + t),
								  0.5f + 0.5f * Sin(chosen * 6.28318f + 2.094f + t * 0.7f),
								  0.5f + 0.5f * Sin(chosen * 6.28318f + 4.188f + t * 0.4f));
		Float3 color = Mix(base, wave, 0.35f + 0.35f * warp);
		color		 = color * exposure;
		color		 = Clamp(color, 0.0f, 1.0f);
		tex.Write(px, py, MakeFloat4(color, 1.0f));
	});

	int	  mode		 = 0;
	int	  iterations = 96;
	float zoom		 = 1.15f;
	float speed		 = 1.0f;
	float warp		 = 0.55f;
	float exposure	 = 1.0f;
	float mousePower = 0.65f;
	float tint[3]	 = {1.0f, 0.92f, 0.78f};
	float accent[3]	 = {0.18f, 0.72f, 1.0f};
	bool  paused	 = false;
	bool  mouseDrive = true;
	bool  showStats	 = true;
	float elapsed	 = 0.0f;
	float fps		 = 0.0f;
	int	  frameCount = 0;
	auto  lastFrame	 = std::chrono::steady_clock::now();
	auto  lastFps	 = lastFrame;

	const char *modes[] = {"Plasma Field", "Julia Probe", "Radial Bands", "Hybrid"};

	while (window.IsOpen()) {
		window.PollEvents();
		WindowEvent event;
		while (window.PollEvent(event)) {
			if (auto *key = std::get_if<KeyEvent>(&event)) {
				if (key->key == Key::Escape && key->pressed && !ui.WantCaptureKeyboard()) {
					window.Close();
				}
				if (key->key == Key::Space && key->pressed && !ui.WantCaptureKeyboard()) {
					paused = !paused;
				}
			}
		}

		auto now = std::chrono::steady_clock::now();
		float dt = std::chrono::duration<float>(now - lastFrame).count();
		lastFrame = now;
		if (!paused) {
			elapsed += dt;
		}

		auto [mouseX, mouseY] = window.MousePosition();
		float mx = std::clamp(static_cast<float>(mouseX) / static_cast<float>(std::max(window.Width(), 1u)), 0.0f, 1.0f);
		float my = 1.0f - std::clamp(static_cast<float>(mouseY) / static_cast<float>(std::max(window.Height(), 1u)), 0.0f, 1.0f);
		if (!mouseDrive) {
			mx = 0.5f + 0.28f * std::sin(elapsed * 0.37f);
			my = 0.5f + 0.28f * std::cos(elapsed * 0.31f);
		}

		timeUniform		  = elapsed;
		mouseUniform	  = Vec2(mx, my);
		modeUniform		  = mode;
		iterationsUniform = iterations;
		zoomUniform		  = zoom;
		speedUniform	  = speed;
		warpUniform		  = warp;
		exposureUniform	  = exposure;
		mousePowerUniform = mousePower;
		tintUniform		  = Vec3(tint[0], tint[1], tint[2]);
		accentUniform	  = Vec3(accent[0], accent[1], accent[2]);

		labKernel.Dispatch((Width + 15) / 16, (Height + 15) / 16);

		ui.Render([&]() {
			ImGui::SetNextWindowSize(ImVec2(390.0f, 0.0f), ImGuiCond_FirstUseEver);
			ImGui::Begin("EasyGPU ImGui Lab");
			ImGui::Combo("Mode", &mode, modes, static_cast<int>(sizeof(modes) / sizeof(modes[0])));
			ImGui::Checkbox("Paused", &paused);
			ImGui::SameLine();
			if (ImGui::Button("Reset")) {
				elapsed = 0.0f;
			}
			ImGui::Checkbox("Mouse drive", &mouseDrive);
			ImGui::SliderFloat("Speed", &speed, 0.0f, 4.0f);
			ImGui::SliderFloat("Zoom", &zoom, 0.35f, 3.0f);
			ImGui::SliderFloat("Warp", &warp, 0.0f, 1.5f);
			ImGui::SliderInt("Iterations", &iterations, 8, 192);
			ImGui::SliderFloat("Mouse power", &mousePower, 0.0f, 1.5f);
			ImGui::SliderFloat("Exposure", &exposure, 0.1f, 2.5f);
			ImGui::ColorEdit3("Tint", tint);
			ImGui::ColorEdit3("Accent", accent);
			ImGui::Separator();
			ImGui::Checkbox("Stats", &showStats);
			if (showStats) {
				ImGui::Text("FPS %.1f", fps);
				ImGui::Text("Mouse %.3f %.3f", mx, my);
				ImGui::Text("Texture %u x %u", Width, Height);
			}
			ImGui::End();
		});

		presenter.Present(renderTarget);

		++frameCount;
		auto fpsElapsed = std::chrono::duration<float>(now - lastFps).count();
		if (fpsElapsed >= 0.5f) {
			fps		   = static_cast<float>(frameCount) / fpsElapsed;
			frameCount = 0;
			lastFps	   = now;
			window.SetTitle(std::format("EasyGPU ImGui Lab - {:.1f} FPS", fps));
		}
	}

	std::cout << "EasyGPU ImGui Lab closed." << std::endl;
	return 0;
}
