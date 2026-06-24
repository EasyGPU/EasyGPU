/**
 * @file main.cpp
 * @brief Show Mandelbrot kernel GLSL at Raw / Aggressive / Ultra optimization levels.
 *
 * The Mandelbrot example has rich control flow:
 *   - Callable functions (Mandelbrot, GetColor)
 *   - For-loop with conditional Break
 *   - If/Else branches
 *   - Vector math (Float3, Sin, Cos, Pow, Clamp)
 *
 * This is where Ultra's custom passes (LoopUnroll, IfConversion, SSA+GVN,
 * ScalarReplacement, VectorDCE) can show measurable impact beyond -O.
 */

#include <Callable/Callable.h>
#include <Flow/BreakFlow.h>
#include <Flow/ForFlow.h>
#include <Flow/IfFlow.h>
#include <Flow/ReturnFlow.h>
#include <GPU.h>
#include <Utility/Math.h>

#include <iostream>
#include <string>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Runtime;
using namespace GPU::Callables;
using namespace GPU::Flow;

constexpr int IMAGE_WIDTH    = 1024;
constexpr int IMAGE_HEIGHT   = 1024;
constexpr int MAX_ITERATIONS = 256;
constexpr float CENTER_X     = -0.5f;
constexpr float CENTER_Y     = 0.0f;
constexpr float ZOOM         = 1.5f;

static int countLines(const std::string &s) {
	int n = 0;
	for (char c : s) if (c == '\n') n++;
	return n;
}

int main() {
	// Build the Mandelbrot kernel (same as the real example, but inspect only)

	// Color mapping — Callable with If/Else, Sin/Cos/Pow vector math
	Callable<Float3(Int)> GetColor = [](Int &iter) {
		Float3 color;
		If(iter == MAX_ITERATIONS, [&]() {
			color = MakeFloat3(0.02f, 0.02f, 0.05f);
		}).Else([&]() {
			Float t    = Expr<float>(iter) / float(MAX_ITERATIONS);
			Float freq = MakeFloat(6.28318f);

			Float r = 0.5f + 0.5f * Sin(freq * t + 0.0f) * Cos(freq * t * 0.5f);
			Float g = 0.5f + 0.5f * Sin(freq * t + 2.094f) * Cos(freq * t * 0.3f + 1.0f);
			Float b = 0.5f + 0.5f * Sin(freq * t + 4.188f) * Cos(freq * t * 0.7f + 2.0f);

			r = Pow(Clamp(r, 0.0f, 1.0f), 0.8f);
			g = Pow(Clamp(g, 0.0f, 1.0f), 0.8f);
			b = Pow(Clamp(b, 0.0f, 1.0f), 0.8f);

			Float intensity = MakeFloat(1.2f);
			r = Clamp(r * intensity, 0.0f, 1.0f);
			g = Clamp(g * intensity, 0.0f, 1.0f);
			b = Clamp(b * intensity, 0.0f, 1.0f);

			color = MakeFloat3(r, g, b);
		});
		Return(color);
	};

	// Mandelbrot iteration — Callable with For-loop + conditional Break
	Callable<Int(Float, Float)> Mandelbrot = [](Float &cx, Float &cy) {
		Float zx   = MakeFloat(0.0f);
		Float zy   = MakeFloat(0.0f);
		Int  iter  = MakeInt(0);

		For(0, MAX_ITERATIONS, [&](Int &i) {
			Float zx2 = zx * zx;
			Float zy2 = zy * zy;

			If(zx2 + zy2 > 4.0f, [&]() {
				iter = Expr<int>(i);
				Break();
			});

			zy   = 2.0f * zx * zy + cy;
			zx   = zx2 - zy2 + cx;
			iter = Expr<int>(i);
		});
		Return(iter);
	};

	Buffer<Vec4> image(IMAGE_WIDTH * IMAGE_HEIGHT, BufferMode::Write);
	float        aspectRatio = static_cast<float>(IMAGE_WIDTH) / static_cast<float>(IMAGE_HEIGHT);
	float        scaleX      = ZOOM * aspectRatio;
	float        scaleY      = ZOOM;

	Kernel::Kernel2D kernel([&](Int &px, Int &py) {
		auto  img = image.Bind();
		Float u   = (Expr<float>(px) + 0.5f) / IMAGE_WIDTH;
		Float v   = (Expr<float>(py) + 0.5f) / IMAGE_HEIGHT;
		Float cx  = CENTER_X + (u * 2.0f - 1.0f) * scaleX;
		Float cy  = CENTER_Y + (v * 2.0f - 1.0f) * scaleY;
		Int   iter = Mandelbrot(cx, cy);
		Float3 col = GetColor(iter);
		Int   idx  = py * IMAGE_WIDTH + px;
		img[idx]   = MakeFloat4(col.x(), col.y(), col.z(), 1.0f);
	});

	// =========================================================================
	// Collect GLSL at all four levels
	// =========================================================================
	kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::None);
	std::string rawGLSL = kernel.GetOptimizedGLSL();

	kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Aggressive);
	std::string aggressiveGLSL = kernel.GetOptimizedGLSL();

	kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Ultra);
	std::string ultraGLSL = kernel.GetOptimizedGLSL();

	kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Extreme);
	std::string extremeGLSL = kernel.GetOptimizedGLSL();

	// =========================================================================
	// Print
	// =========================================================================
	std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
	std::cout << "║        RAW GLSL — No SPIR-V Optimization (Mandelbrot)            ║\n";
	std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
	std::cout << rawGLSL << "\n";

	std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
	std::cout << "║   AGGRESSIVE GLSL — SPIRV-Tools RegisterPerformancePasses (-O)   ║\n";
	std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
	std::cout << aggressiveGLSL << "\n";

	std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
	std::cout << "║     ULTRA GLSL — Custom 20-pass Pipeline (GPU compute tuned)     ║\n";
	std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
	std::cout << ultraGLSL << "\n";

	std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
		std::cout << "║   EXTREME GLSL — Ultra + FP16 + LoopFusion + CanonicalizeIds      ║\n";
	std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
	std::cout << extremeGLSL << "\n";

	// =========================================================================
	// Stats
	// =========================================================================
	std::cout << "┌────────────────────────────┬──────────┬───────────┐\n";
	std::cout << "│ Level                      │ Lines    │ Size (B)  │\n";
	std::cout << "├────────────────────────────┼──────────┼───────────┤\n";
	std::cout << "│ Raw (None)                 │ " << countLines(rawGLSL) << "      │ "
			  << rawGLSL.size() << "       │\n";
	std::cout << "│ Aggressive (-O)            │ " << countLines(aggressiveGLSL) << "      │ "
			  << aggressiveGLSL.size() << "       │\n";
	std::cout << "│ Ultra (20 GPU passes)      │ " << countLines(ultraGLSL) << "      │ "
			  << ultraGLSL.size() << "       │\n";
	std::cout << "│ Extreme (Ultra + FP16 etc)  │ " << countLines(extremeGLSL) << "      │ "
			  << extremeGLSL.size() << "       │\n";
	std::cout << "└────────────────────────────┴──────────┴───────────┘\n";

	return 0;
}
