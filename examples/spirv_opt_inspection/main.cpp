/**
 * @file main.cpp
 * @brief Inspect and time Mandelbrot GLSL at the available optimization levels.
 *
 * The Mandelbrot example has rich control flow:
 *   - Callable functions (Mandelbrot, GetColor)
 *   - For-loop with conditional Break
 *   - If/Else branches
 *   - Vector math (Float3, Sin, Cos, Pow, Clamp)
 *
 * The reported duration covers glslang, SPIRV-Tools, and SPIRV-Cross on the
 * host. It is a compilation/inspection measurement, not a GPU runtime result.
 */

#include <Callable/Callable.h>
#include <Flow/BreakFlow.h>
#include <Flow/ForFlow.h>
#include <Flow/IfFlow.h>
#include <Flow/ReturnFlow.h>
#include <GPU.h>
#include <Utility/Math.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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

struct InspectionResult {
	std::string glsl;
	double		coldMilliseconds = 0.0;
	double		warmMilliseconds = 0.0;
	uint64_t	cacheHits = 0;
	uint64_t	cacheMisses = 0;
};

static void setCacheDirectory(const std::filesystem::path &directory) {
#ifdef _WIN32
	_putenv_s("EASYGPU_SHADER_CACHE_DIR", directory.string().c_str());
#else
	setenv("EASYGPU_SHADER_CACHE_DIR", directory.string().c_str(), 1);
#endif
}

static void clearCacheDirectoryOverride() {
#ifdef _WIN32
	_putenv_s("EASYGPU_SHADER_CACHE_DIR", "");
#else
	unsetenv("EASYGPU_SHADER_CACHE_DIR");
#endif
}

static double median(std::vector<double> samples) {
	std::sort(samples.begin(), samples.end());
	return samples[samples.size() / 2];
}

int main() {
	const auto cacheDirectory = std::filesystem::temp_directory_path() /
		("easygpu-spirv-inspection-" +
		 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	std::filesystem::remove_all(cacheDirectory);
	setCacheDirectory(cacheDirectory);

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
	// Measure seven forced cache misses and seven disk-cache hits per level.
	// =========================================================================
	auto inspect = [&](Backend::ShaderOptimizationLevel level) {
		constexpr int sampleCount = 7;
		kernel.SetOptimizationLevel(level);
		auto *backend = Runtime::Context::GetBackend();
		backend->ResetShaderCompilationStats();

		InspectionResult   result;
		std::vector<double> coldSamples;
		std::vector<double> warmSamples;
		coldSamples.reserve(sampleCount);
		warmSamples.reserve(sampleCount);
		for (int i = 0; i < sampleCount; ++i) {
			std::filesystem::remove_all(cacheDirectory);
			const auto start = std::chrono::steady_clock::now();
			auto	   glsl  = kernel.GetOptimizedGLSL();
			const auto end   = std::chrono::steady_clock::now();
			coldSamples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
			result.glsl = std::move(glsl);
		}
		for (int i = 0; i < sampleCount; ++i) {
			const auto start = std::chrono::steady_clock::now();
			result.glsl = kernel.GetOptimizedGLSL();
			const auto end = std::chrono::steady_clock::now();
			warmSamples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
		}

		const auto stats = backend->GetShaderCompilationStats();
		result.coldMilliseconds = median(std::move(coldSamples));
		result.warmMilliseconds = median(std::move(warmSamples));
		result.cacheHits = stats.diskCacheHits;
		result.cacheMisses = stats.diskCacheMisses;
		return result;
	};

	const auto raw		= inspect(Backend::ShaderOptimizationLevel::None);
	const auto aggressive = inspect(Backend::ShaderOptimizationLevel::Aggressive);
	const auto ultra		= inspect(Backend::ShaderOptimizationLevel::Ultra);
	const auto extreme	= inspect(Backend::ShaderOptimizationLevel::Extreme);
	clearCacheDirectoryOverride();
	std::filesystem::remove_all(cacheDirectory);

	// =========================================================================
	// Print
	// =========================================================================
	std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
	std::cout << "║        RAW GLSL — No SPIR-V Optimization (Mandelbrot)            ║\n";
	std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
	std::cout << raw.glsl << "\n";

	std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
	std::cout << "║   AGGRESSIVE GLSL — SPIRV-Tools RegisterPerformancePasses (-O)   ║\n";
	std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
	std::cout << aggressive.glsl << "\n";

	std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
	std::cout << "║     ULTRA GLSL — Maintained -O + conservative optimization tail ║\n";
	std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
	std::cout << ultra.glsl << "\n";

	std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
	std::cout << "║   EXTREME GLSL — Ultra + speculative loop/precision transforms  ║\n";
	std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
	std::cout << extreme.glsl << "\n";

	// =========================================================================
	// Stats
	// =========================================================================
	auto printStats = [](const char *label, const InspectionResult &result) {
		const auto cacheCounts = std::to_string(result.cacheHits) + "/" + std::to_string(result.cacheMisses);
		std::cout << std::left << std::setw(20) << label << std::right << std::setw(8) << countLines(result.glsl)
				  << std::setw(12) << result.glsl.size() << std::setw(14) << std::fixed << std::setprecision(3)
				  << result.coldMilliseconds << std::setw(14) << result.warmMilliseconds << std::setw(12) << cacheCounts
				  << '\n';
	};

	std::cout << "\nHost compilation/inspection statistics (median of 7 runs)\n";
	std::cout << std::left << std::setw(20) << "Level" << std::right << std::setw(8) << "Lines" << std::setw(12)
			  << "Bytes" << std::setw(14) << "Cold (ms)" << std::setw(14) << "Warm (ms)" << std::setw(12)
			  << "Hit/Miss" << '\n';
	printStats("Raw (None)", raw);
	printStats("Aggressive (-O)", aggressive);
	printStats("Ultra", ultra);
	printStats("Extreme", extreme);
	std::cout << "Cold timing includes glslang, SPIRV-Tools, cache write, and SPIRV-Cross.\n";
	std::cout << "Warm timing includes cache read/validation and SPIRV-Cross. Neither is GPU execution time.\n";

	return 0;
}
