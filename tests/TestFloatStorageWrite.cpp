/**
 * @file TestFloatStorageWrite.cpp
 * @brief Compute-shader storage writes to floating-point textures.
 *
 * The existing image-store coverage in TestTextureFormats.cpp only asserts RGBA8, whose UNORM
 * encoding hides both range and precision problems. GPU-resident simulation keeps its state in
 * float textures instead, so these tests write values a UNORM format could not represent --
 * negative numbers, values well above one, and small deltas -- and read them back exactly.
 */

#include <GPU.h>
#include <cmath>
#include <iostream>
#include <vector>

#define TEST(name)                                                                                                   \
	void test_##name() {                                                                                               \
		std::cout << "\n[TEST] " #name " ... ";                                                                        \
		try {

#define END_TEST                                                                                                       \
	std::cout << "PASSED\n";                                                                                           \
	}                                                                                                                  \
	catch (const std::exception &e) {                                                                                  \
		std::cout << "FAILED: " << e.what() << "\n";                                                                   \
		throw;                                                                                                         \
	}                                                                                                                  \
	}

#define ASSERT(cond)                                                                                                   \
	if (!(cond)) {                                                                                                     \
		throw std::runtime_error("Assertion failed: " #cond);                                                          \
	}

#define ASSERT_NEAR(a, b, eps)                                                                                         \
	if (std::abs((a) - (b)) > (eps)) {                                                                                  \
		throw std::runtime_error("Assertion failed: |" #a " - " #b "| > " #eps);                                        \
	}

#define ASSERT_FINITE(value)                                                                                           \
	if (!std::isfinite(value)) {                                                                                       \
		throw std::runtime_error("Assertion failed: " #value " is not finite");                                        \
	}

// =============================================================================
// R32F storage write: values outside the UNORM range must survive intact.
// =============================================================================
TEST(r32f_kernel_storage_write)
constexpr int						  W = 8;
constexpr int						  H = 8;
Texture2D<PixelFormat::R32F> tex(W, H);

// Spans negative, sub-unit and large magnitudes; an 8-bit UNORM target would clamp all three.
Kernel2D							  kernel(
	 [&, W](Var<int> &x, Var<int> &y) {
		 auto	   img	 = tex.Bind();
		 Var<int>   idx	 = y * W + x;
		 Var<float> value = ToFloat(idx) * 2.5f - 40.0f;
		 img.Write(x, y, MakeFloat4(value, 0.0f, 0.0f, 0.0f));
	 },
	 8, 8);

kernel.Dispatch(1, 1, true);

// R32F is one float per pixel: Download sizes by GetBytesPerPixel(), not by four channels.
std::vector<float> output(W * H);
tex.Download(output);

for (int index = 0; index < W * H; ++index) {
	const float expected = static_cast<float>(index) * 2.5f - 40.0f;
	ASSERT_FINITE(output[index]);
	ASSERT_NEAR(output[index], expected, 1e-4f);
}

// Guard against a silently clamped or UNORM-backed target.
ASSERT(output[0] < -1.0f);
ASSERT(output[W * H - 1] > 1.0f);
END_TEST

// =============================================================================
// RG32F storage write: both channels must be independent.
// =============================================================================
TEST(rg32f_kernel_storage_write)
constexpr int						   W = 8;
constexpr int						   H = 8;
Texture2D<PixelFormat::RG32F> tex(W, H);

// A height/velocity field is the motivating case, so the channels carry unrelated values.
Kernel2D							   kernel(
	 [&, W](Var<int> &x, Var<int> &y) {
		 auto	   img	  = tex.Bind();
		 Var<int>   idx	  = y * W + x;
		 Var<float> height	  = ToFloat(idx) * 0.5f - 16.0f;
		 Var<float> velocity = ToFloat(idx) * -0.25f + 8.0f;
		 img.Write(x, y, MakeFloat4(height, velocity, 0.0f, 0.0f));
	 },
	 8, 8);

kernel.Dispatch(1, 1, true);

std::vector<float> output(W * H * 2);
tex.Download(output);

for (int index = 0; index < W * H; ++index) {
	const float expectedHeight	 = static_cast<float>(index) * 0.5f - 16.0f;
	const float expectedVelocity = static_cast<float>(index) * -0.25f + 8.0f;
	ASSERT_FINITE(output[index * 2 + 0]);
	ASSERT_FINITE(output[index * 2 + 1]);
	ASSERT_NEAR(output[index * 2 + 0], expectedHeight, 1e-4f);
	ASSERT_NEAR(output[index * 2 + 1], expectedVelocity, 1e-4f);
}
END_TEST

// =============================================================================
// RGBA32F storage write: full four-channel HDR range.
// =============================================================================
TEST(rgba32f_kernel_storage_write)
constexpr int							 W = 8;
constexpr int							 H = 8;
Texture2D<PixelFormat::RGBA32F> tex(W, H);

Kernel2D								 kernel(
	 [&, W](Var<int> &x, Var<int> &y) {
		 auto	   img	 = tex.Bind();
		 Var<int>   idx	 = y * W + x;
		 Var<float> base = ToFloat(idx);
		 img.Write(x, y, MakeFloat4(base * 100.0f, base * -100.0f, base + 0.125f, 4.0f));
	 },
	 8, 8);

kernel.Dispatch(1, 1, true);

std::vector<float> output(W * H * 4);
tex.Download(output);

for (int index = 0; index < W * H; ++index) {
	const float base = static_cast<float>(index);
	ASSERT_NEAR(output[index * 4 + 0], base * 100.0f, 1e-2f);
	ASSERT_NEAR(output[index * 4 + 1], base * -100.0f, 1e-2f);
	ASSERT_NEAR(output[index * 4 + 2], base + 0.125f, 1e-4f);
	ASSERT_NEAR(output[index * 4 + 3], 4.0f, 1e-4f);
}

// Alpha above one would be clamped by a UNORM target.
ASSERT(output[3] > 1.0f);
END_TEST

// =============================================================================
// R32F read-after-write across dispatches: the dependency simulation relies on.
// =============================================================================
TEST(r32f_storage_write_then_read_across_dispatches)
constexpr int						  W = 8;
constexpr int						  H = 8;
Texture2D<PixelFormat::R32F> source(W, H);
Texture2D<PixelFormat::R32F> target(W, H);

// The first dispatch writes; the second samples those results. Correctness here means the
// backend inserted the layout transition and barrier between the two dispatches.
Kernel2D							  produce(
	 [&, W](Var<int> &x, Var<int> &y) {
		 auto	  img = source.Bind();
		 Var<int> idx = y * W + x;
		 img.Write(x, y, MakeFloat4(ToFloat(idx) - 32.0f, 0.0f, 0.0f, 0.0f));
	 },
	 8, 8);

Kernel2D							  consume(
	 [&](Var<int> &x, Var<int> &y) {
		 auto	    src	  = source.Bind();
		 auto	    dst	  = target.Bind();
		 Var<float> value = src.Read(x, y).x();
		 dst.Write(x, y, MakeFloat4(value * 3.0f, 0.0f, 0.0f, 0.0f));
	 },
	 8, 8);

produce.Dispatch(1, 1, true);
consume.Dispatch(1, 1, true);

std::vector<float> output(W * H);
target.Download(output);

for (int index = 0; index < W * H; ++index) {
	const float expected = (static_cast<float>(index) - 32.0f) * 3.0f;
	ASSERT_FINITE(output[index]);
	ASSERT_NEAR(output[index], expected, 1e-3f);
}
END_TEST

// =============================================================================
// R32F ping-pong: many dependent dispatches alternating read and write targets.
// =============================================================================
TEST(r32f_ping_pong_iteration)
constexpr int						  W		 = 16;
constexpr int						  H		 = 16;
constexpr int						  kSteps = 8;

// A checkerboard is the worst case for a diffusion kernel, so smoothing is unambiguous.
std::vector<float>					  seed(W * H, 0.0f);
for (int y = 0; y < H; ++y) {
	for (int x = 0; x < W; ++x) {
		seed[y * W + x] = ((x + y) % 2 == 0) ? 100.0f : 0.0f;
	}
}
std::vector<float>					  zero(W * H, 0.0f);

Texture2D<PixelFormat::R32F> ping(W, H, seed.data());
Texture2D<PixelFormat::R32F> pong(W, H, zero.data());

TextureSlot<PixelFormat::R32F> readSlot;
TextureSlot<PixelFormat::R32F> writeSlot;

Kernel2D								relax(
	   [&](Var<int> &x, Var<int> &y) {
		   auto	     src = readSlot.Bind();
		   auto	     dst = writeSlot.Bind();
		   Var<float> sum = MakeFloat(0.0f);
		   sum			  = sum + src.Read(Clamp(x - 1, 0, W - 1), y).x();
		   sum			  = sum + src.Read(Clamp(x + 1, 0, W - 1), y).x();
		   sum			  = sum + src.Read(x, Clamp(y - 1, 0, H - 1)).x();
		   sum			  = sum + src.Read(x, Clamp(y + 1, 0, H - 1)).x();
		   dst.Write(x, y, MakeFloat4(sum / 4.0f, 0.0f, 0.0f, 0.0f));
	   });

auto									 neighbourSpread = [&](const std::vector<float> &values) {
	 float total = 0.0f;
	 for (int y = 0; y < H - 1; ++y) {
		 for (int x = 0; x < W - 1; ++x) {
			 total += std::abs(values[y * W + x] - values[y * W + x + 1]);
			 total += std::abs(values[y * W + x] - values[(y + 1) * W + x]);
		 }
	 }
	 return total;
};

// Record the spread after each step. An unweighted 4-neighbour average converges slowly, so
// asserting a fixed final threshold would encode the convergence rate; requiring that every
// step strictly reduces the spread is both stronger and rate-independent.
std::vector<float> spreadPerStep;
spreadPerStep.reserve(kSteps);
std::vector<float> result(W * H);

for (int step = 0; step < kSteps; ++step) {
	if (step % 2 == 0) {
		readSlot.Attach(ping);
		writeSlot.Attach(pong);
	} else {
		readSlot.Attach(pong);
		writeSlot.Attach(ping);
	}
	relax.Dispatch(1, 1, true);

	// Read whichever texture this step wrote.
	if (step % 2 == 0) {
		pong.Download(result);
	} else {
		ping.Download(result);
	}
	spreadPerStep.push_back(neighbourSpread(result));
}

for (float value : result) {
	ASSERT_FINITE(value);
}

// Every iteration must contribute. A kernel dispatched only once, or one whose target was
// never read back, would leave the spread flat after the first step.
const float seededSpread = neighbourSpread(seed);
ASSERT(spreadPerStep.front() < seededSpread);
for (std::size_t step = 1; step < spreadPerStep.size(); ++step) {
	ASSERT(spreadPerStep[step] < spreadPerStep[step - 1]);
}

// The checkerboard must be measurably relaxed toward its mean by the end.
ASSERT(spreadPerStep.back() < seededSpread * 0.8f);
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Float Storage Write Tests     " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_r32f_kernel_storage_write();
		test_rg32f_kernel_storage_write();
		test_rgba32f_kernel_storage_write();
		test_r32f_storage_write_then_read_across_dispatches();
		test_r32f_ping_pong_iteration();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All float storage write tests passed! " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
