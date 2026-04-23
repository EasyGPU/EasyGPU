/**
 * @file gaussian_blur.cpp
 * @brief 2-pass separable Gaussian blur example
 *
 * Demonstrates multi-pass image processing with Buffer reuse
 * and Uniform parameters. Uses a separable 1D Gaussian kernel
 * applied horizontally then vertically to reduce computation.
 *
 * Techniques shown:
 * - Buffer<int> read/write with packed RGBA pixels
 * - Ping-pong between two buffers for multi-pass algorithms
 * - Uniform<int> for dynamic image dimensions
 * - Clamp for boundary-safe sampling
 */

#include <GPU.h>

#include <cstdio>
#include <vector>

using namespace GPU;

int main() {
	// ========================================================================
	// Configuration
	// ========================================================================
	constexpr int WIDTH  = 8;
	constexpr int HEIGHT = 8;
	constexpr int PIXEL_COUNT = WIDTH * HEIGHT;

	// 1D Gaussian kernel (radius = 1, sigma ~ 1.0), normalized
	// Weights: [0.25, 0.5, 0.25]
	constexpr float WEIGHT_CENTER = 0.5f;
	constexpr float WEIGHT_SIDE   = 0.25f;

	// ========================================================================
	// Host Data Preparation
	// ========================================================================
	// Create a simple gradient image
	std::vector<int> host_input(PIXEL_COUNT);
	for (int y = 0; y < HEIGHT; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			int r = (x * 255) / (WIDTH - 1);
			int g = (y * 255) / (HEIGHT - 1);
			int b = 128;
			int a = 255;
			host_input[y * WIDTH + x] = static_cast<int>((a << 24) | (r << 16) | (g << 8) | b);
		}
	}

	std::vector<int> host_temp(PIXEL_COUNT);
	std::vector<int> host_output(PIXEL_COUNT);

	// ========================================================================
	// Device Data Upload
	// ========================================================================
	Buffer<int> input_image(host_input);
	Buffer<int> temp_image(PIXEL_COUNT);
	Buffer<int> output_image(PIXEL_COUNT);

	// Uniforms for dimensions
	Uniform<int> u_width(WIDTH);
	Uniform<int> u_height(HEIGHT);

	// ========================================================================
	// Pass 1: Horizontal Blur
	// ========================================================================
	Kernel2D horizontal_blur(
		"HorizontalBlur",
		[&](Int x, Int y) {
			If(x < WIDTH && y < HEIGHT, [&]() {
				auto in  = input_image.Bind();
				auto out = temp_image.Bind();
				auto w   = u_width.Load();

				Int idx = y * w + x;

				// Sample left, center, right with clamping
				Int xl = Clamp(x - 1, 0, w - 1);
				Int xr = Clamp(x + 1, 0, w - 1);

				Int pl = in[y * w + xl];
				Int pc = in[y * w + x];
				Int pr = in[y * w + xr];

				// Apply weights per channel
				auto apply_weight = [&](Int p, Float weight) -> Int {
					Int b = p & 0xFF;
					Int g = (p >> 8) & 0xFF;
					Int r = (p >> 16) & 0xFF;
					Int a = (p >> 24) & 0xFF;

					Int nb = ToInt(ToFloat(b) * weight);
					Int ng = ToInt(ToFloat(g) * weight);
					Int nr = ToInt(ToFloat(r) * weight);
					Int na = ToInt(ToFloat(a) * weight);

					return (na << 24) | (nr << 16) | (ng << 8) | nb;
				};

				Int sum = apply_weight(pl, MakeFloat(WEIGHT_SIDE))
						+ apply_weight(pc, MakeFloat(WEIGHT_CENTER))
						+ apply_weight(pr, MakeFloat(WEIGHT_SIDE));

				out[idx] = sum;
			});
		},
		16, 16);

	// ========================================================================
	// Pass 2: Vertical Blur
	// ========================================================================
	Kernel2D vertical_blur(
		"VerticalBlur",
		[&](Int x, Int y) {
			If(x < WIDTH && y < HEIGHT, [&]() {
				auto in  = temp_image.Bind();
				auto out = output_image.Bind();
				auto w   = u_width.Load();
				auto h   = u_height.Load();

				Int idx = y * w + x;

				// Sample top, center, bottom with clamping
				Int yt = Clamp(y - 1, 0, h - 1);
				Int yb = Clamp(y + 1, 0, h - 1);

				Int pt = in[yt * w + x];
				Int pc = in[y * w + x];
				Int pb = in[yb * w + x];

				// Apply weights per channel
				auto apply_weight = [&](Int p, Float weight) -> Int {
					Int b = p & 0xFF;
					Int g = (p >> 8) & 0xFF;
					Int r = (p >> 16) & 0xFF;
					Int a = (p >> 24) & 0xFF;

					Int nb = ToInt(ToFloat(b) * weight);
					Int ng = ToInt(ToFloat(g) * weight);
					Int nr = ToInt(ToFloat(r) * weight);
					Int na = ToInt(ToFloat(a) * weight);

					return (na << 24) | (nr << 16) | (ng << 8) | nb;
				};

				Int sum = apply_weight(pt, MakeFloat(WEIGHT_SIDE))
						+ apply_weight(pc, MakeFloat(WEIGHT_CENTER))
						+ apply_weight(pb, MakeFloat(WEIGHT_SIDE));

				out[idx] = sum;
			});
		},
		16, 16);

	// ========================================================================
	// Kernel Dispatch
	// ========================================================================
	int groups_x = (WIDTH + 15) / 16;
	int groups_y = (HEIGHT + 15) / 16;

	horizontal_blur.Dispatch(groups_x, groups_y, true);
	vertical_blur.Dispatch(groups_x, groups_y, true);

	// ========================================================================
	// Result Verification
	// ========================================================================
	output_image.Download(host_output);

	// Verify: output should differ from input (blur must have modified pixels)
	bool any_changed = false;
	for (int i = 0; i < PIXEL_COUNT; ++i) {
		if (host_output[i] != host_input[i]) {
			any_changed = true;
			break;
		}
	}

	if (any_changed) {
		std::printf("Success! 2-pass Gaussian blur completed on %dx%d image.\n", WIDTH, HEIGHT);
	} else {
		std::printf("Failed! Output is identical to input; blur was not applied.\n");
		return 1;
	}

	return 0;
}
