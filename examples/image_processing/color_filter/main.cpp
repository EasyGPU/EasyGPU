/**
 * @file main.cpp
 * @brief Single-pixel color filter example.
 */

#include <GPU.h>

#include <cstdio>
#include <vector>

using namespace GPU;

namespace {
constexpr int PackRGBA(int r, int g, int b, int a = 255) {
	return r | (g << 8) | (b << 16) | (a << 24);
}
} // namespace

int main() {
	// ========================================================================
	// Configuration
	// ========================================================================
	constexpr int	 WIDTH		 = 4;
	constexpr int	 HEIGHT		 = 4;
	constexpr int	 PIXEL_COUNT = WIDTH * HEIGHT;

	// ========================================================================
	// Host Data Preparation
	// ========================================================================
	// Packed RGBA pixels in byte order: R | G << 8 | B << 16 | A << 24
	std::vector<int> host_input	 = {
		PackRGBA(255, 0, 0),	  PackRGBA(0, 255, 0),	  PackRGBA(0, 0, 255),	PackRGBA(255, 255, 255),
		PackRGBA(127, 127, 127), PackRGBA(0x12, 0x34, 0x56), PackRGBA(0xAB, 0xCD, 0xEF),
		PackRGBA(0, 0, 0),		  PackRGBA(255, 0, 255),	  PackRGBA(255, 255, 0), PackRGBA(0, 255, 255),
		PackRGBA(0x11, 0x22, 0x33), PackRGBA(0x44, 0x55, 0x66), PackRGBA(0x77, 0x88, 0x99),
		PackRGBA(0xAA, 0xBB, 0xCC), PackRGBA(0xDD, 0xEE, 0xFF),
	};

	std::vector<int> host_output(PIXEL_COUNT);

	// ========================================================================
	// Device Data Upload
	// ========================================================================
	Buffer<int>		 input_image(host_input);
	Buffer<int>		 output_image(PIXEL_COUNT);

	// ========================================================================
	// Kernel Definition
	// ========================================================================
	// Apply a red tint: boost red channel by 50%, keep others
	Kernel2D		 color_filter(
		"ColorFilter",
		[&](Int x, Int y) {
			If(x < WIDTH && y < HEIGHT, [&]() {
				auto in	   = input_image.Bind();
				auto out   = output_image.Bind();

				Int	 idx   = y * WIDTH + x;
				Int	 pixel = in[idx];

				// Extract channels using bitwise ops
				Int	 r	   = pixel & 0xFF;
				Int	 g	   = (pixel >> 8) & 0xFF;
				Int	 b	   = (pixel >> 16) & 0xFF;
				Int	 a	   = (pixel >> 24) & 0xFF;

				// Boost red channel (clamp to 255)
				Int	 new_r = r + (r >> 1);
				If(new_r > 255, [&]() { new_r = 255; });

				// Repack channels
				Int result = (a << 24) | (b << 16) | (g << 8) | new_r;

				out[idx]   = result;
			});
		},
		16, 16);

	// ========================================================================
	// Kernel Dispatch
	// ========================================================================
	int groups_x = (WIDTH + 15) / 16;
	int groups_y = (HEIGHT + 15) / 16;
	color_filter.Dispatch(groups_x, groups_y, true);

	// ========================================================================
	// Result Verification
	// ========================================================================
	output_image.Download(host_output);

	bool all_correct = true;
	for (int i = 0; i < PIXEL_COUNT; ++i) {
		int in_pixel   = host_input[i];
		int out_pixel  = host_output[i];

		int r		   = in_pixel & 0xFF;
		int g		   = (in_pixel >> 8) & 0xFF;
		int b		   = (in_pixel >> 16) & 0xFF;
		int a		   = (in_pixel >> 24) & 0xFF;

		int expected_r = r + (r >> 1);
		if (expected_r > 255)
			expected_r = 255;
		int expected = (a << 24) | (b << 16) | (g << 8) | expected_r;

		if (out_pixel != expected) {
			all_correct = false;
			std::printf("Mismatch at pixel %d: got 0x%08X, expected 0x%08X\n", i, out_pixel, expected);
			break;
		}
	}

	if (all_correct) {
		std::printf("Success! Color filter applied correctly to %d pixels.\n", PIXEL_COUNT);
	} else {
		std::printf("Failed! Result verification encountered errors.\n");
		return 1;
	}

	return 0;
}
