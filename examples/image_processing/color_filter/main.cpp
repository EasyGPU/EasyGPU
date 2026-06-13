/**
 * @file main.cpp
 * @brief Single-pixel color filter example.
 */

#include <GPU.h>

#include <cstdio>
#include <vector>

using namespace GPU;

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
	// Packed RGBA pixels: 0xAARRGGBB
	std::vector<int> host_input	 = {
		static_cast<int>(0xFFFF0000), static_cast<int>(0xFF00FF00), static_cast<int>(0xFF0000FF),
		static_cast<int>(0xFFFFFFFF), static_cast<int>(0xFF7F7F7F), static_cast<int>(0xFF123456),
		static_cast<int>(0xFFABCDEF), static_cast<int>(0xFF000000), static_cast<int>(0xFFFF00FF),
		static_cast<int>(0xFFFFFF00), static_cast<int>(0xFF00FFFF), static_cast<int>(0xFF112233),
		static_cast<int>(0xFF445566), static_cast<int>(0xFF778899), static_cast<int>(0xFFAABBCC),
		static_cast<int>(0xFFDDEEFF),
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
				Int	 b	   = pixel & 0xFF;
				Int	 g	   = (pixel >> 8) & 0xFF;
				Int	 r	   = (pixel >> 16) & 0xFF;
				Int	 a	   = (pixel >> 24) & 0xFF;

				// Boost red channel (clamp to 255)
				Int	 new_r = r + (r >> 1);
				If(new_r > 255, [&]() { new_r = 255; });

				// Repack channels
				Int result = (a << 24) | (new_r << 16) | (g << 8) | b;

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

		int b		   = in_pixel & 0xFF;
		int g		   = (in_pixel >> 8) & 0xFF;
		int r		   = (in_pixel >> 16) & 0xFF;
		int a		   = (in_pixel >> 24) & 0xFF;

		int expected_r = r + (r >> 1);
		if (expected_r > 255)
			expected_r = 255;
		int expected = (a << 24) | (expected_r << 16) | (g << 8) | b;

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
