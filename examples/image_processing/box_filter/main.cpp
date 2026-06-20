/**
 * @file main.cpp
 * @brief Box mean filter with nested For loops.
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
	constexpr int	 WIDTH		 = 6;
	constexpr int	 HEIGHT		 = 6;
	constexpr int	 PIXEL_COUNT = WIDTH * HEIGHT;
	constexpr int	 KERNEL_SIZE = 3; // 3x3 box filter
	constexpr int	 KERNEL_AREA = KERNEL_SIZE * KERNEL_SIZE;

	// ========================================================================
	// Host Data Preparation
	// ========================================================================
	// Create a checkerboard pattern
	std::vector<int> host_input(PIXEL_COUNT);
	for (int y = 0; y < HEIGHT; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			bool black				  = ((x + y) % 2) == 0;
			int	 c					  = black ? 0 : 255;
			host_input[y * WIDTH + x] = PackRGBA(c, c, c);
		}
	}

	std::vector<int> host_output(PIXEL_COUNT);

	// ========================================================================
	// Device Data Upload
	// ========================================================================
	Buffer<int>		 input_image(host_input);
	Buffer<int>		 output_image(PIXEL_COUNT);

	// ========================================================================
	// Kernel Definition
	// ========================================================================
	Kernel2D		 box_filter(
		"BoxFilter",
		[&](Int x, Int y) {
			If(x < WIDTH && y < HEIGHT, [&]() {
				auto in	   = input_image.Bind();
				auto out   = output_image.Bind();

				Int	 idx   = y * WIDTH + x;

				// Accumulate per-channel sums
				Int	 sum_r = MakeInt(0);
				Int	 sum_g = MakeInt(0);
				Int	 sum_b = MakeInt(0);
				Int	 sum_a = MakeInt(0);

				// 3x3 neighborhood
				For(-1, 2, [&](Int &dy) {
					For(-1, 2, [&](Int &dx) {
						Int sx = Clamp(x + dx, 0, WIDTH - 1);
						Int sy = Clamp(y + dy, 0, HEIGHT - 1);

						Int p  = in[sy * WIDTH + sx];

						sum_r  = sum_r + (p & 0xFF);
						sum_g  = sum_g + ((p >> 8) & 0xFF);
						sum_b  = sum_b + ((p >> 16) & 0xFF);
						sum_a  = sum_a + ((p >> 24) & 0xFF);
					});
				});

				// Average (integer division)
				Int avg_b  = sum_b / KERNEL_AREA;
				Int avg_g  = sum_g / KERNEL_AREA;
				Int avg_r  = sum_r / KERNEL_AREA;
				Int avg_a  = sum_a / KERNEL_AREA;

				// Repack
				Int result = (avg_a << 24) | (avg_b << 16) | (avg_g << 8) | avg_r;
				out[idx]   = result;
			});
		},
		16, 16);

	// ========================================================================
	// Kernel Dispatch
	// ========================================================================
	int groups_x = (WIDTH + 15) / 16;
	int groups_y = (HEIGHT + 15) / 16;
	box_filter.Dispatch(groups_x, groups_y, true);

	// ========================================================================
	// Result Verification
	// ========================================================================
	output_image.Download(host_output);

	bool all_correct = true;
	for (int y = 0; y < HEIGHT; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			int idx	  = y * WIDTH + x;

			// CPU reference: 3x3 box filter with clamp
			int sum_r = 0, sum_g = 0, sum_b = 0, sum_a = 0;
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					int sx	= std::clamp(x + dx, 0, WIDTH - 1);
					int sy	= std::clamp(y + dy, 0, HEIGHT - 1);
					int p	= host_input[sy * WIDTH + sx];
					sum_r  += p & 0xFF;
					sum_g  += (p >> 8) & 0xFF;
					sum_b  += (p >> 16) & 0xFF;
					sum_a  += (p >> 24) & 0xFF;
				}
			}
			int expected = ((sum_a / KERNEL_AREA) << 24) | ((sum_b / KERNEL_AREA) << 16) |
						   ((sum_g / KERNEL_AREA) << 8) | (sum_r / KERNEL_AREA);

			if (host_output[idx] != expected) {
				all_correct = false;
				std::printf("Mismatch at (%d,%d): got 0x%08X, expected 0x%08X\n", x, y, host_output[idx], expected);
				break;
			}
		}
		if (!all_correct)
			break;
	}

	if (all_correct) {
		std::printf("Success! Box filter applied correctly to %dx%d image.\n", WIDTH, HEIGHT);
	} else {
		std::printf("Failed! Result verification encountered errors.\n");
		return 1;
	}

	return 0;
}
