/**
 * TestTextureFormats.cpp
 * Comprehensive read/write tests for all supported texture pixel formats.
 */

#include <GPU.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;

#define TEST(name)                                                                                                     \
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

#define ASSERT_EQ(a, b)                                                                                                \
	if ((a) != (b)) {                                                                                                  \
		throw std::runtime_error("Assertion failed: " #a " != " #b);                                                   \
	}

#define ASSERT_NEAR(a, b, eps)                                                                                         \
	if (std::abs((a) - (b)) > (eps)) {                                                                                 \
		throw std::runtime_error("Assertion failed: |" #a " - " #b "| > " #eps);                                       \
	}

// =============================================================================
// Helper: fill a CPU buffer with a gradient pattern
// =============================================================================
template <typename T> static std::vector<T> makeGradientData(size_t count, T minVal, T maxVal) {
	std::vector<T> data(count);
	for (size_t i = 0; i < count; ++i) {
		float t = static_cast<float>(i) / static_cast<float>(std::max<size_t>(count - 1, 1));
		data[i] = static_cast<T>(minVal + static_cast<float>(maxVal - minVal) * t);
	}
	return data;
}

// =============================================================================
// R8 Texture
// =============================================================================
TEST(texture_r8_upload_download)
constexpr int		 W = 32;
constexpr int		 H = 32;
std::vector<uint8_t> data(W *H);
for (int i = 0; i < W * H; ++i) {
	data[i] = static_cast<uint8_t>(i % 256);
}

Runtime::Texture2D<PixelFormat::R8> tex(W, H);
tex.Upload(data.data());

std::vector<uint8_t> output(W *H);
tex.Download(output.data());

for (int i = 0; i < W * H; ++i) {
	ASSERT_EQ(data[i], output[i]);
}
END_TEST

// =============================================================================
// RG8 Texture
// =============================================================================
TEST(texture_rg8_upload_download)
constexpr int		 W = 16;
constexpr int		 H = 16;
std::vector<uint8_t> data(W *H * 2);
for (int i = 0; i < W * H; ++i) {
	data[i * 2 + 0] = static_cast<uint8_t>(i % 256);
	data[i * 2 + 1] = static_cast<uint8_t>((i * 3) % 256);
}

Runtime::Texture2D<PixelFormat::RG8> tex(W, H);
tex.Upload(data.data());

std::vector<uint8_t> output(W *H * 2);
tex.Download(output.data());

for (int i = 0; i < W * H * 2; ++i) {
	ASSERT_EQ(data[i], output[i]);
}
END_TEST

// =============================================================================
// RGBA8 Texture
// =============================================================================
TEST(texture_rgba8_upload_download)
constexpr int		 W = 16;
constexpr int		 H = 16;
std::vector<uint8_t> data(W *H * 4);
for (int i = 0; i < W * H; ++i) {
	data[i * 4 + 0] = static_cast<uint8_t>(i % 256);
	data[i * 4 + 1] = static_cast<uint8_t>((i * 2) % 256);
	data[i * 4 + 2] = static_cast<uint8_t>((i * 3) % 256);
	data[i * 4 + 3] = 255;
}

Runtime::Texture2D<PixelFormat::RGBA8> tex(W, H);
tex.Upload(data.data());

std::vector<uint8_t> output(W *H * 4);
tex.Download(output.data());

for (int i = 0; i < W * H * 4; ++i) {
	ASSERT_EQ(data[i], output[i]);
}
END_TEST

// =============================================================================
// RGBA16F Texture
// =============================================================================
TEST(texture_rgba16f_upload_download)
constexpr int		  W = 8;
constexpr int		  H = 8;
std::vector<uint16_t> data(W *H * 4);
for (int i = 0; i < W * H; ++i) {
	// Simple half-float values (1.0, 2.0, 0.5, 1.0)
	data[i * 4 + 0] = 0x3C00; // 1.0 in half-float
	data[i * 4 + 1] = 0x4000; // 2.0 in half-float
	data[i * 4 + 2] = 0x3800; // 0.5 in half-float
	data[i * 4 + 3] = 0x3C00; // 1.0 in half-float
}

Runtime::Texture2D<PixelFormat::RGBA16F> tex(W, H);
tex.Upload(data.data());

std::vector<uint16_t> output(W *H * 4);
tex.Download(output.data());

for (int i = 0; i < W * H * 4; ++i) {
	ASSERT_EQ(data[i], output[i]);
}
END_TEST

// =============================================================================
// R32I Texture (signed integer)
// =============================================================================
TEST(texture_r32i_upload_download)
constexpr int		 W = 8;
constexpr int		 H = 8;
std::vector<int32_t> data(W *H);
for (int i = 0; i < W * H; ++i) {
	data[i] = static_cast<int32_t>(i - 32);
}

Runtime::Texture2D<PixelFormat::R32I> tex(W, H);
tex.Upload(data.data());

std::vector<int32_t> output(W *H);
tex.Download(output.data());

for (int i = 0; i < W * H; ++i) {
	ASSERT_EQ(data[i], output[i]);
}
END_TEST

// =============================================================================
// RGBA32UI Texture (unsigned integer)
// =============================================================================
TEST(texture_rgba32ui_upload_download)
constexpr int		  W = 4;
constexpr int		  H = 4;
std::vector<uint32_t> data(W *H * 4);
for (int i = 0; i < W * H; ++i) {
	data[i * 4 + 0] = static_cast<uint32_t>(i);
	data[i * 4 + 1] = static_cast<uint32_t>(i * 2);
	data[i * 4 + 2] = static_cast<uint32_t>(i * 3);
	data[i * 4 + 3] = 0xFFFFFFFFu;
}

Runtime::Texture2D<PixelFormat::RGBA32UI> tex(W, H);
tex.Upload(data.data());

std::vector<uint32_t> output(W *H * 4);
tex.Download(output.data());

for (int i = 0; i < W * H * 4; ++i) {
	ASSERT_EQ(data[i], output[i]);
}
END_TEST

// =============================================================================
// 3D Texture - RGBA8
// =============================================================================
TEST(texture3d_rgba8_upload_download)
constexpr int		 W = 8;
constexpr int		 H = 8;
constexpr int		 D = 8;
std::vector<uint8_t> data(W *H *D * 4);
for (int i = 0; i < W * H * D; ++i) {
	data[i * 4 + 0] = static_cast<uint8_t>(i % 256);
	data[i * 4 + 1] = static_cast<uint8_t>((i * 7) % 256);
	data[i * 4 + 2] = static_cast<uint8_t>((i * 13) % 256);
	data[i * 4 + 3] = 255;
}

Runtime::Texture3D<PixelFormat::RGBA8> tex(W, H, D);
tex.Upload(data.data());

std::vector<uint8_t> output(W *H *D * 4);
tex.Download(output.data());

for (int i = 0; i < W * H * D * 4; ++i) {
	ASSERT_EQ(data[i], output[i]);
}
END_TEST

// =============================================================================
// Texture SubRegion Upload
// =============================================================================
TEST(texture_subregion_rgba8)
constexpr int						   W = 16;
constexpr int						   H = 16;
Runtime::Texture2D<PixelFormat::RGBA8> tex(W, H);

// Initialize with zeros
std::vector<uint8_t>				   zeros(W *H * 4, 0);
tex.Upload(zeros.data());

// Upload a 4x4 red block at (2, 3)
std::vector<uint8_t> block(4 * 4 * 4, 0);
for (int i = 0; i < 4 * 4; ++i) {
	block[i * 4 + 0] = 255;
	block[i * 4 + 1] = 0;
	block[i * 4 + 2] = 0;
	block[i * 4 + 3] = 255;
}
tex.UploadSubRegion(2, 3, 4, 4, block.data());

std::vector<uint8_t> output(W *H * 4);
tex.Download(output.data());

// Check the block area
for (int y = 3; y < 3 + 4; ++y) {
	for (int x = 2; x < 2 + 4; ++x) {
		int idx = (y * W + x) * 4;
		ASSERT_EQ(output[idx + 0], 255);
		ASSERT_EQ(output[idx + 1], 0);
		ASSERT_EQ(output[idx + 2], 0);
		ASSERT_EQ(output[idx + 3], 255);
	}
}

// Check an area outside the block is still zero
int outsideIdx = (0 * W + 0) * 4;
ASSERT_EQ(output[outsideIdx + 0], 0);
ASSERT_EQ(output[outsideIdx + 1], 0);
ASSERT_EQ(output[outsideIdx + 2], 0);
ASSERT_EQ(output[outsideIdx + 3], 0);
END_TEST

// =============================================================================
// Texture2D Kernel ImageStore / ImageLoad
// =============================================================================
TEST(texture_kernel_image_store_load)
constexpr int						   W = 8;
constexpr int						   H = 8;
Runtime::Texture2D<PixelFormat::RGBA8> tex(W, H);

Kernel2D							   kernel(
								  [&, W, H](Var<int> &x, Var<int> &y) {
		  auto		 img = tex.Bind();
		  Var<int>	 idx = y * W + x;
		  Var<float> r	 = ToFloat(idx % 256) / 255.0f;
		  Var<float> g	 = ToFloat((idx * 2) % 256) / 255.0f;
		  Var<float> b	 = ToFloat((idx * 3) % 256) / 255.0f;
		  img.Write(x, y, MakeFloat4(r, g, b, 1.0f));
								  },
								  8, 8);

kernel.Dispatch(1, 1, true);

std::vector<uint8_t> output(W *H * 4);
tex.Download(output.data());

for (int y = 0; y < H; ++y) {
	for (int x = 0; x < W; ++x) {
		int idx = (y * W + x) * 4;
		int i	= y * W + x;
		ASSERT_EQ(output[idx + 0], static_cast<uint8_t>(i % 256));
		ASSERT_EQ(output[idx + 1], static_cast<uint8_t>((i * 2) % 256));
		ASSERT_EQ(output[idx + 2], static_cast<uint8_t>((i * 3) % 256));
		ASSERT_EQ(output[idx + 3], 255);
	}
}
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Texture Format Tests          " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_texture_r8_upload_download();
		test_texture_rg8_upload_download();
		test_texture_rgba8_upload_download();
		test_texture_rgba16f_upload_download();
		test_texture_r32i_upload_download();
		test_texture_rgba32ui_upload_download();
		test_texture3d_rgba8_upload_download();
		test_texture_subregion_rgba8();
		test_texture_kernel_image_store_load();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All texture format tests passed!      " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
