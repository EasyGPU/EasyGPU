/**
 * TestTexture3D.cpp:
 *      @Descripiton    :   Test for GPU Texture3D functionality
 *      @Author         :   EasyGPU
 *      @Date           :   4/4/2026
 */
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <GPU.h>

using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Runtime;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name)                                                                                                     \
	void test_##name() {                                                                                               \
		std::cout << "\n[TEST] " #name " ... ";                                                                        \
		test_count++;                                                                                                    \
		try {

#define END_TEST                                                                                                       \
		pass_count++;                                                                                                    \
		std::cout << "PASSED\n";                                                                                         \
		}                                                                                                                \
		catch (const GPU::Runtime::ShaderCompileException &e) {                                                          \
			std::cout << "FAILED: Shader compilation error\n";                                                             \
			std::cout << e.GetBeautifulOutput() << "\n";                                                                   \
		}                                                                                                                \
		catch (const GPU::Runtime::ShaderException &e) {                                                                 \
			std::cout << "FAILED: Shader error - " << e.what() << "\n";                                                    \
		}                                                                                                                \
		catch (const std::exception &e) {                                                                                \
			std::cout << "FAILED: " << e.what() << "\n";                                                                   \
		}                                                                                                                \
		catch (...) {                                                                                                    \
			std::cout << "FAILED: Unknown exception\n";                                                                    \
		}                                                                                                                \
		}

#define ASSERT(cond)                                                                                                   \
	if (!(cond)) {                                                                                                     \
		throw std::runtime_error("Assertion failed: " #cond);                                                          \
	}

// =============================================================================
// Test 1: Basic 3D texture creation
// =============================================================================
TEST(texture3d_create_empty)
Texture3D<PixelFormat::RGBA8> tex(16, 16, 16);
ASSERT(tex.GetWidth() == 16);
ASSERT(tex.GetHeight() == 16);
ASSERT(tex.GetDepth() == 16);
ASSERT(tex.GetHandle() != 0);
std::cout << "Created 16x16x16 RGBA8 texture";
END_TEST

// =============================================================================
// Test 2: 3D texture creation from raw buffer
// =============================================================================
TEST(texture3d_create_from_buffer)
const int W = 8, H = 8, D = 8;
std::vector<uint8_t> voxels(W * H * D * 4);
for (int i = 0; i < W * H * D; ++i) {
	voxels[i * 4 + 0] = 255;
	voxels[i * 4 + 1] = 128;
	voxels[i * 4 + 2] = 64;
	voxels[i * 4 + 3] = 255;
}
Texture3D<PixelFormat::RGBA8> tex(W, H, D, voxels.data());
ASSERT(tex.GetWidth() == W);
ASSERT(tex.GetHeight() == H);
ASSERT(tex.GetDepth() == D);
ASSERT(tex.GetHandle() != 0);
std::cout << "Created " << W << "x" << H << "x" << D << " texture from buffer";
END_TEST

// =============================================================================
// Test 3: 3D texture upload/download
// =============================================================================
TEST(texture3d_upload_download)
const int W = 8, H = 8, D = 8;
std::vector<uint8_t> uploadVoxels(W * H * D * 4);
std::vector<uint8_t> downloadVoxels(W * H * D * 4);

for (int z = 0; z < D; ++z) {
	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			int idx = ((z * H + y) * W + x) * 4;
			uploadVoxels[idx + 0] = static_cast<uint8_t>(x * 32);
			uploadVoxels[idx + 1] = static_cast<uint8_t>(y * 32);
			uploadVoxels[idx + 2] = static_cast<uint8_t>(z * 32);
			uploadVoxels[idx + 3] = 255;
		}
	}
}

Texture3D<PixelFormat::RGBA8> tex(W, H, D);
tex.Upload(uploadVoxels.data());
tex.Download(downloadVoxels.data());

bool match = true;
for (int i = 0; i < W * H * D * 4; ++i) {
	if (std::abs(uploadVoxels[i] - downloadVoxels[i]) > 2) {
		match = false;
		std::cout << "Mismatch at byte " << i << ": uploaded " << (int)uploadVoxels[i] << ", downloaded "
				  << (int)downloadVoxels[i];
		break;
	}
}
ASSERT(match);
std::cout << "Upload/Download verified for " << W << "x" << H << "x" << D << " texture";
END_TEST

// =============================================================================
// Test 4: 3D texture sub-region upload
// =============================================================================
TEST(texture3d_subregion_upload)
const int W = 8, H = 8, D = 8;
std::vector<uint8_t> uploadVoxels(W * H * D * 4, 0);
std::vector<uint8_t> subVoxels(4 * 4 * 4 * 4, 255);
for (int i = 0; i < 4 * 4 * 4; ++i) {
	subVoxels[i * 4 + 0] = 10;
	subVoxels[i * 4 + 1] = 20;
	subVoxels[i * 4 + 2] = 30;
	subVoxels[i * 4 + 3] = 40;
}

Texture3D<PixelFormat::RGBA8> tex(W, H, D);
tex.UploadSubRegion(2, 2, 2, 4, 4, 4, subVoxels.data());

std::vector<uint8_t> downloadVoxels(W * H * D * 4);
tex.Download(downloadVoxels.data());

bool correct = true;
for (int z = 2; z < 6; ++z) {
	for (int y = 2; y < 6; ++y) {
		for (int x = 2; x < 6; ++x) {
			int idx = ((z * H + y) * W + x) * 4;
			if (downloadVoxels[idx + 0] != 10 || downloadVoxels[idx + 1] != 20 ||
				downloadVoxels[idx + 2] != 30 || downloadVoxels[idx + 3] != 40) {
				correct = false;
				break;
			}
		}
	}
}
ASSERT(correct);
std::cout << "Sub-region upload verified";
END_TEST

// =============================================================================
// Test 5: 3D texture move semantics
// =============================================================================
TEST(texture3d_move)
Texture3D<PixelFormat::RGBA8> tex1(8, 8, 8);
uint32_t handle1 = tex1.GetHandle();

Texture3D<PixelFormat::RGBA8> tex2(std::move(tex1));
uint32_t handle2 = tex2.GetHandle();

ASSERT(handle1 == handle2);
ASSERT(tex1.GetHandle() == 0);
ASSERT(tex2.GetDepth() == 8);

Texture3D<PixelFormat::RGBA8> tex3(4, 4, 4);
tex3 = std::move(tex2);
ASSERT(tex3.GetHandle() == handle1);
ASSERT(tex2.GetHandle() == 0);
ASSERT(tex3.GetDepth() == 8);

std::cout << "Move semantics verified!";
END_TEST

// =============================================================================
// Test 6: 3D texture Bind API (InspectorKernel)
// =============================================================================
TEST(texture3d_bind_api_inspector)
Texture3D<PixelFormat::RGBA8> tex(8, 8, 8);

GPU::Kernel::InspectorKernel kernel([&](Var<int> &id) {
	auto vol = tex.Bind();

	Var<int> x = id % 8;
	Var<int> y = (id / 8) % 8;
	Var<int> z = id / 64;

	Var<Vec4> color = vol.Read(x, y, z);
	vol.Write(x, y, z, Vec4(1.0f) - color);
});

std::cout << "\n=== Generated GLSL (Texture3D Bind API) ===\n";
kernel.PrintCode();
std::cout << "=========================================\n";

ASSERT(true);
END_TEST

// =============================================================================
// Test 7: End-to-end GPU 3D texture operation - volume fill
// =============================================================================
TEST(gpu_texture3d_fill)
const int W = 8, H = 8, D = 8;
Texture3D<PixelFormat::RGBA8> tex(W, H, D);

GPU::Kernel::Kernel1D kernel(
	[&](Var<int> &id) {
		auto vol = tex.Bind();

		Var<int> x = id % W;
		Var<int> y = (id / W) % H;
		Var<int> z = id / (W * H);

		Var<float> r = Expr<float>(x) / static_cast<float>(W);
		Var<float> g = Expr<float>(y) / static_cast<float>(H);
		Var<float> b = Expr<float>(z) / static_cast<float>(D);

		vol.Write(x, y, z, Expr<Vec4>(r, g, b, 1.0f));
	},
	64);

kernel.Dispatch((W * H * D + 63) / 64, true);

std::vector<uint8_t> resultVoxels(W * H * D * 4);
tex.Download(resultVoxels.data());

bool correct = true;
for (int z = 0; z < D && correct; ++z) {
	for (int y = 0; y < H && correct; ++y) {
		for (int x = 0; x < W && correct; ++x) {
			int idx = ((z * H + y) * W + x) * 4;
			uint8_t r = resultVoxels[idx + 0];
			uint8_t g = resultVoxels[idx + 1];
			uint8_t b = resultVoxels[idx + 2];
			uint8_t a = resultVoxels[idx + 3];

			uint8_t expectedR = static_cast<uint8_t>((x / static_cast<float>(W)) * 255.0f);
			uint8_t expectedG = static_cast<uint8_t>((y / static_cast<float>(H)) * 255.0f);
			uint8_t expectedB = static_cast<uint8_t>((z / static_cast<float>(D)) * 255.0f);

			if (std::abs(r - expectedR) > 5 || std::abs(g - expectedG) > 5 || std::abs(b - expectedB) > 5 || a != 255) {
				correct = false;
				std::cout << "Voxel (" << x << "," << y << "," << z << ") mismatch: got (" << (int)r << "," << (int)g
						  << "," << (int)b << "," << (int)a << "), expected (~" << (int)expectedR << ",~" << (int)expectedG
						  << ",~" << (int)expectedB << ",255)";
			}
		}
	}
}

ASSERT(correct);
std::cout << "Volume fill verified!";
END_TEST

// =============================================================================
// Test 8: Float 3D texture format (RGBA32F)
// =============================================================================
TEST(texture3d_rgba32f_format)
const int W = 4, H = 4, D = 4;

std::vector<float> floatVoxels(W * H * D * 4);
for (int i = 0; i < W * H * D; ++i) {
	floatVoxels[i * 4 + 0] = 0.25f;
	floatVoxels[i * 4 + 1] = 0.5f;
	floatVoxels[i * 4 + 2] = 0.75f;
	floatVoxels[i * 4 + 3] = 1.0f;
}

Texture3D<PixelFormat::RGBA32F> floatTex(W, H, D, floatVoxels.data());

GPU::Kernel::Kernel1D kernel(
	[&](Var<int> &id) {
		auto vol = floatTex.Bind();

		Var<int> x = id % W;
		Var<int> y = (id / W) % H;
		Var<int> z = id / (W * H);

		Var<Vec4> color = vol.Read(x, y, z);
		vol.Write(x, y, z, color * 2.0f);
	},
	32);

kernel.Dispatch((W * H * D + 31) / 32, true);

std::vector<float> resultVoxels(W * H * D * 4);
floatTex.Download(resultVoxels.data());

bool correct = true;
for (int i = 0; i < W * H * D; ++i) {
	if (std::abs(resultVoxels[i * 4 + 0] - 0.5f) > 0.01f || std::abs(resultVoxels[i * 4 + 1] - 1.0f) > 0.01f ||
		std::abs(resultVoxels[i * 4 + 2] - 1.5f) > 0.01f || std::abs(resultVoxels[i * 4 + 3] - 2.0f) > 0.01f) {
		correct = false;
		break;
	}
}
ASSERT(correct);
std::cout << "Float 3D texture verified!";
END_TEST

// =============================================================================
// Test 9: 3D texture with R32F format
// =============================================================================
TEST(texture3d_r32f_format)
const int W = 4, H = 4, D = 4;

std::vector<float> floatVoxels(W * H * D);
for (int i = 0; i < W * H * D; ++i) {
	floatVoxels[i] = static_cast<float>(i);
}

Texture3D<PixelFormat::R32F> floatTex(W, H, D, floatVoxels.data());

GPU::Kernel::Kernel1D kernel(
	[&](Var<int> &id) {
		auto vol = floatTex.Bind();

		Var<int> x = id % W;
		Var<int> y = (id / W) % H;
		Var<int> z = id / (W * H);

		Var<Vec4> color = vol.Read(x, y, z);
		vol.Write(x, y, z, MakeFloat4(color.x() + 1.0f, 0.0f, 0.0f, 0.0f));
	},
	32);

kernel.Dispatch((W * H * D + 31) / 32, true);

std::vector<float> resultVoxels(W * H * D);
floatTex.Download(resultVoxels.data());

bool correct = true;
for (int i = 0; i < W * H * D; ++i) {
	if (std::abs(resultVoxels[i] - (static_cast<float>(i) + 1.0f)) > 0.01f) {
		correct = false;
		break;
	}
}
ASSERT(correct);
std::cout << "R32F 3D texture verified!";
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================\n";
	std::cout << "  EasyGPU Texture3D Test Suite          \n";
	std::cout << "========================================\n";

	try {
		test_texture3d_create_empty();
		test_texture3d_create_from_buffer();
		test_texture3d_upload_download();
		test_texture3d_subregion_upload();
		test_texture3d_move();
		test_texture3d_bind_api_inspector();
		test_gpu_texture3d_fill();
		test_texture3d_rgba32f_format();
		test_texture3d_r32f_format();

		std::cout << "\n========================================\n";
		std::cout << "  Results: " << pass_count << "/" << test_count << " tests passed\n";
		std::cout << "========================================\n";

		return (pass_count == test_count) ? 0 : 1;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << "\n";
		return 1;
	}
}
