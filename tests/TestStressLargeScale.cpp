/**
 * @file TestStressLargeScale.cpp
 * @brief Stress tests for large data sets, repeated dispatches, and resource pressure.
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

#define ASSERT_NEAR(a, b, eps)                                                                                         \
	if (std::abs((a) - (b)) > (eps)) {                                                                                 \
		throw std::runtime_error("Assertion failed: |" #a " - " #b "| > " #eps);                                       \
	}

// =============================================================================
// Large-Scale Buffer Tests
// =============================================================================

TEST(buffer_million_elements_upload_download)
constexpr size_t   N = 1'000'000;
std::vector<float> input(N);
for (size_t i = 0; i < N; ++i) {
	input[i] = static_cast<float>(i % 1000) / 1000.0f;
}

Runtime::Buffer<float> buf(input);
std::vector<float>	   output(N);
buf.Download(output.data(), N);

for (size_t i = 0; i < N; ++i) {
	ASSERT_NEAR(input[i], output[i], 0.0001f);
}
END_TEST

TEST(buffer_large_int_array)
constexpr size_t N = 500'000;
std::vector<int> input(N);
for (size_t i = 0; i < N; ++i) {
	input[i] = static_cast<int>(i);
}
Runtime::Buffer<int> buf(input);
std::vector<int>	 output(N);
buf.Download(output.data(), N);
for (size_t i = 0; i < N; ++i) {
	ASSERT(input[i] == output[i]);
}
END_TEST

// =============================================================================
// Large Dispatch Tests
// =============================================================================

TEST(kernel_dispatch_large_1d)
constexpr int	   N = 1'000'000;
std::vector<float> input(N);
for (int i = 0; i < N; ++i) {
	input[i] = static_cast<float>(i);
}
Runtime::Buffer<float> bufIn(input);
Runtime::Buffer<float> bufOut(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto in  = bufIn.Bind();
		  auto out = bufOut.Bind();
		  out[id]  = in[id] + MakeFloat(1.0f);
				  },
				  256);

int groups = (N + 255) / 256;
kernel.Dispatch(groups, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], input[i] + 1.0f, 0.0001f);
}
END_TEST

TEST(kernel_dispatch_multiple_workgroups_2d)
constexpr int		   W = 512;
constexpr int		   H = 512;
Runtime::Buffer<float> buf(W *H);

Kernel2D			   kernel(
				  [&, W, H](Var<int> &x, Var<int> &y) {
		  auto	   out = buf.Bind();
		  Var<int> idx = y * W + x;
		  out[idx]	   = ToFloat(x) + ToFloat(y);
				  },
				  16, 16);

kernel.Dispatch((W + 15) / 16, (H + 15) / 16, true);

std::vector<float> output(W *H);
buf.Download(output.data(), W *H);
for (int y = 0; y < H; ++y) {
	for (int x = 0; x < W; ++x) {
		float expected = static_cast<float>(x + y);
		ASSERT_NEAR(output[y * W + x], expected, 0.0001f);
	}
}
END_TEST

// =============================================================================
// Repeated Dispatch Stress
// =============================================================================

TEST(repeated_dispatch_same_kernel)
constexpr int		   N = 1024;
std::vector<float>	   data(N, 0.0f);
Runtime::Buffer<float> buf(data);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto b = buf.Bind();
		  b[id]	 = b[id] + MakeFloat(1.0f);
				  },
				  256);

int groups = (N + 255) / 256;
for (int iter = 0; iter < 50; ++iter) {
	kernel.Dispatch(groups, true);
}

std::vector<float> output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], 50.0f, 0.0001f);
}
END_TEST

TEST(rapid_kernel_creation_and_destruction)
constexpr int		   N = 1024;
std::vector<float>	   data(N, 1.0f);
Runtime::Buffer<float> buf(data);
std::vector<float>	   output(N);

for (int iter = 0; iter < 20; ++iter) {
	Kernel1D kernel(
		[&, N](Var<int> &id) {
			auto b = buf.Bind();
			b[id]  = b[id] * MakeFloat(2.0f);
		},
		256);
	kernel.Dispatch((N + 255) / 256, true);
}

buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], std::pow(2.0f, 20.0f), 0.1f);
}
END_TEST

// =============================================================================
// Resource Reuse / Rebinding Stress
// =============================================================================

TEST(buffer_repeated_upload_download_cycles)
constexpr int		   N = 100'000;
Runtime::Buffer<float> buf(N);
std::vector<float>	   data(N);
std::vector<float>	   output(N);

for (int iter = 0; iter < 10; ++iter) {
	for (int i = 0; i < N; ++i) {
		data[i] = static_cast<float>(iter * 1000 + i);
	}
	buf.Upload(data.data(), N);
	buf.Download(output.data(), N);
	for (int i = 0; i < N; ++i) {
		ASSERT_NEAR(data[i], output[i], 0.0001f);
	}
}
END_TEST

TEST(multiple_buffers_max_binding_slots)
// Create many buffers and bind them in a single kernel to stress binding slots.
constexpr int										 NUM_BUFFERS = 8;
constexpr int										 N			 = 1024;
std::vector<std::unique_ptr<Runtime::Buffer<float>>> buffers;
for (int b = 0; b < NUM_BUFFERS; ++b) {
	std::vector<float> data(N, static_cast<float>(b));
	buffers.push_back(std::make_unique<Runtime::Buffer<float>>(data));
}

Runtime::Buffer<float> out(N);
Kernel1D			   kernel(
				  [&, N, NUM_BUFFERS](Var<int> &id) {
		  Var<float> sum = MakeFloat(0.0f);
		  for (int b = 0; b < NUM_BUFFERS; ++b) {
			  sum = sum + buffers[b]->Bind()[id];
		  }
		  auto o = out.Bind();
		  o[id]	 = sum;
				  },
				  256);

kernel.Dispatch((N + 255) / 256, true);

std::vector<float> output(N);
out.Download(output.data(), N);
float expected = 0.0f;
for (int b = 0; b < NUM_BUFFERS; ++b) {
	expected += static_cast<float>(b);
}
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], expected, 0.0001f);
}
END_TEST

// =============================================================================
// Large-Element Type Stress (Vec4)
// =============================================================================

TEST(large_vec4_buffer)
constexpr int			N = 100'000;
std::vector<Math::Vec4> input(N);
for (int i = 0; i < N; ++i) {
	input[i] = Math::Vec4{static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2),
						  static_cast<float>(i + 3)};
}
Runtime::Buffer<Math::Vec4> buf(input);
std::vector<Math::Vec4>		output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(input[i].x, output[i].x, 0.0001f);
	ASSERT_NEAR(input[i].y, output[i].y, 0.0001f);
	ASSERT_NEAR(input[i].z, output[i].z, 0.0001f);
	ASSERT_NEAR(input[i].w, output[i].w, 0.0001f);
}
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Large-Scale Stress Tests      " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_buffer_million_elements_upload_download();
		test_buffer_large_int_array();
		test_kernel_dispatch_large_1d();
		test_kernel_dispatch_multiple_workgroups_2d();
		test_repeated_dispatch_same_kernel();
		test_rapid_kernel_creation_and_destruction();
		test_buffer_repeated_upload_download_cycles();
		test_multiple_buffers_max_binding_slots();
		test_large_vec4_buffer();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All stress tests passed!               " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
