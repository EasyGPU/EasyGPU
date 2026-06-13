/**
 * @file TestMultiKernelSequence.cpp
 * @brief Tests multi-kernel execution pipelines and resource isolation.
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
// Two-Kernel Pipeline (Read -> Process -> Write)
// =============================================================================

TEST(two_kernel_sequence)
constexpr int	   N = 1024;
std::vector<float> input(N);
for (int i = 0; i < N; ++i) {
	input[i] = static_cast<float>(i);
}

Runtime::Buffer<float> bufA(input);
Runtime::Buffer<float> bufB(N);
Runtime::Buffer<float> bufC(N);

// Kernel 1: A -> B (square)
Kernel1D			   kernel1(
	[&, N](Var<int> &id) {
		auto	   a = bufA.Bind();
		auto	   b = bufB.Bind();
		Var<float> v = a[id];
		b[id]		 = v * v;
	},
	256);

// Kernel 2: B -> C (add 1)
Kernel1D kernel2(
	[&, N](Var<int> &id) {
		auto b = bufB.Bind();
		auto c = bufC.Bind();
		c[id]  = b[id] + MakeFloat(1.0f);
	},
	256);

int groups = (N + 255) / 256;
kernel1.Dispatch(groups, true);
kernel2.Dispatch(groups, true);

std::vector<float> output(N);
bufC.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	float expected = input[i] * input[i] + 1.0f;
	ASSERT_NEAR(output[i], expected, 0.0001f);
}
END_TEST

// =============================================================================
// Three-Kernel Pipeline with Reuse
// =============================================================================

TEST(three_kernel_pipeline_with_reuse)
constexpr int		   N = 512;
std::vector<float>	   data(N, 2.0f);
Runtime::Buffer<float> buf(data);

Kernel1D			   kernelMul(
	[&, N](Var<int> &id) {
		auto b = buf.Bind();
		b[id]  = b[id] * MakeFloat(3.0f);
	},
	256);

Kernel1D kernelAdd(
	[&, N](Var<int> &id) {
		auto b = buf.Bind();
		b[id]  = b[id] + MakeFloat(1.0f);
	},
	256);

Kernel1D kernelDiv(
	[&, N](Var<int> &id) {
		auto b = buf.Bind();
		b[id]  = b[id] / MakeFloat(2.0f);
	},
	256);

int groups = (N + 255) / 256;
// ((2 * 3) + 1) / 2 = 3.5
kernelMul.Dispatch(groups, true);
kernelAdd.Dispatch(groups, true);
kernelDiv.Dispatch(groups, true);

std::vector<float> output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], 3.5f, 0.0001f);
}
END_TEST

// =============================================================================
// Resource Isolation Between Kernels
// =============================================================================

TEST(resource_isolation)
constexpr int		   N = 256;
std::vector<float>	   dataA(N, 1.0f);
std::vector<float>	   dataB(N, 2.0f);
Runtime::Buffer<float> bufA(dataA);
Runtime::Buffer<float> bufB(dataB);
Runtime::Buffer<float> bufOutA(N);
Runtime::Buffer<float> bufOutB(N);

// Kernel writes to bufOutA using bufA
Kernel1D			   kernelA(
	[&, N](Var<int> &id) {
		auto in	 = bufA.Bind();
		auto out = bufOutA.Bind();
		out[id]	 = in[id] + MakeFloat(10.0f);
	},
	256);

// Kernel writes to bufOutB using bufB
Kernel1D kernelB(
	[&, N](Var<int> &id) {
		auto in	 = bufB.Bind();
		auto out = bufOutB.Bind();
		out[id]	 = in[id] + MakeFloat(20.0f);
	},
	256);

kernelA.Dispatch(1, true);
kernelB.Dispatch(1, true);

std::vector<float> outA(N);
std::vector<float> outB(N);
bufOutA.Download(outA.data(), N);
bufOutB.Download(outB.data(), N);

for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(outA[i], 11.0f, 0.0001f);
	ASSERT_NEAR(outB[i], 22.0f, 0.0001f);
}
END_TEST

// =============================================================================
// Kernel Reuse (same kernel object dispatched multiple times)
// =============================================================================

TEST(kernel_object_reuse)
constexpr int		   N = 256;
std::vector<float>	   data(N, 1.0f);
Runtime::Buffer<float> buf(data);

Kernel1D			   kernel(
	[&, N](Var<int> &id) {
		auto b = buf.Bind();
		b[id]  = b[id] * MakeFloat(2.0f);
	},
	256);

kernel.Dispatch(1, true); // 2
kernel.Dispatch(1, true); // 4
kernel.Dispatch(1, true); // 8

std::vector<float> output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], 8.0f, 0.0001f);
}
END_TEST

// =============================================================================
// Uniform Change Between Dispatches
// =============================================================================

TEST(uniform_change_between_dispatches)
constexpr int		   N = 256;
std::vector<float>	   data(N, 0.0f);
Runtime::Buffer<float> buf(data);
Uniform<float>		   factor(1.0f);

Kernel1D			   kernel(
	[&, N](Var<int> &id) {
		auto	   b = buf.Bind();
		Var<float> f = factor.Load();
		b[id]		 = b[id] + f;
	},
	256);

kernel.Dispatch(1, true); // +1
factor = 2.0f;
kernel.Dispatch(1, true); // +2
factor = 3.0f;
kernel.Dispatch(1, true); // +3

std::vector<float> output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], 6.0f, 0.0001f); // 1 + 2 + 3
}
END_TEST

// =============================================================================
// Ping-Pong Between Two Buffers Using Multiple Kernels
// =============================================================================

TEST(ping_pong_buffers)
constexpr int		   N = 256;
std::vector<float>	   dataA(N, 1.0f);
std::vector<float>	   dataB(N, 0.0f);
Runtime::Buffer<float> bufA(dataA);
Runtime::Buffer<float> bufB(dataB);

Kernel1D			   kernelAtoB(
	[&, N](Var<int> &id) {
		auto a = bufA.Bind();
		auto b = bufB.Bind();
		b[id]  = a[id] + MakeFloat(1.0f);
	},
	256);

Kernel1D kernelBtoA(
	[&, N](Var<int> &id) {
		auto a = bufA.Bind();
		auto b = bufB.Bind();
		a[id]  = b[id] + MakeFloat(1.0f);
	},
	256);

// A=1 -> B=2 -> A=3 -> B=4 -> A=5
kernelAtoB.Dispatch(1, true);
kernelBtoA.Dispatch(1, true);
kernelAtoB.Dispatch(1, true);
kernelBtoA.Dispatch(1, true);
kernelAtoB.Dispatch(1, true);

std::vector<float> outputA(N);
std::vector<float> outputB(N);
bufA.Download(outputA.data(), N);
bufB.Download(outputB.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(outputA[i], 5.0f, 0.0001f);
	ASSERT_NEAR(outputB[i], 6.0f, 0.0001f);
}
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Multi-Kernel Sequence Tests   " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_two_kernel_sequence();
		test_three_kernel_pipeline_with_reuse();
		test_resource_isolation();
		test_kernel_object_reuse();
		test_uniform_change_between_dispatches();
		test_ping_pong_buffers();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All multi-kernel tests passed!        " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
