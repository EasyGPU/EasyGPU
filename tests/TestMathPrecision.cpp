/**
 * @file TestMathPrecision.cpp
 * @brief Runtime precision tests for math functions and edge-case inputs.
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

#define ASSERT_NEAR(a, b, eps)                                                                                         \
	if (std::abs((a) - (b)) > (eps)) {                                                                                 \
		throw std::runtime_error("Assertion failed: |" #a " - " #b "| > " #eps);                                       \
	}

#define ASSERT(cond)                                                                                                   \
	if (!(cond)) {                                                                                                     \
		throw std::runtime_error("Assertion failed: " #cond);                                                          \
	}

// =============================================================================
// Trigonometric Precision
// =============================================================================

TEST(sin_precision)
constexpr int	   N = 256;
std::vector<float> input(N);
for (int i = 0; i < N; ++i) {
	input[i] = static_cast<float>(i) * 0.1f;
}
Runtime::Buffer<float> bufIn(input);
Runtime::Buffer<float> bufOut(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto in  = bufIn.Bind();
		  auto out = bufOut.Bind();
		  out[id]  = Sin(in[id]);
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], std::sin(input[i]), 0.0001f);
}
END_TEST

TEST(cos_precision)
constexpr int	   N = 256;
std::vector<float> input(N);
for (int i = 0; i < N; ++i) {
	input[i] = static_cast<float>(i) * 0.1f;
}
Runtime::Buffer<float> bufIn(input);
Runtime::Buffer<float> bufOut(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto in  = bufIn.Bind();
		  auto out = bufOut.Bind();
		  out[id]  = Cos(in[id]);
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], std::cos(input[i]), 0.0001f);
}
END_TEST

TEST(atan2_quadrants)
constexpr int		   N	 = 4;
std::vector<float>	   xVals = {1.0f, -1.0f, -1.0f, 1.0f};
std::vector<float>	   yVals = {1.0f, 1.0f, -1.0f, -1.0f};
Runtime::Buffer<float> bufX(xVals);
Runtime::Buffer<float> bufY(yVals);
Runtime::Buffer<float> bufOut(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto x   = bufX.Bind();
		  auto y   = bufY.Bind();
		  auto out = bufOut.Bind();
		  out[id]  = Atan2(y[id], x[id]);
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], std::atan2(yVals[i], xVals[i]), 0.0001f);
}
END_TEST

// =============================================================================
// Exponential / Logarithmic Precision
// =============================================================================

TEST(sqrt_precision)
constexpr int	   N = 256;
std::vector<float> input(N);
for (int i = 0; i < N; ++i) {
	input[i] = static_cast<float>(i + 1);
}
Runtime::Buffer<float> bufIn(input);
Runtime::Buffer<float> bufOut(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto in  = bufIn.Bind();
		  auto out = bufOut.Bind();
		  out[id]  = Sqrt(in[id]);
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], std::sqrt(input[i]), 0.0001f);
}
END_TEST

TEST(exp_log_roundtrip)
constexpr int	   N = 64;
std::vector<float> input(N);
for (int i = 0; i < N; ++i) {
	input[i] = static_cast<float>(i) * 0.1f;
}
Runtime::Buffer<float> bufIn(input);
Runtime::Buffer<float> bufOut(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto in  = bufIn.Bind();
		  auto out = bufOut.Bind();
		  // log(exp(x)) should be x
		  out[id]  = Log(Exp(in[id]));
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], input[i], 0.001f);
}
END_TEST

TEST(pow_edge_cases)
constexpr int		   N	= 5;
std::vector<float>	   base = {2.0f, 4.0f, 8.0f, 1.0f, 0.0f};
std::vector<float>	   exp	= {3.0f, 0.5f, -1.0f, 100.0f, 5.0f};
Runtime::Buffer<float> bufBase(base);
Runtime::Buffer<float> bufExp(exp);
Runtime::Buffer<float> bufOut(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto b   = bufBase.Bind();
		  auto e   = bufExp.Bind();
		  auto out = bufOut.Bind();
		  out[id]  = Pow(b[id], e[id]);
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], std::pow(base[i], exp[i]), 0.0001f);
}
END_TEST

// =============================================================================
// Common Functions
// =============================================================================

TEST(clamp_bounds)
constexpr int		   N	 = 5;
std::vector<float>	   input = {-1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
Runtime::Buffer<float> bufIn(input);
Runtime::Buffer<float> bufOut(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto in  = bufIn.Bind();
		  auto out = bufOut.Bind();
		  out[id]  = Clamp(in[id], 0.0f, 1.0f);
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
ASSERT_NEAR(output[0], 0.0f, 0.0001f);
ASSERT_NEAR(output[1], 0.0f, 0.0001f);
ASSERT_NEAR(output[2], 0.5f, 0.0001f);
ASSERT_NEAR(output[3], 1.0f, 0.0001f);
ASSERT_NEAR(output[4], 1.0f, 0.0001f);
END_TEST

TEST(mod_behavior)
constexpr int		   N	 = 5;
std::vector<float>	   input = {5.0f, -5.0f, 3.5f, 7.0f, 0.0f};
Runtime::Buffer<float> bufIn(input);
Runtime::Buffer<float> bufOut(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto in  = bufIn.Bind();
		  auto out = bufOut.Bind();
		  out[id]  = Mod(in[id], 2.0f);
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	// GLSL mod(x,y) = x - y * floor(x/y)
	float expected = input[i] - 2.0f * std::floor(input[i] / 2.0f);
	ASSERT_NEAR(output[i], expected, 0.0001f);
}
END_TEST

TEST(min_max_vector)
constexpr int		   N = 4;
std::vector<float>	   a = {1.0f, 5.0f, 3.0f, -1.0f};
std::vector<float>	   b = {2.0f, 2.0f, 10.0f, 0.0f};
Runtime::Buffer<float> bufA(a);
Runtime::Buffer<float> bufB(b);
Runtime::Buffer<float> bufMin(N);
Runtime::Buffer<float> bufMax(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto av = bufA.Bind();
		  auto bv = bufB.Bind();
		  auto mn = bufMin.Bind();
		  auto mx = bufMax.Bind();
		  mn[id]  = Min(av[id], bv[id]);
		  mx[id]  = Max(av[id], bv[id]);
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> outMin(N);
std::vector<float> outMax(N);
bufMin.Download(outMin.data(), N);
bufMax.Download(outMax.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(outMin[i], std::min(a[i], b[i]), 0.0001f);
	ASSERT_NEAR(outMax[i], std::max(a[i], b[i]), 0.0001f);
}
END_TEST

TEST(mix_precision)
constexpr int		   N = 3;
std::vector<float>	   a = {0.0f, 10.0f, 5.0f};
std::vector<float>	   b = {10.0f, 0.0f, 15.0f};
Runtime::Buffer<float> bufA(a);
Runtime::Buffer<float> bufB(b);
Runtime::Buffer<float> bufOut(N);

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto av  = bufA.Bind();
		  auto bv  = bufB.Bind();
		  auto out = bufOut.Bind();
		  out[id]  = Mix(av[id], bv[id], 0.5f);
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], a[i] * 0.5f + b[i] * 0.5f, 0.0001f);
}
END_TEST

// =============================================================================
// Geometric Functions
// =============================================================================

TEST(length_normalize)
constexpr int			N	  = 3;
std::vector<Math::Vec3> input = {
	Math::Vec3{3.0f, 4.0f, 0.0f},
	Math::Vec3{1.0f, 1.0f, 1.0f},
	Math::Vec3{0.0f, 0.0f, 5.0f},
};
Runtime::Buffer<Math::Vec3> bufIn(input);
Runtime::Buffer<float>		bufLen(N);
Runtime::Buffer<Math::Vec3> bufNorm(N);

Kernel1D					kernel(
					   [&, N](Var<int> &id) {
		   auto in	 = bufIn.Bind();
		   auto len	 = bufLen.Bind();
		   auto norm = bufNorm.Bind();
		   len[id]	 = Length(in[id]);
		   norm[id]	 = Normalize(in[id]);
					   },
					   256);

kernel.Dispatch(1, true);

std::vector<float>		outLen(N);
std::vector<Math::Vec3> outNorm(N);
bufLen.Download(outLen.data(), N);
bufNorm.Download(outNorm.data(), N);

ASSERT_NEAR(outLen[0], 5.0f, 0.0001f);
ASSERT_NEAR(outLen[1], std::sqrt(3.0f), 0.0001f);
ASSERT_NEAR(outLen[2], 5.0f, 0.0001f);

for (int i = 0; i < N; ++i) {
	float nLen = std::sqrt(outNorm[i].x * outNorm[i].x + outNorm[i].y * outNorm[i].y + outNorm[i].z * outNorm[i].z);
	ASSERT_NEAR(nLen, 1.0f, 0.0001f);
}
END_TEST

TEST(dot_cross)
constexpr int			N = 2;
std::vector<Math::Vec3> a = {
	Math::Vec3{1.0f, 0.0f, 0.0f},
	Math::Vec3{1.0f, 2.0f, 3.0f},
};
std::vector<Math::Vec3> b = {
	Math::Vec3{0.0f, 1.0f, 0.0f},
	Math::Vec3{4.0f, 5.0f, 6.0f},
};
Runtime::Buffer<Math::Vec3> bufA(a);
Runtime::Buffer<Math::Vec3> bufB(b);
Runtime::Buffer<float>		bufDot(N);
Runtime::Buffer<Math::Vec3> bufCross(N);

Kernel1D					kernel(
					   [&, N](Var<int> &id) {
		   auto av = bufA.Bind();
		   auto bv = bufB.Bind();
		   auto d  = bufDot.Bind();
		   auto c  = bufCross.Bind();
		   d[id]   = Dot(av[id], bv[id]);
		   c[id]   = Cross(av[id], bv[id]);
					   },
					   256);

kernel.Dispatch(1, true);

std::vector<float>		outDot(N);
std::vector<Math::Vec3> outCross(N);
bufDot.Download(outDot.data(), N);
bufCross.Download(outCross.data(), N);

ASSERT_NEAR(outDot[0], 0.0f, 0.0001f);
ASSERT_NEAR(outDot[1], 1.0f * 4.0f + 2.0f * 5.0f + 3.0f * 6.0f, 0.0001f);

ASSERT_NEAR(outCross[0].x, 0.0f, 0.0001f);
ASSERT_NEAR(outCross[0].y, 0.0f, 0.0001f);
ASSERT_NEAR(outCross[0].z, 1.0f, 0.0001f);
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Math Precision Tests           " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_sin_precision();
		test_cos_precision();
		test_atan2_quadrants();
		test_sqrt_precision();
		test_exp_log_roundtrip();
		test_pow_edge_cases();
		test_clamp_bounds();
		test_mod_behavior();
		test_min_max_vector();
		test_mix_precision();
		test_length_normalize();
		test_dot_cross();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All math precision tests passed!      " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
