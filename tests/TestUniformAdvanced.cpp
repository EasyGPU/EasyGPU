/**
 * @file TestUniformAdvanced.cpp
 * @brief Advanced uniform tests: dynamic updates, multi-kernel uniform changes,.
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

#define ASSERT_EQ(a, b)                                                                                                \
	if ((a) != (b)) {                                                                                                  \
		throw std::runtime_error("Assertion failed: " #a " != " #b);                                                   \
	}

// =============================================================================
// Uniform Dynamic Update
// =============================================================================

TEST(uniform_scalar_dynamic_update)
constexpr int		   N = 64;
std::vector<float>	   data(N, 0.0f);
Runtime::Buffer<float> buf(data);
Uniform<float>		   u(1.0f);

Kernel1D			   kernel(
	[&, N](Var<int> &id) {
		auto	   b = buf.Bind();
		Var<float> v = u.Load();
		b[id]		 = b[id] + v;
	},
	256);

kernel.Dispatch(1, true);
u = 2.0f;
kernel.Dispatch(1, true);
u = 3.0f;
kernel.Dispatch(1, true);

std::vector<float> output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], 6.0f, 0.0001f); // 1 + 2 + 3
}
END_TEST

TEST(uniform_vec4_dynamic_update)
constexpr int				N = 64;
std::vector<Math::Vec4>		data(N, Math::Vec4{0.0f, 0.0f, 0.0f, 0.0f});
Runtime::Buffer<Math::Vec4> buf(data);
Uniform<Math::Vec4>			u(Math::Vec4{1.0f, 2.0f, 3.0f, 4.0f});

Kernel1D					kernel(
	[&, N](Var<int> &id) {
		auto	  b = buf.Bind();
		Var<Vec4> v = u.Load();
		b[id]		= b[id] + v;
	},
	256);

kernel.Dispatch(1, true);
u = Math::Vec4{10.0f, 20.0f, 30.0f, 40.0f};
kernel.Dispatch(1, true);

std::vector<Math::Vec4> output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i].x, 11.0f, 0.0001f);
	ASSERT_NEAR(output[i].y, 22.0f, 0.0001f);
	ASSERT_NEAR(output[i].z, 33.0f, 0.0001f);
	ASSERT_NEAR(output[i].w, 44.0f, 0.0001f);
}
END_TEST

TEST(uniform_int_dynamic_update)
constexpr int		 N = 64;
std::vector<int>	 data(N, 0);
Runtime::Buffer<int> buf(data);
Uniform<int>		 u(5);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto	 b = buf.Bind();
		Var<int> v = u.Load();
		b[id]	   = b[id] + v;
	},
	256);

kernel.Dispatch(1, true);
u = 10;
kernel.Dispatch(1, true);

std::vector<int> output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_EQ(output[i], 15);
}
END_TEST

// =============================================================================
// Multiple Uniforms in Same Kernel
// =============================================================================

TEST(multiple_uniforms_same_kernel)
constexpr int		   N = 64;
std::vector<float>	   data(N, 1.0f);
Runtime::Buffer<float> buf(data);
Uniform<float>		   scale(2.0f);
Uniform<float>		   offset(3.0f);

Kernel1D			   kernel(
	[&, N](Var<int> &id) {
		auto	   b = buf.Bind();
		Var<float> s = scale.Load();
		Var<float> o = offset.Load();
		b[id]		 = b[id] * s + o;
	},
	256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], 1.0f * 2.0f + 3.0f, 0.0001f);
}
END_TEST

// =============================================================================
// Uniform Changed Between Different Kernel Objects
// =============================================================================

TEST(uniform_shared_across_kernels)
constexpr int		   N = 64;
std::vector<float>	   data(N, 0.0f);
Runtime::Buffer<float> buf(data);
Uniform<float>		   factor(2.0f);

Kernel1D			   kernelA(
	[&, N](Var<int> &id) {
		auto	   b = buf.Bind();
		Var<float> f = factor.Load();
		b[id]		 = b[id] + f;
	},
	256);

Kernel1D kernelB(
	[&, N](Var<int> &id) {
		auto	   b = buf.Bind();
		Var<float> f = factor.Load();
		b[id]		 = b[id] * f;
	},
	256);

kernelA.Dispatch(1, true); // +2 -> 2
factor = 3.0f;
kernelB.Dispatch(1, true); // *3 -> 6
factor = 4.0f;
kernelA.Dispatch(1, true); // +4 -> 10

std::vector<float> output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], 10.0f, 0.0001f);
}
END_TEST

// =============================================================================
// Uniform with Bool Type
// =============================================================================

TEST(uniform_bool)
constexpr int		   N = 64;
std::vector<float>	   data(N, 1.0f);
Runtime::Buffer<float> buf(data);
Uniform<bool>		   flag(true);

Kernel1D			   kernel(
	[&, N](Var<int> &id) {
		auto	  b = buf.Bind();
		Var<bool> f = flag.Load();
		If(f, [&]() { b[id] = b[id] * MakeFloat(2.0f); }).Else([&]() { b[id] = b[id] * MakeFloat(3.0f); });
	},
	256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
buf.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i], 2.0f, 0.0001f);
}
END_TEST

// =============================================================================
// Uniform Vec2 and Vec3
// =============================================================================

TEST(uniform_vec2_vec3)
constexpr int				N = 32;
std::vector<Math::Vec2>		data2(N, Math::Vec2{0.0f, 0.0f});
std::vector<Math::Vec3>		data3(N, Math::Vec3{0.0f, 0.0f, 0.0f});
Runtime::Buffer<Math::Vec2> buf2(data2);
Runtime::Buffer<Math::Vec3> buf3(data3);
Uniform<Math::Vec2>			u2(Math::Vec2{1.0f, 2.0f});
Uniform<Math::Vec3>			u3(Math::Vec3{3.0f, 4.0f, 5.0f});

Kernel1D					kernel(
	[&, N](Var<int> &id) {
		auto b2 = buf2.Bind();
		auto b3 = buf3.Bind();
		b2[id]	= b2[id] + u2.Load();
		b3[id]	= b3[id] + u3.Load();
	},
	256);

kernel.Dispatch(1, true);

std::vector<Math::Vec2> out2(N);
std::vector<Math::Vec3> out3(N);
buf2.Download(out2.data(), N);
buf3.Download(out3.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(out2[i].x, 1.0f, 0.0001f);
	ASSERT_NEAR(out2[i].y, 2.0f, 0.0001f);
	ASSERT_NEAR(out3[i].x, 3.0f, 0.0001f);
	ASSERT_NEAR(out3[i].y, 4.0f, 0.0001f);
	ASSERT_NEAR(out3[i].z, 5.0f, 0.0001f);
}
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Uniform Advanced Tests        " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_uniform_scalar_dynamic_update();
		test_uniform_vec4_dynamic_update();
		test_uniform_int_dynamic_update();
		test_multiple_uniforms_same_kernel();
		test_uniform_shared_across_kernels();
		test_uniform_bool();
		test_uniform_vec2_vec3();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All uniform advanced tests passed!    " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
