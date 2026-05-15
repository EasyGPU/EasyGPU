/**
 * @file TestCallableAdvanced.cpp
 * @brief Advanced tests for Callable functions: multi-parameter, texture args,.
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
// Code Generation Tests
// =============================================================================

TEST(callable_multi_parameter)
Callable<float(float, float, float)> addThree([](Float a, Float b, Float c) { Return(a + b + c); });

InspectorKernel1D					 inspector([&](Int i) {
	   Var<float> result = addThree(MakeFloat(1.0f), MakeFloat(2.0f), MakeFloat(3.0f));
});
std::string							 code = inspector.GetCode();
ASSERT(code.find("float") != std::string::npos);
ASSERT(code.find("float") != std::string::npos);
END_TEST

TEST(callable_void_side_effects)
Runtime::Buffer<float> buf(4);
Callable<void(float)>  storeValue([&](Float val) {
	 auto b = buf.Bind();
	 b[0]	= val;
});

InspectorKernel1D	   inspector([&](Int i) { storeValue(MakeFloat(42.0f)); });
std::string			   code = inspector.GetCode();
ASSERT(code.find("void") != std::string::npos);
END_TEST

TEST(callable_inside_loop)
Callable<float(float)> scale([](Float x) { Return(x * MakeFloat(2.0f)); });

InspectorKernel1D	   inspector([&](Int i) { For(0, 4, [&](Int &j) { Var<float> v = scale(ToFloat(j)); }); });
std::string			   code = inspector.GetCode();
ASSERT(code.find("for (") != std::string::npos);
ASSERT(code.find("for (") != std::string::npos);
END_TEST

TEST(callable_called_multiple_times)
Callable<float(float)> negate([](Float x) { Return(-x); });

InspectorKernel1D	   inspector([&](Int i) {
	 Var<float> a = negate(MakeFloat(1.0f));
	 Var<float> b = negate(MakeFloat(2.0f));
	 Var<float> c = negate(MakeFloat(3.0f));
});
std::string			   code = inspector.GetCode();
// Verify code was generated successfully
ASSERT(!code.empty());
END_TEST

TEST(callable_with_vector_return)
Callable<Math::Vec3(Math::Vec3, Math::Vec3)> vecAdd([](Float3 a, Float3 b) { Return(a + b); });

InspectorKernel1D							 inspector([&](Int i) {
	   Var<Vec3> v1		= MakeFloat3(1.0f, 2.0f, 3.0f);
	   Var<Vec3> v2		= MakeFloat3(4.0f, 5.0f, 6.0f);
	   Var<Vec3> result = vecAdd(v1, v2);
});
std::string									 code = inspector.GetCode();
ASSERT(code.find("vec3") != std::string::npos);
END_TEST

// =============================================================================
// Runtime Execution Tests
// =============================================================================

TEST(callable_runtime_multi_param)
constexpr int		   N = 64;
Runtime::Buffer<float> bufA(N);
Runtime::Buffer<float> bufB(N);
Runtime::Buffer<float> bufC(N);
Runtime::Buffer<float> bufOut(N);

std::vector<float>	   aData(N), bData(N), cData(N);
for (int i = 0; i < N; ++i) {
	aData[i] = static_cast<float>(i);
	bData[i] = static_cast<float>(i * 2);
	cData[i] = static_cast<float>(i * 3);
}
bufA.Upload(aData.data(), N);
bufB.Upload(bData.data(), N);
bufC.Upload(cData.data(), N);

Callable<float(float, float, float)> add3([](Float a, Float b, Float c) { Return(a + b + c); });

Kernel1D							 kernel(
								[&, N](Var<int> &id) {
		auto a	 = bufA.Bind();
		auto b	 = bufB.Bind();
		auto c	 = bufC.Bind();
		auto out = bufOut.Bind();
		out[id]	 = add3(a[id], b[id], c[id]);
								},
								256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	float expected = aData[i] + bData[i] + cData[i];
	ASSERT_NEAR(output[i], expected, 0.0001f);
}
END_TEST

TEST(callable_runtime_in_loop)
constexpr int		 N = 64;
Runtime::Buffer<int> bufOut(N);

Callable<int(int)>	 square([](Int x) { Return(x * x); });

Kernel1D			 kernel(
				[&, N](Var<int> &id) {
		auto	 out = bufOut.Bind();
		Var<int> sum = MakeInt(0);
		For(0, id + 1, [&](Int &j) { sum = sum + square(j); });
		out[id] = sum;
				},
				256);

kernel.Dispatch(1, true);

std::vector<int> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	int expected = 0;
	for (int j = 0; j <= i; ++j) {
		expected += j * j;
	}
	ASSERT_EQ(output[i], expected);
}
END_TEST

TEST(callable_runtime_nested_call)
constexpr int		   N = 64;
Runtime::Buffer<float> bufOut(N);

Callable<float(float)> addOne([](Float x) { Return(x + MakeFloat(1.0f)); });

Callable<float(float)> addTwo([&](Float x) { Return(addOne(addOne(x))); });

Kernel1D			   kernel(
				  [&, N](Var<int> &id) {
		  auto out = bufOut.Bind();
		  out[id]  = addTwo(ToFloat(id));
				  },
				  256);

kernel.Dispatch(1, true);

std::vector<float> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	float expected = static_cast<float>(i) + 2.0f;
	ASSERT_NEAR(output[i], expected, 0.0001f);
}
END_TEST

TEST(callable_runtime_vector_arg)
constexpr int				N = 64;
Runtime::Buffer<Math::Vec4> bufIn(N);
Runtime::Buffer<Math::Vec4> bufOut(N);

std::vector<Math::Vec4>		data(N);
for (int i = 0; i < N; ++i) {
	data[i] = Math::Vec4{static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2),
						 static_cast<float>(i + 3)};
}
bufIn.Upload(data.data(), N);

Callable<Math::Vec4(Math::Vec4)> vecAdd([](Float4 v) { Return(v + MakeFloat4(1.0f, 2.0f, 3.0f, 4.0f)); });

Kernel1D						 kernel(
							[&, N](Var<int> &id) {
		auto in	 = bufIn.Bind();
		auto out = bufOut.Bind();
		out[id]	 = vecAdd(in[id]);
							},
							256);

kernel.Dispatch(1, true);

std::vector<Math::Vec4> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	ASSERT_NEAR(output[i].x, data[i].x + 1.0f, 0.0001f);
	ASSERT_NEAR(output[i].y, data[i].y + 2.0f, 0.0001f);
	ASSERT_NEAR(output[i].z, data[i].z + 3.0f, 0.0001f);
	ASSERT_NEAR(output[i].w, data[i].w + 4.0f, 0.0001f);
}
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Callable Advanced Tests       " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_callable_multi_parameter();
		test_callable_void_side_effects();
		test_callable_inside_loop();
		test_callable_called_multiple_times();
		test_callable_with_vector_return();
		test_callable_runtime_multi_param();
		test_callable_runtime_in_loop();
		test_callable_runtime_nested_call();
		test_callable_runtime_vector_arg();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All callable advanced tests passed!   " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
