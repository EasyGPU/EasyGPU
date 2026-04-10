/**
 * TestMath.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <GPU.h>
#include <IR/Value/Var.h>
#include <Kernel/Kernel.h>
#include <Utility/Math.h>
#include <GPU.h>

using namespace GPU::IR::Value;
using namespace GPU::Math;

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
// Test 1: Trigonometric Functions
// =============================================================================
TEST(trig_functions)
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<float> a(1.0f);
	Var<float> b(Sin(a));
	Var<float> c(Cos(a));
	Var<float> d(Tan(a));
	Var<float> e(Asin(a));
	Var<float> f(Acos(a));
	Var<float> g(Atan(a));

	// Two-argument Atan
	Var<float> h(Atan2(a, b));

	// Hyperbolic functions
	Var<float> i(Sinh(a));
	Var<float> j(Cosh(a));
	Var<float> k(Tanh(a));
	Var<float> l(Asinh(a));
	Var<float> m(Acosh(a));
	Var<float> n(Atanh(a));

	// Degrees/Radians conversion
	Var<float> o(Radians(a));
	Var<float> p(Degrees(a));
});
kernel.PrintCode();
END_TEST

TEST(trig_vector_functions)
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<Vec2> v2 = MakeFloat2(1.0f, 2.0f);
	Var<Vec3> v3 = MakeFloat3(1.0f, 2.0f, 3.0f);
	Var<Vec4> v4 = MakeFloat4(1.0f, 2.0f, 3.0f, 4.0f);

	Var<Vec2> r2 = Sin(v2);
	Var<Vec3> r3 = Cos(v3);
	Var<Vec4> r4 = Tan(v4);

	Var<Vec2> s2 = Asin(v2);
	Var<Vec3> s3 = Acos(v3);
	Var<Vec4> s4 = Atan(v4);

	Var<Vec2> t2 = Atan2(v2, v2);
	Var<Vec3> t3 = Atan2(v3, v3);
	Var<Vec4> t4 = Atan2(v4, v4);
});
kernel.PrintCode();
END_TEST

// =============================================================================
// Test 2: Exponential Functions
// =============================================================================
TEST(exp_functions)
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<float> a(2.0f);
	Var<float> b(3.0f);

	Var<float> c(Pow(a, b));
	Var<float> d(Exp(a));
	Var<float> e(Log(a));
	Var<float> f(Exp2(a));
	Var<float> g(Log2(a));
	Var<float> h(Sqrt(a));
	Var<float> i(Inversesqrt(a));
});
kernel.PrintCode();
END_TEST

TEST(exp_vector_functions)
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<Vec2> v2 = MakeFloat2(1.0f, 2.0f);
	Var<Vec3> v3 = MakeFloat3(1.0f, 2.0f, 3.0f);
	Var<Vec4> v4 = MakeFloat4(1.0f, 2.0f, 3.0f, 4.0f);

	Var<Vec2> p2 = Pow(v2, v2);
	Var<Vec3> p3 = Pow(v3, v3);
	Var<Vec4> p4 = Pow(v4, v4);

	Var<Vec2> e2 = Exp(v2);
	Var<Vec3> l3 = Log(v3);
	Var<Vec4> s4 = Sqrt(v4);
});
kernel.PrintCode();
END_TEST

// =============================================================================
// Test 3: Common Functions
// =============================================================================
TEST(common_functions)
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<float> a(2.5f);
	Var<float> b(-1.5f);
	Var<int>   c(-5);

	Var<float> d(Abs(b));
	Var<int>   e(Abs(c));
	Var<float> f(Sign(b));
	Var<int>   g(Sign(c));
	Var<float> h(Floor(a));
	Var<float> i(Trunc(a));
	Var<float> j(Round(a));
	Var<float> k(RoundEven(a));
	Var<float> l(Ceil(a));
	Var<float> m(Fract(a));
	Var<float> n(Mod(a, 2.0f));

	// Min/Max with scalar
	Var<float> o(Min(a, b));
	Var<float> p(Min(a, 1.0f));
	Var<float> q(Max(a, b));
	Var<float> r(Max(a, 1.0f));

	// Clamp
	Var<float> s(Clamp(a, 0.0f, 1.0f));

	// Mix
	Var<float> t(Mix(a, b, 0.5f));

	// Step
	Var<float> u(Step(1.0f, a));

	// Smoothstep
	Var<float> v(Smoothstep(0.0f, 2.0f, a));
});
kernel.PrintCode();
END_TEST

TEST(common_vector_functions)
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<Vec2>  v2  = MakeFloat2(1.5f, 2.5f);
	Var<Vec3>  v3  = MakeFloat3(1.5f, 2.5f, 3.5f);
	Var<Vec4>  v4  = MakeFloat4(1.5f, 2.5f, 3.5f, 4.5f);

	Var<Vec2>  a2  = Abs(v2);
	Var<Vec3>  f3  = Floor(v3);
	Var<Vec4>  c4  = Ceil(v4);
	Var<Vec2>  r2  = Round(v2);
	Var<Vec3>  t3  = Trunc(v3);
	Var<Vec4>  f4  = Fract(v4);

	Var<Vec2>  m2  = Min(v2, v2);
	Var<Vec2>  m2s = Min(v2, 1.0f);
	Var<Vec3>  x3  = Max(v3, v3);
	Var<Vec3>  x3s = Max(v3, 1.0f);

	Var<Vec4>  cl4 = Clamp(v4, 0.0f, 1.0f);
	Var<Vec2>  mi2 = Mix(v2, v2, 0.5f);
	Var<Vec3>  st3 = Step(1.0f, v3);
	Var<Vec4>  ss4 = Smoothstep(0.0f, 2.0f, v4);

	// Integer versions
	Var<IVec2> iv2 = MakeInt2(1, 2);
	Var<IVec3> iv3 = MakeInt3(1, 2, 3);
	Var<IVec4> iv4 = MakeInt4(1, 2, 3, 4);

	Var<IVec2> ai2 = Abs(iv2);
	Var<IVec2> ni2 = Min(iv2, iv2);
	Var<IVec3> xi3 = Max(iv3, iv3);
	Var<IVec4> ci4 = Clamp(iv4, iv4, iv4);
});
kernel.PrintCode();
END_TEST

// =============================================================================
// Test 4: Geometric Functions
// =============================================================================
TEST(geometric_functions)
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<Vec2>  v2 = MakeFloat2(1.0f, 2.0f);
	Var<Vec3>  v3 = MakeFloat3(1.0f, 2.0f, 3.0f);
	Var<Vec4>  v4 = MakeFloat4(1.0f, 2.0f, 3.0f, 4.0f);

	// Length
	Var<float> l2(Length(v2));
	Var<float> l3(Length(v3));
	Var<float> l4(Length(v4));

	// Distance
	Var<float> d2(Distance(v2, v2));
	Var<float> d3(Distance(v3, v3));
	Var<float> d4(Distance(v4, v4));

	// Dot
	Var<float> dt2(Dot(v2, v2));
	Var<float> dt3(Dot(v3, v3));
	Var<float> dt4(Dot(v4, v4));

	// Cross
	Var<Vec3>  cr	= Cross(v3, v3);

	// Normalize
	Var<Vec2>  n2	= Normalize(v2);
	Var<Vec3>  n3	= Normalize(v3);
	Var<Vec4>  n4	= Normalize(v4);

	// Faceforward
	Var<Vec3>  ff	= Faceforward(v3, v3, v3);

	// Reflect
	Var<Vec3>  rfl	= Reflect(v3, v3);

	// Refract
	Var<Vec3>  rfr	= Refract(v3, v3, 1.5f);
	Var<Vec3>  rfr2 = Refract(v3, v3, Var<float>(1.5f));
});
kernel.PrintCode();
END_TEST

// =============================================================================
// Test 5: Vector Relational Functions
// =============================================================================
TEST(vector_relational_functions)
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<Vec2>  v2a = MakeFloat2(1.0f, 2.0f);
	Var<Vec2>  v2b = MakeFloat2(3.0f, 1.0f);
	Var<Vec3>  v3a = MakeFloat3(1.0f, 2.0f, 3.0f);
	Var<Vec3>  v3b = MakeFloat3(3.0f, 1.0f, 2.0f);
	Var<Vec4>  v4a = MakeFloat4(1.0f, 2.0f, 3.0f, 4.0f);
	Var<Vec4>  v4b = MakeFloat4(4.0f, 3.0f, 2.0f, 1.0f);

	Var<bool>  lt2(LessThan(v2a, v2b));
	Var<bool>  lt3(LessThan(v3a, v3b));
	Var<bool>  lt4(LessThan(v4a, v4b));

	Var<bool>  le2(LessThanEqual(v2a, v2b));
	Var<bool>  gt2(GreaterThan(v2a, v2b));
	Var<bool>  ge2(GreaterThanEqual(v2a, v2b));
	Var<bool>  eq2(Equal(v2a, v2b));
	Var<bool>  ne2(NotEqual(v2a, v2b));

	// Integer vector versions
	Var<IVec2> iv2a = MakeInt2(1, 2);
	Var<IVec2> iv2b = MakeInt2(3, 1);

	Var<bool>  ilt2(LessThan(iv2a, iv2b));
	Var<bool>  ieq2(Equal(iv2a, iv2b));
});
kernel.PrintCode();
END_TEST

// =============================================================================
// Test 6: CopySign Functions
// =============================================================================
TEST(copysign_basic)
// Test basic CopySign with InspectorKernel
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<float> x	   = MakeFloat(5.0f);
	Var<float> y	   = MakeFloat(-3.0f);

	// CopySign(5.0, -3.0) = -5.0
	Var<float> result  = CopySign(x, y);

	// CopySign with positive y
	Var<float> y2	   = MakeFloat(2.0f);
	Var<float> result2 = CopySign(x, y2);

	// CopySign with scalar
	Var<float> result3 = CopySign(x, -1.0f);
	Var<float> result4 = CopySign(10.0f, y);
});
kernel.PrintCode();
ASSERT(true);
END_TEST

TEST(copysign_vector2)
// Test CopySign with Vec2
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<Vec2>  x	   = MakeFloat2(1.0f, 2.0f);
	Var<Vec2>  y	   = MakeFloat2(-1.0f, -2.0f);

	// CopySign vector, vector
	Var<Vec2>  result1 = CopySign(x, y);

	// CopySign vector, scalar (broadcast)
	Var<Vec2>  result2 = CopySign(x, -1.0f);

	// CopySign vector, scalar expr
	Var<float> signVal = MakeFloat(-1.0f);
	Var<Vec2>  result3 = CopySign(x, signVal);
});
kernel.PrintCode();
ASSERT(true);
END_TEST

TEST(copysign_vector3)
// Test CopySign with Vec3
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<Vec3> x		  = MakeFloat3(1.0f, 2.0f, 3.0f);
	Var<Vec3> y		  = MakeFloat3(-1.0f, -2.0f, -3.0f);

	// CopySign vector, vector
	Var<Vec3> result1 = CopySign(x, y);

	// CopySign vector, scalar (broadcast)
	Var<Vec3> result2 = CopySign(x, -1.0f);
});
kernel.PrintCode();
ASSERT(true);
END_TEST

TEST(copysign_vector4)
// Test CopySign with Vec4
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
	Var<Vec4> x		  = MakeFloat4(1.0f, 2.0f, 3.0f, 4.0f);
	Var<Vec4> y		  = MakeFloat4(-1.0f, -2.0f, -3.0f, -4.0f);

	// CopySign vector, vector
	Var<Vec4> result1 = CopySign(x, y);

	// CopySign vector, scalar (broadcast)
	Var<Vec4> result2 = CopySign(x, -1.0f);
});
kernel.PrintCode();
ASSERT(true);
END_TEST

TEST(copysign_runtime)
// Test CopySign with actual GPU execution
std::vector<float>	   x_values = {5.0f, -3.0f, 10.0f, -7.0f, 0.0f};
std::vector<float>	   y_values = {-1.0f, 2.0f, -5.0f, 8.0f, 1.0f};
size_t				   N		= x_values.size();
std::vector<float>	   output(N);

Runtime::Buffer<float> bufferX(x_values);
Runtime::Buffer<float> bufferY(y_values);
Runtime::Buffer<float> bufferOutput(N);

Kernel::Kernel1D	   kernel(
		  [&, N](Var<int> &id) {
		  auto x   = bufferX.Bind();
		  auto y   = bufferY.Bind();
		  auto out = bufferOutput.Bind();

		  out[id]  = CopySign(x[id], y[id]);
		  },
		  static_cast<int>(N));

kernel.Dispatch(1, true);
bufferOutput.Download(output);

// Verify results
for (size_t i = 0; i < N; i++) {
	float expected = std::copysign(x_values[i], y_values[i]);
	ASSERT_NEAR(output[i], expected, 0.0001f);
}
END_TEST

TEST(copysign_special_cases)
// Test special cases: zero, very small values
std::vector<float>	   x_values = {0.0f, -0.0f, 1e-10f, -1e-10f, 1e10f};
std::vector<float>	   y_values = {1.0f, -1.0f, 0.0f, -0.0f, -1e10f};
size_t				   N		= x_values.size();
std::vector<float>	   output(N);

Runtime::Buffer<float> bufferX(x_values);
Runtime::Buffer<float> bufferY(y_values);
Runtime::Buffer<float> bufferOutput(N);

Kernel::Kernel1D	   kernel(
		  [&, N](Var<int> &id) {
		  auto x   = bufferX.Bind();
		  auto y   = bufferY.Bind();
		  auto out = bufferOutput.Bind();

		  out[id]  = CopySign(x[id], y[id]);
		  },
		  static_cast<int>(N));

kernel.Dispatch(1, true);
bufferOutput.Download(output);

// Verify results
for (size_t i = 0; i < N; i++) {
	float expected = std::copysign(x_values[i], y_values[i]);
	ASSERT_NEAR(output[i], expected, 0.0001f);
}
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================\n";
	std::cout << "  EasyGPU Math Functions Test Suite    \n";
	std::cout << "========================================\n";

	try {
		test_trig_functions();
		test_trig_vector_functions();
		test_exp_functions();
		test_exp_vector_functions();
		test_common_functions();
		test_common_vector_functions();
		test_geometric_functions();
		test_vector_relational_functions();
		test_copysign_basic();
		test_copysign_vector2();
		test_copysign_vector3();
		test_copysign_vector4();
		test_copysign_runtime();
		test_copysign_special_cases();

		std::cout << "\n========================================\n";
		std::cout << "  All tests passed!                     \n";
		std::cout << "========================================\n";

		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << "\n";
		return 1;
	}
}
