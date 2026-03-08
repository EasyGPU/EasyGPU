/**
 * TestTernary.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/8/2026
 */
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <GPU.h>

// =============================================================================
// Test Macros
// =============================================================================
static int test_count = 0;
static int pass_count = 0;

#define TEST(name)                                                                                                     \
	void test_##name() {                                                                                               \
		std::cout << "\n[TEST] " #name " ... ";                                                                        \
		test_count++;                                                                                                  \
		try {

#define END_TEST                                                                                                       \
	pass_count++;                                                                                                      \
	std::cout << "PASSED\n";                                                                                           \
	}                                                                                                                  \
	catch (const std::exception &e) {                                                                                  \
		std::cout << "FAILED: " << e.what() << "\n";                                                                   \
	}                                                                                                                  \
	}

#define ASSERT(cond)                                                                                                   \
	if (!(cond)) {                                                                                                     \
		throw std::runtime_error("Assertion failed: " #cond);                                                          \
	}

#define ASSERT_EQ(a, b)                                                                                                \
	if ((a) != (b)) {                                                                                                  \
		throw std::runtime_error("Assertion failed: " #a " == " #b);                                                    \
	}

#define ASSERT_NEAR(a, b, eps)                                                                                         \
	if (std::abs((a) - (b)) > (eps)) {                                                                                 \
		throw std::runtime_error("Assertion failed: |" #a " - " #b "| > " #eps);                                       \
	}

// =============================================================================
// Test Suite: Basic Ternary Operations
// =============================================================================

TEST(basic_scalar_ternary)
	// Test basic scalar ternary with InspectorKernel
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		Var<float> a = MakeFloat(5.0f);
		Var<float> b = MakeFloat(-3.0f);

		// Simple ternary: select a if a > 0 else b
		Expr<bool> cond = a > 0.0f;
		Expr<float> result = Select(cond, Expr<float>(a), Expr<float>(b));
		Var<float> selected = result;

		// Ternary with literal values (both Expr)
		Var<float> absB = Select(b > 0.0f, Expr<float>(b), Expr<float>(-b));

		// Nested ternary
		Var<float> signA = Select(a > 0.0f, Expr<float>(MakeFloat(1.0f)),
									Select(a < 0.0f, Expr<float>(MakeFloat(-1.0f)), Expr<float>(MakeFloat(0.0f))));
	});
	kernel.PrintCode();
	ASSERT(true);
END_TEST

TEST(ternary_with_var_arguments)
	// Test ternary with Var arguments (implicit conversion to Expr)
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		Var<float> a = MakeFloat(10.0f);
		Var<float> b = MakeFloat(20.0f);
		Var<bool> cond = a < b;

		// Var, Var
		Var<float> r1 = Select(Expr<bool>(cond), a, b);

		// Expr, Var
		Var<float> r2 = Select(a > 0.0f, a + 1.0f, b);

		// Var, Expr
		Var<float> r3 = Select(b > 0.0f, a, b * 2.0f);

		// Expr, Expr
		Var<float> r4 = Select(a > b, a + b, a - b);
	});
	kernel.PrintCode();
	ASSERT(true);
END_TEST

TEST(ternary_int_type)
	// Test ternary with integer type
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		Var<int> a = MakeInt(5);
		Var<int> b = MakeInt(10);

		Var<int> maxVal = Select(a > b, Expr<int>(a), Expr<int>(b));
		Var<int> minVal = Select(a < b, Expr<int>(a), Expr<int>(b));
		Var<int> absVal = Select(a < 0, Expr<int>(-a), Expr<int>(a));
	});
	kernel.PrintCode();
	ASSERT(true);
END_TEST

TEST(ternary_bool_type)
	// Test ternary with bool type
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		Var<bool> flag = MakeBool(true);
		Var<bool> result = Select(flag, MakeBool(false), MakeBool(true));
	});
	kernel.PrintCode();
	ASSERT(true);
END_TEST

TEST(ternary_vector_types)
	// Test ternary with vector types
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		Var<Vec2> v1 = MakeFloat2(1.0f, 2.0f);
		Var<Vec2> v2 = MakeFloat2(3.0f, 4.0f);
		Var<bool> useV1 = MakeBool(true);

		Var<Vec2> selected = Select(useV1, v1, v2);

		// Vec3
		Var<Vec3> v3 = MakeFloat3(1.0f, 2.0f, 3.0f);
		Var<Vec3> v4 = MakeFloat3(0.0f, 0.0f, 0.0f);
		Var<Vec3> result3 = Select(v3.x() > 0.0f, v3, v4);

		// Vec4
		Var<Vec4> v5 = MakeFloat4(1.0f, 2.0f, 3.0f, 4.0f);
		Var<Vec4> v6 = MakeFloat4(0.0f, 0.0f, 0.0f, 0.0f);
		Var<Vec4> result4 = Select(v5.w() > 0.0f, v5, v6);
	});
	kernel.PrintCode();
	ASSERT(true);
END_TEST

TEST(ternary_ivec_types)
	// Test ternary with integer vector types
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		Var<IVec2> iv1 = MakeInt2(1, 2);
		Var<IVec2> iv2 = MakeInt2(3, 4);
		Var<IVec2> result = Select(iv1.x() > 0, iv1, iv2);

		Var<IVec3> iv3 = MakeInt3(1, 2, 3);
		Var<IVec3> iv4 = MakeInt3(0, 0, 0);
		Var<IVec3> result3 = Select(iv3.x() > 0, iv3, iv4);

		Var<IVec4> iv5 = MakeInt4(1, 2, 3, 4);
		Var<IVec4> iv6 = MakeInt4(0, 0, 0, 0);
		Var<IVec4> result4 = Select(iv5.x() > 0, iv5, iv6);
	});
	kernel.PrintCode();
	ASSERT(true);
END_TEST

TEST(ternary_in_expressions)
	// Test ternary used within larger expressions
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		Var<float> a = MakeFloat(5.0f);
		Var<float> b = MakeFloat(3.0f);

		// Ternary in arithmetic expression
		Var<float> result1 = Select(a > b, Expr<float>(a), Expr<float>(b)) + 10.0f;
		Var<float> result2 = 2.0f * Select(a < 0.0f, Expr<float>(-a), Expr<float>(a));

		// Ternary as part of complex expression
		Var<float> result3 = Select(a > 0.0f, Expr<float>(a), Expr<float>(MakeFloat(0.0f))) +
							 Select(b > 0.0f, Expr<float>(b), Expr<float>(MakeFloat(0.0f)));

		// Chained operations with ternary
		Var<float> result4 = (Select(a > b, Expr<float>(a), Expr<float>(b)) + 1.0f) * 2.0f;
	});
	kernel.PrintCode();
	ASSERT(true);
END_TEST

TEST(ternary_with_buffer)
	// Test ternary with buffer operations
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		// This test just verifies compilation - actual buffers would be bound at runtime
		Var<float> inputVal = MakeFloat(10.0f);
		Var<float> threshold = MakeFloat(5.0f);

		// Clamp using ternary
		Var<float> clamped = Select(inputVal > threshold, Expr<float>(threshold), Expr<float>(inputVal));
		clamped = Select(clamped < 0.0f, MakeFloat(0.0f), clamped);
	});
	kernel.PrintCode();
	ASSERT(true);
END_TEST

TEST(deeply_nested_ternary)
	// Test deeply nested ternary expressions
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		Var<int> score = MakeInt(85);

		// Grade calculation using nested ternary
		Var<int> grade = Select(score >= 90, Expr<int>(MakeInt(4)),
								Select(score >= 80, Expr<int>(MakeInt(3)),
									   Select(score >= 70, Expr<int>(MakeInt(2)),
											  Select(score >= 60, Expr<int>(MakeInt(1)),
													 Expr<int>(MakeInt(0))))));
	});
	kernel.PrintCode();
	ASSERT(true);
END_TEST

TEST(ternary_runtime_execution)
	// Test ternary operator with actual GPU execution
	std::vector<float> input = {-5.0f, 3.0f, -1.0f, 10.0f, 0.0f, -8.0f, 7.0f, -2.0f};
	size_t N = input.size();
	std::vector<float> output(N);

	Runtime::Buffer<float> inputBuffer(input);
	Runtime::Buffer<float> outputBuffer(N);

	Kernel::Kernel1D kernel(
		[&, N](Var<int> &id) {
			auto in = inputBuffer.Bind();
			auto out = outputBuffer.Bind();

			// Compute absolute value using ternary
			Var<float> val = in[id];
			out[id] = Select(val < 0.0f, Expr<float>(-val), Expr<float>(val));
		},
		static_cast<int>(N));

	kernel.Dispatch(1, true);
	outputBuffer.Download(output);

	// Verify results
	for (size_t i = 0; i < N; i++) {
		float expected = std::abs(input[i]);
		ASSERT_NEAR(output[i], expected, 0.0001f);
	}
END_TEST

TEST(ternary_max_min_runtime)
	// Test ternary for max/min with GPU execution
	std::vector<float> a = {1.0f, 5.0f, 3.0f, 8.0f, 2.0f};
	std::vector<float> b = {4.0f, 2.0f, 6.0f, 3.0f, 7.0f};
	size_t N = a.size();
	std::vector<float> maxResult(N);
	std::vector<float> minResult(N);

	Runtime::Buffer<float> bufferA(a);
	Runtime::Buffer<float> bufferB(b);
	Runtime::Buffer<float> bufferMax(N);
	Runtime::Buffer<float> bufferMin(N);

	Kernel::Kernel1D kernel(
		[&, N](Var<int> &id) {
			auto bufA = bufferA.Bind();
			auto bufB = bufferB.Bind();
			auto bufMax = bufferMax.Bind();
			auto bufMin = bufferMin.Bind();

			Var<float> valA = bufA[id];
			Var<float> valB = bufB[id];

			bufMax[id] = Select(valA > valB, Expr<float>(valA), Expr<float>(valB));
			bufMin[id] = Select(valA < valB, Expr<float>(valA), Expr<float>(valB));
		},
		static_cast<int>(N));

	kernel.Dispatch(1, true);
	bufferMax.Download(maxResult);
	bufferMin.Download(minResult);

	// Verify results
	for (size_t i = 0; i < N; i++) {
		ASSERT_NEAR(maxResult[i], std::max(a[i], b[i]), 0.0001f);
		ASSERT_NEAR(minResult[i], std::min(a[i], b[i]), 0.0001f);
	}
END_TEST

TEST(ternary_clamp_runtime)
	// Test ternary for clamping values
	std::vector<float> input = {-2.0f, 0.5f, 1.5f, 2.0f, 3.5f, 5.0f};
	size_t N = input.size();
	std::vector<float> output(N);

	Runtime::Buffer<float> inputBuffer(input);
	Runtime::Buffer<float> outputBuffer(N);

	Kernel::Kernel1D kernel(
		[&, N](Var<int> &id) {
			auto in = inputBuffer.Bind();
			auto out = outputBuffer.Bind();

			Var<float> val = in[id];
			// Clamp to [0, 1] range using nested ternary
			Var<float> clamped = Select(val < 0.0f, Expr<float>(MakeFloat(0.0f)),
										Select(val > 1.0f, Expr<float>(MakeFloat(1.0f)), Expr<float>(val)));
			out[id] = clamped;
		},
		static_cast<int>(N));

	kernel.Dispatch(1, true);
	outputBuffer.Download(output);

	// Verify results
	for (size_t i = 0; i < N; i++) {
		float expected = std::max(0.0f, std::min(1.0f, input[i]));
		ASSERT_NEAR(output[i], expected, 0.0001f);
	}
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================\n";
	std::cout << "  EasyGPU Ternary Operator Test Suite   \n";
	std::cout << "========================================\n";

	try {
		test_basic_scalar_ternary();
		test_ternary_with_var_arguments();
		test_ternary_int_type();
		test_ternary_bool_type();
		test_ternary_vector_types();
		test_ternary_ivec_types();
		test_ternary_in_expressions();
		test_ternary_with_buffer();
		test_deeply_nested_ternary();
		test_ternary_runtime_execution();
		test_ternary_max_min_runtime();
		test_ternary_clamp_runtime();

		std::cout << "\n========================================\n";
		std::cout << "  Results: " << pass_count << "/" << test_count << " tests passed\n";
		std::cout << "========================================\n";

		return (pass_count == test_count) ? 0 : 1;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << "\n";
		return 1;
	}
}
