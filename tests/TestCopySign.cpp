/**
 * TestCopySign.cpp:
 *      @Author         :   Test suite for CopySign functions
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

#define ASSERT_NEAR(a, b, eps)                                                                                         \
	if (std::abs((a) - (b)) > (eps)) {                                                                                 \
		throw std::runtime_error("Assertion failed: |" #a " - " #b "| > " #eps);                                       \
	}

// =============================================================================
// Test Suite: CopySign Functions
// =============================================================================

TEST(copysign_basic)
	// Test basic CopySign with InspectorKernel
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		Var<float> x = MakeFloat(5.0f);
		Var<float> y = MakeFloat(-3.0f);

		// CopySign(5.0, -3.0) = -5.0
		Var<float> result = CopySign(x, y);

		// CopySign with positive y
		Var<float> y2 = MakeFloat(2.0f);
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
		Var<Vec2> x = MakeFloat2(1.0f, 2.0f);
		Var<Vec2> y = MakeFloat2(-1.0f, -2.0f);

		// CopySign vector, vector
		Var<Vec2> result1 = CopySign(x, y);

		// CopySign vector, scalar (broadcast)
		Var<Vec2> result2 = CopySign(x, -1.0f);

		// CopySign vector, scalar expr
		Var<float> signVal = MakeFloat(-1.0f);
		Var<Vec2> result3 = CopySign(x, signVal);
	});
	kernel.PrintCode();
	ASSERT(true);
END_TEST

TEST(copysign_vector3)
	// Test CopySign with Vec3
	GPU::Kernel::InspectorKernel kernel([](Var<int> &id) {
		Var<Vec3> x = MakeFloat3(1.0f, 2.0f, 3.0f);
		Var<Vec3> y = MakeFloat3(-1.0f, -2.0f, -3.0f);

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
		Var<Vec4> x = MakeFloat4(1.0f, 2.0f, 3.0f, 4.0f);
		Var<Vec4> y = MakeFloat4(-1.0f, -2.0f, -3.0f, -4.0f);

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
	std::vector<float> x_values = {5.0f, -3.0f, 10.0f, -7.0f, 0.0f};
	std::vector<float> y_values = {-1.0f, 2.0f, -5.0f, 8.0f, 1.0f};
	size_t N = x_values.size();
	std::vector<float> output(N);

	Runtime::Buffer<float> bufferX(x_values);
	Runtime::Buffer<float> bufferY(y_values);
	Runtime::Buffer<float> bufferOutput(N);

	Kernel::Kernel1D kernel(
		[&, N](Var<int> &id) {
			auto x = bufferX.Bind();
			auto y = bufferY.Bind();
			auto out = bufferOutput.Bind();

			out[id] = CopySign(x[id], y[id]);
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
	std::vector<float> x_values = {0.0f, -0.0f, 1e-10f, -1e-10f, 1e10f};
	std::vector<float> y_values = {1.0f, -1.0f, 0.0f, -0.0f, -1e10f};
	size_t N = x_values.size();
	std::vector<float> output(N);

	Runtime::Buffer<float> bufferX(x_values);
	Runtime::Buffer<float> bufferY(y_values);
	Runtime::Buffer<float> bufferOutput(N);

	Kernel::Kernel1D kernel(
		[&, N](Var<int> &id) {
			auto x = bufferX.Bind();
			auto y = bufferY.Bind();
			auto out = bufferOutput.Bind();

			out[id] = CopySign(x[id], y[id]);
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
	std::cout << "  EasyGPU CopySign Test Suite          \n";
	std::cout << "========================================\n";

	try {
		test_copysign_basic();
		test_copysign_vector2();
		test_copysign_vector3();
		test_copysign_vector4();
		test_copysign_runtime();
		test_copysign_special_cases();

		std::cout << "\n========================================\n";
		std::cout << "  Results: " << pass_count << "/" << test_count << " tests passed\n";
		std::cout << "========================================\n";

		return (pass_count == test_count) ? 0 : 1;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << "\n";
		return 1;
	}
}
