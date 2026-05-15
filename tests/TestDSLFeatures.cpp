/**
 * @file TestDSLFeatures.cpp
 * @brief Tests for recently added DSL features.
 */

#include <GPU.h>
#include <cassert>
#include <iostream>
#include <string>

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

// =============================================================================
// Test 1: Float prefix increment (standalone emits code)
// =============================================================================
TEST(float_prefix_increment)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x(0.0f);
	++x;
	(void)id;
});
std::string					   code = kernel.GetCode();
assert(code.find("++") != std::string::npos && "Prefix ++ should generate ++ in GLSL");
END_TEST

// =============================================================================
// Test 2: Float postfix increment (only emits when result is used)
// =============================================================================
TEST(float_postfix_increment)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x(0.0f);
	Var<float> y = x++ + 1.0f;
	(void)y;
	(void)id;
});
std::string					   code = kernel.GetCode();
assert(code.find("++") != std::string::npos && "Postfix ++ in expression should generate ++ in GLSL");
END_TEST

// =============================================================================
// Test 3: Float prefix decrement
// =============================================================================
TEST(float_prefix_decrement)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x(5.0f);
	--x;
	(void)id;
});
std::string					   code = kernel.GetCode();
assert(code.find("--") != std::string::npos && "Prefix -- should generate -- in GLSL");
END_TEST

// =============================================================================
// Test 4: Float postfix decrement in expression
// =============================================================================
TEST(float_postfix_decrement)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x(5.0f);
	Var<float> y = x-- + 1.0f;
	(void)y;
	(void)id;
});
std::string					   code = kernel.GetCode();
assert(code.find("--") != std::string::npos && "Postfix -- in expression should generate -- in GLSL");
END_TEST

// =============================================================================
// Test 5: Int increment/decrement still compiles and emits code
// =============================================================================
TEST(int_increment)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<int> x(Expr<int>(0));
	++x;
	Var<int> y = x++ + Expr<int>(1);
	--x;
	Var<int> z = x-- + Expr<int>(1);
	(void)y;
	(void)z;
	(void)id;
});
std::string					   code = kernel.GetCode();
assert(code.find("++") != std::string::npos && "Int ++ should still work");
assert(code.find("--") != std::string::npos && "Int -- should still work");
END_TEST

// =============================================================================
// Test 6: Float increment in expression context
// =============================================================================
TEST(float_increment_in_expression)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x(0.0f);
	Var<float> y = ++x + 1.0f;
	Var<float> z = x++ + 2.0f;
	(void)y;
	(void)z;
	(void)id;
});
std::string					   code = kernel.GetCode();
assert(code.find("++") != std::string::npos && "Float ++ in expression should work");
END_TEST

// =============================================================================
// Test 7: RawCode API
// =============================================================================
TEST(rawcode_api)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	// Store raw code result in a Var so it gets emitted
	Var<int> x(Expr<int>(0));
	x = Expr<int>(RawCode("my_custom_glsl_function()"));
	(void)x;
	(void)id;
});
std::string					   code = kernel.GetCode();
assert(code.find("my_custom_glsl_function()") != std::string::npos && "RawCode should insert custom GLSL into output");
END_TEST

// =============================================================================
// Test 8: RawCode with Expr cast
// =============================================================================
TEST(rawcode_with_expr_cast)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<int> y(Expr<int>(0));
	y = Expr<int>(RawCode("someFunction(42)"));
	(void)y;
	(void)id;
});
std::string					   code = kernel.GetCode();
assert(code.find("someFunction(42)") != std::string::npos && "RawCode via Expr cast should generate inline GLSL");
END_TEST

// =============================================================================
// Test 9: Double literal conversion in ValueToString
// =============================================================================
TEST(double_literal)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x(Expr<float>(3.14));
	(void)x;
	(void)id;
});
std::string					   code = kernel.GetCode();
assert(code.find("float") != std::string::npos && "Double literal should generate float() cast in GLSL");
END_TEST

// =============================================================================
// Test 10: Float increment inside If body
// =============================================================================
TEST(float_increment_in_if)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x(0.0f);
	Var<bool>  cond(true);
	If(cond, [&]() { ++x; });
	(void)id;
});
std::string					   code = kernel.GetCode();
assert(code.find("++") != std::string::npos && "Float ++ inside If should work");
END_TEST

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "DSL Features Test Suite" << std::endl;
	std::cout << "========================================" << std::endl;

	int	 passed = 0;
	int	 total	= 0;

	auto run	= [&](auto func, const char *name) {
		   total++;
		   try {
			   func();
			   passed++;
		   } catch (const std::exception &) {
			   // Already printed by TEST/END_TEST macros
		   }
	};

	run(test_float_prefix_increment, "Float prefix increment");
	run(test_float_postfix_increment, "Float postfix increment");
	run(test_float_prefix_decrement, "Float prefix decrement");
	run(test_float_postfix_decrement, "Float postfix decrement");
	run(test_int_increment, "Int increment/decrement");
	run(test_float_increment_in_expression, "Float increment in expression");
	run(test_rawcode_api, "RawCode API");
	run(test_rawcode_with_expr_cast, "RawCode with Expr cast");
	run(test_double_literal, "Double literal");
	run(test_float_increment_in_if, "Float increment in If");

	std::cout << "\n========================================" << std::endl;
	std::cout << "Test Results: " << passed << "/" << total << " passed" << std::endl;
	std::cout << "========================================" << std::endl;

	return (passed == total) ? 0 : 1;
}
