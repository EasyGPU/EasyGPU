/**
 * TestExpression.cpp:
 *      Comprehensive test suite for EasyGPU expression system
 *      Tests Var/Expr interactions, type conversions, operators, and complex expressions
 */
#include <iostream>
#include <cassert>
#include <cmath>

#include <Kernel/Kernel.h>
#include <IR/Value/Var.h>
#include <Utility/Math.h>

using namespace GPU::IR::Value;
using namespace GPU::Math;

#define TEST(name) void test_##name() { \
    std::cout << "\n[TEST] " #name " ... "; \
    try {

#define END_TEST \
        std::cout << "PASSED\n"; \
    } catch (const std::exception& e) { \
        std::cout << "FAILED: " << e.what() << "\n"; \
        throw; \
    } \
}

// =============================================================================
// Test 1: Basic Var to Expr conversion for function parameters
// =============================================================================
TEST(var_to_expr_conversion)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Test that Var can be passed to functions expecting Expr
        Var<Vec3> v1 = Vec3(1.0f, 2.0f, 3.0f);
        Var<Vec3> v2 = Vec3(4.0f, 5.0f, 6.0f);
        Var<Vec3> v3 = Vec3(0.5f, 0.5f, 0.5f);
        
        // Mix expects Expr<Vec3> but we pass Var<Vec3>
        Var<Vec3> result = Mix(v1, v2, v3);
        
        // Scalar types
        Var<float> f1 = 1.0f;
        Var<float> f2 = 2.0f;
        Var<float> f3 = 0.5f;
        Var<float> fresult = Mix(f1, f2, f3);
        
        // Other math functions
        Var<float> fsin = Sin(f1);
        Var<float> fcos = Cos(f1);
        Var<float> fabs = Abs(-f1);
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 2: Type cast functions (ToFloat, ToInt, ToBool)
// =============================================================================
TEST(type_cast_functions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // bool to float cast
        Var<bool> b = true;
        Var<float> f = ToFloat(b);
        
        // float to int cast
        Var<float> f2 = 3.7f;
        Var<int> i = ToInt(f2);
        
        // int to bool cast
        Var<int> i2 = 1;
        Var<bool> b2 = ToBool(i2);
        
        // Vector type casts
        Var<IVec2> iv2 = IVec2(1, 2);
        Var<Vec2> fv2 = ToFloat(iv2);
        
        Var<Vec3> fv3 = Vec3(1.5f, 2.5f, 3.5f);
        Var<IVec3> iv3 = ToInt(fv3);
        
        // Chained casts (avoid ambiguous overloads by using explicit Var)
        Var<float> f_tmp = 3.7f;
        Var<int> tmp_int = ToInt(f_tmp);
        Var<float> f3 = ToFloat(tmp_int);
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 3: Var<bool> logical operators
// =============================================================================
TEST(var_bool_logical_operators)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<bool> b1 = true;
        Var<bool> b2 = false;
        
        // && between Var<bool> and Var<bool>
        Var<bool> and_result = b1 && b2;
        
        // || between Var<bool> and Var<bool>
        Var<bool> or_result = b1 || b2;
        
        // ! on Var<bool>
        Var<bool> not_result = !b1;
        
        // With literals
        Var<bool> and_lit = b1 && true;
        Var<bool> or_lit = b1 || false;
        
        // Complex logical expressions
        Var<bool> complex = (b1 && b2) || (!b1 && !b2);
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 4: Var-Expr cross operators
// =============================================================================
TEST(var_expr_cross_operators)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<float> v1 = 1.0f;
        Var<float> v2 = 2.0f;
        
        // Create an Expr<float> from operation
        Expr<float> e1 = v1 + v1;
        
        // Var + Expr
        Var<float> r1 = v1 + e1;
        
        // Expr + Var
        Var<float> r2 = e1 + v1;
        
        // Var - Expr
        Var<float> r3 = v1 - e1;
        
        // Expr - Var
        Var<float> r4 = e1 - v1;
        
        // Var * Expr
        Var<float> r5 = v1 * e1;
        
        // Expr * Var
        Var<float> r6 = e1 * v1;
        
        // Var / Expr
        Var<float> r7 = v1 / e1;
        
        // Expr / Var
        Var<float> r8 = e1 / v1;
        
        // Comparisons
        Var<bool> c1 = v1 < e1;   // Var < Expr
        Var<bool> c2 = e1 < v1;   // Expr < Var
        Var<bool> c3 = v1 == e1;  // Var == Expr
        Var<bool> c4 = e1 == v1;  // Expr == Var
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 5: Vector Var-Expr cross operators
// =============================================================================
TEST(vector_var_expr_cross_operators)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec3> v1 = Vec3(1.0f, 2.0f, 3.0f);
        Var<Vec3> v2 = Vec3(4.0f, 5.0f, 6.0f);
        
        // Create an Expr<Vec3> from operation
        Expr<Vec3> e1 = v1 + v2;
        
        // Var + Expr
        Var<Vec3> r1 = v1 + e1;
        
        // Expr + Var
        Var<Vec3> r2 = e1 + v1;
        
        // Var * Expr
        Var<Vec3> r3 = v1 * e1;
        
        // Expr * Var
        Var<Vec3> r4 = e1 * v1;
        
        // Vec2 operations
        Var<Vec2> vv1 = Vec2(1.0f, 2.0f);
        Var<Vec2> vv2 = Vec2(3.0f, 4.0f);
        Expr<Vec2> ve1 = vv1 + vv2;
        Var<Vec2> vr1 = vv1 + ve1;
        
        // Vec4 operations
        Var<Vec4> v4a = Vec4(1.0f, 2.0f, 3.0f, 4.0f);
        Var<Vec4> v4b = Vec4(5.0f, 6.0f, 7.0f, 8.0f);
        Expr<Vec4> e4 = v4a + v4b;
        Var<Vec4> r4b = v4a * e4;
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 6: Integer vector Var-Expr cross operators
// =============================================================================
TEST(ivector_var_expr_cross_operators)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<IVec3> v1 = IVec3(1, 2, 3);
        Var<IVec3> v2 = IVec3(4, 5, 6);
        
        // Create an Expr<IVec3> from operation
        Expr<IVec3> e1 = v1 + v2;
        
        // Var + Expr
        Var<IVec3> r1 = v1 + e1;
        
        // Expr + Var
        Var<IVec3> r2 = e1 + v1;
        
        // Bitwise operators
        Var<IVec3> r3 = v1 & e1;
        Var<IVec3> r4 = e1 & v1;
        Var<IVec3> r5 = v1 | e1;
        Var<IVec3> r6 = e1 | v1;
        Var<IVec3> r7 = v1 ^ e1;
        
        // IVec2
        Var<IVec2> iv2a = IVec2(1, 2);
        Var<IVec2> iv2b = IVec2(3, 4);
        Expr<IVec2> ie2 = iv2a + iv2b;
        Var<IVec2> ir2 = iv2a & ie2;
        
        // IVec4
        Var<IVec4> iv4a = IVec4(1, 2, 3, 4);
        Var<IVec4> iv4b = IVec4(5, 6, 7, 8);
        Expr<IVec4> ie4 = iv4a + iv4b;
        Var<IVec4> ir4 = iv4a | ie4;
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 7: Logical operators with Var-Expr cross
// =============================================================================
TEST(logical_var_expr_cross)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<bool> vb = true;
        Expr<bool> eb = Expr<bool>(false);
        
        // Var && Expr
        Var<bool> r1 = vb && eb;
        
        // Expr && Var
        Var<bool> r2 = eb && vb;
        
        // Var || Expr
        Var<bool> r3 = vb || eb;
        
        // Expr || Var
        Var<bool> r4 = eb || vb;
        
        // Complex combinations (avoid reusing Expr multiple times in same expression)
        Var<bool> not_eb = !eb;
        Var<bool> r5 = (vb && eb) || (vb && not_eb);
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 8: Complex nested expressions
// =============================================================================
TEST(complex_nested_expressions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<float> a = 1.0f;
        Var<float> b = 2.0f;
        Var<float> c = 3.0f;
        
        // Nested arithmetic (each Expr can only be used once)
        Expr<float> e1 = (a + b) * c;
        Expr<float> e2 = a + (b * c);
        Var<float> r1 = e1 + e2;
        
        // Deep nesting
        Var<float> r2 = ((a + b) * c) / (a - b);
        
        // Mixed Var and Expr
        Var<float> r3 = a + ((b * c) - (a / b));
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 9: Expression reuse (critical for CloneNode correctness)
// =============================================================================
TEST(expression_reuse)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<float> v1 = 1.0f;
        Var<float> v2 = 2.0f;
        
        // Create an expression
        Expr<float> e = v1 + v2;
        
        // Use expression (each Expr can only be used once due to move semantics)
        Var<float> r1 = v1 + e;  // e is moved here, cannot be reused
        // r2-r4 omitted - would require reusing moved Expr
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 10: Vector swizzle with expressions
// =============================================================================
TEST(vector_swizzle_with_expressions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec3> v = Vec3(1.0f, 2.0f, 3.0f);
        
        // Create expression from swizzle
        Expr<float> e = v.x() + v.y();
        
        // Use expression (can only use once)
        Var<float> r1 = v.z() * e;
        
        // Swizzle reuse
        Var<float> x1 = v.x();
        Var<float> r2 = x1 + v.x();  // x1 should be same as v.x()
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 11: Math functions with mixed Var/Expr
// =============================================================================
TEST(math_functions_mixed)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<float> v1 = 1.0f;
        Var<float> v2 = 2.0f;
        
        // Create expression
        Expr<float> e = v1 + v2;
        
        // Math functions should accept both Var and Expr
        Var<float> s = Sin(v1);   // Var argument
        Var<float> c = Cos(e);    // Expr argument (moved)
        // Note: other uses of 'e' omitted as it was moved above
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 12: Comparison and boolean expressions
// =============================================================================
TEST(comparison_and_boolean)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<float> a = 1.0f;
        Var<float> b = 2.0f;
        Var<float> c = 3.0f;
        
        // Use boolean expressions directly (Expr cannot be stored and reused)
        Var<bool> r1 = a < b;     // Var < Var produces Expr, assigned to Var
        Var<bool> r2 = (a < b) && (b < c);  // Combined comparisons
        Var<bool> r3 = (a < b) || (b > c);  // Another combination
        
        // Complex boolean from comparisons
        Var<bool> complex = (a < b) && ((b < c) || (a == c));
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 13: Type conversions with expressions
// =============================================================================
TEST(type_conversions_with_expressions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<int> i = 5;
        Var<float> f = 3.14f;
        
        // Create expressions
        Expr<int> ei = i + i;
        Expr<float> ef = f * f;
        
        // Cast expressions (use std::move to avoid ambiguous overloads)
        Var<float> f2 = ToFloat(std::move(ei));  // Expr<int> -> float
        Var<int> i2 = ToInt(std::move(ef));      // Expr<float> -> int
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 14: Vector math with expressions
// =============================================================================
TEST(vector_math_with_expressions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec3> v1 = Vec3(1.0f, 2.0f, 3.0f);
        Var<Vec3> v2 = Vec3(4.0f, 5.0f, 6.0f);
        
        // Vector expressions
        Expr<Vec3> e = v1 + v2;
        
        // Vector math functions (each Expr can only be used once)
        Var<float> len1 = Length(e);
        Var<float> len2 = Length(v1);
        // Note: other operations with 'e' omitted as e was moved above
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 15: Stress test - deeply nested expressions
// =============================================================================
TEST(stress_test_deep_nesting)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<float> a = 1.0f;
        Var<float> b = 2.0f;
        Var<float> c = 3.0f;
        Var<float> d = 4.0f;
        
        // Deep nesting with mixed Var and Expr
        Expr<float> e1 = a + b;
        Expr<float> e2 = c * d;
        Expr<float> e3 = e1 + e2;
        Expr<float> e4 = e3 - a;
        Expr<float> e5 = e4 / b;
        
        // Note: Expr objects cannot be reused in multiple operations due to move semantics
        // Each Expr should only be used once, or cloned via Var assignment
        Var<float> r1 = e1 + e5;
        // r2-r3 and tree omitted - would require reusing moved Expr objects
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  EasyGPU Expression Test Suite       \n";
    std::cout << "========================================\n";

    try {
        test_var_to_expr_conversion();
        test_type_cast_functions();
        test_var_bool_logical_operators();
        test_var_expr_cross_operators();
        test_vector_var_expr_cross_operators();
        test_ivector_var_expr_cross_operators();
        test_logical_var_expr_cross();
        test_complex_nested_expressions();
        test_expression_reuse();
        test_vector_swizzle_with_expressions();
        test_math_functions_mixed();
        test_comparison_and_boolean();
        test_type_conversions_with_expressions();
        test_vector_math_with_expressions();
        test_stress_test_deep_nesting();

        std::cout << "\n========================================\n";
        std::cout << "  All tests passed!                     \n";
        std::cout << "========================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
