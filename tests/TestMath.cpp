/**
 * TestMath.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
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
// Test 1: Trigonometric Functions
// =============================================================================
TEST(trig_functions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<float> a = 1.0f;
        Var<float> b = Sin(a);
        Var<float> c = Cos(a);
        Var<float> d = Tan(a);
        Var<float> e = Asin(a);
        Var<float> f = Acos(a);
        Var<float> g = Atan(a);
        
        // Two-argument Atan
        Var<float> h = Atan2(a, b);
        
        // Hyperbolic functions
        Var<float> i = Sinh(a);
        Var<float> j = Cosh(a);
        Var<float> k = Tanh(a);
        Var<float> l = Asinh(a);
        Var<float> m = Acosh(a);
        Var<float> n = Atanh(a);
        
        // Degrees/Radians conversion
        Var<float> o = Radians(a);
        Var<float> p = Degrees(a);
    });
    kernel.Dispatch(true);
END_TEST

TEST(trig_vector_functions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec2> v2 = Vec2(1.0f, 2.0f);
        Var<Vec3> v3 = Vec3(1.0f, 2.0f, 3.0f);
        Var<Vec4> v4 = Vec4(1.0f, 2.0f, 3.0f, 4.0f);
        
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
    kernel.Dispatch(true);
END_TEST

// =============================================================================
// Test 2: Exponential Functions
// =============================================================================
TEST(exp_functions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<float> a = 2.0f;
        Var<float> b = 3.0f;
        
        Var<float> c = Pow(a, b);
        Var<float> d = Exp(a);
        Var<float> e = Log(a);
        Var<float> f = Exp2(a);
        Var<float> g = Log2(a);
        Var<float> h = Sqrt(a);
        Var<float> i = Inversesqrt(a);
    });
    kernel.Dispatch(true);
END_TEST

TEST(exp_vector_functions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec2> v2 = Vec2(1.0f, 2.0f);
        Var<Vec3> v3 = Vec3(1.0f, 2.0f, 3.0f);
        Var<Vec4> v4 = Vec4(1.0f, 2.0f, 3.0f, 4.0f);
        
        Var<Vec2> p2 = Pow(v2, v2);
        Var<Vec3> p3 = Pow(v3, v3);
        Var<Vec4> p4 = Pow(v4, v4);
        
        Var<Vec2> e2 = Exp(v2);
        Var<Vec3> l3 = Log(v3);
        Var<Vec4> s4 = Sqrt(v4);
    });
    kernel.Dispatch(true);
END_TEST

// =============================================================================
// Test 3: Common Functions
// =============================================================================
TEST(common_functions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<float> a = 2.5f;
        Var<float> b = -1.5f;
        Var<int> c = -5;
        
        Var<float> d = Abs(b);
        Var<int> e = Abs(c);
        Var<float> f = Sign(b);
        Var<int> g = Sign(c);
        Var<float> h = Floor(a);
        Var<float> i = Trunc(a);
        Var<float> j = Round(a);
        Var<float> k = RoundEven(a);
        Var<float> l = Ceil(a);
        Var<float> m = Fract(a);
        Var<float> n = Mod(a, 2.0f);
        
        // Min/Max with scalar
        Var<float> o = Min(a, b);
        Var<float> p = Min(a, 1.0f);
        Var<float> q = Max(a, b);
        Var<float> r = Max(a, 1.0f);
        
        // Clamp
        Var<float> s = Clamp(a, 0.0f, 1.0f);
        
        // Mix
        Var<float> t = Mix(a, b, 0.5f);
        
        // Step
        Var<float> u = Step(1.0f, a);
        
        // Smoothstep
        Var<float> v = Smoothstep(0.0f, 2.0f, a);
    });
    kernel.Dispatch(true);
END_TEST

TEST(common_vector_functions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec2> v2 = Vec2(1.5f, 2.5f);
        Var<Vec3> v3 = Vec3(1.5f, 2.5f, 3.5f);
        Var<Vec4> v4 = Vec4(1.5f, 2.5f, 3.5f, 4.5f);
        
        Var<Vec2> a2 = Abs(v2);
        Var<Vec3> f3 = Floor(v3);
        Var<Vec4> c4 = Ceil(v4);
        Var<Vec2> r2 = Round(v2);
        Var<Vec3> t3 = Trunc(v3);
        Var<Vec4> f4 = Fract(v4);
        
        Var<Vec2> m2 = Min(v2, v2);
        Var<Vec2> m2s = Min(v2, 1.0f);
        Var<Vec3> x3 = Max(v3, v3);
        Var<Vec3> x3s = Max(v3, 1.0f);
        
        Var<Vec4> cl4 = Clamp(v4, 0.0f, 1.0f);
        Var<Vec2> mi2 = Mix(v2, v2, 0.5f);
        Var<Vec3> st3 = Step(1.0f, v3);
        Var<Vec4> ss4 = Smoothstep(0.0f, 2.0f, v4);
        
        // Integer versions
        Var<IVec2> iv2 = IVec2(1, 2);
        Var<IVec3> iv3 = IVec3(1, 2, 3);
        Var<IVec4> iv4 = IVec4(1, 2, 3, 4);
        
        Var<IVec2> ai2 = Abs(iv2);
        Var<IVec2> ni2 = Min(iv2, iv2);
        Var<IVec3> xi3 = Max(iv3, iv3);
        Var<IVec4> ci4 = Clamp(iv4, iv4, iv4);
    });
    kernel.Dispatch(true);
END_TEST

// =============================================================================
// Test 4: Geometric Functions
// =============================================================================
TEST(geometric_functions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec2> v2 = Vec2(1.0f, 2.0f);
        Var<Vec3> v3 = Vec3(1.0f, 2.0f, 3.0f);
        Var<Vec4> v4 = Vec4(1.0f, 2.0f, 3.0f, 4.0f);
        
        // Length
        Var<float> l2 = Length(v2);
        Var<float> l3 = Length(v3);
        Var<float> l4 = Length(v4);
        
        // Distance
        Var<float> d2 = Distance(v2, v2);
        Var<float> d3 = Distance(v3, v3);
        Var<float> d4 = Distance(v4, v4);
        
        // Dot
        Var<float> dt2 = Dot(v2, v2);
        Var<float> dt3 = Dot(v3, v3);
        Var<float> dt4 = Dot(v4, v4);
        
        // Cross
        Var<Vec3> cr = Cross(v3, v3);
        
        // Normalize
        Var<Vec2> n2 = Normalize(v2);
        Var<Vec3> n3 = Normalize(v3);
        Var<Vec4> n4 = Normalize(v4);
        
        // Faceforward
        Var<Vec3> ff = Faceforward(v3, v3, v3);
        
        // Reflect
        Var<Vec3> rfl = Reflect(v3, v3);
        
        // Refract
        Var<Vec3> rfr = Refract(v3, v3, 1.5f);
        Var<Vec3> rfr2 = Refract(v3, v3, Var<float>(1.5f));
    });
    kernel.Dispatch(true);
END_TEST

// =============================================================================
// Test 5: Vector Relational Functions
// =============================================================================
TEST(vector_relational_functions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec2> v2a = Vec2(1.0f, 2.0f);
        Var<Vec2> v2b = Vec2(3.0f, 1.0f);
        Var<Vec3> v3a = Vec3(1.0f, 2.0f, 3.0f);
        Var<Vec3> v3b = Vec3(3.0f, 1.0f, 2.0f);
        Var<Vec4> v4a = Vec4(1.0f, 2.0f, 3.0f, 4.0f);
        Var<Vec4> v4b = Vec4(4.0f, 3.0f, 2.0f, 1.0f);
        
        Var<bool> lt2 = LessThan(v2a, v2b);
        Var<bool> lt3 = LessThan(v3a, v3b);
        Var<bool> lt4 = LessThan(v4a, v4b);
        
        Var<bool> le2 = LessThanEqual(v2a, v2b);
        Var<bool> gt2 = GreaterThan(v2a, v2b);
        Var<bool> ge2 = GreaterThanEqual(v2a, v2b);
        Var<bool> eq2 = Equal(v2a, v2b);
        Var<bool> ne2 = NotEqual(v2a, v2b);
        
        // Integer vector versions
        Var<IVec2> iv2a = IVec2(1, 2);
        Var<IVec2> iv2b = IVec2(3, 1);
        
        Var<bool> ilt2 = LessThan(iv2a, iv2b);
        Var<bool> ieq2 = Equal(iv2a, iv2b);
    });
    kernel.Dispatch(true);
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

        std::cout << "\n========================================\n";
        std::cout << "  All tests passed!                     \n";
        std::cout << "========================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
