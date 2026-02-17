/**
 * TestDSL.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#include <iostream>
#include <cassert>
#include <cmath>

#include <GPU.h>

// =============================================================================
// Test 1: Basic Scalar Types
// =============================================================================
static int test_count = 0;
static int pass_count = 0;

#define TEST(name) void test_##name() { \
    std::cout << "\n[TEST] " #name " ... "; \
    test_count++; \
    try {

#define END_TEST \
        pass_count++; \
        std::cout << "PASSED\n"; \
    } catch (const std::exception& e) { \
        std::cout << "FAILED: " << e.what() << "\n"; \
    } \
}

#define ASSERT(cond) if (!(cond)) { \
    throw std::runtime_error("Assertion failed: " #cond); \
}

// =============================================================================
// Test 2: Struct Definition for Complex Tests
// =============================================================================
EASYGPU_STRUCT(Particle,
    (GPU::Math::Vec3, position),
    (GPU::Math::Vec3, velocity),
    (float, life),
    (int, type)
);

EASYGPU_STRUCT(Material,
    (GPU::Math::Vec3, albedo),
    (float, roughness),
    (float, metallic)
);

// =============================================================================
// Test Suites
// =============================================================================

TEST(basic_scalar_types)
    // Test integer variables
    GPU::Kernel::InspectorKernel kernel1([](Var<int>& id) {
        Var<int> a;
        Var<int> b = MakeInt(10);
        Var<int> c = a + b;
        Var<int> d = c - 5;
        Var<int> e = d * 2;
        Var<int> f = e / 3;
        Var<int> g = f % 4;
        
        // Comparison
        Var<bool> eq = (a == b);
        Var<bool> ne = (a != b);
        Var<bool> lt = (a < b);
        Var<bool> gt = (a > b);
        Var<bool> le = (a <= b);
        Var<bool> ge = (a >= b);
        
        // Bitwise operations
        Var<int> h = b & 7;
        Var<int> i = b | 8;
        Var<int> j = b ^ 15;
        Var<int> k = ~b;
        Var<int> l = b << 2;
        Var<int> m = b >> 1;
        
        // Compound assignment
        a += 1;
        a -= 1;
        a *= 2;
        a /= 2;
        a %= 3;
        a &= 7;
        a |= 8;
        a ^= 15;
        a <<= 1;
        a >>= 1;
        
        // Increment/Decrement
        ++a;
        auto _unused1 = a++;
        --a;
        auto _unused2 = a--;
    });
    kernel1.PrintCode();
    ASSERT(true);
END_TEST

TEST(float_operations)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<float> a;
        Var<float> b = MakeFloat(3.14f);
        Var<float> c = a + b;
        Var<float> d = c - 1.0f;
        Var<float> e = d * 2.5f;
        Var<float> f = e / 2.0f;
        
        // Comparison
        Var<bool> eq = (a == b);
        Var<bool> lt = (a < b);
        Var<bool> gt = (a > 1.0f);
        Var<bool> le = (a <= b);
        Var<bool> ge = (a >= 0.0f);
        
        // Compound assignment
        a += 1.0f;
        a -= 0.5f;
        a *= 2.0f;
        a /= 2.0f;
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(bool_operations)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<bool> a;
        Var<bool> b = MakeBool(true);
        Var<bool> c = a == b;
        Var<bool> d = a != b;
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(vec2_operations)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec2> a;
        Var<Vec2> b = MakeFloat2(1.0f, 2.0f);
        Var<Vec2> c = a + b;
        Var<Vec2> d = c - b;
        Var<Vec2> e = d * 2.0f;  // Scalar multiplication
        Var<Vec2> f = 3.0f * e;  // Reverse scalar multiplication
        Var<Vec2> g = f / 2.0f;
        
        // Swizzle access
        Var<float> x = b.x();
        Var<float> y = b.y();
        Var<Vec2> xx = b.xx();
        Var<Vec2> xy = b.xy();
        Var<Vec2> yy = b.yy();
        
        // Subscript access
        Var<float> comp0 = b[0];
        Var<float> comp1 = b[1];
        
        // Comparison
        Var<bool> eq = (a == b);
        Var<bool> ne = (a != b);
        Var<bool> lt = (a.x() < b.x());
        
        // Assignment from expression
        Var<Vec2> h = a + b;
        
        // Compound assignment
        a += b;
        a -= b;
        a *= 2.0f;
        a /= 2.0f;
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(vec3_operations)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec3> a;
        Var<Vec3> b = MakeFloat3(1.0f, 2.0f, 3.0f);
        Var<Vec3> c = a + b;
        Var<Vec3> d = c - b;
        Var<Vec3> e = d * 2.0f;
        Var<Vec3> f = 3.0f * e;
        Var<Vec3> g = f / 2.0f;

        // Swizzle access - single component
        Var<float> x = b.x();
        Var<float> y = b.y();
        Var<float> z = b.z();
        
        // Swizzle access - 2-component
        Var<Vec2> xx = b.xx();
        Var<Vec2> xy = b.xy();
        Var<Vec2> xz = b.xz();
        Var<Vec2> yy = b.yy();
        Var<Vec2> yz = b.yz();
        Var<Vec2> zz = b.zz();
        
        // Swizzle access - 3-component
        Var<Vec3> xxx = b.xxx();
        Var<Vec3> xyz = b.xyz();
        Var<Vec3> zyx = b.zyx();
        
        // Subscript access
        Var<float> comp0 = b[0];
        Var<float> comp1 = b[1];
        Var<float> comp2 = b[2];
        
        // Expression swizzle
        Var<Vec3> res = (a + b).xyz();
        Var<Vec2> res2 = (a + b).xy();
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(vec4_operations)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<Vec4> a;
        Var<Vec4> b = MakeFloat4(1.0f, 2.0f, 3.0f, 4.0f);
        Var<Vec4> c = a + b;
        Var<Vec4> d = c - b;
        Var<Vec4> e = d * 2.0f;
        Var<Vec4> f = 3.0f * e;
        Var<Vec4> g = f / 2.0f;
        
        // Swizzle access - single component
        Var<float> x = b.x();
        Var<float> y = b.y();
        Var<float> z = b.z();
        Var<float> w = b.w();
        
        // Swizzle access - 2-component
        Var<Vec2> xy = b.xy();
        Var<Vec2> zw = b.zw();
        Var<Vec2> wx = b.wx();
        
        // Swizzle access - 3-component
        Var<Vec3> xyz = b.xyz();
        
        // Swizzle access - 4-component
        Var<Vec4> xyzw = b.xyzw();
        
        // Expression swizzle
        Var<Vec2> res = (a + b).xy();
        Var<Vec3> res2 = (a + b).xyz();
        Var<Vec4> res3 = (a + b).xyzw();
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(ivec_operations)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // IVec2
        Var<IVec2> iv2 = MakeInt2(1, 2);
        Var<IVec2> iv2_add = iv2 + IVec2(3, 4);
        Var<IVec2> iv2_mul = iv2 * 2;  // Scalar multiplication
        Var<int> iv2_x = iv2.x();
        Var<IVec2> iv2_xx = iv2.xx();
        
        // Bitwise operations on integer vectors
        Var<IVec2> iv2b = MakeInt2(7, 7);
        Var<IVec2> iv2_and = iv2 & iv2b;
        Var<IVec2> iv2_or = iv2 | iv2b;
        Var<IVec2> iv2_xor = iv2 ^ iv2b;
        Var<IVec2> iv2_not = ~iv2;
        
        // IVec3
        Var<IVec3> iv3 = MakeInt3(1, 2, 3);
        Var<IVec3> iv3_mul = iv3 * 2;
        Var<IVec3> iv3_xyz = iv3.xyz();
        
        // IVec4
        Var<IVec4> iv4 = MakeInt4(1, 2, 3, 4);
        Var<IVec4> iv4_mul = iv4 * 2;
        Var<IVec4> iv4_xyzw = iv4.xyzw();
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(matrix_operations)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Mat2
        Var<Mat2> m2;
        Var<Mat2> m2b = MakeMat2(Mat2::Identity());
        Var<Mat2> m2_add = m2 + m2b;
        Var<Mat2> m2_sub = m2 - m2b;
        Var<Mat2> m2_mul = m2 * 2.0f;
        Var<Mat2> m2_mul2 = 3.0f * m2;
        
        // Mat3
        Var<Mat3> m3;
        Var<Mat3> m3b = MakeMat3(Mat3::Identity());
        Var<Mat3> m3_mul = m3 * 2.0f;
        
        // Mat4
        Var<Mat4> m4;
        Var<Mat4> m4b = MakeMat4(Mat4::Identity());
        Var<Mat4> m4_mul = m4 * 2.0f;
        
        // Column access
        Var<Vec2> m2_col0 = m2[0];
        Var<Vec3> m3_col0 = m3[0];
        Var<Vec4> m4_col0 = m4[0];
        
        // Rectangular matrices
        Var<Mat2x3> m2x3;
        Var<Mat2x3> m2x3_mul = m2x3 * 2.0f;
        Var<Vec3> m2x3_col = m2x3[0];
        
        Var<Mat3x2> m3x2;
        Var<Mat3x2> m3x2_mul = m3x2 * 2.0f;
        Var<Vec2> m3x2_col = m3x2[0];
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(matrix_vector_multiplication)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Mat2 * Vec2
        Var<Mat2> m2;
        Var<Vec2> v2;
        Var<Vec2> m2v2 = m2 * v2;
        
        // Mat3 * Vec3
        Var<Mat3> m3;
        Var<Vec3> v3;
        Var<Vec3> m3v3 = m3 * v3;
        
        // Mat4 * Vec4
        Var<Mat4> m4;
        Var<Vec4> v4;
        Var<Vec4> m4v4 = m4 * v4;
        
        // Rectangular matrix * vector
        Var<Mat2x3> m2x3;
        Var<Vec3> m2x3_v2 = m2x3 * v2;  // Mat2x3 * Vec2 -> Vec3
        
        Var<Mat3x2> m3x2;
        Var<Vec2> m3x2_v3 = m3x2 * v3;  // Mat3x2 * Vec3 -> Vec2
        
        // Expression matrix * expression vector
        Var<Vec3> expr_mul = (m3 * 2.0f) * (v3 * 3.0f);
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(array_operations)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Float array
        VarArray<float, 10> float_arr;
        Var<float> f0 = float_arr[0];
        Var<float> f1 = float_arr[1];
        float_arr[0] = 1.0f;
        float_arr[1] = float_arr[0] + 1.0f;
        
        // Int array
        VarArray<int, 5> int_arr;
        Var<int> i0 = int_arr[0];
        int_arr[0] = 10;
        int_arr[1] = int_arr[0] + 5;
        
        // Vec3 array
        VarArray<Vec3, 8> vec3_arr;
        Var<Vec3> v = vec3_arr[0];
        vec3_arr[0] = Vec3(1.0f, 2.0f, 3.0f);
        Var<float> x = vec3_arr[0].x();
        
        // Dynamic index using expression
        Var<float> f_dyn = float_arr[Expr<int>(id)];
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(struct_operations)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Create struct variable
        Var<Particle> p;
        
        // Member access
        p.position() = Vec3(1.0f, 2.0f, 3.0f);
        p.velocity() = Vec3(0.0f, 1.0f, 0.0f);
        p.life() = 100.0f;
        p.type() = 1;
        
        // Member swizzle
        p.position().x() = 5.0f;
        Var<Vec2> pos_xy = p.position().xy();
        Var<Vec3> pos_xyz = p.position().xyz();
        
        // Arithmetic on members
        p.position() = p.position() + p.velocity() * 0.016f;
        p.life() = p.life() - 1.0f;
        
        // Nested struct
        Var<Material> mat;
        mat.albedo() = Vec3(1.0f, 0.5f, 0.0f);
        mat.roughness() = 0.5f;
        mat.metallic() = 0.0f;
        
        // Member of member
        mat.albedo().x() = 1.0f;
        Var<float> rough = mat.roughness();
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(struct_cpu_capture)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // CPU struct capture
        Particle cpu_particle;
        cpu_particle.position = Vec3(10.0f, 20.0f, 30.0f);
        cpu_particle.velocity = Vec3(1.0f, 2.0f, 3.0f);
        cpu_particle.life = 5.0f;
        cpu_particle.type = 2;
        
        Var<Particle> p = cpu_particle;
        
        // Modify
        p.position() = p.position() + p.velocity() * 0.016f;
        p.life() = p.life() - 0.01f;
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(var_assignment)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Assignment from Var to Var
        Var<int> a = MakeInt(10);
        Var<int> b;
        b = a;
        
        // Assignment from scalar
        Var<float> f;
        f = 3.14f;
        
        // Assignment from expression
        Var<Vec3> v1 = MakeFloat3(1.0f, 2.0f, 3.0f);
        Var<Vec3> v2;
        v2 = v1 + Vec3(1.0f, 1.0f, 1.0f);
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(expression_operations)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Complex expressions
        Var<int> a = MakeInt(1);
        Var<int> b = MakeInt(2);
        Var<int> c = MakeInt(3);
        
        Var<int> result1 = a + b * c;  // Should be 7
        Var<int> result2 = (a + b) * c;  // Should be 9
        Var<int> result3 = a + b + c;
        Var<int> result4 = a * b * c;
        
        // Comparisons with logical operators
        Var<bool> cmp1 = (a < b) && (b < c);
        Var<bool> cmp2 = (a == 1) || (b == 3);
        Var<bool> cmp3 = !(a == b);
        Var<bool> cmp4 = (a < b) && (b < c) && (c > a);
        
        // Float expressions
        Var<float> f1 = MakeFloat(1.5f);
        Var<float> f2 = MakeFloat(2.5f);
        Var<float> f3 = (f1 + f2) * 2.0f;
        Var<float> f4 = f1 * f2 + f3;
        
        // Vector expressions with swizzle
        Var<Vec3> v1 = MakeFloat3(1.0f, 2.0f, 3.0f);
        Var<Vec3> v2 = MakeFloat3(4.0f, 5.0f, 6.0f);
        Var<Vec2> v3 = (v1 + v2).xy();
        Var<Vec3> v4 = (v1 * 2.0f).xyz();
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

TEST(mixed_type_expressions)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Int and float comparison (promotion)
        Var<int> i = MakeInt(5);
        Var<float> f = MakeFloat(5.0f);
        Var<bool> eq = (i == i);  // Same type comparison
        Var<bool> eq2 = (f == f);
        
        // Scalar and vector multiplication
        Var<Vec3> v = MakeFloat3(1.0f, 2.0f, 3.0f);
        Var<Vec3> v2 = v * 2.0f;
        Var<Vec3> v3 = 3.0f * v;
        
        // Matrix and scalar
        Var<Mat4> m;
        Var<Mat4> m2 = m * 2.0f;
        Var<Mat4> m3 = 3.0f * m;
    });
    kernel.PrintCode();
    ASSERT(true);
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  EasyGPU DSL Comprehensive Test Suite  \n";
    std::cout << "========================================\n";
    
    try {
        test_basic_scalar_types();
        test_float_operations();
        test_bool_operations();
        test_vec2_operations();
        test_vec3_operations();
        test_vec4_operations();
        test_ivec_operations();
        test_matrix_operations();
        test_matrix_vector_multiplication();
        test_array_operations();
        test_struct_operations();
        test_struct_cpu_capture();
        test_var_assignment();
        test_expression_operations();
        test_mixed_type_expressions();
        
        std::cout << "\n========================================\n";
        std::cout << "  Results: " << pass_count << "/" << test_count << " tests passed\n";
        std::cout << "========================================\n";
        
        return (pass_count == test_count) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
