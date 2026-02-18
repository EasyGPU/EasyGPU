/**
 * TestBroadcastConstruct.cpp:
 *      Test suite for broadcast construction helpers (MakeFloat3(Float2, z), etc.)
 */
#include <iostream>
#include <cassert>

#include <GPU.h>

using namespace GPU;

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
// Test 1: MakeFloat3 from Float2 + float
// =============================================================================
TEST(make_float3_from_float2)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Create Float2
        Var<Vec2> v2 = MakeFloat2(1.0f, 2.0f);
        
        // MakeFloat3 from Float2 + scalar
        Var<Vec3> v3 = MakeFloat3(v2, 3.0f);
        
        // Verify components
        Var<float> x = v3.x();
        Var<float> y = v3.y();
        Var<float> z = v3.z();
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 2: MakeFloat4 from Float2 + float + float
// =============================================================================
TEST(make_float4_from_float2)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Create Float2
        Var<Vec2> v2 = MakeFloat2(1.0f, 2.0f);
        
        // MakeFloat4 from Float2 + scalar + scalar
        Var<Vec4> v4 = MakeFloat4(v2, 3.0f, 4.0f);
        
        // Verify components
        Var<float> x = v4.x();
        Var<float> y = v4.y();
        Var<float> z = v4.z();
        Var<float> w = v4.w();
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 3: MakeFloat4 from Float3 + float
// =============================================================================
TEST(make_float4_from_float3)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Create Float3
        Var<Vec3> v3 = MakeFloat3(1.0f, 2.0f, 3.0f);
        
        // MakeFloat4 from Float3 + scalar
        Var<Vec4> v4 = MakeFloat4(v3, 4.0f);
        
        // Verify components
        Var<float> x = v4.x();
        Var<float> y = v4.y();
        Var<float> z = v4.z();
        Var<float> w = v4.w();
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 4: MakeInt3 from Int2 + int
// =============================================================================
TEST(make_int3_from_int2)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Create Int2
        Var<IVec2> v2 = MakeInt2(1, 2);
        
        // MakeInt3 from Int2 + scalar
        Var<IVec3> v3 = MakeInt3(v2, 3);
        
        // Verify components
        Var<int> x = v3.x();
        Var<int> y = v3.y();
        Var<int> z = v3.z();
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 5: MakeInt4 from Int2 + int + int
// =============================================================================
TEST(make_int4_from_int2)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Create Int2
        Var<IVec2> v2 = MakeInt2(1, 2);
        
        // MakeInt4 from Int2 + scalar + scalar
        Var<IVec4> v4 = MakeInt4(v2, 3, 4);
        
        // Verify components
        Var<int> x = v4.x();
        Var<int> y = v4.y();
        Var<int> z = v4.z();
        Var<int> w = v4.w();
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 6: MakeInt4 from Int3 + int
// =============================================================================
TEST(make_int4_from_int3)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Create Int3
        Var<IVec3> v3 = MakeInt3(1, 2, 3);
        
        // MakeInt4 from Int3 + scalar
        Var<IVec4> v4 = MakeInt4(v3, 4);
        
        // Verify components
        Var<int> x = v4.x();
        Var<int> y = v4.y();
        Var<int> z = v4.z();
        Var<int> w = v4.w();
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Test 7: Mixed usage with Expr types
// =============================================================================
TEST(broadcast_with_expr_types)
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        // Create Float2 from expression
        Var<Vec2> v2 = MakeFloat2(1.0f, 2.0f);
        Expr<Vec2> e2 = v2 + Vec2(0.5f, 0.5f);
        
        // MakeFloat3 from Expr<Vec2> + Expr<float>
        Expr<float> fz = 3.0f;
        Var<Vec3> v3 = MakeFloat3(e2, fz);
        
        // Similarly for Vec4
        Var<Vec3> v3b = MakeFloat3(1.0f, 2.0f, 3.0f);
        Expr<Vec3> e3 = v3b * 2.0f;
        Var<Vec4> v4 = MakeFloat4(e3, 1.0f);
    });
    kernel.PrintCode();
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  Broadcast Construction Test Suite    \n";
    std::cout << "========================================\n";

    try {
        test_make_float3_from_float2();
        test_make_float4_from_float2();
        test_make_float4_from_float3();
        test_make_int3_from_int2();
        test_make_int4_from_int2();
        test_make_int4_from_int3();
        test_broadcast_with_expr_types();

        std::cout << "\n========================================\n";
        std::cout << "  All tests passed!                     \n";
        std::cout << "========================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
