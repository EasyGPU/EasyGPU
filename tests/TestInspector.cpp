/**
 * TestInspector.cpp:
 *      @Descripiton    :   Test Inspector Kernels for all dimensions
 *      @Author         :   Margoo
 *      @Date           :   2/15/2026
 */
#include <Kernel/Kernel.h>
#include <IR/Value/Var.h>
#include <IR/Value/BufferRef.h>
#include <Runtime/Buffer.h>
#include <Utility/Helpers.h>
#include <Flow/ForFlow.h>
#include <Flow/IfFlow.h>
#include <Utility/Vec.h>

#include <iostream>
#include <cassert>
#include <string>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Flow;
using namespace GPU::Runtime;

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
// Test InspectorKernel1D
// =============================================================================
TEST(inspector_1d_basic)
    GPU::Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id * 2;
        Var<float> f = Expr<float>(x) + 1.5f;
    });
    
    std::string code = kernel.GetCode();
    ASSERT(!code.empty());
    ASSERT(code.find("local_size_x") != std::string::npos);
    ASSERT(code.find("gl_GlobalInvocationID.x") != std::string::npos);
    std::cout << "\n[Generated Code]:\n" << code << "\n";
END_TEST

TEST(inspector_1d_with_worksize)
    GPU::Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id;
    }, 128);  // Custom work size
    
    std::string code = kernel.GetCode();
    ASSERT(code.find("local_size_x = 128") != std::string::npos);
END_TEST

TEST(inspector_1d_buffer)
    Buffer<float> buffer(256, BufferMode::ReadWrite);
    
    GPU::Kernel::InspectorKernel1D kernel([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = Expr<float>(id) * 2.0f;
    });
    
    std::string code = kernel.GetCode();
    ASSERT(code.find("buffer") != std::string::npos || code.find("Buffer") != std::string::npos);
END_TEST

// =============================================================================
// Test InspectorKernel2D
// =============================================================================
TEST(inspector_2d_basic)
    GPU::Kernel::InspectorKernel2D kernel([](Var<int>& x, Var<int>& y) {
        Var<int> idx = y * 100 + x;
        Var<float> value = Expr<float>(idx);
    });
    
    std::string code = kernel.GetCode();
    ASSERT(!code.empty());
    ASSERT(code.find("local_size_x") != std::string::npos);
    ASSERT(code.find("local_size_y") != std::string::npos);
    ASSERT(code.find("gl_GlobalInvocationID.x") != std::string::npos);
    ASSERT(code.find("gl_GlobalInvocationID.y") != std::string::npos);
    std::cout << "\n[Generated Code]:\n" << code << "\n";
END_TEST

TEST(inspector_2d_custom_worksize)
    GPU::Kernel::InspectorKernel2D kernel([](Var<int>& x, Var<int>& y) {
        Var<int> sum = x + y;
    }, 32, 32);
    
    std::string code = kernel.GetCode();
    ASSERT(code.find("local_size_x = 32") != std::string::npos);
    ASSERT(code.find("local_size_y = 32") != std::string::npos);
END_TEST

TEST(inspector_2d_vector_ops)
    GPU::Kernel::InspectorKernel2D kernel([](Var<int>& x, Var<int>& y) {
        Var<Vec3> color = MakeFloat3(
            Expr<float>(x) / 100.0f,
            Expr<float>(y) / 100.0f,
            0.5f
        );
    });
    
    std::string code = kernel.GetCode();
    ASSERT(code.find("vec3") != std::string::npos);
END_TEST

// =============================================================================
// Test InspectorKernel3D
// =============================================================================
TEST(inspector_3d_basic)
    GPU::Kernel::InspectorKernel3D kernel([](Var<int>& x, Var<int>& y, Var<int>& z) {
        Var<int> idx = (z * 100 + y) * 100 + x;
    });
    
    std::string code = kernel.GetCode();
    ASSERT(!code.empty());
    ASSERT(code.find("local_size_x") != std::string::npos);
    ASSERT(code.find("local_size_y") != std::string::npos);
    ASSERT(code.find("local_size_z") != std::string::npos);
    ASSERT(code.find("gl_GlobalInvocationID.x") != std::string::npos);
    ASSERT(code.find("gl_GlobalInvocationID.y") != std::string::npos);
    ASSERT(code.find("gl_GlobalInvocationID.z") != std::string::npos);
    std::cout << "\n[Generated Code]:\n" << code << "\n";
END_TEST

TEST(inspector_3d_custom_worksize)
    GPU::Kernel::InspectorKernel3D kernel([](Var<int>& x, Var<int>& y, Var<int>& z) {
        Var<int> sum = x + y + z;
    }, 4, 4, 4);
    
    std::string code = kernel.GetCode();
    ASSERT(code.find("local_size_x = 4") != std::string::npos);
    ASSERT(code.find("local_size_y = 4") != std::string::npos);
    ASSERT(code.find("local_size_z = 4") != std::string::npos);
END_TEST

// =============================================================================
// Test Backward Compatibility (InspectorKernel alias)
// =============================================================================
TEST(inspector_backward_compat)
    // InspectorKernel should be an alias for InspectorKernel1D
    GPU::Kernel::InspectorKernel kernel([](Var<int>& id) {
        Var<int> x = id;
    });
    
    std::string code = kernel.GetCode();
    ASSERT(!code.empty());
    ASSERT(code.find("local_size_x = 256") != std::string::npos);
END_TEST

// =============================================================================
// Test PrintCode
// =============================================================================
TEST(inspector_print_code)
    GPU::Kernel::InspectorKernel2D kernel([](Var<int>& x, Var<int>& y) {
        Var<int> sum = x + y;
    });
    
    // Should not throw
    kernel.PrintCode();
    ASSERT(true);
END_TEST

// =============================================================================
// Test Compile API
// =============================================================================
TEST(inspector_compile_1d)
    std::cout << "\n  Testing InspectorKernel1D::Compile()...\n";
    
    GPU::Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id * 2;
        Var<float> f = Expr<float>(x) + 1.5f;
    });
    
    std::string errorMsg;
    bool compiled = kernel.Compile(errorMsg);
    
    if (!compiled) {
        std::cout << "  Error: " << errorMsg << "\n";
    }
    ASSERT(compiled);
    std::cout << "  ✓ 1D kernel compiled successfully!\n";
END_TEST

TEST(inspector_compile_2d)
    std::cout << "\n  Testing InspectorKernel2D::Compile()...\n";
    
    GPU::Kernel::InspectorKernel2D kernel([](Var<int>& x, Var<int>& y) {
        Var<int> idx = y * 100 + x;
        Var<Vec3> color = MakeFloat3(
            Expr<float>(x) / 100.0f,
            Expr<float>(y) / 100.0f,
            0.5f
        );
    });
    
    std::string errorMsg;
    bool compiled = kernel.Compile(errorMsg);
    
    if (!compiled) {
        std::cout << "  Error: " << errorMsg << "\n";
    }
    ASSERT(compiled);
    std::cout << "  ✓ 2D kernel compiled successfully!\n";
END_TEST

TEST(inspector_compile_3d)
    std::cout << "\n  Testing InspectorKernel3D::Compile()...\n";
    
    GPU::Kernel::InspectorKernel3D kernel([](Var<int>& x, Var<int>& y, Var<int>& z) {
        Var<int> idx = (z * 100 + y) * 100 + x;
        Var<Vec3> pos = MakeFloat3(
            Expr<float>(x),
            Expr<float>(y),
            Expr<float>(z)
        );
    });
    
    std::string errorMsg;
    bool compiled = kernel.Compile(errorMsg);
    
    if (!compiled) {
        std::cout << "  Error: " << errorMsg << "\n";
    }
    ASSERT(compiled);
    std::cout << "  ✓ 3D kernel compiled successfully!\n";
END_TEST

TEST(inspector_compile_simple_version)
    std::cout << "\n  Testing Compile() without error message...\n";
    
    GPU::Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id + 1;
    });
    
    // Simple version without error message
    bool compiled = kernel.Compile();
    ASSERT(compiled);
    std::cout << "  ✓ Simple Compile() works!\n";
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "   Inspector Kernel Test Suite          \n";
    std::cout << "========================================\n";
    
    try {
        test_inspector_1d_basic();
        test_inspector_1d_with_worksize();
        test_inspector_1d_buffer();
        test_inspector_2d_basic();
        test_inspector_2d_custom_worksize();
        test_inspector_2d_vector_ops();
        test_inspector_3d_basic();
        test_inspector_3d_custom_worksize();
        test_inspector_backward_compat();
        test_inspector_print_code();
        test_inspector_compile_1d();
        test_inspector_compile_2d();
        test_inspector_compile_3d();
        test_inspector_compile_simple_version();
        
        std::cout << "\n========================================\n";
        std::cout << "  Results: " << pass_count << "/" << test_count << " tests passed\n";
        std::cout << "========================================\n";
        
        return (pass_count == test_count) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
