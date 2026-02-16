/**
 * TestDispatch.cpp:
 *      @Descripiton    :   Test Kernel Dispatch with sync/async modes
 *      @Author         :   Margoo
 *      @Date           :   2/15/2026
 */
#include <Kernel/Kernel.h>
#include <IR/Value/Var.h>
#include <Runtime/Buffer.h>
#include <Flow/ForFlow.h>

#include <iostream>
#include <cassert>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;
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
// Test Basic Dispatch
// =============================================================================
TEST(dispatch_1d_basic)
    Buffer<int> buffer(256, BufferMode::ReadWrite);
    std::vector<int> data(256);
    for (int i = 0; i < 256; i++) data[i] = i;
    buffer.Upload(data);
    
    Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = buf[id] * 2;
    });
    
    // Sync dispatch (blocking)
    kernel.Dispatch(1, true);
    
    // Verify results
    std::vector<int> result;
    buffer.Download(result);
    ASSERT(result[0] == 0);
    ASSERT(result[1] == 2);
    ASSERT(result[100] == 200);
END_TEST

TEST(dispatch_2d_basic)
    Buffer<int> buffer(16 * 16, BufferMode::ReadWrite);
    std::vector<int> data(16 * 16, 0);
    buffer.Upload(data);
    
    Kernel::Kernel2D kernel([&](Var<int>& x, Var<int>& y) {
        auto buf = buffer.Bind();
        Var<int> idx = y * 16 + x;
        buf[idx] = idx;
    });
    
    kernel.Dispatch(1, 1, true);
    
    std::vector<int> result;
    buffer.Download(result);
    ASSERT(result[0] == 0);
    ASSERT(result[15] == 15);
    ASSERT(result[16] == 16);
END_TEST

TEST(dispatch_3d_basic)
    Buffer<int> buffer(8 * 8 * 8, BufferMode::ReadWrite);
    std::vector<int> data(8 * 8 * 8, 0);
    buffer.Upload(data);
    
    Kernel::Kernel3D kernel([&](Var<int>& x, Var<int>& y, Var<int>& z) {
        auto buf = buffer.Bind();
        Var<int> idx = (z * 8 + y) * 8 + x;
        buf[idx] = x + y + z;
    });
    
    kernel.Dispatch(1, 1, 1, true);
    
    std::vector<int> result;
    buffer.Download(result);
    ASSERT(result[0] == 0);  // x=0, y=0, z=0
END_TEST

// =============================================================================
// Test Async Dispatch (non-blocking)
// =============================================================================
TEST(dispatch_async_1d)
    Buffer<int> buffer(256, BufferMode::ReadWrite);
    std::vector<int> data(256, 1);
    buffer.Upload(data);
    
    Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = buf[id] + 100;
    });
    
    // Async dispatch (non-blocking)
    kernel.Dispatch(1, false);
    
    // Manual sync using RuntimeBarrier
    Kernel::KernelBase::RuntimeBarrier();
    
    // Verify results
    std::vector<int> result;
    buffer.Download(result);
    ASSERT(result[0] == 101);
    ASSERT(result[100] == 101);
END_TEST

TEST(dispatch_async_2d)
    Buffer<int> buffer(16 * 16, BufferMode::ReadWrite);
    std::vector<int> data(16 * 16, 0);
    buffer.Upload(data);
    
    Kernel::Kernel2D kernel([&](Var<int>& x, Var<int>& y) {
        auto buf = buffer.Bind();
        Var<int> idx = y * 16 + x;
        buf[idx] = 42;
    });
    
    // Async dispatch
    kernel.Dispatch(1, 1, false);
    
    // Manual barrier
    Kernel::KernelBase::RuntimeBarrier();
    
    std::vector<int> result;
    buffer.Download(result);
    ASSERT(result[0] == 42);
    ASSERT(result[100] == 42);
END_TEST

TEST(dispatch_async_3d)
    Buffer<int> buffer(64, BufferMode::ReadWrite);
    std::vector<int> data(64, 0);
    buffer.Upload(data);
    
    Kernel::Kernel3D kernel([&](Var<int>& x, Var<int>& y, Var<int>& z) {
        auto buf = buffer.Bind();
        Var<int> idx = (z * 4 + y) * 4 + x;
        buf[idx] = 123;
    });
    
    // Async dispatch
    kernel.Dispatch(1, 1, 1, false);
    
    // Manual barrier
    Kernel::KernelBase::RuntimeBarrier();
    
    std::vector<int> result;
    buffer.Download(result);
    ASSERT(result[0] == 123);
END_TEST

// =============================================================================
// Test Default Dispatch (sync=false by default)
// =============================================================================
TEST(dispatch_default_async)
    Buffer<int> buffer(256, BufferMode::ReadWrite);
    std::vector<int> data(256, 5);
    buffer.Upload(data);
    
    Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = buf[id] * 3;
    });
    
    // Default is async (sync=false)
    kernel.Dispatch(1);
    
    // Need explicit barrier
    Kernel::KernelBase::RuntimeBarrier();
    
    std::vector<int> result;
    buffer.Download(result);
    ASSERT(result[0] == 15);
    ASSERT(result[100] == 15);
END_TEST

// =============================================================================
// Test Large Dispatch
// =============================================================================
TEST(dispatch_large_1d)
    const int N = 1024 * 1024;  // 1M elements
    Buffer<int> buffer(N, BufferMode::ReadWrite);
    std::vector<int> data(N);
    for (int i = 0; i < N; i++) data[i] = i;
    buffer.Upload(data);
    
    Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = buf[id] + 1;
    });
    
    int groups = (N + 255) / 256;
    kernel.Dispatch(groups, true);
    
    std::vector<int> result;
    buffer.Download(result);
    ASSERT(result[0] == 1);
    ASSERT(result[1000] == 1001);
    ASSERT(result[N-1] == N);
END_TEST

TEST(dispatch_large_2d)
    const int W = 1024;
    const int H = 1024;
    Buffer<int> buffer(W * H, BufferMode::ReadWrite);
    std::vector<int> data(W * H, 0);
    buffer.Upload(data);
    
    Kernel::Kernel2D kernel([&](Var<int>& x, Var<int>& y) {
        auto buf = buffer.Bind();
        Var<int> idx = y * W + x;
        buf[idx] = x + y;
    });
    
    int groupsX = (W + 15) / 16;
    int groupsY = (H + 15) / 16;
    kernel.Dispatch(groupsX, groupsY, true);
    
    std::vector<int> result;
    buffer.Download(result);
    ASSERT(result[0] == 0);        // x=0, y=0
    ASSERT(result[1] == 1);        // x=1, y=0
    ASSERT(result[W] == 1);        // x=0, y=1
    ASSERT(result[W + 1] == 2);    // x=1, y=1
END_TEST

// =============================================================================
// Test GetCode
// =============================================================================
TEST(dispatch_get_code_1d)
    Kernel::Kernel1D kernel([&](Var<int>& id) {
        Var<int> x = id * 2;
    });
    
    std::string code = kernel.GetCode();
    ASSERT(!code.empty());
    ASSERT(code.find("local_size_x") != std::string::npos);
    ASSERT(code.find("void main()") != std::string::npos);
END_TEST

TEST(dispatch_get_code_2d)
    Kernel::Kernel2D kernel([&](Var<int>& x, Var<int>& y) {
        Var<int> sum = x + y;
    });
    
    std::string code = kernel.GetCode();
    ASSERT(!code.empty());
    ASSERT(code.find("local_size_x") != std::string::npos);
    ASSERT(code.find("local_size_y") != std::string::npos);
END_TEST

TEST(dispatch_get_code_3d)
    Kernel::Kernel3D kernel([&](Var<int>& x, Var<int>& y, Var<int>& z) {
        Var<int> sum = x + y + z;
    });
    
    std::string code = kernel.GetCode();
    ASSERT(!code.empty());
    ASSERT(code.find("local_size_x") != std::string::npos);
    ASSERT(code.find("local_size_y") != std::string::npos);
    ASSERT(code.find("local_size_z") != std::string::npos);
END_TEST

// =============================================================================
// Test Barrier Functions (Code Generation Only)
// =============================================================================
TEST(barrier_workgroup)
    Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id;
        // Insert workgroup barrier
        Kernel::KernelBase::WorkgroupBarrier();
        Var<int> y = x + 1;
    });
    
    std::string code = kernel.GetCode();
    ASSERT(code.find("barrier()") != std::string::npos);
END_TEST

TEST(barrier_memory)
    Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id;
        // Insert memory barrier
        Kernel::KernelBase::MemoryBarrier();
        Var<int> y = x + 1;
    });
    
    std::string code = kernel.GetCode();
    ASSERT(code.find("memoryBarrier()") != std::string::npos);
END_TEST

TEST(barrier_full)
    Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id;
        // Insert full barrier
        Kernel::KernelBase::FullBarrier();
        Var<int> y = x + 1;
    });
    
    std::string code = kernel.GetCode();
    ASSERT(code.find("memoryBarrier()") != std::string::npos);
    ASSERT(code.find("barrier()") != std::string::npos);
END_TEST

// =============================================================================
// Test Barrier Compilation (Actually Compile the Shader)
// =============================================================================
TEST(barrier_compile_workgroup)
    std::cout << "\n  Testing workgroup barrier compilation...\n";
    
    Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id;
        Kernel::KernelBase::WorkgroupBarrier();
        Var<int> y = x + 1;
    });
    
    std::string errorMsg;
    bool compiled = kernel.Compile(errorMsg);
    
    if (!compiled) {
        std::cout << "  Compilation failed: " << errorMsg << "\n";
    }
    ASSERT(compiled);
    std::cout << "  ✓ Workgroup barrier shader compiled successfully!\n";
END_TEST

TEST(barrier_compile_memory)
    std::cout << "\n  Testing memory barrier compilation...\n";
    
    Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id;
        Kernel::KernelBase::MemoryBarrier();
        Var<int> y = x + 1;
    });
    
    std::string errorMsg;
    bool compiled = kernel.Compile(errorMsg);
    
    if (!compiled) {
        std::cout << "  Compilation failed: " << errorMsg << "\n";
    }
    ASSERT(compiled);
    std::cout << "  ✓ Memory barrier shader compiled successfully!\n";
END_TEST

TEST(barrier_compile_full)
    std::cout << "\n  Testing full barrier compilation...\n";
    
    Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id;
        Kernel::KernelBase::FullBarrier();
        Var<int> y = x + 1;
    });
    
    std::string errorMsg;
    bool compiled = kernel.Compile(errorMsg);
    
    if (!compiled) {
        std::cout << "  Compilation failed: " << errorMsg << "\n";
    }
    ASSERT(compiled);
    std::cout << "  ✓ Full barrier shader compiled successfully!\n";
END_TEST

TEST(barrier_compile_2d)
    std::cout << "\n  Testing 2D kernel with barrier compilation...\n";
    
    Kernel::InspectorKernel2D kernel([](Var<int>& x, Var<int>& y) {
        Var<int> sum = x + y;
        Kernel::KernelBase::FullBarrier();
        Var<int> product = x * y;
    });
    
    std::string errorMsg;
    bool compiled = kernel.Compile(errorMsg);
    
    if (!compiled) {
        std::cout << "  Compilation failed: " << errorMsg << "\n";
    }
    ASSERT(compiled);
    std::cout << "  ✓ 2D barrier shader compiled successfully!\n";
END_TEST

TEST(barrier_compile_3d)
    std::cout << "\n  Testing 3D kernel with barrier compilation...\n";
    
    Kernel::InspectorKernel3D kernel([](Var<int>& x, Var<int>& y, Var<int>& z) {
        Var<int> sum = x + y + z;
        Kernel::KernelBase::FullBarrier();
        Var<int> product = x * y * z;
    });
    
    std::string errorMsg;
    bool compiled = kernel.Compile(errorMsg);
    
    if (!compiled) {
        std::cout << "  Compilation failed: " << errorMsg << "\n";
    }
    ASSERT(compiled);
    std::cout << "  ✓ 3D barrier shader compiled successfully!\n";
END_TEST

TEST(barrier_compile_with_buffer)
    std::cout << "\n  Testing kernel with barrier and buffer operations...\n";
    
    // Note: This test uses a dummy buffer for syntax, but doesn't actually bind it
    // since InspectorKernel doesn't execute
    Kernel::InspectorKernel1D kernel([](Var<int>& id) {
        Var<int> x = id;
        Kernel::KernelBase::MemoryBarrier();
        // Simulate some compute work
        For(0, 10, [&](Var<int>& i) {
            x = x + i;
        });
        Kernel::KernelBase::WorkgroupBarrier();
        Var<int> y = x * 2;
    });
    
    std::string errorMsg;
    bool compiled = kernel.Compile(errorMsg);
    
    if (!compiled) {
        std::cout << "  Compilation failed: " << errorMsg << "\n";
    }
    ASSERT(compiled);
    std::cout << "  ✓ Complex barrier shader compiled successfully!\n";
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "   Kernel Dispatch Test Suite           \n";
    std::cout << "========================================\n";
    
    try {
        test_dispatch_1d_basic();
        test_dispatch_2d_basic();
        test_dispatch_3d_basic();
        test_dispatch_async_1d();
        test_dispatch_async_2d();
        test_dispatch_async_3d();
        test_dispatch_default_async();
        test_dispatch_large_1d();
        test_dispatch_large_2d();
        test_dispatch_get_code_1d();
        test_dispatch_get_code_2d();
        test_dispatch_get_code_3d();
        test_barrier_workgroup();
        test_barrier_memory();
        test_barrier_full();
        test_barrier_compile_workgroup();
        test_barrier_compile_memory();
        test_barrier_compile_full();
        test_barrier_compile_2d();
        test_barrier_compile_3d();
        test_barrier_compile_with_buffer();
        
        std::cout << "\n========================================\n";
        std::cout << "  Results: " << pass_count << "/" << test_count << " tests passed\n";
        std::cout << "========================================\n";
        
        return (pass_count == test_count) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
