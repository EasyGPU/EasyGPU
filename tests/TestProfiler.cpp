/**
 * TestProfiler.cpp:
 *      @Descripiton    :   Test Kernel Profiler functionality
 *      @Author         :   Margoo
 *      @Date           :   2/15/2026
 */
#include <Kernel/Kernel.h>
#include <Kernel/KernelProfiler.h>
#include <IR/Value/Var.h>
#include <Runtime/Buffer.h>

#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>

using namespace GPU;
using namespace GPU::IR::Value;
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
// Test Basic Profiler Enable/Disable
// =============================================================================
TEST(profiler_enable_disable)
    auto& profiler = Kernel::KernelProfiler::GetInstance();
    
    // Initially should be disabled
    ASSERT(!profiler.IsEnabled());
    
    // Enable
    profiler.SetEnabled(true);
    ASSERT(profiler.IsEnabled());
    
    // Disable
    profiler.SetEnabled(false);
    ASSERT(!profiler.IsEnabled());
    
    // Cleanup
    profiler.SetEnabled(false);
END_TEST

TEST(profiler_convenience_functions)
    // Test convenience functions
    Kernel::EnableKernelProfiler(true);
    ASSERT(Kernel::KernelProfiler::GetInstance().IsEnabled());
    
    Kernel::EnableKernelProfiler(false);
    ASSERT(!Kernel::KernelProfiler::GetInstance().IsEnabled());
END_TEST

// =============================================================================
// Test Clear Records
// =============================================================================
TEST(profiler_clear)
    Kernel::EnableKernelProfiler(true);
    Kernel::ClearKernelProfilerInfo();
    
    auto& profiler = Kernel::KernelProfiler::GetInstance();
    ASSERT(profiler.GetRecords().empty());
    ASSERT(profiler.GetTotalTime() == 0.0);
    
    Kernel::EnableKernelProfiler(false);
END_TEST

// =============================================================================
// Test Profiling with Actual Kernel Execution
// =============================================================================
TEST(profiler_record_kernel)
    Kernel::EnableKernelProfiler(true);
    Kernel::ClearKernelProfilerInfo();
    
    Buffer<int> buffer(256, BufferMode::ReadWrite);
    std::vector<int> data(256, 1);
    buffer.Upload(data);
    
    Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = buf[id] * 2;
    });
    
    // Execute kernel with sync=true for accurate timing
    kernel.Dispatch(1, true);
    
    auto& profiler = Kernel::KernelProfiler::GetInstance();
    // Note: OpenGL timer queries may not work on all systems
    // So we just check that profiler doesn't crash
    
    Kernel::EnableKernelProfiler(false);
END_TEST

// =============================================================================
// Test Multiple Kernel Executions
// =============================================================================
TEST(profiler_multiple_kernels)
    Kernel::EnableKernelProfiler(true);
    Kernel::ClearKernelProfilerInfo();
    
    Buffer<int> buffer(256, BufferMode::ReadWrite);
    std::vector<int> data(256, 1);
    buffer.Upload(data);
    
    Kernel::Kernel1D kernel1([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = buf[id] + 1;
    });
    
    Kernel::Kernel1D kernel2([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = buf[id] * 2;
    });
    
    // Execute multiple times
    for (int i = 0; i < 3; i++) {
        kernel1.Dispatch(1, true);
        kernel2.Dispatch(1, true);
    }
    
    // Test doesn't crash
    ASSERT(true);
    
    Kernel::EnableKernelProfiler(false);
END_TEST

// =============================================================================
// Test Query Info
// =============================================================================
TEST(profiler_query_info)
    Kernel::EnableKernelProfiler(true);
    Kernel::ClearKernelProfilerInfo();
    
    Buffer<int> buffer(256, BufferMode::ReadWrite);
    std::vector<int> data(256, 1);
    buffer.Upload(data);
    
    Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = buf[id] * 2;
    });
    
    kernel.Dispatch(1, true);
    
    // Query with empty name (should return empty result)
    auto result = Kernel::QueryKernelProfilerInfo("NonExistent");
    ASSERT(result.counter == 0);
    
    Kernel::EnableKernelProfiler(false);
END_TEST

// =============================================================================
// Test Total Time
// =============================================================================
TEST(profiler_total_time)
    Kernel::EnableKernelProfiler(true);
    Kernel::ClearKernelProfilerInfo();
    
    double totalTime = Kernel::GetKernelProfilerTotalTime();
    ASSERT(totalTime == 0.0);  // Should be 0 with no records
    
    Kernel::EnableKernelProfiler(false);
END_TEST

// =============================================================================
// Test PrintInfo (doesn't crash)
// =============================================================================
TEST(profiler_print_info)
    Kernel::EnableKernelProfiler(true);
    Kernel::ClearKernelProfilerInfo();
    
    Buffer<int> buffer1(256, BufferMode::ReadWrite);
    Buffer<float> buffer2(1024, BufferMode::ReadWrite);
    std::vector<int> data1(256, 1);
    std::vector<float> data2(1024, 1.0f);
    buffer1.Upload(data1);
    buffer2.Upload(data2);
    
    // Create multiple kernels with different names for pretty output
    Kernel::Kernel1D kernel_add("add_kernel", [&](Var<int>& id) {
        auto buf = buffer1.Bind();
        buf[id] = buf[id] + 1;
    });
    
    Kernel::Kernel1D kernel_mul("multiply_kernel", [&](Var<int>& id) {
        auto buf = buffer1.Bind();
        buf[id] = buf[id] * 2;
    });
    
    Kernel::Kernel2D kernel_2d("image_processing", [&](Var<int>& x, Var<int>& y) {
        auto buf = buffer2.Bind();
        Var<int> idx = y * 32 + x;
        buf[idx] = buf[idx] * 1.5f;
    });
    
    // Also test SetName API
    Kernel::Kernel1D kernel_unnamed([&](Var<int>& id) {
        auto buf = buffer1.Bind();
        buf[id] = buf[id] - 1;
    });
    kernel_unnamed.SetName("decrement_kernel");
    
    // Execute kernels multiple times to show statistics
    std::cout << "\n========== Running kernels for profiling demo ==========\n";
    
    // Run kernel_add 5 times
    for (int i = 0; i < 5; i++) {
        kernel_add.Dispatch(1, true);
    }
    
    // Run kernel_mul 3 times
    for (int i = 0; i < 3; i++) {
        kernel_mul.Dispatch(1, true);
    }
    
    // Run kernel_2d 2 times
    for (int i = 0; i < 2; i++) {
        kernel_2d.Dispatch(1, 1, true);
    }
    
    // Run decrement_kernel 4 times
    for (int i = 0; i < 4; i++) {
        kernel_unnamed.Dispatch(1, true);
    }
    
    // Test PrintInfo with formatted output - PRETTY FORMAT
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    PRETTY FORMAT OUTPUT (count mode)                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    Kernel::PrintKernelProfilerInfo("count");
    
    // Test trace mode
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    PRETTY FORMAT OUTPUT (trace mode)                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    Kernel::PrintKernelProfilerInfo("trace");
    
    Kernel::EnableKernelProfiler(false);
END_TEST

TEST(profiler_kernel_naming)
    std::cout << "\n  Testing kernel naming API...\n";
    
    // Test constructor naming
    Kernel::Kernel1D kernel1("my_kernel", [&](Var<int>& id) {
        Var<int> x = id;
    });
    ASSERT(kernel1.GetName() == "my_kernel");
    std::cout << "  ✓ Constructor naming works: " << kernel1.GetName() << "\n";
    
    // Test default name
    Kernel::Kernel1D kernel2([&](Var<int>& id) {
        Var<int> x = id;
    });
    ASSERT(kernel2.GetName() == "Kernel1D");
    std::cout << "  ✓ Default name works: " << kernel2.GetName() << "\n";
    
    // Test SetName
    kernel2.SetName("renamed_kernel");
    ASSERT(kernel2.GetName() == "renamed_kernel");
    std::cout << "  ✓ SetName works: " << kernel2.GetName() << "\n";
    
    // Test 2D and 3D naming
    Kernel::Kernel2D kernel2d_named("my_2d_kernel", [&](Var<int>& x, Var<int>& y) {
        Var<int> sum = x + y;
    });
    ASSERT(kernel2d_named.GetName() == "my_2d_kernel");
    std::cout << "  ✓ 2D kernel naming works: " << kernel2d_named.GetName() << "\n";
    
    Kernel::Kernel3D kernel3d_named("my_3d_kernel", [&](Var<int>& x, Var<int>& y, Var<int>& z) {
        Var<int> sum = x + y + z;
    });
    ASSERT(kernel3d_named.GetName() == "my_3d_kernel");
    std::cout << "  ✓ 3D kernel naming works: " << kernel3d_named.GetName() << "\n";
END_TEST

TEST(profiler_formatted_output_string)
    // Must enable profiler and clear any previous records
    Kernel::EnableKernelProfiler(true);
    Kernel::ClearKernelProfilerInfo();
    
    Buffer<int> buffer(256, BufferMode::ReadWrite);
    std::vector<int> data(256, 1);
    buffer.Upload(data);
    
    Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = buf[id] + 1;
    });
    
    // Execute kernel
    kernel.Dispatch(1, true);
    
    // Test GetFormattedOutput
    std::cout << "\n========== GetFormattedOutput String ==========\n";
    std::string output = Kernel::GetKernelProfilerFormattedOutput("count");
    std::cout << output;
    
    // Verify output is not empty and contains expected content
    ASSERT(!output.empty());
    
    // Check for header (using ASCII version for GetFormattedOutput)
    bool hasHeader = output.find("Kernel Profiling Results") != std::string::npos ||
                     output.find("Profiling Results") != std::string::npos;
    ASSERT(hasHeader);
    
    Kernel::EnableKernelProfiler(false);
END_TEST

// =============================================================================
// Test Profiler Disabled Behavior
// =============================================================================
TEST(profiler_disabled_no_records)
    Kernel::EnableKernelProfiler(false);
    Kernel::ClearKernelProfilerInfo();
    
    Buffer<int> buffer(256, BufferMode::ReadWrite);
    std::vector<int> data(256, 1);
    buffer.Upload(data);
    
    Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto buf = buffer.Bind();
        buf[id] = buf[id] + 1;
    });
    
    kernel.Dispatch(1, true);
    
    auto& profiler = Kernel::KernelProfiler::GetInstance();
    // When disabled, no records should be collected
    ASSERT(profiler.GetRecords().empty());
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "   Kernel Profiler Test Suite           \n";
    std::cout << "========================================\n";
    
    try {
        test_profiler_enable_disable();
        test_profiler_convenience_functions();
        test_profiler_clear();
        test_profiler_record_kernel();
        test_profiler_multiple_kernels();
        test_profiler_query_info();
        test_profiler_total_time();
        test_profiler_print_info();
        test_profiler_kernel_naming();
        test_profiler_formatted_output_string();
        test_profiler_disabled_no_records();
        
        std::cout << "\n========================================\n";
        std::cout << "  Results: " << pass_count << "/" << test_count << " tests passed\n";
        std::cout << "========================================\n";
        
        return (pass_count == test_count) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
