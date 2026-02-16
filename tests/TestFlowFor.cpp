/**
 * TestFlowFor.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

#include <Kernel/Kernel.h>
#include <IR/Value/Var.h>
#include <Flow/For.h>
#include <Flow/Continue.h>
#include <Runtime/Buffer.h>

using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Flow;
using namespace GPU::Runtime;

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

#define ASSERT(cond) if (!(cond)) { \
    throw std::runtime_error("Assertion failed: " #cond); \
}

#define ASSERT_NEAR(a, b, eps) if (std::abs((a) - (b)) > (eps)) { \
    throw std::runtime_error(std::format("Assertion failed: |{} - {}| > {}", a, b, eps)); \
}

// =============================================================================
// Test 1: Simple for loop summation
// =============================================================================
TEST(for_sum)
    // Create input/output buffer
    std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> outputData(inputData.size(), 0.0f);

    Buffer<float> inputBuffer(inputData, BufferMode::Read);
    Buffer<float> outputBuffer(inputData.size(), BufferMode::Write);

    // Kernel: sum of first i+1 elements for each output[i]
    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto input = inputBuffer.Bind();
        auto output = outputBuffer.Bind();
        
        Var<int> idx = id;
        Var<float> sum = 0.0f;
        
        For(0, idx + 1, [&](Var<int> i) {
            sum = sum + input[i];
        });
        
        output[idx] = sum;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: output[i] should be sum of input[0..i]
    float expectedSum = 0.0f;
    for (size_t i = 0; i < inputData.size(); ++i) {
        expectedSum += inputData[i];
        ASSERT_NEAR(outputData[i], expectedSum, 0.001f);
    }
END_TEST

// =============================================================================
// Test 2: For loop with explicit step
// =============================================================================
TEST(for_step)
    std::vector<float> outputData(10, 0.0f);
    Buffer<float> outputBuffer(outputData.size(), BufferMode::Write);

    // Kernel: count even indices
    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> count = 0;
        For(0, 10, 2, [&](Var<int> i) {
            count = count + 1;
        });
        
        output[id] = Expr<float>(count);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: should count 5 even numbers (0, 2, 4, 6, 8)
    ASSERT_NEAR(outputData[0], 5.0f, 0.001f);
END_TEST

// =============================================================================
// Test 3: Nested for loops - matrix transpose
// =============================================================================
TEST(for_nested_transpose)
    // 4x4 matrix
    std::vector<float> matrixData = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    std::vector<float> outputData(16, 0.0f);

    Buffer<float> inputBuffer(matrixData, BufferMode::Read);
    Buffer<float> outputBuffer(16, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto input = inputBuffer.Bind();
        auto output = outputBuffer.Bind();
        
        For(0, 4, [&](Var<int> i) {
            For(0, 4, [&](Var<int> j) {
                Var<int> srcIdx = i * 4 + j;
                Var<int> dstIdx = j * 4 + i;
                output[dstIdx] = input[srcIdx];
            });
        });
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify transpose
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float original = matrixData[i * 4 + j];
            float transposed = outputData[j * 4 + i];
            ASSERT_NEAR(original, transposed, 0.001f);
        }
    }
END_TEST

// =============================================================================
// Test 4: For loop with Var<int> bounds
// =============================================================================
TEST(for_var_bounds)
    std::vector<float> outputData(10, 0.0f);
    Buffer<float> outputBuffer(outputData.size(), BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> start = 2;
        Var<int> end = 7;
        Var<float> sum = 0.0f;
        
        For(start, end, [&](Var<int> i) {
            sum = sum + 1.0f;
        });
        
        output[id] = sum;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: loop from 2 to 6 (5 iterations)
    ASSERT_NEAR(outputData[0], 5.0f, 0.001f);
END_TEST

// =============================================================================
// Test 5: For loop with expression bounds
// =============================================================================
TEST(for_expr_bounds)
    std::vector<float> outputData(10, 0.0f);
    Buffer<float> outputBuffer(outputData.size(), BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> base = 3;
        Var<float> sum = 0.0f;
        
        // Loop from 0 to base*2 (6)
        For(0, base * 2, [&](Var<int> i) {
            sum = sum + 1.0f;
        });
        
        output[id] = sum;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: loop from 0 to 5 (6 iterations)
    ASSERT_NEAR(outputData[0], 6.0f, 0.001f);
END_TEST

// =============================================================================
// Test 6: Factorial calculation
// =============================================================================
TEST(for_factorial)
    std::vector<float> outputData(6, 0.0f); // 0!, 1!, 2!, 3!, 4!, 5!
    Buffer<float> outputBuffer(outputData.size(), BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> n = id;
        Var<float> result = 1.0f;
        
        For(1, n + 1, [&](Var<int> i) {
            result = result * Expr<float>(i);
        });
        
        output[n] = result;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify factorials
    ASSERT_NEAR(outputData[0], 1.0f, 0.001f);   // 0! = 1
    ASSERT_NEAR(outputData[1], 1.0f, 0.001f);   // 1! = 1
    ASSERT_NEAR(outputData[2], 2.0f, 0.001f);   // 2! = 2
    ASSERT_NEAR(outputData[3], 6.0f, 0.001f);   // 3! = 6
    ASSERT_NEAR(outputData[4], 24.0f, 0.001f);  // 4! = 24
    ASSERT_NEAR(outputData[5], 120.0f, 0.001f); // 5! = 120
END_TEST

// =============================================================================
// Test 7: Multiple work items with independent loops
// =============================================================================
TEST(for_multiple_work_items)
    const int numItems = 64;
    std::vector<float> outputData(numItems, 0.0f);
    Buffer<float> outputBuffer(numItems, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> myId = id;
        Var<float> sum = 0.0f;
        
        // Each work item sums up to its own ID + 1
        For(0, myId + 1, [&](Var<int> i) {
            sum = sum + 1.0f;
        });
        
        output[myId] = sum;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: output[i] should be i + 1
    for (int i = 0; i < 8; ++i) { // Check first 8 items
        ASSERT_NEAR(outputData[i], static_cast<float>(i + 1), 0.001f);
    }
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  EasyGPU Flow For Test (GPU Execution) \n";
    std::cout << "========================================\n";

    try {
        test_for_sum();
        test_for_step();
        test_for_nested_transpose();
        test_for_var_bounds();
        test_for_expr_bounds();
        test_for_factorial();
        test_for_multiple_work_items();

        std::cout << "\n========================================\n";
        std::cout << "  All tests passed!                     \n";
        std::cout << "========================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
