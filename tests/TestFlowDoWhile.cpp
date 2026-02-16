/**
 * TestFlowDoWhile.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

#include <Kernel/Kernel.h>
#include <IR/Value/Var.h>
#include <Flow/DoWhileFlow.h>
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
// Test 1: Simple do-while - executes at least once
// =============================================================================
TEST(dowhile_simple)
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> count = 0;
        
        DoWhile([&]() {
            count = count + 1;
        }, count < 5);
        
        output[0] = Expr<float>(count);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: loop runs while count < 5, so count becomes 5
    ASSERT_NEAR(outputData[0], 5.0f, 0.001f);
END_TEST

// =============================================================================
// Test 2: Do-while executes at least once even if condition is initially false
// =============================================================================
TEST(dowhile_once)
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> count = 0;
        Var<bool> flag = false;  // Condition starts as false
        
        // Body executes once even though flag is false
        DoWhile([&]() {
            count = count + 1;
        }, flag);
        
        output[0] = Expr<float>(count);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: body executes exactly once (condition false from start)
    ASSERT_NEAR(outputData[0], 1.0f, 0.001f);
END_TEST

// =============================================================================
// Test 3: Do-while for summation
// =============================================================================
TEST(dowhile_sum)
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> i = 1;
        Var<float> sum = 0.0f;
        
        DoWhile([&]() {
            sum = sum + Expr<float>(i);
            i = i + 1;
        }, i <= 10);
        
        output[0] = sum;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: sum of 1 to 10 = 55
    ASSERT_NEAR(outputData[0], 55.0f, 0.001f);
END_TEST

// =============================================================================
// Test 4: Do-while with Var<bool> condition
// =============================================================================
TEST(dowhile_var_condition)
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> count = 0;
        Var<bool> continueLoop = true;
        
        DoWhile([&]() {
            count = count + 1;
            If(count >= 3, [&]() {
                continueLoop = false;
            });
        }, continueLoop);
        
        output[0] = Expr<float>(count);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: runs 3 times
    ASSERT_NEAR(outputData[0], 3.0f, 0.001f);
END_TEST

// =============================================================================
// Test 5: Multiple work items with independent do-while
// =============================================================================
TEST(dowhile_multiple_work_items)
    const int numItems = 8;
    std::vector<float> outputData(numItems, 0.0f);
    Buffer<float> outputBuffer(numItems, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> myId = id;
        Var<int> count = 0;
        
        // Each work item runs do-while (id + 1) times
        DoWhile([&]() {
            count = count + 1;
        }, count <= myId);
        
        output[myId] = Expr<float>(count);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: output[i] = i + 1 (runs once, then while count <= id)
    for (int i = 0; i < numItems; ++i) {
        ASSERT_NEAR(outputData[i], static_cast<float>(i + 1), 0.001f);
    }
END_TEST

// =============================================================================
// Test 6: Nested do-while (rare but valid)
// =============================================================================
TEST(dowhile_nested)
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> outer = 0;
        Var<int> total = 0;
        
        DoWhile([&]() {
            Var<int> inner = 0;
            DoWhile([&]() {
                total = total + 1;
                inner = inner + 1;
            }, inner < 3);
            outer = outer + 1;
        }, outer < 4);
        
        output[0] = Expr<float>(total);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: 4 * 3 = 12 iterations
    ASSERT_NEAR(outputData[0], 12.0f, 0.001f);
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  EasyGPU Flow DoWhile Test (GPU)       \n";
    std::cout << "========================================\n";

    try {
        test_dowhile_simple();
        test_dowhile_once();
        test_dowhile_sum();
        test_dowhile_var_condition();
        test_dowhile_multiple_work_items();
        test_dowhile_nested();

        std::cout << "\n========================================\n";
        std::cout << "  All tests passed!                     \n";
        std::cout << "========================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
