/**
 * TestFlowIf.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

#include <Kernel/Kernel.h>
#include <IR/Value/Var.h>
#include <Flow/IfFlow.h>
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
// Test 1: Simple if - take true branch
// =============================================================================
TEST(if_true_branch)
    std::vector<float> outputData(4, 0.0f);
    Buffer<float> outputBuffer(outputData.size(), BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> idx = id;
        Var<float> value = 0.0f;
        
        If(idx < 2, [&]() {
            value = 1.0f;  // True branch
        });
        
        output[idx] = value;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: first 2 elements should be 1.0, others 0.0
    ASSERT_NEAR(outputData[0], 1.0f, 0.001f);
    ASSERT_NEAR(outputData[1], 1.0f, 0.001f);
    ASSERT_NEAR(outputData[2], 0.0f, 0.001f);
    ASSERT_NEAR(outputData[3], 0.0f, 0.001f);
END_TEST

// =============================================================================
// Test 2: If-Else - both branches
// =============================================================================
TEST(if_else_branches)
    std::vector<float> outputData(4, 0.0f);
    Buffer<float> outputBuffer(outputData.size(), BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> idx = id;
        Var<float> value = 0.0f;
        
        If(idx < 2, [&]() {
            value = 1.0f;  // True branch
        }).Else([&]() {
            value = 2.0f;  // False branch
        });
        
        output[idx] = value;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: first 2 elements should be 1.0, others 2.0
    ASSERT_NEAR(outputData[0], 1.0f, 0.001f);
    ASSERT_NEAR(outputData[1], 1.0f, 0.001f);
    ASSERT_NEAR(outputData[2], 2.0f, 0.001f);
    ASSERT_NEAR(outputData[3], 2.0f, 0.001f);
END_TEST

// =============================================================================
// Test 3: If-Elif-Else - grade calculation
// =============================================================================
TEST(if_elif_else_grade)
    // Input scores: 95, 85, 75, 65, 55
    std::vector<float> scores = {95.0f, 85.0f, 75.0f, 65.0f, 55.0f};
    std::vector<float> outputData(5, 0.0f);
    
    Buffer<float> scoreBuffer(scores, BufferMode::Read);
    Buffer<float> outputBuffer(5, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto input = scoreBuffer.Bind();
        auto output = outputBuffer.Bind();
        
        Var<int> idx = id;
        Var<float> score = input[idx];
        Var<float> grade = 0.0f;
        
        If(score >= 90.0f, [&]() {
            grade = 5.0f;  // A
        }).Elif(score >= 80.0f, [&]() {
            grade = 4.0f;  // B
        }).Elif(score >= 70.0f, [&]() {
            grade = 3.0f;  // C
        }).Elif(score >= 60.0f, [&]() {
            grade = 2.0f;  // D
        }).Else([&]() {
            grade = 1.0f;  // F
        });
        
        output[idx] = grade;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify grades
    ASSERT_NEAR(outputData[0], 5.0f, 0.001f);  // 95 -> A
    ASSERT_NEAR(outputData[1], 4.0f, 0.001f);  // 85 -> B
    ASSERT_NEAR(outputData[2], 3.0f, 0.001f);  // 75 -> C
    ASSERT_NEAR(outputData[3], 2.0f, 0.001f);  // 65 -> D
    ASSERT_NEAR(outputData[4], 1.0f, 0.001f);  // 55 -> F
END_TEST

// =============================================================================
// Test 4: Multiple Elif without Else
// =============================================================================
TEST(if_elif_no_else)
    std::vector<float> outputData(5, 0.0f);
    Buffer<float> outputBuffer(5, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> idx = id;
        Var<float> value = 0.0f;
        
        If(idx == 0, [&]() {
            value = 10.0f;
        }).Elif(idx == 1, [&]() {
            value = 20.0f;
        }).Elif(idx == 2, [&]() {
            value = 30.0f;
        });
        // No Else - value stays 0.0 for idx > 2
        
        output[idx] = value;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify
    ASSERT_NEAR(outputData[0], 10.0f, 0.001f);
    ASSERT_NEAR(outputData[1], 20.0f, 0.001f);
    ASSERT_NEAR(outputData[2], 30.0f, 0.001f);
    ASSERT_NEAR(outputData[3], 0.0f, 0.001f);   // No match
    ASSERT_NEAR(outputData[4], 0.0f, 0.001f);   // No match
END_TEST

// =============================================================================
// Test 5: If with vector operations
// =============================================================================
TEST(if_vector_ops)
    std::vector<float> outputData(4, 0.0f);
    Buffer<float> outputBuffer(4, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> idx = id;
        Var<Vec3> pos = Vec3(1.0f, 2.0f, 3.0f);
        Var<float> result = 0.0f;
        
        // Check if y component > 1.5
        If(pos.y() > 1.5f, [&]() {
            pos = pos * 2.0f;  // Double the vector
            result = pos.x() + pos.y() + pos.z();  // 2 + 4 + 6 = 12
        }).Else([&]() {
            pos = pos + Vec3(1.0f, 1.0f, 1.0f);
            result = pos.x() + pos.y() + pos.z();
        });
        
        output[idx] = result;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // All work items have same pos, so all should take true branch
    // Result: 2 + 4 + 6 = 12
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(outputData[i], 12.0f, 0.001f);
    }
END_TEST

// =============================================================================
// Test 6: Complex condition with logical operators
// =============================================================================
TEST(if_complex_condition)
    std::vector<float> inputData = {5.0f, 15.0f, 25.0f, 35.0f};
    std::vector<float> outputData(4, 0.0f);
    
    Buffer<float> inputBuffer(inputData, BufferMode::Read);
    Buffer<float> outputBuffer(4, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto input = inputBuffer.Bind();
        auto output = outputBuffer.Bind();
        
        Var<int> idx = id;
        Var<float> val = input[idx];
        Var<float> result = 0.0f;
        
        // Range check: 10 <= val < 30
        If((val >= 10.0f) && (val < 30.0f), [&]() {
            result = 1.0f;  // In range
        }).Else([&]() {
            result = 0.0f;  // Out of range
        });
        
        output[idx] = result;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: 5 (out), 15 (in), 25 (in), 35 (out)
    ASSERT_NEAR(outputData[0], 0.0f, 0.001f);
    ASSERT_NEAR(outputData[1], 1.0f, 0.001f);
    ASSERT_NEAR(outputData[2], 1.0f, 0.001f);
    ASSERT_NEAR(outputData[3], 0.0f, 0.001f);
END_TEST

// =============================================================================
// Test 7: Multiple work items with independent conditions
// =============================================================================
TEST(if_multiple_work_items)
    const int numItems = 64;
    std::vector<float> outputData(numItems, 0.0f);
    Buffer<float> outputBuffer(numItems, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> myId = id;
        Var<float> value = 0.0f;
        
        // Even ids get 1.0, odd ids get 2.0
        If((myId % 2) == 0, [&]() {
            value = 1.0f;
        }).Else([&]() {
            value = 2.0f;
        });
        
        output[myId] = value;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify pattern
    for (int i = 0; i < 8; ++i) {
        float expected = (i % 2 == 0) ? 1.0f : 2.0f;
        ASSERT_NEAR(outputData[i], expected, 0.001f);
    }
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  EasyGPU Flow If Test (GPU Execution)  \n";
    std::cout << "========================================\n";

    try {
        test_if_true_branch();
        test_if_else_branches();
        test_if_elif_else_grade();
        test_if_elif_no_else();
        test_if_vector_ops();
        test_if_complex_condition();
        test_if_multiple_work_items();

        std::cout << "\n========================================\n";
        std::cout << "  All tests passed!                     \n";
        std::cout << "========================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
