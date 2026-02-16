/**
 * TestBreakContinue.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/15/2026
 */
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

#include <Kernel/Kernel.h>
#include <IR/Value/Var.h>
#include <Flow/For.h>
#include <Flow/While.h>
#include <Flow/DoWhile.h>
#include <Flow/If.h>
#include <Flow/Break.h>
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
// Test 1: Continue in For loop - sum only odd numbers
// =============================================================================
TEST(continue_for_sum_odds)
    // Sum of odd numbers from 0 to 9: 1 + 3 + 5 + 7 + 9 = 25
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<float> sum = 0.0f;
        
        For(0, 10, [&](Var<int>& i) {
            // Skip even numbers
            If(i % 2 == 0, [&]() {
                Continue();
            });
            sum = sum + Expr<float>(i);
        });
        
        output[0] = sum;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: sum of odd numbers 1+3+5+7+9 = 25
    ASSERT_NEAR(outputData[0], 25.0f, 0.001f);
END_TEST

// =============================================================================
// Test 2: Break in For loop - sum until condition met
// =============================================================================
TEST(break_for_sum_until)
    // Sum from 0 to 4 (break when i >= 5): 0+1+2+3+4 = 10
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<float> sum = 0.0f;
        
        For(0, 100, [&](Var<int>& i) {
            If(i >= 5, [&]() {
                Break();
            });
            sum = sum + Expr<float>(i);
        });
        
        output[0] = sum;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: sum of 0+1+2+3+4 = 10
    ASSERT_NEAR(outputData[0], 10.0f, 0.001f);
END_TEST

// =============================================================================
// Test 3: Continue and Break together in For loop
// =============================================================================
TEST(continue_break_for_combined)
    // Sum odd numbers from 0 to 19, but stop when sum exceeds 50
    // Odds: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
    // 1+3+5+7+9+11 = 36 (<50)
    // 1+3+5+7+9+11+13 = 49 (<50)
    // 1+3+5+7+9+11+13+15 = 64 (>50, break before adding 15)
    // Expected: 49
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<float> sum = 0.0f;
        
        For(0, 20, [&](Var<int>& i) {
            // Skip even numbers
            If(i % 2 == 0, [&]() {
                Continue();
            });
            
            // Check if adding this odd would exceed 50
            Var<float> nextSum = sum + Expr<float>(i);
            If(nextSum > 50.0f, [&]() {
                Break();
            });
            
            sum = nextSum;
        });
        
        output[0] = sum;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: sum should be 49 (1+3+5+7+9+11+13)
    ASSERT_NEAR(outputData[0], 49.0f, 0.001f);
END_TEST

// =============================================================================
// Test 4: Continue in While loop - count iterations with condition
// =============================================================================
TEST(continue_while_count_iterations)
    // Count numbers 1-10 that are NOT divisible by 3
    // Expected: 1,2,4,5,7,8,10 = 7 numbers
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> count = 0;
        Var<int> i = 1;
        
        While(i <= 10, [&]() {
            // Skip multiples of 3
            If(i % 3 == 0, [&]() {
                i = i + 1;
                Continue();
            });
            count = count + 1;
            i = i + 1;
        });
        
        output[0] = Expr<float>(count);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: count should be 7
    ASSERT_NEAR(outputData[0], 7.0f, 0.001f);
END_TEST

// =============================================================================
// Test 5: Break in While loop - find first match
// =============================================================================
TEST(break_while_find_first)
    // Find first number > 5 that is divisible by 7
    // Numbers: 6,7,8... 7 is divisible by 7
    // Expected: 7
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> result = -1;
        Var<int> i = 6;
        
        While(i < 100, [&]() {
            If(i % 7 == 0, [&]() {
                result = i;
                Break();
            });
            i = i + 1;
        });
        
        output[0] = Expr<float>(result);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: first match is 7
    ASSERT_NEAR(outputData[0], 7.0f, 0.001f);
END_TEST

// =============================================================================
// Test 6: Continue in DoWhile loop
// =============================================================================
TEST(continue_dowhile_skip)
    // Sum 1 to 10, but skip multiples of 3
    // Sum: 1+2+4+5+7+8+10 = 37
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<float> sum = 0.0f;
        Var<int> i = 1;
        
        DoWhile([&]() {
            // Skip multiples of 3
            If(i % 3 == 0, [&]() {
                i = i + 1;
                Continue();
            });
            sum = sum + Expr<float>(i);
            i = i + 1;
        }, i <= 10);
        
        output[0] = sum;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: 1+2+4+5+7+8+10 = 37
    ASSERT_NEAR(outputData[0], 37.0f, 0.001f);
END_TEST

// =============================================================================
// Test 7: Break in DoWhile loop
// =============================================================================
TEST(break_dowhile_early_exit)
    // Sum 1 to 10, but break when sum exceeds 20
    // 1+2+3+4+5+6 = 21 (>20, break at 6)
    // Actually: sum after adding 6 is 21, so we should have added up to 5
    // Let's verify: 1+2+3+4+5 = 15, next is 6, 15+6=21>20, so break
    // Expected: 15
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<float> sum = 0.0f;
        Var<int> i = 1;
        
        DoWhile([&]() {
            Var<float> nextSum = sum + Expr<float>(i);
            If(nextSum > 20.0f, [&]() {
                Break();
            });
            sum = nextSum;
            i = i + 1;
        }, i <= 10);
        
        output[0] = sum;
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: 1+2+3+4+5 = 15
    ASSERT_NEAR(outputData[0], 15.0f, 0.001f);
END_TEST

// =============================================================================
// Test 8: Nested loops with Break (inner only)
// =============================================================================
TEST(break_nested_inner)
    // Outer: i = 0 to 2
    // Inner: j = 0 to 10, break when j >= 3
    // Count total inner iterations: 3 + 3 + 3 = 9
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> count = 0;
        
        For(0, 3, [&](Var<int>& i) {
            For(0, 10, [&](Var<int>& j) {
                If(j >= 3, [&]() {
                    Break();
                });
                count = count + 1;
            });
        });
        
        output[0] = Expr<float>(count);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: 3 iterations per outer loop * 3 outer loops = 9
    ASSERT_NEAR(outputData[0], 9.0f, 0.001f);
END_TEST

// =============================================================================
// Test 9: Nested loops with Continue (inner only)
// =============================================================================
TEST(continue_nested_inner)
    // Outer: i = 0 to 2
    // Inner: j = 0 to 4, count only odd j
    // Per outer: j=1,3 counted = 2
    // Total: 2 * 3 = 6
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> count = 0;
        
        For(0, 3, [&](Var<int>& i) {
            For(0, 5, [&](Var<int>& j) {
                If(j % 2 == 0, [&]() {
                    Continue();
                });
                count = count + 1;
            });
        });
        
        output[0] = Expr<float>(count);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: 2 odd numbers per inner loop * 3 outer loops = 6
    ASSERT_NEAR(outputData[0], 6.0f, 0.001f);
END_TEST

// =============================================================================
// Test 10: Complex case - Prime number detection using Break
// =============================================================================
TEST(break_prime_detection)
    // Check if numbers are prime using trial division
    // A number is prime if no divisor found from 2 to sqrt(n)
    // Test numbers: 2,3,4,5,6,7,8,9,10
    // Expected: 2,3,5,7 are prime (1), others not (0)
    std::vector<int> inputData = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> outputData(inputData.size(), 0.0f);
    
    Buffer<int> inputBuffer(inputData, BufferMode::Read);
    Buffer<float> outputBuffer(inputData.size(), BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto input = inputBuffer.Bind();
        auto output = outputBuffer.Bind();
        
        Var<int> n = input[id];
        Var<bool> isPrime = true;
        
        // Numbers less than 2 are not prime
        If(n < 2, [&]() {
            isPrime = false;
        });
        
        // Check divisibility from 2 to n-1
        // Using For with Break for early exit
        Var<bool> foundDivisor = false;
        For(2, n, [&](Var<int>& d) {
            If(foundDivisor, [&]() {
                Break();  // Already found a divisor
            });
            If(n % d == 0, [&]() {
                foundDivisor = true;
                isPrime = false;
                Break();  // No need to check further
            });
        });
        
        // Output: 1.0 if prime, 0.0 if not
        If(isPrime, [&]() {
            output[id] = 1.0f;
        }).Else([&]() {
            output[id] = 0.0f;
        });
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify primes: 2,3,5,7 are prime (1.0)
    ASSERT_NEAR(outputData[0], 1.0f, 0.001f);  // 2 is prime
    ASSERT_NEAR(outputData[1], 1.0f, 0.001f);  // 3 is prime
    ASSERT_NEAR(outputData[2], 0.0f, 0.001f);  // 4 is not
    ASSERT_NEAR(outputData[3], 1.0f, 0.001f);  // 5 is prime
    ASSERT_NEAR(outputData[4], 0.0f, 0.001f);  // 6 is not
    ASSERT_NEAR(outputData[5], 1.0f, 0.001f);  // 7 is prime
    ASSERT_NEAR(outputData[6], 0.0f, 0.001f);  // 8 is not
    ASSERT_NEAR(outputData[7], 0.0f, 0.001f);  // 9 is not
    ASSERT_NEAR(outputData[8], 0.0f, 0.001f);  // 10 is not
END_TEST

// =============================================================================
// Test 11: Multiple Continue conditions in same loop
// =============================================================================
TEST(continue_multiple_conditions)
    // Count 1-20, skip multiples of 2, 3, and 5
    // Remaining: 1,7,11,13,17,19 = 6 numbers
    std::vector<float> outputData(1, 0.0f);
    Buffer<float> outputBuffer(1, BufferMode::Write);

    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto output = outputBuffer.Bind();
        
        Var<int> count = 0;
        
        For(1, 21, [&](Var<int>& i) {
            If(i % 2 == 0, [&]() {
                Continue();
            });
            If(i % 3 == 0, [&]() {
                Continue();
            });
            If(i % 5 == 0, [&]() {
                Continue();
            });
            count = count + 1;
        });
        
        output[0] = Expr<float>(count);
    }, 64);

    kernel.Dispatch(1, true);
    outputBuffer.Download(outputData);

    // Verify: count should be 6 (1,7,11,13,17,19)
    ASSERT_NEAR(outputData[0], 6.0f, 0.001f);
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  EasyGPU Break/Continue Test Suite     \n";
    std::cout << "========================================\n";

    try {
        test_continue_for_sum_odds();
        test_break_for_sum_until();
        test_continue_break_for_combined();
        test_continue_while_count_iterations();
        test_break_while_find_first();
        test_continue_dowhile_skip();
        test_break_dowhile_early_exit();
        test_break_nested_inner();
        test_continue_nested_inner();
        test_break_prime_detection();
        test_continue_multiple_conditions();

        std::cout << "\n========================================\n";
        std::cout << "  All tests passed!                     \n";
        std::cout << "========================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
