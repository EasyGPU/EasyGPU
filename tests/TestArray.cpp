/**
 * TestArray.cpp:
 *      @Description    :   Test for GPU VarArray functionality
 *      @Author         :   EasyGPU Team
 *      @Date           :   2026
 */
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <array>

#include <GPU.h>

using namespace GPU::IR::Value;
using namespace GPU::Math;
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
    } catch (const GPU::Runtime::ShaderCompileException& e) { \
        std::cout << "FAILED: Shader compilation error\n"; \
        std::cout << e.GetBeautifulOutput() << "\n"; \
    } catch (const GPU::Runtime::ShaderException& e) { \
        std::cout << "FAILED: Shader error - " << e.what() << "\n"; \
    } catch (const std::exception& e) { \
        std::cout << "FAILED: " << e.what() << "\n"; \
    } catch (...) { \
        std::cout << "FAILED: Unknown exception\n"; \
    } \
}

#define ASSERT(cond) if (!(cond)) { \
    throw std::runtime_error("Assertion failed: " #cond); \
}

// =============================================================================
// Test 1: Basic VarArray creation and element access
// =============================================================================
TEST(basic_vararray)
        Buffer<float> output(10, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            // Create local array of 10 floats
            VarArray<float, 10> localArr;
            
            // Initialize with values
            For(0, 10, [&](Int& i) {
                localArr[i] = Expr<float>(i) * 2.0f;
            });
            
            // Write to output buffer
            auto out = output.Bind();
            out[id] = localArr[id];
        }, 10);

        kernel.Dispatch(1, true);
        
        std::vector<float> result(10);
        output.Download(result.data(), 10);
        
        // Verify: localArr[i] should be i * 2
        bool correct = true;
        for (int i = 0; i < 10; i++) {
            float expected = i * 2.0f;
            if (std::abs(result[i] - expected) > 0.0001f) {
                correct = false;
                std::cout << "Mismatch at " << i << ": got " << result[i] 
                          << ", expected " << expected << "\n";
                break;
            }
        }
        
        ASSERT(correct);
        std::cout << "Basic VarArray operations verified!";
END_TEST

// =============================================================================
// Test 2: VarArray initialized from CPU std::array
// =============================================================================
TEST(vararray_from_cpu_array)
        // CPU-side lookup table
        std::array<float, 5> lookupTable = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
        
        Buffer<int> indices(std::vector<int>{0, 1, 2, 3, 4}, BufferMode::Read);
        Buffer<float> output(5, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&, lookupTable](Var<int>& id) {
            // Copy lookup table to local array
            VarArray<float, 5> table(lookupTable);
            
            auto idx = indices.Bind();
            auto out = output.Bind();
            
            // Lookup value from table
            Int index = MakeInt(idx[id]);
            index = Clamp(index, 0, 4);
            out[id] = table[index];
        }, 5);

        kernel.Dispatch(1, true);
        
        std::vector<float> result(5);
        output.Download(result.data(), 5);
        
        // Verify
        bool correct = true;
        for (int i = 0; i < 5; i++) {
            float expected = lookupTable[i];
            if (std::abs(result[i] - expected) > 0.0001f) {
                correct = false;
                std::cout << "Mismatch at " << i << ": got " << result[i] 
                          << ", expected " << expected << "\n";
                break;
            }
        }
        
        ASSERT(correct);
        std::cout << "VarArray from CPU array verified!";
END_TEST

// =============================================================================
// Test 3: VarArray with dynamic indexing in loops
// =============================================================================
TEST(vararray_dynamic_indexing)
        Buffer<float> output(5, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            VarArray<float, 5> arr;
            
            // Initialize using dynamic index from loop
            For(0, 5, [&](Int& i) {
                arr[i] = Expr<float>(i) * Expr<float>(i);  // i^2
            });
            
            // Sum all elements
            Float sum = MakeFloat(0.0f);
            For(0, 5, [&](Int& i) {
                sum = sum + arr[i];
            });
            
            auto out = output.Bind();
            out[id] = sum;  // Each thread writes the same sum
        }, 5);

        kernel.Dispatch(1, true);
        
        std::vector<float> result(5);
        output.Download(result.data(), 5);
        
        // Expected: 0^2 + 1^2 + 2^2 + 3^2 + 4^2 = 0 + 1 + 4 + 9 + 16 = 30
        float expected = 30.0f;
        bool correct = std::abs(result[0] - expected) < 0.0001f;
        
        ASSERT(correct);
        std::cout << "VarArray dynamic indexing verified! Sum = " << result[0];
END_TEST

// =============================================================================
// Test 4: VarArray for stencil computation (sliding window)
// =============================================================================
TEST(vararray_stencil_computation)
        // Input: 1D signal
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f};
        const int N = input.size();
        
        Buffer<float> inputBuf(input, BufferMode::Read);
        Buffer<float> outputBuf(N, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto in = inputBuf.Bind();
            auto out = outputBuf.Bind();
            
            // Load neighborhood into local array (window size 3)
            VarArray<float, 3> window;
            
            For(0, 3, [&](Int& j) {
                Int srcIdx = id + j - 1;  // -1, 0, +1 offset
                If(srcIdx >= 0 && srcIdx < N, [&]() {
                    window[j] = in[srcIdx];
                }).Else([&]() {
                    window[j] = MakeFloat(0.0f);  // Zero padding
                });
            });
            
            // Apply 1D blur kernel: [0.25, 0.5, 0.25]
            Float blurred = window[0] * 0.25f + window[1] * 0.5f + window[2] * 0.25f;
            out[id] = blurred;
        }, 8);

        kernel.Dispatch(1, true);
        
        std::vector<float> result(N);
        outputBuf.Download(result.data(), N);
        
        // Verify: expected values with zero padding
        // Position 0: (0*0.25 + 1*0.5 + 2*0.25) = 1.0
        // Position 1: (1*0.25 + 2*0.5 + 3*0.25) = 2.0
        // Position 2: (2*0.25 + 3*0.5 + 4*0.25) = 3.0
        // Position 3: (3*0.25 + 4*0.5 + 5*0.25) = 4.0
        // Position 4: (4*0.25 + 5*0.5 + 4*0.25) = 4.5
        // Position 5: (5*0.25 + 4*0.5 + 3*0.25) = 4.0
        // Position 6: (4*0.25 + 3*0.5 + 2*0.25) = 3.0
        // Position 7: (3*0.25 + 2*0.5 + 0*0.25) = 1.75
        bool correct = true;
        std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 4.5f, 4.0f, 3.0f, 1.75f};
        for (int i = 0; i < N; i++) {
            if (std::abs(result[i] - expected[i]) > 0.0001f) {
                correct = false;
                std::cout << "Mismatch at " << i << ": got " << result[i] 
                          << ", expected " << expected[i] << "\n";
                break;
            }
        }
        
        ASSERT(correct);
        std::cout << "VarArray stencil computation verified!";
END_TEST

// =============================================================================
// Test 5: VarArray with integer type
// =============================================================================
TEST(vararray_integer)
        Buffer<int> output(10, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            VarArray<int, 10> intArr;
            
            // Initialize with Fibonacci sequence
            intArr[0] = MakeInt(0);
            intArr[1] = MakeInt(1);
            
            For(2, 10, [&](Int& i) {
                intArr[i] = intArr[i - 1] + intArr[i - 2];
            });
            
            auto out = output.Bind();
            out[id] = intArr[id];
        }, 10);

        kernel.Dispatch(1, true);
        
        std::vector<int> result(10);
        output.Download(result.data(), 10);
        
        // Expected: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
        int expected[] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34};
        bool correct = true;
        for (int i = 0; i < 10; i++) {
            if (result[i] != expected[i]) {
                correct = false;
                std::cout << "Mismatch at " << i << ": got " << result[i] 
                          << ", expected " << expected[i] << "\n";
                break;
            }
        }
        
        ASSERT(correct);
        std::cout << "Integer VarArray verified!";
END_TEST

// =============================================================================
// Test 6: VarArray for histogram (per-thread local histogram)
// =============================================================================
TEST(vararray_histogram)
        // Input data (values 0-3 for 4 bins)
        std::vector<int> data = {0, 1, 2, 3, 0, 1, 2, 0, 1, 0};
        const int NUM_BINS = 4;
        const int CHUNK_SIZE = 10;
        
        Buffer<int> inputBuf(data, BufferMode::Read);
        Buffer<int> histBuf(NUM_BINS, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto in = inputBuf.Bind();
            auto hist = histBuf.Bind();
            
            // Local histogram array
            VarArray<int, 4> localHist;
            
            // Initialize to zero
            For(0, NUM_BINS, [&](Int& j) {
                localHist[j] = MakeInt(0);
            });
            
            // Count occurrences in chunk
            For(0, CHUNK_SIZE, [&](Int& j) {
                Int val = MakeInt(in[j]);
                val = Clamp(val, 0, NUM_BINS - 1);
                localHist[val] = localHist[val] + 1;
            });
            
            // Write to global histogram (only thread 0)
            If(id == 0, [&]() {
                For(0, NUM_BINS, [&](Int& j) {
                    hist[j] = localHist[j];
                });
            });
        }, 10);

        kernel.Dispatch(1, true);
        
        std::vector<int> result(NUM_BINS);
        histBuf.Download(result.data(), NUM_BINS);
        
        // Expected: bin 0: 4, bin 1: 3, bin 2: 2, bin 3: 1
        int expected[] = {4, 3, 2, 1};
        bool correct = true;
        for (int i = 0; i < NUM_BINS; i++) {
            if (result[i] != expected[i]) {
                correct = false;
                std::cout << "Mismatch at bin " << i << ": got " << result[i] 
                          << ", expected " << expected[i] << "\n";
                break;
            }
        }
        
        ASSERT(correct);
        std::cout << "VarArray histogram verified!";
END_TEST

// =============================================================================
// Test 7: VarArray with vector type (Vec3)
// =============================================================================
TEST(vararray_vec3)
        Buffer<Vec3> output(4, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            VarArray<Vec3, 4> vectors;
            
            // Initialize 4 direction vectors
            vectors[0] = MakeFloat3(1.0f, 0.0f, 0.0f);  // +X
            vectors[1] = MakeFloat3(-1.0f, 0.0f, 0.0f); // -X
            vectors[2] = MakeFloat3(0.0f, 1.0f, 0.0f);  // +Y
            vectors[3] = MakeFloat3(0.0f, -1.0f, 0.0f); // -Y
            
            // Compute normalized sum of all vectors
            Float3 sum = MakeFloat3(0.0f, 0.0f, 0.0f);
            For(0, 4, [&](Int& i) {
                sum = sum + vectors[i];
            });
            
            auto out = output.Bind();
            out[id] = sum;  // Should be (0, 0, 0)
        }, 4);

        kernel.Dispatch(1, true);
        
        std::vector<Vec3> result(4);
        output.Download(result.data(), 4);
        
        // All elements should be approximately (0, 0, 0)
        bool correct = true;
        for (int i = 0; i < 4; i++) {
            if (result[i].Length() > 0.0001f) {
                correct = false;
                std::cout << "Mismatch at " << i << ": got (" 
                          << result[i].x << ", " << result[i].y << ", " << result[i].z << ")\n";
                break;
            }
        }
        
        ASSERT(correct);
        std::cout << "Vec3 VarArray verified!";
END_TEST

// =============================================================================
// Test 8: Multiple VarArrays in same kernel
// =============================================================================
TEST(multiple_vararrays)
        Buffer<float> output(5, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            // Two local arrays
            VarArray<float, 5> arr1;
            VarArray<float, 5> arr2;
            
            // Initialize first array
            For(0, 5, [&](Int& i) {
                arr1[i] = Expr<float>(i);
            });
            
            // Copy and square into second array
            For(0, 5, [&](Int& i) {
                Float val = arr1[i];
                arr2[i] = val * val;
            });
            
            auto out = output.Bind();
            out[id] = arr2[id];
        }, 5);

        kernel.Dispatch(1, true);
        
        std::vector<float> result(5);
        output.Download(result.data(), 5);
        
        // Expected: 0^2, 1^2, 2^2, 3^2, 4^2 = 0, 1, 4, 9, 16
        float expected[] = {0, 1, 4, 9, 16};
        bool correct = true;
        for (int i = 0; i < 5; i++) {
            if (std::abs(result[i] - expected[i]) > 0.0001f) {
                correct = false;
                std::cout << "Mismatch at " << i << ": got " << result[i] 
                          << ", expected " << expected[i] << "\n";
                break;
            }
        }
        
        ASSERT(correct);
        std::cout << "Multiple VarArrays verified!";
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "EasyGPU VarArray Test Suite\n";
    std::cout << "========================================\n";

    test_basic_vararray();
    test_vararray_from_cpu_array();
    test_vararray_dynamic_indexing();
    test_vararray_stencil_computation();
    test_vararray_integer();
    test_vararray_histogram();
    test_vararray_vec3();
    test_multiple_vararrays();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << pass_count << "/" << test_count << " tests passed\n";
    std::cout << "========================================\n";

    return (pass_count == test_count) ? 0 : 1;
}
