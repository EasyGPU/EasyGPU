/**
 * TestBufferSlotBasic.cpp:
 *      @Descripiton    :   Basic BufferSlot functionality tests
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/6/2026
 */
#include <GPU.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Helper function for approximate equality
bool FloatEq(float a, float b, float epsilon = 0.001f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    try {
        std::cout << "=== BufferSlot Basic Tests ===" << std::endl;
        int testsPassed = 0;
        int testsTotal = 0;
        
        // ==================================================================
        // Test 1: Basic Attach/Detach functionality
        // ==================================================================
        {
            std::cout << "\n[Test 1] Attach/Detach functionality..." << std::flush;
            testsTotal++;
            
            BufferSlot<float> slot;
            std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            Buffer<float> buf(data);
            
            // Initially not attached
            assert(!slot.IsAttached() && "Slot should not be attached initially");
            assert(slot.GetAttached() == nullptr && "GetAttached should return nullptr");
            
            // After Attach
            slot.Attach(buf);
            assert(slot.IsAttached() && "Slot should be attached after Attach()");
            assert(slot.GetAttached() == &buf && "GetAttached should return the attached buffer");
            assert(slot.GetHandle() == buf.GetHandle() && "GetHandle should match buffer handle");
            
            // After Detach
            slot.Detach();
            assert(!slot.IsAttached() && "Slot should not be attached after Detach()");
            assert(slot.GetAttached() == nullptr && "GetAttached should return nullptr after Detach");
            
            std::cout << " PASS" << std::endl;
            testsPassed++;
        }
        
        // ==================================================================
        // Test 2: Basic kernel execution with slot
        // ==================================================================
        {
            std::cout << "[Test 2] Basic kernel execution..." << std::flush;
            testsTotal++;
            
            BufferSlot<float> inputSlot;
            BufferSlot<float> outputSlot;
            
            Kernel1D kernel([&](Int i) {
                auto in = inputSlot.Bind();
                auto out = outputSlot.Bind();
                out[i] = in[i] * 2.0f;
            });
            
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
            std::vector<float> result(5);
            
            Buffer<float> inputBuf(input);
            Buffer<float> outputBuf(5);
            
            inputSlot.Attach(inputBuf);
            outputSlot.Attach(outputBuf);
            kernel.Dispatch(1, true);
            
            outputBuf.Download(result);
            
            bool pass = true;
            for (size_t i = 0; i < 5; ++i) {
                if (!FloatEq(result[i], expected[i])) {
                    pass = false;
                    std::cout << "\n  Mismatch at " << i << ": got " << result[i] 
                              << ", expected " << expected[i];
                }
            }
            
            if (pass) {
                std::cout << " PASS" << std::endl;
                testsPassed++;
            } else {
                std::cout << " FAIL" << std::endl;
            }
        }
        
        // ==================================================================
        // Test 3: Switch buffers between dispatches (no recompilation)
        // ==================================================================
        {
            std::cout << "[Test 3] Buffer switching (no recompilation)..." << std::flush;
            testsTotal++;
            
            BufferSlot<float> inputSlot;
            BufferSlot<float> outputSlot;
            
            Kernel1D kernel([&](Int i) {
                auto in = inputSlot.Bind();
                auto out = outputSlot.Bind();
                out[i] = in[i] + 10.0f;
            });
            
            std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
            std::vector<float> data2 = {100.0f, 200.0f, 300.0f};
            std::vector<float> result(3);
            
            Buffer<float> buf1(data1);
            Buffer<float> buf2(data2);
            Buffer<float> outBuf(3);
            
            outputSlot.Attach(outBuf);
            
            // First dispatch with buf1
            inputSlot.Attach(buf1);
            kernel.Dispatch(1, true);
            outBuf.Download(result);
            
            bool pass1 = FloatEq(result[0], 11.0f) && FloatEq(result[1], 12.0f) && FloatEq(result[2], 13.0f);
            
            // Second dispatch with buf2 (should use cached kernel)
            inputSlot.Attach(buf2);
            kernel.Dispatch(1, true);
            outBuf.Download(result);
            
            bool pass2 = FloatEq(result[0], 110.0f) && FloatEq(result[1], 210.0f) && FloatEq(result[2], 310.0f);
            
            // Third dispatch back to buf1
            inputSlot.Attach(buf1);
            kernel.Dispatch(1, true);
            outBuf.Download(result);
            
            bool pass3 = FloatEq(result[0], 11.0f) && FloatEq(result[1], 12.0f) && FloatEq(result[2], 13.0f);
            
            if (pass1 && pass2 && pass3) {
                std::cout << " PASS" << std::endl;
                testsPassed++;
            } else {
                std::cout << " FAIL (pass1=" << pass1 << ", pass2=" << pass2 << ", pass3=" << pass3 << ")" << std::endl;
            }
        }
        
        // ==================================================================
        // Test 4: Multiple slots in same kernel
        // ==================================================================
        {
            std::cout << "[Test 4] Multiple slots in same kernel..." << std::flush;
            testsTotal++;
            
            BufferSlot<float> slotA;
            BufferSlot<float> slotB;
            BufferSlot<float> slotC;
            
            Kernel1D kernel([&](Int i) {
                auto a = slotA.Bind();
                auto b = slotB.Bind();
                auto c = slotC.Bind();
                c[i] = a[i] + b[i];
            });
            
            std::vector<float> dataA = {1.0f, 2.0f, 3.0f};
            std::vector<float> dataB = {10.0f, 20.0f, 30.0f};
            std::vector<float> result(3);
            
            Buffer<float> bufA(dataA);
            Buffer<float> bufB(dataB);
            Buffer<float> bufC(3);
            
            slotA.Attach(bufA);
            slotB.Attach(bufB);
            slotC.Attach(bufC);
            
            kernel.Dispatch(1, true);
            bufC.Download(result);
            
            bool pass = FloatEq(result[0], 11.0f) && FloatEq(result[1], 22.0f) && FloatEq(result[2], 33.0f);
            
            if (pass) {
                std::cout << " PASS" << std::endl;
                testsPassed++;
            } else {
                std::cout << " FAIL (got " << result[0] << ", " << result[1] << ", " << result[2] << ")" << std::endl;
            }
        }
        
        // ==================================================================
        // Test 5: Large data processing
        // ==================================================================
        {
            std::cout << "[Test 5] Large data processing (1M elements)..." << std::flush;
            testsTotal++;
            
            const size_t N = 1024 * 1024;
            BufferSlot<float> inputSlot;
            BufferSlot<float> outputSlot;
            
            Kernel1D kernel([&](Int i) {
                auto in = inputSlot.Bind();
                auto out = outputSlot.Bind();
                out[i] = in[i] * 3.14159f;
            });
            
            std::vector<float> input(N);
            for (size_t i = 0; i < N; ++i) input[i] = static_cast<float>(i);
            std::vector<float> result(N);
            
            Buffer<float> inputBuf(input);
            Buffer<float> outputBuf(N);
            
            inputSlot.Attach(inputBuf);
            outputSlot.Attach(outputBuf);
            
            kernel.Dispatch((N + 255) / 256, true);
            outputBuf.Download(result);
            
            bool pass = true;
            // Spot check
            for (size_t i : {0, 1, 100, 10000, N/2, N-1}) {
                if (!FloatEq(result[i], input[i] * 3.14159f, 0.01f)) {
                    pass = false;
                    std::cout << "\n  Mismatch at " << i << ": got " << result[i] 
                              << ", expected " << input[i] * 3.14159f;
                    break;
                }
            }
            
            if (pass) {
                std::cout << " PASS" << std::endl;
                testsPassed++;
            } else {
                std::cout << " FAIL" << std::endl;
            }
        }
        
        // ==================================================================
        // Test 6: Vector type BufferSlot
        // ==================================================================
        {
            std::cout << "[Test 6] Vec4 BufferSlot..." << std::flush;
            testsTotal++;
            
            BufferSlot<Vec4> inputSlot;
            BufferSlot<Vec4> outputSlot;
            
            Kernel1D kernel([&](Int i) {
                auto in = inputSlot.Bind();
                auto out = outputSlot.Bind();
                Vec4 v = in[i];
                out[i] = MakeFloat4(v.x() * 2.0f, v.y() * 2.0f, v.z() * 2.0f, v.w() * 2.0f);
            });
            
            std::vector<Vec4> input = {
                Vec4(1.0f, 2.0f, 3.0f, 4.0f),
                Vec4(5.0f, 6.0f, 7.0f, 8.0f)
            };
            std::vector<Vec4> result(2);
            
            Buffer<Vec4> inputBuf(input);
            Buffer<Vec4> outputBuf(2);
            
            inputSlot.Attach(inputBuf);
            outputSlot.Attach(outputBuf);
            kernel.Dispatch(1, true);
            outputBuf.Download(result);
            
            bool pass = FloatEq(result[0].x, 2.0f) && FloatEq(result[0].y, 4.0f) && 
                        FloatEq(result[0].z, 6.0f) && FloatEq(result[0].w, 8.0f) &&
                        FloatEq(result[1].x, 10.0f) && FloatEq(result[1].y, 12.0f);
            
            if (pass) {
                std::cout << " PASS" << std::endl;
                testsPassed++;
            } else {
                std::cout << " FAIL" << std::endl;
            }
        }
        
        // ==================================================================
        // Summary
        // ==================================================================
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test Results: " << testsPassed << "/" << testsTotal << " passed" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return (testsPassed == testsTotal) ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
