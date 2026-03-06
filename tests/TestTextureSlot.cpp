/**
 * TestTextureSlot.cpp:
 *      @Descripiton    :   TextureSlot (2D) functionality tests
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/6/2026
 */
#include <GPU.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>

// Helper for approximate float comparison
bool FloatEq(float a, float b, float epsilon = 0.01f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    try {
        std::cout << "=== TextureSlot 2D Tests ===" << std::endl;
        int testsPassed = 0;
        int testsTotal = 0;
        
        // ==================================================================
        // Test 1: Basic Attach/Detach
        // ==================================================================
        {
            std::cout << "\n[Test 1] TextureSlot Attach/Detach..." << std::flush;
            testsTotal++;
            
            TextureSlot<PixelFormat::RGBA8> slot;
            Texture2D<PixelFormat::RGBA8> tex(64, 64);
            
            assert(!slot.IsAttached() && "Slot should not be attached initially");
            assert(slot.GetAttached() == nullptr && "GetAttached should return nullptr");
            
            slot.Attach(tex);
            assert(slot.IsAttached() && "Slot should be attached after Attach()");
            assert(slot.GetAttached() == &tex && "GetAttached should return the attached texture");
            assert(slot.GetHandle() == tex.GetHandle() && "GetHandle should match texture handle");
            
            slot.Detach();
            assert(!slot.IsAttached() && "Slot should not be attached after Detach()");
            
            std::cout << " PASS" << std::endl;
            testsPassed++;
        }
        
        // ==================================================================
        // Test 2: Basic read/write with slot
        // ==================================================================
        {
            std::cout << "[Test 2] Basic read/write..." << std::flush;
            testsTotal++;
            
            TextureSlot<PixelFormat::RGBA8> inputSlot;
            TextureSlot<PixelFormat::RGBA8> outputSlot;
            
            Kernel2D kernel([&](Int x, Int y) {
                auto in = inputSlot.Bind();
                auto out = outputSlot.Bind();
                Var<Vec4> color = in.Read(x, y);
                // Invert colors
                out.Write(x, y, MakeFloat4(1.0f - color.x(), 1.0f - color.y(), 
                                           1.0f - color.z(), color.w()));
            });
            
            // Create test pattern
            std::vector<uint8_t> inputData(64 * 64 * 4);
            for (int y = 0; y < 64; ++y) {
                for (int x = 0; x < 64; ++x) {
                    int idx = (y * 64 + x) * 4;
                    inputData[idx + 0] = static_cast<uint8_t>(x * 4);     // R
                    inputData[idx + 1] = static_cast<uint8_t>(y * 4);     // G
                    inputData[idx + 2] = 128;                             // B
                    inputData[idx + 3] = 255;                             // A
                }
            }
            
            TextureRGBA8 inputTex(64, 64, inputData.data());
            TextureRGBA8 outputTex(64, 64);
            
            inputSlot.Attach(inputTex);
            outputSlot.Attach(outputTex);
            kernel.Dispatch(4, 4, true);
            
            // Download and verify
            std::vector<uint8_t> outputData(64 * 64 * 4);
            outputTex.Download(outputData.data());
            
            // Check a few pixels
            bool pass = true;
            for (int checkY : {0, 31, 63}) {
                for (int checkX : {0, 31, 63}) {
                    int idx = (checkY * 64 + checkX) * 4;
                    uint8_t inR = static_cast<uint8_t>(checkX * 4);
                    uint8_t outR = outputData[idx + 0];
                    uint8_t expectedR = 255 - inR;
                    
                    if (std::abs(static_cast<int>(outR) - expectedR) > 2) {
                        pass = false;
                        std::cout << "\n  Mismatch at (" << checkX << "," << checkY << "): "
                                  << "R got " << (int)outR << " expected " << (int)expectedR;
                    }
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
        // Test 3: Switch textures between dispatches
        // ==================================================================
        {
            std::cout << "[Test 3] Texture switching..." << std::flush;
            testsTotal++;
            
            TextureSlot<PixelFormat::RGBA8> inputSlot;
            TextureSlot<PixelFormat::RGBA8> outputSlot;
            
            Kernel2D kernel([&](Int x, Int y) {
                auto in = inputSlot.Bind();
                auto out = outputSlot.Bind();
                Var<Vec4> color = in.Read(x, y);
                // Multiply by 0.5
                out.Write(x, y, color * 0.5f);
            });
            
            // Texture A: solid red
            std::vector<uint8_t> dataA(16 * 16 * 4, 0);
            for (int i = 0; i < 16 * 16; ++i) {
                dataA[i * 4 + 0] = 255;  // R
                dataA[i * 4 + 3] = 255;  // A
            }
            
            // Texture B: solid green
            std::vector<uint8_t> dataB(16 * 16 * 4, 0);
            for (int i = 0; i < 16 * 16; ++i) {
                dataB[i * 4 + 1] = 200;  // G
                dataB[i * 4 + 3] = 255;  // A
            }
            
            TextureRGBA8 texA(16, 16, dataA.data());
            TextureRGBA8 texB(16, 16, dataB.data());
            TextureRGBA8 outputTex(16, 16);
            
            outputSlot.Attach(outputTex);
            
            // First dispatch: process A
            inputSlot.Attach(texA);
            kernel.Dispatch(1, 1, true);
            
            std::vector<uint8_t> result(16 * 16 * 4);
            outputTex.Download(result.data());
            bool passA = (result[0] == 127 || result[0] == 128);  // 255 * 0.5 = ~127
            
            // Second dispatch: process B (no recompilation)
            inputSlot.Attach(texB);
            kernel.Dispatch(1, 1, true);
            outputTex.Download(result.data());
            bool passB = (result[1] == 100);  // 200 * 0.5 = 100
            
            if (passA && passB) {
                std::cout << " PASS" << std::endl;
                testsPassed++;
            } else {
                std::cout << " FAIL (passA=" << passA << ", passB=" << passB << ")" << std::endl;
            }
        }
        
        // ==================================================================
        // Test 4: Multiple texture slots
        // ==================================================================
        {
            std::cout << "[Test 4] Multiple texture slots..." << std::flush;
            testsTotal++;
            
            TextureSlot<PixelFormat::R32F> slotA;
            TextureSlot<PixelFormat::R32F> slotB;
            TextureSlot<PixelFormat::R32F> slotC;
            
            Kernel2D kernel([&](Int x, Int y) {
                auto a = slotA.Bind();
                auto b = slotB.Bind();
                auto c = slotC.Bind();
                Float va = a.Read(x, y).x();
                Float vb = b.Read(x, y).x();
                c.Write(x, y, MakeFloat4(va + vb, 0.0f, 0.0f, 1.0f));
            });
            
            std::vector<float> dataA(16 * 16, 10.0f);
            std::vector<float> dataB(16 * 16, 20.0f);
            
            TextureR32F texA(16, 16, dataA.data());
            TextureR32F texB(16, 16, dataB.data());
            TextureR32F texC(16, 16);
            
            slotA.Attach(texA);
            slotB.Attach(texB);
            slotC.Attach(texC);
            
            kernel.Dispatch(1, 1, true);
            
            std::vector<float> result(16 * 16 * 4);  // RGBA
            texC.Download(result.data());
            
            bool pass = FloatEq(result[0], 30.0f, 0.1f);
            
            if (pass) {
                std::cout << " PASS" << std::endl;
                testsPassed++;
            } else {
                std::cout << " FAIL (got " << result[0] << ", expected 30.0)" << std::endl;
            }
        }
        
        // ==================================================================
        // Test 5: Different pixel formats
        // ==================================================================
        {
            std::cout << "[Test 5] Different pixel formats (RGBA32F)..." << std::flush;
            testsTotal++;
            
            TextureSlot<PixelFormat::RGBA32F> slot;
            
            Kernel2D kernel([&](Int x, Int y) {
                auto tex = slot.Bind();
                Var<Vec4> color = tex.Read(x, y);
                tex.Write(x, y, color * 2.0f);
            });
            
            std::vector<float> data(16 * 16 * 4);
            for (int i = 0; i < 16 * 16; ++i) {
                data[i * 4 + 0] = 0.5f;
                data[i * 4 + 1] = 0.25f;
                data[i * 4 + 2] = 0.125f;
                data[i * 4 + 3] = 1.0f;
            }
            
            TextureRGBA32F tex(16, 16, data.data());
            slot.Attach(tex);
            kernel.Dispatch(1, 1, true);
            
            std::vector<float> result(16 * 16 * 4);
            tex.Download(result.data());
            
            bool pass = FloatEq(result[0], 1.0f) && FloatEq(result[1], 0.5f) && 
                        FloatEq(result[2], 0.25f);
            
            if (pass) {
                std::cout << " PASS" << std::endl;
                testsPassed++;
            } else {
                std::cout << " FAIL (got " << result[0] << ", " << result[1] << ", " << result[2] << ")" << std::endl;
            }
        }
        
        // ==================================================================
        // Test 6: Larger texture
        // ==================================================================
        {
            std::cout << "[Test 6] Large texture (1024x1024)..." << std::flush;
            testsTotal++;
            
            TextureSlot<PixelFormat::R32F> inputSlot;
            TextureSlot<PixelFormat::R32F> outputSlot;
            
            Kernel2D kernel([&](Int x, Int y) {
                auto in = inputSlot.Bind();
                auto out = outputSlot.Bind();
                Float v = in.Read(x, y).x();
                out.Write(x, y, MakeFloat4(v + 1.0f, 0.0f, 0.0f, 1.0f));
            });
            
            const int size = 1024;
            std::vector<float> data(size * size);
            for (int i = 0; i < size * size; ++i) data[i] = static_cast<float>(i % 100);
            
            TextureR32F inputTex(size, size, data.data());
            TextureR32F outputTex(size, size);
            
            inputSlot.Attach(inputTex);
            outputSlot.Attach(outputTex);
            
            kernel.Dispatch(64, 64, true);
            
            // For R32F format, each pixel is 1 float (not 4)
            std::vector<float> result(size * size);
            outputTex.Download(result.data());
            
            // Spot check
            bool pass = true;
            for (int checkY : {0, 512, 1023}) {
                for (int checkX : {0, 512, 1023}) {
                    int idx = checkY * size + checkX;
                    float expected = (idx % 100) + 1.0f;
                    if (!FloatEq(result[idx], expected, 0.1f)) {
                        pass = false;
                        std::cout << "\n  Mismatch at (" << checkX << "," << checkY << "): got " 
                                  << result[idx] << " expected " << expected;
                        break;
                    }
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
