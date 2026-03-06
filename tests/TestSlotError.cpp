/**
 * TestSlotError.cpp:
 *      @Descripiton    :   Error handling tests for Slots
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/6/2026
 */
#include <GPU.h>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Slot Error Handling Tests ===" << std::endl;
    int testsPassed = 0;
    int testsTotal = 0;
    
    // ==================================================================
    // Test 1: Dispatch without attaching BufferSlot
    // ==================================================================
    {
        std::cout << "\n[Test 1] Dispatch without Attach (should throw)..." << std::flush;
        testsTotal++;
        
        BufferSlot<float> slot;
        
        Kernel1D kernel([&](Int i) {
            auto buf = slot.Bind();
            buf[i] = buf[i] * 2.0f;
        });
        
        bool caught = false;
        try {
            // slot is not attached - should throw
            kernel.Dispatch(1, true);
        } catch (const std::exception& e) {
            caught = true;
            std::cout << " (caught: " << e.what() << ")" << std::flush;
        }
        
        if (caught) {
            std::cout << " PASS" << std::endl;
            testsPassed++;
        } else {
            std::cout << " FAIL (no exception thrown)" << std::endl;
        }
    }
    
    // ==================================================================
    // Test 2: Dispatch without attaching TextureSlot
    // ==================================================================
    {
        std::cout << "[Test 2] TextureSlot without Attach (should throw)..." << std::flush;
        testsTotal++;
        
        TextureSlot<PixelFormat::R32F> slot;
        
        Kernel2D kernel([&](Int x, Int y) {
            auto tex = slot.Bind();
            tex.Write(x, y, MakeFloat4(1.0f, 0.0f, 0.0f, 1.0f));
        });
        
        bool caught = false;
        try {
            kernel.Dispatch(1, 1, true);
        } catch (const std::exception& e) {
            caught = true;
            std::cout << " (caught: " << e.what() << ")" << std::flush;
        }
        
        if (caught) {
            std::cout << " PASS" << std::endl;
            testsPassed++;
        } else {
            std::cout << " FAIL (no exception thrown)" << std::endl;
        }
    }
    
    // ==================================================================
    // Test 3: Partial attachment (one slot attached, one not)
    // ==================================================================
    {
        std::cout << "[Test 4] Partial attachment (should throw)..." << std::flush;
        testsTotal++;
        
        BufferSlot<float> slotA;
        BufferSlot<float> slotB;
        
        Kernel1D kernel([&](Int i) {
            auto a = slotA.Bind();
            auto b = slotB.Bind();
            b[i] = a[i];
        });
        
        // Only attach slotA
        std::vector<float> data(8, 1.0f);
        Buffer<float> buf(data);
        slotA.Attach(buf);
        // slotB NOT attached
        
        bool caught = false;
        try {
            kernel.Dispatch(1, true);
        } catch (const std::exception& e) {
            caught = true;
            std::cout << " (caught)" << std::flush;
        }
        
        if (caught) {
            std::cout << " PASS" << std::endl;
            testsPassed++;
        } else {
            std::cout << " FAIL (no exception thrown)" << std::endl;
        }
    }
    
    // ==================================================================
    // Test 5: Attach then Detach before dispatch
    // ==================================================================
    {
        std::cout << "[Test 5] Attach then Detach (should throw)..." << std::flush;
        testsTotal++;
        
        BufferSlot<float> slot;
        
        Kernel1D kernel([&](Int i) {
            auto buf = slot.Bind();
            buf[i] = 1.0f;
        });
        
        std::vector<float> data(8, 0.0f);
        Buffer<float> buf(data);
        
        slot.Attach(buf);
        slot.Detach();  // Detach before dispatch
        
        bool caught = false;
        try {
            kernel.Dispatch(1, true);
        } catch (const std::exception& e) {
            caught = true;
            std::cout << " (caught)" << std::flush;
        }
        
        if (caught) {
            std::cout << " PASS" << std::endl;
            testsPassed++;
        } else {
            std::cout << " FAIL (no exception thrown)" << std::endl;
        }
    }
    
    // ==================================================================
    // Test 6: Successful run after fixing attachment issue
    // ==================================================================
    {
        std::cout << "[Test 6] Recovery after fixing attachment..." << std::flush;
        testsTotal++;
        
        BufferSlot<float> slot;
        
        Kernel1D kernel([&](Int i) {
            auto buf = slot.Bind();
            buf[i] = buf[i] + 5.0f;
        });
        
        std::vector<float> data(8, 10.0f);
        Buffer<float> buf(data);
        std::vector<float> result(8);
        
        // First try without attachment (should fail)
        bool firstFailed = false;
        try {
            kernel.Dispatch(1, true);
        } catch (...) {
            firstFailed = true;
        }
        
        // Now attach and retry
        slot.Attach(buf);
        kernel.Dispatch(1, true);
        buf.Download(result.data());
        
        bool pass = firstFailed && (result[0] == 15.0f);
        
        if (pass) {
            std::cout << " PASS" << std::endl;
            testsPassed++;
        } else {
            std::cout << " FAIL (firstFailed=" << firstFailed << ", result=" << result[0] << ")" << std::endl;
        }
    }
    
    // ==================================================================
    // Test 7: Slot state verification
    // ==================================================================
    {
        std::cout << "[Test 7] Slot state verification..." << std::flush;
        testsTotal++;
        
        BufferSlot<float> bufSlot;
        TextureSlot<PixelFormat::R32F> texSlot;
        
        // All should be initially detached
        bool check1 = !bufSlot.IsAttached() && !texSlot.IsAttached();
        
        // Check handles are 0 when detached
        bool check2 = (bufSlot.GetHandle() == 0) && (texSlot.GetHandle() == 0);
        
        std::vector<float> data(8, 1.0f);
        Buffer<float> buf(data);
        TextureR32F tex(8, 8, data.data());
        
        bufSlot.Attach(buf);
        texSlot.Attach(tex);
        
        // All should be attached now
        bool check3 = bufSlot.IsAttached() && texSlot.IsAttached();
        
        // Handles should match
        bool check4 = (bufSlot.GetHandle() == buf.GetHandle()) &&
                      (texSlot.GetHandle() == tex.GetHandle());
        
        if (check1 && check2 && check3 && check4) {
            std::cout << " PASS" << std::endl;
            testsPassed++;
        } else {
            std::cout << " FAIL (" << check1 << check2 << check3 << check4 << ")" << std::endl;
        }
    }
    
    // ==================================================================
    // Summary
    // ==================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Results: " << testsPassed << "/" << testsTotal << " passed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (testsPassed == testsTotal) ? 0 : 1;
}
