/**
 * @file TestIfBuffer.cpp
 * @brief Test: BufferSlot::Bind() and buffer writes inside If body
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <Flow/IfFlow.h>
#include <GPU.h>
#include <IR/Value/Var.h>
#include <Kernel/Kernel.h>
#include <Runtime/Buffer.h>
#include <Runtime/BufferSlot.h>

using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Flow;
using namespace GPU::Runtime;
using namespace GPU::Kernel;

#define TEST(name) \
    std::cout << "\n[TEST] " #name " ... "; \
    try {

#define END_TEST \
    std::cout << "PASSED"; \
    } catch (const std::exception &e) { \
        std::cout << "FAILED: " << e.what(); \
        throw; \
    }

int main() {
    std::cout << "=== If + Buffer/BufferSlot Tests ===\n";
    int passed = 0;

    // Test 1: Buffer::Bind outside, buffer write inside If
    TEST(buffer_outside_write_inside_if)
        std::vector<float> outputData(4, 0.0f);
        Buffer<float> outputBuffer(outputData.size(), BufferMode::Write);

        Kernel1D kernel([&](Var<int> &id) {
            auto output = outputBuffer.Bind();
            Int idx(id);

            If(idx < 2, [&]() {
                output[idx] = 1.0f;
            });
        }, 64);

        kernel.Dispatch(1, true);
        outputBuffer.Download(outputData);
        assert(std::abs(outputData[0] - 1.0f) < 0.01f);
        assert(std::abs(outputData[1] - 1.0f) < 0.01f);
        assert(std::abs(outputData[2] - 0.0f) < 0.01f);
        assert(std::abs(outputData[3] - 0.0f) < 0.01f);
        passed++;
    END_TEST

    // Test 2: BufferSlot::Bind inside If, write inside If
    TEST(bufferslot_bind_inside_if)
        std::vector<float> gradData(12, 0.0f);
        Buffer<float> gradBuffer(gradData.size(), BufferMode::ReadWrite);
        BufferSlot<float> gradSlot;
        gradSlot.Attach(gradBuffer);

        Kernel1D kernel([&](Var<int> &id) {
            Int i(id);

            If(i < 4, [&]() {
                auto posGrad = gradSlot.Bind();
                Int poff = i * 3;
                posGrad[poff + 0] = 1.0f;
                posGrad[poff + 1] = 1.0f;
                posGrad[poff + 2] = 1.0f;
            });
        }, 64);

        kernel.Dispatch(1, true);
        gradBuffer.Download(gradData);
        for (int j = 0; j < 12; j++) {
            assert(std::abs(gradData[j] - 1.0f) < 0.01f);
        }
        for (int j = 12; j < (int)gradData.size(); j++) {
            assert(std::abs(gradData[j] - 0.0f) < 0.01f);
        }
        passed++;
    END_TEST

    // Test 3: BufferSlot::Bind outside If, write inside If with computation
    TEST(bufferslot_outside_write_inside_if)
        std::vector<float> gradData(12, 0.0f);
        Buffer<float> gradBuffer(gradData.size(), BufferMode::ReadWrite);
        BufferSlot<float> gradSlot;
        gradSlot.Attach(gradBuffer);

        Kernel1D kernel([&](Var<int> &id) {
            auto posGrad = gradSlot.Bind();
            Int i(id);

            If(i < 4, [&]() {
                Int poff = i * 3;
                posGrad[poff + 0] = 1.0f;
                posGrad[poff + 1] = 1.0f;
                posGrad[poff + 2] = 1.0f;
            });
        }, 64);

        kernel.Dispatch(1, true);
        gradBuffer.Download(gradData);
        for (int j = 0; j < 12; j++) {
            assert(std::abs(gradData[j] - 1.0f) < 0.01f);
        }
        passed++;
    END_TEST

    // Test 4: If-Else with BufferSlot inside both branches
    TEST(if_else_bufferslot_both_branches)
        std::vector<float> data(8, 0.0f);
        Buffer<float> buffer(data.size(), BufferMode::ReadWrite);
        BufferSlot<float> slot;
        slot.Attach(buffer);

        Kernel1D kernel([&](Var<int> &id) {
            Int i(id);

            If(i < 4, [&]() {
                auto buf = slot.Bind();
                buf[i] = 1.0f;
            }).Else([&]() {
                auto buf = slot.Bind();
                buf[i] = 2.0f;
            });
        }, 64);

        kernel.Dispatch(1, true);
        buffer.Download(data);
        for (int j = 0; j < 4; j++) {
            assert(std::abs(data[j] - 1.0f) < 0.01f);
        }
        for (int j = 4; j < 8; j++) {
            assert(std::abs(data[j] - 2.0f) < 0.01f);
        }
        passed++;
    END_TEST

    std::cout << "\n\n=== " << passed << "/4 tests passed ===\n";
    return (passed == 4) ? 0 : 1;
}
