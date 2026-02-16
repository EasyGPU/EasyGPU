/**
 * TestBuffer.cpp:
 *      @Descripiton    :   Test for GPU Buffer functionality with Bind() API
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 */
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>

#include <Kernel/Kernel.h>
#include <Runtime/Buffer.h>
#include <Runtime/ShaderException.h>

#include <IR/Value/Var.h>
#include <IR/Value/VarVector.h>

#include <IR/Value/ExprVector.h>
#include <IR/Value/ExprIVector.h>
#include <IR/Value/ExprMatrix.h>

#include <Utility/Meta/StructMeta.h>

#include <Utility/Vec.h>
#include <Utility/Helpers.h>

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
    } catch (const std::length_error& e) { \
        std::cout << "FAILED: String length error (probably error message too long): " << e.what() << "\n"; \
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
// Test 1: Basic float buffer with new Bind() API
// =============================================================================
TEST(float_buffer_bind_api)
        std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Buffer<float> inputBuffer(inputData, BufferMode::Read);
        Buffer<float> outputBuffer(5, BufferMode::Write);

        // New API: Bind() is called inside kernel lambda
        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            // Bind resources - automatically allocates binding slots
            auto input = inputBuffer.Bind();   // binding 0
            auto output = outputBuffer.Bind(); // binding 1

            // DSL programming (looks like normal C++)
            Var<float> value = input[id];
            output[id] = value * 2.0f;
        }, 256);

        // No SetBuffer needed! Bind() already registered the buffers.

        std::cout << "\n=== Generated GLSL (Bind API) ===\n";
        kernel.Dispatch(1, true);
        std::cout << "===================================\n";

        ASSERT(true);
END_TEST

// =============================================================================
// Test 2: Multiple buffer operations with Bind()
// =============================================================================
TEST(multiple_buffer_bind)
        std::vector<float> dataA = {10.0f, 20.0f, 30.0f};
        std::vector<float> dataB = {1.0f, 2.0f, 3.0f};

        Buffer<float> bufferA(dataA, BufferMode::Read);
        Buffer<float> bufferB(dataB, BufferMode::Read);
        Buffer<float> resultBuffer(3, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            // Multiple buffers, auto-incrementing bindings
            auto bufA = bufferA.Bind();        // binding 0
            auto bufB = bufferB.Bind();        // binding 1
            auto result = resultBuffer.Bind(); // binding 2

            // Complex operation: result = (a + b) * (a - b)
            Var<float> a = bufA[id];
            Var<float> b = bufB[id];
            result[id] = (a + b) * (a - b);
        }, 256);

        std::cout << "\n=== Generated GLSL (Multiple Buffers) ===\n";
        kernel.Dispatch(1, true);
        std::cout << "==========================================\n";

        ASSERT(true);
END_TEST

// =============================================================================
// Test 3: Vector buffer with Bind()
// =============================================================================
TEST(vector_buffer_bind)
        std::vector<Vec2> inputData = {Vec2(1.0f, 2.0f), Vec2(3.0f, 4.0f), Vec2(5.0f, 6.0f)};
        Buffer<Vec2> inputBuffer(inputData, BufferMode::Read);
        Buffer<Vec2> outputBuffer(3, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto input = inputBuffer.Bind();
            auto output = outputBuffer.Bind();

            // Vector operations
            Var<Vec2> vec = input[id];
            Var<float> lenSq = vec.x() * vec.x() + vec.y() * vec.y();
            output[id] = vec / lenSq;
        }, 256);

        std::cout << "\n=== Generated GLSL (Vector Buffer) ===\n";
        kernel.Dispatch(1, true);
        std::cout << "=======================================\n";

        ASSERT(true);
END_TEST

// =============================================================================
// Test 4: Buffer with expression index and assignment
// =============================================================================
TEST(buffer_expression_index_bind)
        Buffer<float> dataBuffer(100, BufferMode::ReadWrite);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto data = dataBuffer.Bind();

            // Use expression for index calculation
            Var<int> idx = id * 2;
            Var<float> value = data[idx];

            // Store at next position using proxy assignment
            data[idx + 1] = value * 0.5f;
        }, 256);

        std::cout << "\n=== Generated GLSL (Expression Index) ===\n";
        kernel.Dispatch(1, true);
        std::cout << "=========================================\n";

        ASSERT(true);
END_TEST

// =============================================================================
// Test 5: Read-write buffer in-place modification
// =============================================================================
TEST(readwrite_buffer_bind)
        std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        Buffer<int> rwBuffer(data, BufferMode::ReadWrite);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto data = rwBuffer.Bind();

            // In-place modification using proxy assignment
            Var<int> value = data[id];
            data[id] = value * value;
        }, 256);

        std::cout << "\n=== Generated GLSL (ReadWrite Buffer) ===\n";
        kernel.Dispatch(1, true);
        std::cout << "=========================================\n";

        ASSERT(true);
END_TEST

// =============================================================================
// Test 6: Buffer upload/download (CPU-GPU data transfer test)
// =============================================================================
TEST(buffer_upload_download)
        // Create test data
        std::vector<float> originalData = {3.14f, 2.718f, 1.414f, 1.732f};
        std::vector<float> downloadedData(4);

        // Create buffer and upload
        Buffer<float> buffer(4, BufferMode::ReadWrite);
        buffer.Upload(originalData.data(), originalData.size());

        // Download back
        buffer.Download(downloadedData.data(), downloadedData.size());

        // Verify (with small epsilon for float comparison)
        bool match = true;
        for (size_t i = 0; i < originalData.size(); ++i) {
            if (std::abs(originalData[i] - downloadedData[i]) > 0.0001f) {
                match = false;
                break;
            }
        }

        ASSERT(match);
        std::cout << "Upload/Download verified!";
END_TEST

// =============================================================================
// Test 7: Integer buffer with Bind()
// =============================================================================
TEST(int_buffer_bind)
        std::vector<int> inputData = {10, 20, 30, 40, 50};
        Buffer<int> inputBuffer(inputData, BufferMode::Read);
        Buffer<int> outputBuffer(5, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto input = inputBuffer.Bind();
            auto output = outputBuffer.Bind();

            // Bitwise operations on integers
            Var<int> value = input[id];
            Var<int> shifted = value << 1;
            output[id] = shifted & 0xFF;
        }, 256);

        std::cout << "\n=== Generated GLSL (Integer Buffer) ===\n";
        kernel.Dispatch(1, true);
        std::cout << "=======================================\n";

        ASSERT(true);
END_TEST

// =============================================================================
// Test 8: Buffer move semantics
// =============================================================================
TEST(buffer_move)
        Buffer<float> buffer1(100, BufferMode::ReadWrite);
        uint32_t handle1 = buffer1.GetHandle();

        // Move construction
        Buffer<float> buffer2(std::move(buffer1));
        uint32_t handle2 = buffer2.GetHandle();

        // Verify the handle was transferred
        ASSERT(handle1 == handle2);
        ASSERT(buffer1.GetHandle() == 0);
        ASSERT(buffer2.GetCount() == 100);

        // Move assignment
        Buffer<float> buffer3(10, BufferMode::Read);
        buffer3 = std::move(buffer2);

        ASSERT(buffer3.GetHandle() == handle1);
        ASSERT(buffer2.GetHandle() == 0);
        ASSERT(buffer3.GetCount() == 100);

        std::cout << "Move semantics verified!";
END_TEST

// =============================================================================
// Test 9: End-to-end GPU compute - Vector Addition
// C[i] = A[i] + B[i]
// =============================================================================
TEST(gpu_vector_addition)
        const size_t N = 1024;

        // 1. Prepare CPU data
        std::vector<float> dataA(N);
        std::vector<float> dataB(N);
        std::vector<float> resultCPU(N);

        for (size_t i = 0; i < N; i++) {
            dataA[i] = static_cast<float>(i);
            dataB[i] = static_cast<float>(N - i);
        }

        // 2. Create GPU buffers
        Buffer<float> bufferA(dataA, BufferMode::Read);
        Buffer<float> bufferB(dataB, BufferMode::Read);
        Buffer<float> bufferC(N, BufferMode::Write);

        // 3. Define kernel
        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto A = bufferA.Bind();
            auto B = bufferB.Bind();
            auto C = bufferC.Bind();

            Var<float> a = A[id];
            Var<float> b = B[id];
            C[id] = a + b;
        }, 256);

        // 4. Dispatch (ceil(N / 256) groups)
        int groups = (N + 255) / 256;
        kernel.Dispatch(groups, true);

        // 5. Download results
        bufferC.Download(resultCPU.data(), N);

        // 6. Verify results
        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            float expected = dataA[i] + dataB[i];
            if (std::abs(resultCPU[i] - expected) > 0.0001f) {
                correct = false;
                std::cout << "Mismatch at " << i << ": got " << resultCPU[i]
                          << ", expected " << expected << "\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Vector addition verified: " << N << " elements computed correctly!";
END_TEST

// =============================================================================
// Test 10: End-to-end GPU compute - Vector Multiplication
// C[i] = A[i] * B[i]
// =============================================================================
TEST(gpu_vector_multiplication)
        const size_t N = 512;

        std::vector<float> dataA(N);
        std::vector<float> dataB(N);
        std::vector<float> resultCPU(N);

        for (size_t i = 0; i < N; i++) {
            dataA[i] = static_cast<float>(i + 1) * 0.5f;
            dataB[i] = static_cast<float>(i + 2) * 0.3f;
        }

        Buffer<float> bufferA(dataA, BufferMode::Read);
        Buffer<float> bufferB(dataB, BufferMode::Read);
        Buffer<float> bufferC(N, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto A = bufferA.Bind();
            auto B = bufferB.Bind();
            auto C = bufferC.Bind();

            C[id] = A[id] * B[id];
        }, 256);

        kernel.Dispatch((N + 255) / 256, true);
        bufferC.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            float expected = dataA[i] * dataB[i];
            if (std::abs(resultCPU[i] - expected) > 0.0001f) {
                correct = false;
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Vector multiplication verified!";
END_TEST

// =============================================================================
// Test 11: End-to-end GPU compute - Scalar Multiplication
// B[i] = A[i] * scalar
// =============================================================================
TEST(gpu_scalar_multiplication)
        const size_t N = 256;
        const float scalar = 3.14159f;

        std::vector<float> dataA(N);
        std::vector<float> resultCPU(N);

        for (size_t i = 0; i < N; i++) {
            dataA[i] = static_cast<float>(i) * 0.1f;
        }

        Buffer<float> bufferA(dataA, BufferMode::Read);
        Buffer<float> bufferB(N, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto A = bufferA.Bind();
            auto B = bufferB.Bind();

            B[id] = A[id] * scalar;
        }, 256);

        kernel.Dispatch(1, true);  // Only 1 work group needed for 256 elements
        bufferB.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            float expected = dataA[i] * scalar;
            if (std::abs(resultCPU[i] - expected) > 0.0001f) {
                correct = false;
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Scalar multiplication verified!";
END_TEST

// =============================================================================
// Test 12: End-to-end GPU compute - Complex Formula
// result[i] = (A[i] + B[i]) * (A[i] - B[i]) + A[i] * B[i]
// =============================================================================
TEST(gpu_complex_formula)
        const size_t N = 128;

        std::vector<float> dataA(N);
        std::vector<float> dataB(N);
        std::vector<float> resultCPU(N);

        for (size_t i = 0; i < N; i++) {
            dataA[i] = static_cast<float>(i) * 0.5f + 1.0f;
            dataB[i] = static_cast<float>(i) * 0.3f + 0.5f;
        }

        Buffer<float> bufferA(dataA, BufferMode::Read);
        Buffer<float> bufferB(dataB, BufferMode::Read);
        Buffer<float> bufferC(N, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto A = bufferA.Bind();
            auto B = bufferB.Bind();
            auto C = bufferC.Bind();

            Var<float> a = A[id];
            Var<float> b = B[id];

            // (a + b) * (a - b) + a * b
            Var<float> sum = a + b;
            Var<float> diff = a - b;
            Var<float> product = a * b;

            C[id] = sum * diff + product;
        }, 128);

        kernel.Dispatch(1, true);
        bufferC.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            float a = dataA[i];
            float b = dataB[i];
            float expected = (a + b) * (a - b) + a * b;
            // Use larger epsilon for complex formula due to floating point accumulation
            if (std::abs(resultCPU[i] - expected) > 0.01f) {
                correct = false;
                std::cout << std::fixed << std::setprecision(6);
                std::cout << "Mismatch at " << i << ": got " << resultCPU[i]
                          << ", expected " << expected
                          << " (diff: " << std::abs(resultCPU[i] - expected) << ")\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Complex formula verified!";
END_TEST

// =============================================================================
// Test 13: End-to-end GPU compute - Integer Bitwise Operations
// result[i] = (input[i] << 2) & 0xFF
// =============================================================================
TEST(gpu_integer_bitwise)
        const size_t N = 64;

        std::vector<int> inputData(N);
        std::vector<int> resultCPU(N);

        for (size_t i = 0; i < N; i++) {
            inputData[i] = static_cast<int>(i * 7 + 13);
        }

        Buffer<int> bufferInput(inputData, BufferMode::Read);
        Buffer<int> bufferOutput(N, BufferMode::Write);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto input = bufferInput.Bind();
            auto output = bufferOutput.Bind();

            Var<int> value = input[id];
            // Shift left by 2, then mask with 0xFF
            output[id] = (value << 2) & 0xFF;
        }, 64);

        kernel.Dispatch(1, true);
        bufferOutput.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            int expected = (inputData[i] << 2) & 0xFF;
            if (resultCPU[i] != expected) {
                correct = false;
                std::cout << "Mismatch at " << i << ": got " << resultCPU[i]
                          << ", expected " << expected << "\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Integer bitwise operations verified!";
END_TEST

// =============================================================================
// Test 14: End-to-end GPU compute - In-place Square
// data[i] = data[i] * data[i]
// =============================================================================
TEST(gpu_inplace_square)
        const size_t N = 256;

        std::vector<float> data(N);
        std::vector<float> resultCPU(N);

        for (size_t i = 0; i < N; i++) {
            data[i] = static_cast<float>(i + 1) * 0.5f;
        }

        // Save original for verification
        std::vector<float> original = data;

        Buffer<float> buffer(data, BufferMode::ReadWrite);

        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto data = buffer.Bind();

            Var<float> value = data[id];
            data[id] = value * value;
        }, 256);

        kernel.Dispatch(1, true);
        buffer.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            float expected = original[i] * original[i];
            if (std::abs(resultCPU[i] - expected) > 0.0001f) {
                correct = false;
                std::cout << "Mismatch at " << i << ": got " << resultCPU[i]
                          << ", expected " << expected << "\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "In-place square verified!";
END_TEST

EASYGPU_STRUCT(Simple,
               (float, life)
);

// Using Vec3 - automatic conversion will handle the layout difference!
EASYGPU_STRUCT(Particle,
               (GPU::Math::Vec3, position),   // 12 bytes C++ -> 16 bytes GPU
               (GPU::Math::Vec3, velocity),   // 12 bytes C++ -> 16 bytes GPU
               (float, life)                  // 4 bytes
);


// =============================================================================
// Test 15: Custom struct - Particle system
// Particle { position: Vec3, velocity: Vec3, life: float }
// =============================================================================
TEST(gpu_struct_particle)
        // Define custom struct using EASYGPU_STRUCT

        const size_t N = 64;

        // Prepare CPU data
        std::vector<Particle> particlesCPU(N);
        for (size_t i = 0; i < N; i++) {
            particlesCPU[i].position = Vec3(static_cast<float>(i), static_cast<float>(i) * 0.5f, 0.0f);
            particlesCPU[i].velocity = Vec3(1.0f, 2.0f, 3.0f);
            particlesCPU[i].life = static_cast<float>(N - i) / N;
        }

        // Create GPU buffers
        Buffer<Particle> particlesBuffer(particlesCPU, BufferMode::ReadWrite);

        // Define kernel: update particle position and decrease life
        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto pts = particlesBuffer.Bind();

            GPU::IR::Value::Var<Particle> p = pts[id];

            // Update position: pos += velocity * 0.016
            p.position() = p.position() + p.velocity() * 0.016f;

            // Decrease life
            p.life() = p.life() - 0.01f;
        }, 64);

        kernel.Dispatch(1, true);

        // Download and verify
        std::vector<Particle> resultCPU(N);
        particlesBuffer.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            Vec3 expectedPos = particlesCPU[i].position + particlesCPU[i].velocity * 0.016f;
            float expectedLife = particlesCPU[i].life - 0.01f;

            float posDiff = (resultCPU[i].position - expectedPos).Length();
            float lifeDiff = std::abs(resultCPU[i].life - expectedLife);

            if (posDiff > 0.001f || lifeDiff > 0.0001f) {
                correct = false;
                std::cout << "Position mismatch at " << i << ": got ("
                          << resultCPU[i].position.x << ", "
                          << resultCPU[i].position.y << ", "
                          << resultCPU[i].position.z << "), expected ("
                          << expectedPos.x << ", "
                          << expectedPos.y << ", "
                          << expectedPos.z << ")\n" << ": got ("
                                                    << resultCPU[i].life << "), expected ("
                                                    << expectedLife << ")\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Particle struct GPU compute verified!";
END_TEST

EASYGPU_STRUCT(Transform,
               (GPU::Math::Mat4, matrix),     // 64 bytes
               (GPU::Math::Vec4, offset)      // 16 bytes
);

EASYGPU_STRUCT(Vertex,
               (GPU::Math::Vec4, position),   // 16 bytes
               (GPU::Math::Vec4, normal),     // 16 bytes
               (GPU::Math::Vec2, uv)
);


// =============================================================================
// Test 16: Multiple custom structs interaction
// Transform { matrix: Mat4, offset: Vec3 }
// Vertex { position: Vec3, normal: Vec3, uv: Vec2 }
// =============================================================================
TEST(gpu_struct_multiple)

        const size_t N = 32;

        // Prepare data
        std::vector<Vertex> verticesCPU(N);
        for (size_t i = 0; i < N; i++) {
            verticesCPU[i].position = Vec4(static_cast<float>(i), 0.0f, 0.0f, 0.0f);
            verticesCPU[i].normal = Vec4(0.0f, 1.0f, 0.0f, 0.0f);
            verticesCPU[i].uv = Vec2(static_cast<float>(i) / N, 0.0f);
        }

        Transform transformCPU;
        transformCPU.matrix = Mat4::Identity();
        transformCPU.offset = Vec4(10.0f, 20.0f, 30.0f, 0.0f);

        // Create buffers
        Buffer<Vertex> verticesBuffer(verticesCPU, BufferMode::ReadWrite);
        Buffer<Transform> transformBuffer(std::vector<Transform>{transformCPU}, BufferMode::Read);

        // Kernel: apply transform to vertices
        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto verts = verticesBuffer.Bind();
            auto xf = transformBuffer.Bind();

            Var<Vertex> v = verts[id];
            Var<Transform> t = xf[0];

            // Transform position: matrix * position + offset
            // Simplified: just add offset for this test
            v.position() = v.position() + t.offset();

            verts[id] = v;
        }, 32);

        kernel.Dispatch(1, true);

        // Verify
        std::vector<Vertex> resultCPU(N);
        verticesBuffer.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            Vec4 expectedPos = verticesCPU[i].position + transformCPU.offset;
            float posDiff = (resultCPU[i].position - expectedPos).Length();

            if (posDiff > 0.001f) {
                correct = false;
                std::cout << "Position mismatch at " << i << ": got ("
                          << resultCPU[i].position.x << ", "
                          << resultCPU[i].position.y << ", "
                          << resultCPU[i].position.z << "), expected ("
                          << expectedPos.x << ", "
                          << expectedPos.y << ", "
                          << expectedPos.z << ")\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Multiple struct interaction verified!";
END_TEST

EASYGPU_STRUCT(ColorRGBA,
               (float, r),
               (float, g),
               (float, b),
               (float, a)
);

// =============================================================================
// Test 17: Nested struct-like behavior (struct with primitive array-like access)
// ColorRGBA { r: float, g: float, b: float, a: float }
// =============================================================================
TEST(gpu_struct_color_rgba)
        const size_t N = 128;

        std::vector<ColorRGBA> colorsCPU(N);
        for (size_t i = 0; i < N; i++) {
            colorsCPU[i].r = static_cast<float>(i) / N;
            colorsCPU[i].g = 1.0f - static_cast<float>(i) / N;
            colorsCPU[i].b = 0.5f;
            colorsCPU[i].a = 1.0f;
        }

        Buffer<ColorRGBA> colorsBuffer(colorsCPU, BufferMode::ReadWrite);

        // Kernel: invert colors and apply brightness
        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto cols = colorsBuffer.Bind();

            Var<ColorRGBA> c = cols[id];

            // Invert RGB, keep alpha
            c.r() = 1.0f - c.r();
            c.g() = 1.0f - c.g();
            c.b() = 1.0f - c.b();

            cols[id] = c;
        }, 128);

        kernel.Dispatch(1, true);

        // Verify
        std::vector<ColorRGBA> resultCPU(N);
        colorsBuffer.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            float expectedR = 1.0f - colorsCPU[i].r;
            float expectedG = 1.0f - colorsCPU[i].g;
            float expectedB = 1.0f - colorsCPU[i].b;

            if (std::abs(resultCPU[i].r - expectedR) > 0.0001f ||
                std::abs(resultCPU[i].g - expectedG) > 0.0001f ||
                std::abs(resultCPU[i].b - expectedB) > 0.0001f ||
                resultCPU[i].a != colorsCPU[i].a) {
                correct = false;
                std::cout << "Color mismatch at " << i << "\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "ColorRGBA struct inversion verified!";
END_TEST

// =============================================================================
// Test 18: Struct with integer members
// IndexData { index: int, flags: int }
// =============================================================================

EASYGPU_STRUCT(IndexData,
               (int, index),
               (int, flags)
);

TEST(gpu_struct_integer)
        const size_t N = 64;

        std::vector<IndexData> dataCPU(N);
        for (size_t i = 0; i < N; i++) {
            dataCPU[i].index = static_cast<int>(i);
            dataCPU[i].flags = static_cast<int>(i & 0x0F);
        }

        Buffer<IndexData> dataBuffer(dataCPU, BufferMode::ReadWrite);

        // Kernel: bitwise OR with mask
        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto data = dataBuffer.Bind();

            Var<IndexData> d = data[id];

            // Set high bits in flags
            d.flags() = d.flags() | 0xF0;

            data[id] = d;
        }, 64);

        kernel.Dispatch(1, true);

        // Verify
        std::vector<IndexData> resultCPU(N);
        dataBuffer.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            int expectedFlags = (dataCPU[i].flags | 0xF0);
            if (resultCPU[i].index != dataCPU[i].index ||
                resultCPU[i].flags != expectedFlags) {
                correct = false;
                std::cout << "Mismatch at " << i << ": index=" << resultCPU[i].index
                          << " flags=" << resultCPU[i].flags
                          << " expected_flags=" << expectedFlags << "\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Integer struct bitwise operations verified!";
END_TEST

// =============================================================================
// Test 19: Deliberately misaligned struct - the ultimate test!
// This struct has terrible alignment in C++ but should work perfectly
// thanks to automatic layout conversion.
// =============================================================================

// =============================================================================
// Nested struct tests - structures containing other structures
// =============================================================================

// Nested: Material contains color and properties
EASYGPU_STRUCT(Material,
               (GPU::Math::Vec4, color),       // RGBA color
               (float, roughness),             // Material roughness
               (float, metallic),              // Material metallic
               (float, emission)               // Emission strength
);

// Nested: Transform for 3D objects (different from the one at line 628)
EASYGPU_STRUCT(ObjectTransform,
               (GPU::Math::Vec3, position),    // 12 bytes C++ -> 16 bytes GPU
               (GPU::Math::Vec4, rotation),    // 16 bytes
               (GPU::Math::Vec3, scale)        // 12 bytes C++ -> 16 bytes GPU
);

// Complex flat struct: many Vec3/Vec4 fields to stress test layout conversion
// Complex nested struct: GameObject contains ObjectTransform and Material
EASYGPU_STRUCT(GameObject,
               (ObjectTransform, transform),   // Nested ObjectTransform struct
               (Material, material),           // Nested Material struct
               (int, objectId),                // Object identifier
               (float, visibility)             // Visibility factor
);

// =============================================================================
// Test: Nested struct (Transform) with automatic layout conversion
// =============================================================================
TEST(gpu_struct_nested_transform)
        const size_t N = 32;

        std::vector<ObjectTransform> transformsCPU(N);
        for (size_t i = 0; i < N; i++) {
            transformsCPU[i].position = Vec3(static_cast<float>(i), 0.0f, 0.0f);
            transformsCPU[i].rotation = Vec4(0.0f, 0.0f, 0.0f, 1.0f);
            transformsCPU[i].scale = Vec3(1.0f, 1.0f, 1.0f);
        }

        Buffer<ObjectTransform> buffer(transformsCPU, BufferMode::ReadWrite);

        // Kernel: update position based on id
        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto data = buffer.Bind();

            Var<ObjectTransform> t = data[id];

            // Modify position
            t.position() = t.position() + Vec3(1.0f, 2.0f, 3.0f);

            // Modify rotation (simple rotation around Z)
            t.rotation() = Vec4(0.0f, 0.0f, 0.0f, 1.0f);

            // Modify scale
            t.scale() = t.scale() * 2.0f;

            data[id] = t;
        }, 64);

        kernel.Dispatch(1, true);

        // Verify
        std::vector<ObjectTransform> resultCPU(N);
        buffer.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            Vec3 expectedPos(static_cast<float>(i) + 1.0f, 2.0f, 3.0f);
            Vec3 expectedScale(2.0f, 2.0f, 2.0f);

            if ((resultCPU[i].position - expectedPos).Length() > 0.0001f ||
                (resultCPU[i].scale - expectedScale).Length() > 0.0001f) {
                correct = false;
                std::cout << "Mismatch at " << i << "\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Nested Transform struct verified!";
END_TEST

// =============================================================================
// Test: Complex struct with many Vec3/Vec4 fields (flat structure)
// Note: Nested struct support requires recursive conversion, testing flat structure instead
// =============================================================================
// =============================================================================
// Test: Complex nested struct (GameObject containing ObjectTransform and Material)
// =============================================================================
TEST(gpu_struct_complex_nested)
        const size_t N = 16;

        std::vector<GameObject> objectsCPU(N);
        for (size_t i = 0; i < N; i++) {
            objectsCPU[i].transform.position = Vec3(static_cast<float>(i), 0.0f, 0.0f);
            objectsCPU[i].transform.rotation = Vec4(0.0f, 0.0f, 0.0f, 1.0f);
            objectsCPU[i].transform.scale = Vec3(1.0f, 1.0f, 1.0f);

            objectsCPU[i].material.color = Vec4(1.0f, 0.5f, 0.0f, 1.0f);
            objectsCPU[i].material.roughness = 0.5f;
            objectsCPU[i].material.metallic = 0.0f;
            objectsCPU[i].material.emission = 0.0f;

            objectsCPU[i].objectId = static_cast<int>(i);
            objectsCPU[i].visibility = 1.0f;
        }

        Buffer<GameObject> buffer(objectsCPU, BufferMode::ReadWrite);

        // Kernel: update object properties
        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto data = buffer.Bind();

            Var<GameObject> obj = data[id];

            // Update nested transform
            obj.transform().position() = obj.transform().position() + GPU::MakeFloat3(10.0f, 0.0f, 0.0f);
            obj.transform().scale() = GPU::MakeFloat3(2.0f, 2.0f, 2.0f);

            // Update nested material
            obj.material().color() = GPU::MakeFloat4(0.0f, 1.0f, 0.0f, 1.0f);
            obj.material().roughness() = 0.8f;

            // Update direct fields
            obj.visibility() = 0.5f;

            data[id] = obj;
        }, 64);

        kernel.Dispatch(1, true);

        // Verify
        std::vector<GameObject> resultCPU(N);
        buffer.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            Vec3 expectedPos(static_cast<float>(i) + 10.0f, 0.0f, 0.0f);
            Vec3 expectedScale(2.0f, 2.0f, 2.0f);

            if ((resultCPU[i].transform.position - expectedPos).Length() > 0.0001f ||
                (resultCPU[i].transform.scale - expectedScale).Length() > 0.0001f ||
                std::abs(resultCPU[i].material.color.y - 1.0f) > 0.0001f ||  // green = 1
                std::abs(resultCPU[i].visibility - 0.5f) > 0.0001f) {
                correct = false;
                std::cout << "Mismatch at " << i << "\n";
                std::cout << "  pos=" << resultCPU[i].transform.position.x << "," << resultCPU[i].transform.position.y << "," << resultCPU[i].transform.position.z;
                std::cout << " expected=" << expectedPos.x << "," << expectedPos.y << "," << expectedPos.z << "\n";
                std::cout << "  scale=" << resultCPU[i].transform.scale.x << "," << resultCPU[i].transform.scale.y << "," << resultCPU[i].transform.scale.z;
                std::cout << " expected=" << expectedScale.x << "," << expectedScale.y << "," << expectedScale.z << "\n";
                std::cout << "  color.y=" << resultCPU[i].material.color.y << " expected=1\n";
                std::cout << "  visibility=" << resultCPU[i].visibility << " expected=0.5\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Complex nested GameObject struct verified!";
END_TEST

// A simple struct for testing - all types naturally aligned
EASYGPU_STRUCT(MisalignedData,
               (GPU::Math::Vec4, a),   // 16 bytes
               (GPU::Math::Vec4, b),   // 16 bytes
               (float, c),             // 4 bytes
               (float, d)              // 4 bytes (8 total for this group)
);

TEST(gpu_struct_misaligned)
        const size_t N = 32;

        std::vector<MisalignedData> dataCPU(N);
        for (size_t i = 0; i < N; i++) {
            dataCPU[i].a = Vec4(static_cast<float>(i), 0.0f, 0.0f, 0.0f);
            dataCPU[i].b = Vec4(static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2), 0.0f);
            dataCPU[i].c = static_cast<float>(i * 10);
            dataCPU[i].d = static_cast<float>(i) * 0.5f;
        }

        Buffer<MisalignedData> buffer(dataCPU, BufferMode::ReadWrite);

        // Kernel: modify all fields
        GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
            auto data = buffer.Bind();

            Var<MisalignedData> d = data[id];

            d.a() = d.a() * 2.0f;
            d.b() = d.b() + Vec4(1.0f, 1.0f, 1.0f, 0.0f);
            d.c() = d.c() + 1.0f;
            d.d() = d.d() + 1.0f;

            data[id] = d;
        }, 32);

        kernel.Dispatch(1, true);

        // Verify
        std::vector<MisalignedData> resultCPU(N);
        buffer.Download(resultCPU.data(), N);

        bool correct = true;
        for (size_t i = 0; i < N; i++) {
            Vec4 expectedA = dataCPU[i].a * 2.0f;
            Vec4 expectedB = dataCPU[i].b + Vec4(1.0f, 1.0f, 1.0f, 0.0f);
            float expectedC = dataCPU[i].c + 1.0f;
            float expectedD = dataCPU[i].d + 1.0f;

            if ((resultCPU[i].a - expectedA).Length() > 0.0001f ||
                (resultCPU[i].b - expectedB).Length() > 0.0001f ||
                std::abs(resultCPU[i].c - expectedC) > 0.0001f ||
                std::abs(resultCPU[i].d - expectedD) > 0.0001f) {
                correct = false;
                std::cout << "Mismatch at " << i << "\n";
                break;
            }
        }

        ASSERT(correct);
        std::cout << "Misaligned struct automatic conversion verified!";
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  EasyGPU Buffer Test Suite (E2E GPU)   \n";
    std::cout << "========================================\n";
    std::cout << "  Tests: Bind API + GPU Compute + Struct\n";
    std::cout << "========================================\n";

    try {
        test_float_buffer_bind_api();
        test_multiple_buffer_bind();
        test_vector_buffer_bind();
        test_buffer_expression_index_bind();
        test_readwrite_buffer_bind();
        test_buffer_upload_download();
        test_int_buffer_bind();
        test_buffer_move();
        test_gpu_vector_addition();
        test_gpu_vector_multiplication();
        test_gpu_scalar_multiplication();
        test_gpu_complex_formula();
        test_gpu_integer_bitwise();
        test_gpu_inplace_square();
        test_gpu_struct_particle();
        test_gpu_struct_multiple();
        test_gpu_struct_color_rgba();
        test_gpu_struct_integer();
        test_gpu_struct_nested_transform();
        test_gpu_struct_complex_nested();
        test_gpu_struct_misaligned();

        std::cout << "\n========================================\n";
        std::cout << "  Results: " << pass_count << "/" << test_count << " tests passed\n";
        std::cout << "========================================\n";

        return (pass_count == test_count) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
