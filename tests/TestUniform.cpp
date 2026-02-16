/**
 * TestUniform.cpp
 *      Test cases for Uniform API
 *      @author EasyGPU
 *      @date 2026-02-16
 */
#include <GPU.h>
#include <iostream>
#include <vector>
#include <cmath>

// =============================================================================
// Test helpers
// =============================================================================
static int g_testsPassed = 0;
static int g_testsFailed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at line " << __LINE__ << std::endl; \
            g_testsFailed++; \
        } else { \
            g_testsPassed++; \
        } \
    } while(0)

// =============================================================================
// Test 1: Basic Uniform<int>
// =============================================================================
void TestUniformInt() {
    std::cout << "Testing Uniform<int>..." << std::endl;
    
    Uniform<int> offset;
    offset = 100;
    
    std::vector<int> input(64);
    std::vector<int> output(64);
    for (int i = 0; i < 64; i++) {
        input[i] = i;
        output[i] = 0;
    }
    
    Buffer<int> gpuInput(input);
    Buffer<int> gpuOutput(64);
    
    Kernel1D kernel([&](Int i) {
        auto in = gpuInput.Bind();
        auto out = gpuOutput.Bind();
        auto off = offset.Load();  // Load uniform
        out[i] = in[i] + off;
    });
    
    kernel.Dispatch(1, true);
    
    gpuOutput.Download(output);
    
    bool pass = true;
    for (int i = 0; i < 64; i++) {
        if (output[i] != input[i] + 100) {
            pass = false;
            break;
        }
    }
    TEST_ASSERT(pass, "Uniform<int> basic test");
    
    std::cout << "  Uniform<int> basic test: " << (pass ? "PASSED" : "FAILED") << std::endl;
}

// =============================================================================
// Test 2: Basic Uniform<float>
// =============================================================================
void TestUniformFloat() {
    std::cout << "Testing Uniform<float>..." << std::endl;
    
    Uniform<float> scale;
    scale = 2.5f;
    
    std::vector<float> input(64);
    std::vector<float> output(64);
    for (int i = 0; i < 64; i++) {
        input[i] = static_cast<float>(i) * 0.5f;
        output[i] = 0.0f;
    }
    
    Buffer<float> gpuInput(input);
    Buffer<float> gpuOutput(64);
    
    Kernel1D kernel([&](Int i) {
        auto in = gpuInput.Bind();
        auto out = gpuOutput.Bind();
        auto s = scale.Load();  // Load uniform
        out[i] = in[i] * s;
    });
    
    kernel.Dispatch(1, true);
    
    gpuOutput.Download(output);
    
    bool pass = true;
    for (int i = 0; i < 64; i++) {
        float expected = input[i] * 2.5f;
        if (std::abs(output[i] - expected) > 0.001f) {
            pass = false;
            break;
        }
    }
    TEST_ASSERT(pass, "Uniform<float> basic test");
    
    std::cout << "  Uniform<float> basic test: " << (pass ? "PASSED" : "FAILED") << std::endl;
}

// =============================================================================
// Test 3: Multiple Uniforms
// =============================================================================
void TestMultipleUniforms() {
    std::cout << "Testing multiple uniforms..." << std::endl;
    
    Uniform<int> offset;
    Uniform<float> scale;
    Uniform<int> threshold;
    
    offset = 10;
    scale = 0.5f;
    threshold = 20;
    
    std::vector<int> input(64);
    std::vector<int> output(64);
    for (int i = 0; i < 64; i++) {
        input[i] = i;
        output[i] = 0;
    }
    
    Buffer<int> gpuInput(input);
    Buffer<int> gpuOutput(64);
    
    Kernel1D kernel([&](Int i) {
        auto in = gpuInput.Bind();
        auto out = gpuOutput.Bind();
        auto off = offset.Load();
        auto s = scale.Load();
        auto thresh = threshold.Load();
        
        Int val = in[i] + off;
        If(val > thresh, [&]() {
            out[i] = ToInt(ToFloat(val) * s);
        }).Else([&]() {
            out[i] = val;
        });
    });
    
    kernel.Dispatch(1, true);
    
    gpuOutput.Download(output);
    
    bool pass = true;
    for (int i = 0; i < 64; i++) {
        int val = input[i] + 10;
        int expected;
        if (val > 20) {
            expected = static_cast<int>(static_cast<float>(val) * 0.5f);
        } else {
            expected = val;
        }
        if (output[i] != expected) {
            pass = false;
            break;
        }
    }
    TEST_ASSERT(pass, "Multiple uniforms test");
    
    std::cout << "  Multiple uniforms test: " << (pass ? "PASSED" : "FAILED") << std::endl;
}

// =============================================================================
// Test 4: Uniform value change between dispatches
// =============================================================================
void TestUniformValueChange() {
    std::cout << "Testing uniform value change..." << std::endl;
    
    Uniform<int> multiplier;
    multiplier = 2;
    
    std::vector<int> input(64);
    std::vector<int> output(64);
    for (int i = 0; i < 64; i++) {
        input[i] = i;
        output[i] = 0;
    }
    
    Buffer<int> gpuInput(input);
    Buffer<int> gpuOutput(64);
    
    Kernel1D kernel([&](Int i) {
        auto in = gpuInput.Bind();
        auto out = gpuOutput.Bind();
        auto m = multiplier.Load();
        out[i] = in[i] * m;
    });
    
    // First dispatch with multiplier = 2
    kernel.Dispatch(1, true);
    gpuOutput.Download(output);
    
    bool pass1 = true;
    for (int i = 0; i < 64; i++) {
        if (output[i] != input[i] * 2) {
            pass1 = false;
            break;
        }
    }
    TEST_ASSERT(pass1, "First dispatch with multiplier=2");
    std::cout << "  First dispatch (x2): " << (pass1 ? "PASSED" : "FAILED") << std::endl;
    
    // Change uniform value
    multiplier = 5;
    
    // Second dispatch with multiplier = 5
    kernel.Dispatch(1, true);
    gpuOutput.Download(output);
    
    bool pass2 = true;
    for (int i = 0; i < 64; i++) {
        if (output[i] != input[i] * 5) {
            pass2 = false;
            break;
        }
    }
    TEST_ASSERT(pass2, "Second dispatch with multiplier=5");
    std::cout << "  Second dispatch (x5): " << (pass2 ? "PASSED" : "FAILED") << std::endl;
}

// =============================================================================
// Test 5: Uniform with constructor initialization
// =============================================================================
void TestUniformConstructorInit() {
    std::cout << "Testing uniform constructor initialization..." << std::endl;
    
    Uniform<float> factor(3.0f);
    
    std::vector<float> input(64);
    std::vector<float> output(64);
    for (int i = 0; i < 64; i++) {
        input[i] = static_cast<float>(i);
        output[i] = 0.0f;
    }
    
    Buffer<float> gpuInput(input);
    Buffer<float> gpuOutput(64);
    
    Kernel1D kernel([&](Int i) {
        auto in = gpuInput.Bind();
        auto out = gpuOutput.Bind();
        auto f = factor.Load();
        out[i] = in[i] * f;
    });
    
    kernel.Dispatch(1, true);
    
    gpuOutput.Download(output);
    
    bool pass = true;
    for (int i = 0; i < 64; i++) {
        float expected = input[i] * 3.0f;
        if (std::abs(output[i] - expected) > 0.001f) {
            pass = false;
            break;
        }
    }
    TEST_ASSERT(pass, "Uniform constructor initialization test");
    
    std::cout << "  Constructor init test: " << (pass ? "PASSED" : "FAILED") << std::endl;
}

// =============================================================================
// Test 6: Uniform<bool>
// =============================================================================
void TestUniformBool() {
    std::cout << "Testing Uniform<bool>..." << std::endl;
    
    Uniform<bool> condition;
    condition = true;
    
    std::vector<int> input(64);
    std::vector<int> output(64);
    for (int i = 0; i < 64; i++) {
        input[i] = i;
        output[i] = 0;
    }
    
    Buffer<int> gpuInput(input);
    Buffer<int> gpuOutput(64);
    
    Kernel1D kernel([&](Int i) {
        auto in = gpuInput.Bind();
        auto out = gpuOutput.Bind();
        auto cond = condition.Load();
        If(cond, [&]() {
            out[i] = in[i] * 2;
        }).Else([&]() {
            out[i] = in[i];
        });
    });
    
    // First dispatch with condition = true
    kernel.Dispatch(1, true);
    gpuOutput.Download(output);
    
    bool pass1 = true;
    for (int i = 0; i < 64; i++) {
        if (output[i] != input[i] * 2) {
            pass1 = false;
            break;
        }
    }
    TEST_ASSERT(pass1, "Uniform<bool> with true value");
    std::cout << "  Bool=true test: " << (pass1 ? "PASSED" : "FAILED") << std::endl;
    
    // Change to false
    condition = false;
    
    kernel.Dispatch(1, true);
    gpuOutput.Download(output);
    
    bool pass2 = true;
    for (int i = 0; i < 64; i++) {
        if (output[i] != input[i]) {
            pass2 = false;
            break;
        }
    }
    TEST_ASSERT(pass2, "Uniform<bool> with false value");
    std::cout << "  Bool=false test: " << (pass2 ? "PASSED" : "FAILED") << std::endl;
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Uniform API Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    try {
        TestUniformInt();
        std::cout << std::endl;
        
        TestUniformFloat();
        std::cout << std::endl;
        
        TestMultipleUniforms();
        std::cout << std::endl;
        
        TestUniformValueChange();
        std::cout << std::endl;
        
        TestUniformConstructorInit();
        std::cout << std::endl;
        
        TestUniformBool();
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Passed: " << g_testsPassed << std::endl;
    std::cout << "  Failed: " << g_testsFailed << std::endl;
    std::cout << "========================================" << std::endl;
    
    return g_testsFailed > 0 ? 1 : 0;
}
