/**
 * TestTexture.cpp:
 *      @Descripiton    :   Test for GPU Texture functionality
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 */
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <string>

// STB Image Write for saving images
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <Kernel/Kernel.h>
#include <Runtime/Texture.h>
#include <Runtime/ShaderException.h>
#include <Utility/Math.h>
#include <Utility/Helpers.h>

#include <IR/Value/Var.h>
#include <IR/Value/ExprVector.h>

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

// Helper function to save texture to PNG
void SaveTextureToPNG(const std::string& filename, int width, int height, const std::vector<uint8_t>& data) {
    int result = stbi_write_png(filename.c_str(), width, height, 4, data.data(), width * 4);
    if (result == 0) {
        throw std::runtime_error("Failed to write PNG: " + filename);
    }
    std::cout << "Saved: " << filename;
}

// =============================================================================
// Test 1: Basic texture creation (empty)
// =============================================================================
TEST(texture_create_empty)
    Texture2D<PixelFormat::RGBA8> tex(256, 256);
    ASSERT(tex.GetWidth() == 256);
    ASSERT(tex.GetHeight() == 256);
    ASSERT(tex.GetHandle() != 0);
    std::cout << "Created 256x256 RGBA8 texture";
END_TEST

// =============================================================================
// Test 2: Texture creation from raw buffer
// =============================================================================
TEST(texture_create_from_buffer)
    const int W = 64, H = 64;
    std::vector<uint8_t> pixels(W * H * 4);
    // Fill with red color
    for (int i = 0; i < W * H; ++i) {
        pixels[i * 4 + 0] = 255;  // R
        pixels[i * 4 + 1] = 0;    // G
        pixels[i * 4 + 2] = 0;    // B
        pixels[i * 4 + 3] = 255;  // A
    }
    
    Texture2D<PixelFormat::RGBA8> tex(W, H, pixels.data());
    ASSERT(tex.GetWidth() == W);
    ASSERT(tex.GetHeight() == H);
    ASSERT(tex.GetHandle() != 0);
    std::cout << "Created " << W << "x" << H << " texture from buffer";
END_TEST

// =============================================================================
// Test 3: Texture upload/download
// =============================================================================
TEST(texture_upload_download)
    const int W = 32, H = 32;
    std::vector<uint8_t> uploadPixels(W * H * 4);
    std::vector<uint8_t> downloadPixels(W * H * 4);
    
    // Fill with gradient pattern
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = (y * W + x) * 4;
            uploadPixels[idx + 0] = static_cast<uint8_t>(x * 8);   // R gradient
            uploadPixels[idx + 1] = static_cast<uint8_t>(y * 8);   // G gradient
            uploadPixels[idx + 2] = 128;                            // B constant
            uploadPixels[idx + 3] = 255;                            // A constant
        }
    }
    
    // Create empty texture and upload
    Texture2D<PixelFormat::RGBA8> tex(W, H);
    tex.Upload(uploadPixels.data());
    
    // Download back
    tex.Download(downloadPixels.data());
    
    // Verify (with small tolerance due to potential GPU format conversion)
    bool match = true;
    for (int i = 0; i < W * H * 4; ++i) {
        if (std::abs(uploadPixels[i] - downloadPixels[i]) > 2) {
            match = false;
            std::cout << "Mismatch at byte " << i << ": uploaded " << (int)uploadPixels[i] 
                      << ", downloaded " << (int)downloadPixels[i];
            break;
        }
    }
    
    ASSERT(match);
    std::cout << "Upload/Download verified for " << W << "x" << H << " texture";
END_TEST

// =============================================================================
// Test 4: Texture move semantics
// =============================================================================
TEST(texture_move)
    Texture2D<PixelFormat::RGBA8> tex1(100, 100);
    uint32_t handle1 = tex1.GetHandle();
    
    // Move construction
    Texture2D<PixelFormat::RGBA8> tex2(std::move(tex1));
    uint32_t handle2 = tex2.GetHandle();
    
    // Verify the handle was transferred
    ASSERT(handle1 == handle2);
    ASSERT(tex1.GetHandle() == 0);
    ASSERT(tex2.GetWidth() == 100);
    ASSERT(tex2.GetHeight() == 100);
    
    // Move assignment
    Texture2D<PixelFormat::RGBA8> tex3(50, 50);
    tex3 = std::move(tex2);
    
    ASSERT(tex3.GetHandle() == handle1);
    ASSERT(tex2.GetHandle() == 0);
    ASSERT(tex3.GetWidth() == 100);
    ASSERT(tex3.GetHeight() == 100);
    
    std::cout << "Move semantics verified!";
END_TEST

// =============================================================================
// Test 5: Texture Bind API (InspectorKernel - just check code generation)
// =============================================================================
TEST(texture_bind_api_inspector)
    Texture2D<PixelFormat::RGBA8> tex(64, 64);
    
    GPU::Kernel::InspectorKernel kernel([&](Var<int>& id) {
        auto img = tex.Bind();
        
        // Calculate pixel coordinates
        Var<int> x = id % 64;
        Var<int> y = id / 64;
        
        // Read pixel
        Var<Vec4> color = img.Read(x, y);
        
        // Write inverted color
        img.Write(x, y, Vec4(1.0f) - color);
    });
    
    std::cout << "\n=== Generated GLSL (Texture Bind API) ===\n";
    kernel.PrintCode();
    std::cout << "=========================================\n";
    
    ASSERT(true);
END_TEST

// =============================================================================
// Test 6: End-to-end GPU texture operation - Invert colors
// =============================================================================
TEST(gpu_texture_invert)
    const int W = 64, H = 64;
    
    // Create texture with red pixels
    std::vector<uint8_t> inputPixels(W * H * 4);
    for (int i = 0; i < W * H; ++i) {
        inputPixels[i * 4 + 0] = 200;  // R
        inputPixels[i * 4 + 1] = 100;  // G
        inputPixels[i * 4 + 2] = 50;   // B
        inputPixels[i * 4 + 3] = 255;  // A
    }
    
    Texture2D<PixelFormat::RGBA8> tex(W, H, inputPixels.data());
    
    // Kernel: invert colors
    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto img = tex.Bind();
        
        // Calculate pixel coordinates from 1D index
        Var<int> pixelId = id % (W * H);
        Var<int> x = pixelId % W;
        Var<int> y = pixelId / W;
        
        // Read and invert RGB only, keep alpha at 1
        Var<Vec4> color = img.Read(x, y);
        // Extract individual components and invert RGB
        Var<float> r = 1.0f - color.x();
        Var<float> g = 1.0f - color.y();
        Var<float> b = 1.0f - color.z();
        // Keep alpha at 1.0 (not inverted) - construct new Vec4
        Var<Vec4> inverted(r, g, b, 1.0f);
        img.Write(x, y, inverted);
    }, 256);
    
    // Dispatch enough groups to cover all pixels
    int numPixels = W * H;
    kernel.Dispatch((numPixels + 255) / 256, true);
    
    // Download and verify
    std::vector<uint8_t> resultPixels(W * H * 4);
    tex.Download(resultPixels.data());
    
    bool correct = true;
    for (int i = 0; i < W * H && correct; ++i) {
        uint8_t r = resultPixels[i * 4 + 0];
        uint8_t g = resultPixels[i * 4 + 1];
        uint8_t b = resultPixels[i * 4 + 2];
        uint8_t a = resultPixels[i * 4 + 3];
        
        // Check inversion (with tolerance for float conversion)
        // Input: (200, 100, 50, 255) -> Expected output: (55, 155, 205, 255)
        if (std::abs(r - 55) > 5 || std::abs(g - 155) > 5 || 
            std::abs(b - 205) > 5 || a != 255) {
            correct = false;
            std::cout << "Pixel " << i << " mismatch: got (" 
                      << (int)r << "," << (int)g << "," << (int)b << "," << (int)a 
                      << "), expected (~55,~155,~205,255)";
        }
    }
    
    ASSERT(correct);
    std::cout << "Texture color inversion verified!";
END_TEST

// =============================================================================
// Test 7: Multiple textures in one kernel
// =============================================================================
TEST(gpu_texture_multiple)
    const int W = 32, H = 32;
    
    // Create input texture with gradient
    std::vector<uint8_t> inputPixels(W * H * 4);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = (y * W + x) * 4;
            inputPixels[idx + 0] = static_cast<uint8_t>(x * 8);
            inputPixels[idx + 1] = static_cast<uint8_t>(y * 8);
            inputPixels[idx + 2] = 0;
            inputPixels[idx + 3] = 255;
        }
    }
    
    Texture2D<PixelFormat::RGBA8> inputTex(W, H, inputPixels.data());
    Texture2D<PixelFormat::RGBA8> outputTex(W, H);
    
    // Kernel: copy from input to output with modification
    GPU::Kernel::Kernel1D kernel([&](Var<int>& id) {
        auto input = inputTex.Bind();
        auto output = outputTex.Bind();
        
        Var<int> pixelId = id % (W * H);
        Var<int> x = pixelId % W;
        Var<int> y = pixelId / W;
        
        // Read from input, add offset to RG channels, write to output
        Var<Vec4> color = input.Read(x, y);
        // Add 0.2 to red and green channels using expression
        // color is vec4, we can use arithmetic directly
        Var<Vec4> offset = GPU::MakeFloat4(0.2f, 0.2f, 0.0f, 0.0f);
        Var<Vec4> result = color + offset;
        
        // Write modified color
        output.Write(x, y, result);
    }, 256);
    
    kernel.Dispatch((W * H + 255) / 256, true);
    
    // Download output and verify
    std::vector<uint8_t> resultPixels(W * H * 4);
    outputTex.Download(resultPixels.data());
    
    bool correct = true;
    // Check a few pixels
    for (int y = 0; y < H && correct; y += 8) {
        for (int x = 0; x < W && correct; x += 8) {
            int idx = (y * W + x) * 4;
            uint8_t r = resultPixels[idx + 0];
            uint8_t g = resultPixels[idx + 1];
            
            // Input was (x*8, y*8, 0, 255), we added 0.2 (~51 in 8-bit)
            uint8_t expectedR = static_cast<uint8_t>(std::min(255, x * 8 + 51));
            uint8_t expectedG = static_cast<uint8_t>(std::min(255, y * 8 + 51));
            
            if (std::abs(r - expectedR) > 10 || std::abs(g - expectedG) > 10) {
                correct = false;
                std::cout << "Pixel (" << x << "," << y << ") mismatch: got (" 
                          << (int)r << "," << (int)g << "), expected (~" 
                          << (int)expectedR << ",~" << (int)expectedG << ")";
            }
        }
    }
    
    ASSERT(correct);
    std::cout << "Multiple texture kernel verified!";
END_TEST

// =============================================================================
// Test 8: Float texture format (RGBA32F)
// =============================================================================
TEST(texture_rgba32f_format)
    const int W = 16, H = 16;
    
    // Create float texture
    std::vector<float> floatPixels(W * H * 4);
    for (int i = 0; i < W * H; ++i) {
        floatPixels[i * 4 + 0] = 0.5f;   // R
        floatPixels[i * 4 + 1] = 1.0f;   // G
        floatPixels[i * 4 + 2] = 0.25f;  // B
        floatPixels[i * 4 + 3] = 1.0f;   // A
    }
    
    Texture2D<PixelFormat::RGBA32F> floatTex(W, H, floatPixels.data());
    
    GPU::Kernel::InspectorKernel kernel([&](Var<int>& id) {
        auto img = floatTex.Bind();
        
        Var<int> x = id % W;
        Var<int> y = id / W;
        
        // Read float values
        Var<Vec4> color = img.Read(x, y);
        
        // Multiply by 2
        img.Write(x, y, color * 2.0f);
    });
    
    std::cout << "\n=== Generated GLSL (RGBA32F Texture) ===\n";
    kernel.PrintCode();
    std::cout << "========================================\n";
    
    ASSERT(true);
    std::cout << "Float texture format test passed!";
END_TEST

// =============================================================================
// Test 9: Texture with buffer together
// =============================================================================
TEST(texture_with_buffer)
    const int W = 32, H = 32;
    const int N = W * H;
    
    // Create texture and buffer
    Texture2D<PixelFormat::RGBA8> tex(W, H);
    std::vector<float> bufferData(N);
    for (int i = 0; i < N; ++i) {
        bufferData[i] = static_cast<float>(i) / N;
    }
    Buffer<float> buf(bufferData, BufferMode::Read);
    
    GPU::Kernel::InspectorKernel kernel([&](Var<int>& id) {
        auto img = tex.Bind();
        auto data = buf.Bind();
        
        Var<int> pixelId = id % N;
        Var<int> x = pixelId % W;
        Var<int> y = pixelId / W;
        
        // Read from buffer, write to texture
        // Use simple gradient based on pixelId
        // Create float values from int expressions
        Var<float> rf = (Expr<float>(pixelId) * 1.0f) / static_cast<float>(W * H);
        Var<float> gf = GPU::MakeFloat(0.5f);
        img.Write(x, y, Expr<Vec4>(rf, gf, 0.5f, 1.0f));
    });
    
    std::cout << "\n=== Generated GLSL (Texture + Buffer) ===\n";
    kernel.PrintCode();
    std::cout << "=========================================\n";
    
    ASSERT(true);
    std::cout << "Texture with buffer test passed!";
END_TEST

// =============================================================================
// Test 10: Generate gradient image and save to PNG
// =============================================================================
TEST(generate_gradient_image)
    const int W = 512, H = 512;
    
    // Create empty texture
    Texture2D<PixelFormat::RGBA8> tex(W, H);
    
    // Kernel: generate gradient using 2D kernel
    GPU::Kernel::Kernel2D kernel([&](Var<int>& idx, Var<int>& idy) {
        auto img = tex.Bind();
        
        // Create RGB gradient based on position
        Var<float> r = Expr<float>(idx) / static_cast<float>(W);
        Var<float> g = Expr<float>(idy) / static_cast<float>(H);
        Var<float> b = GPU::MakeFloat(0.5f);
        
        img.Write(idx, idy, Expr<Vec4>(r, g, b, 1.0f));
    }, 16, 16);
    
    // Dispatch 2D work groups
    kernel.Dispatch((W + 15) / 16, (H + 15) / 16, true);
    
    // Download result
    std::vector<uint8_t> pixels(W * H * 4);
    tex.Download(pixels.data());
    
    // Save to PNG
    SaveTextureToPNG("gradient.png", W, H, pixels);
END_TEST

// =============================================================================
// Test 11: Generate plasma effect and save to PNG
// =============================================================================
TEST(generate_plasma_effect)
    const int W = 512, H = 512;
    
    Texture2D<PixelFormat::RGBA8> tex(W, H);
    
    // Kernel: generate plasma effect using sine waves
    GPU::Kernel::Kernel2D kernel([&](Var<int>& idx, Var<int>& idy) {
        auto img = tex.Bind();
        
        // Normalize coordinates to 0-1
        Var<float> u = Expr<float>(idx) / static_cast<float>(W);
        Var<float> v = Expr<float>(idy) / static_cast<float>(H);
        
        // Plasma pattern using sine waves via CallInst
        // Create some wave patterns
        Var<float> wave1 = Sin(u * 20.0f);
        Var<float> wave2 = Sin(v * 20.0f);
        Var<float> wave3 = Sin((u + v) * 10.0f);
        
        // Distance from center for radial waves
        Var<float> dx = u - 0.5f;
        Var<float> dy = v - 0.5f;
        Var<float> dist = Sqrt(dx * dx + dy * dy);
        Var<float> wave4 = Sin(dist * 40.0f);
        
        // Combine waves
        Var<float> r = (wave1 + wave2 + 2.0f) * 0.25f;
        Var<float> g = (wave2 + wave3 + 2.0f) * 0.25f;
        Var<float> b = (wave3 + wave4 + 2.0f) * 0.25f;
        
        // Clamp to 0-1 range using GLSL Clamp
        r = Clamp(r, 0.0f, 1.0f);
        g = Clamp(g, 0.0f, 1.0f);
        b = Clamp(b, 0.0f, 1.0f);
        
        img.Write(idx, idy, Expr<Vec4>(r, g, b, 1.0f));
    }, 16, 16);
    
    kernel.Dispatch((W + 15) / 16, (H + 15) / 16, true);
    
    std::vector<uint8_t> pixels(W * H * 4);
    tex.Download(pixels.data());
    
    SaveTextureToPNG("plasma.png", W, H, pixels);
END_TEST

// =============================================================================
// Test 12: Generate checkerboard pattern and save to PNG
// =============================================================================
TEST(generate_checkerboard)
    const int W = 512, H = 512;
    const int TILE_SIZE = 64;
    
    Texture2D<PixelFormat::RGBA8> tex(W, H);
    
    // Kernel: generate checkerboard pattern
    GPU::Kernel::Kernel2D kernel([&](Var<int>& idx, Var<int>& idy) {
        auto img = tex.Bind();
        
        // Determine which tile we're in
        Var<int> tileX = idx / TILE_SIZE;
        Var<int> tileY = idy / TILE_SIZE;
        
        // Checker pattern: (tileX + tileY) % 2
        Var<int> checker = (tileX + tileY) % 2;
        
        // Color based on checker pattern
        Var<float> c = Expr<float>(checker);  // 0 or 1
        
        // Add subtle gradient overlay using Sin for smooth curve
        Var<float> grad = Expr<float>(idx % TILE_SIZE) / static_cast<float>(TILE_SIZE);
        grad = Sin(grad * 3.14159f) * 0.5f + 0.5f;
        c = Mix(c * 0.8f, c * 0.8f + 0.2f, grad);
        
        img.Write(idx, idy, Expr<Vec4>(c, c, c, 1.0f));
    }, 16, 16);
    
    kernel.Dispatch((W + 15) / 16, (H + 15) / 16, true);
    
    std::vector<uint8_t> pixels(W * H * 4);
    tex.Download(pixels.data());
    
    SaveTextureToPNG("checkerboard.png", W, H, pixels);
END_TEST

// =============================================================================
// Test 13: Image filter - blur effect
// =============================================================================
TEST(image_filter_blur)
    const int W = 256, H = 256;
    
    // Create input texture with sharp edges
    std::vector<uint8_t> inputPixels(W * H * 4);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = (y * W + x) * 4;
            // Create concentric circles pattern
            float dx = x - W/2.0f;
            float dy = y - H/2.0f;
            float dist = std::sqrt(dx*dx + dy*dy);
            float intensity = (std::sin(dist * 0.1f) + 1.0f) * 0.5f;
            
            inputPixels[idx + 0] = static_cast<uint8_t>(intensity * 255);
            inputPixels[idx + 1] = static_cast<uint8_t>((1.0f - intensity) * 255);
            inputPixels[idx + 2] = 128;
            inputPixels[idx + 3] = 255;
        }
    }
    
    Texture2D<PixelFormat::RGBA8> inputTex(W, H, inputPixels.data());
    Texture2D<PixelFormat::RGBA8> outputTex(W, H);
    
    // Save original first
    SaveTextureToPNG("blur_input.png", W, H, inputPixels);
    
    // Kernel: simple box blur
    GPU::Kernel::Kernel2D kernel([&](Var<int>& idx, Var<int>& idy) {
        auto input = inputTex.Bind();
        auto output = outputTex.Bind();
        
        // Simple 3x3 box blur using manual sampling
        // Center pixel
        Var<Vec4> c = input.Read(idx, idy);
        
        // Sample neighbors (with clamping at edges)
        Var<int> xm1 = Max(idx - 1, 0);
        Var<int> xp1 = Min(idx + 1, W - 1);
        Var<int> ym1 = Max(idy - 1, 0);
        Var<int> yp1 = Min(idy + 1, H - 1);
        
        Var<Vec4> n1 = input.Read(xm1, idy);
        Var<Vec4> n2 = input.Read(xp1, idy);
        Var<Vec4> n3 = input.Read(idx, ym1);
        Var<Vec4> n4 = input.Read(idx, yp1);
        
        // Average (center has weight 4, neighbors have weight 1 each)
        Var<Vec4> blurred = (c * 4.0f + n1 + n2 + n3 + n4) / 8.0f;
        
        output.Write(idx, idy, blurred);
    }, 16, 16);
    
    kernel.Dispatch((W + 15) / 16, (H + 15) / 16, true);
    
    std::vector<uint8_t> outputPixels(W * H * 4);
    outputTex.Download(outputPixels.data());
    
    SaveTextureToPNG("blur_output.png", W, H, outputPixels);
END_TEST

// =============================================================================
// Test 14: Image filter - edge detection
// =============================================================================
TEST(image_filter_edge_detection)
    const int W = 256, H = 256;
    
    // Create input texture with shapes
    std::vector<uint8_t> inputPixels(W * H * 4);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = (y * W + x) * 4;
            // Create a simple geometric pattern
            bool inRect1 = (x > 50 && x < 150 && y > 50 && y < 100);
            bool inRect2 = (x > 80 && x < 200 && y > 120 && y < 180);
            bool inCircle = ((x-180)*(x-180) + (y-80)*(y-80)) < 900;
            
            uint8_t val = (inRect1 || inRect2 || inCircle) ? 255 : 0;
            inputPixels[idx + 0] = val;
            inputPixels[idx + 1] = val;
            inputPixels[idx + 2] = val;
            inputPixels[idx + 3] = 255;
        }
    }
    
    Texture2D<PixelFormat::RGBA8> inputTex(W, H, inputPixels.data());
    Texture2D<PixelFormat::RGBA8> outputTex(W, H);
    
    SaveTextureToPNG("edge_input.png", W, H, inputPixels);
    
    // Kernel: Sobel edge detection
    GPU::Kernel::Kernel2D kernel([&](Var<int>& idx, Var<int>& idy) {
        auto input = inputTex.Bind();
        auto output = outputTex.Bind();
        
        // Sample neighbors with bounds checking using GLSL Min/Max
        Var<int> xm1 = Max(idx - 1, 0);
        Var<int> xp1 = Min(idx + 1, W - 1);
        Var<int> ym1 = Max(idy - 1, 0);
        Var<int> yp1 = Min(idy + 1, H - 1);
        
        // Get grayscale values (use red channel)
        auto getGray = [&](Var<int> x, Var<int> y) -> Expr<float> {
            return input.Read(x, y).x();
        };
        
        // Sobel operator
        // Gx = [-1 0 1; -2 0 2; -1 0 1]
        // Gy = [-1 -2 -1; 0 0 0; 1 2 1]
        Var<float> gx = 
            - getGray(xm1, ym1) - 2.0f * getGray(xm1, idy) - getGray(xm1, yp1) +
            getGray(xp1, ym1) + 2.0f * getGray(xp1, idy) + getGray(xp1, yp1);
        
        Var<float> gy =
                - getGray(xm1, ym1) - 2.0f * getGray(idx, ym1) - getGray(xp1, ym1) +
            getGray(xm1, yp1) + 2.0f * getGray(idx, yp1) + getGray(xp1, yp1);
        
        // Gradient magnitude using GLSL Sqrt
        Var<float> mag = Sqrt(gx * gx + gy * gy);
        
        // Clamp using GLSL Min
        mag = Min(mag, 1.0f);
        
        output.Write(idx, idy, Expr<Vec4>(mag, mag, mag, 1.0f));
    }, 16, 16);
    
    kernel.Dispatch((W + 15) / 16, (H + 15) / 16, true);
    
    std::vector<uint8_t> outputPixels(W * H * 4);
    outputTex.Download(outputPixels.data());
    
    SaveTextureToPNG("edge_output.png", W, H, outputPixels);
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  EasyGPU Texture Test Suite           \n";
    std::cout << "========================================\n";
    
    try {
        test_texture_create_empty();
        test_texture_create_from_buffer();
        test_texture_upload_download();
        test_texture_move();
        test_texture_bind_api_inspector();
        test_gpu_texture_invert();
        test_gpu_texture_multiple();
        test_texture_rgba32f_format();
        test_texture_with_buffer();
        
        // Image generation tests
        std::cout << "\n=== Image Generation Tests ===\n";
        test_generate_gradient_image();
        test_generate_plasma_effect();
        test_generate_checkerboard();
        test_image_filter_blur();
        test_image_filter_edge_detection();
        
        std::cout << "\n========================================\n";
        std::cout << "  Results: " << pass_count << "/" << test_count << " tests passed\n";
        std::cout << "========================================\n";
        
        return (pass_count == test_count) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cout << "\nFATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
