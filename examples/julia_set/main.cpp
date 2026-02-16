/**
 * JuliaSet:
 *      @Descripiton    :   Julia set fractal renderer
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/15/2026
 */
#include <Kernel/Kernel.h>
#include <Callable/Callable.h>
#include <IR/Value/Var.h>
#include <IR/Value/BufferRef.h>
#include <Runtime/Buffer.h>
#include <Flow/ForFlow.h>
#include <Flow/IfFlow.h>
#include <Flow/BreakFlow.h>
#include <Flow/ReturnFlow.h>
#include <Utility/Vec.h>
#include <Utility/Math.h>
#include <Utility/Helpers.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Runtime;
using namespace GPU::Callables;
using namespace GPU::Flow;

// =============================================================================
// Configuration
// =============================================================================
constexpr int IMAGE_WIDTH = 1024;
constexpr int IMAGE_HEIGHT = 1024;
constexpr int MAX_ITERATIONS = 256;

// Viewport settings
constexpr float CENTER_X = 0.0f;
constexpr float CENTER_Y = 0.0f;
constexpr float ZOOM = 1.5f;

// Julia set parameter c = cx + i*cy
// Classic values that produce interesting patterns:
// (-0.8, 0.156) - Douady rabbit
// (-0.4, 0.6) - Siegel disk
// (0.285, 0.01) - San Marco fractal
// (-0.70176, -0.3842) - Dragon
constexpr float JULIA_CX = -0.8f;
constexpr float JULIA_CY = 0.156f;

// =============================================================================
// Color mapping function - Smooth gradient palette
// =============================================================================
Callable<Vec3(int)> GetColor = [](Int& iter) {
    Float3 color;
    If(iter == MAX_ITERATIONS, [&]() {
        // Inside set - deep navy/black
        color = MakeFloat3(0.02f, 0.02f, 0.05f);
    }).Else([&]() {
        // Outside set - smooth gradient from cyan to blue to purple
        Float t = ToFloat(iter) / MakeFloat(MAX_ITERATIONS);
        
        // Smooth color gradient using cosine palette
        Float freq = MakeFloat(4.71239f); // 1.5 * PI for broader color range
        
        // Cyan to Blue to Purple gradient (cool tones)
        Float r = 0.2f + 0.6f * Sin(freq * t + 4.0f);
        Float g = 0.3f + 0.5f * Sin(freq * t + 2.0f);
        Float b = 0.6f + 0.4f * Sin(freq * t + 0.0f);
        
        // Boost blue channel for ocean-like aesthetic
        r = Clamp(r, 0.0f, 1.0f);
        g = Clamp(g, 0.0f, 1.0f);
        b = Clamp(b * 1.1f, 0.0f, 1.0f);
        
        color = MakeFloat3(r, g, b);
    });
    
    Return(color);
};

// =============================================================================
// Julia set iteration function
// =============================================================================
Callable<int(float, float)> Julia = [](Float& zx, Float& zy) {
    // Julia set: z = z^2 + c, where c is constant
    // z^2 = (zx + i*zy)^2 = zx^2 - zy^2 + 2*i*zx*zy
    
    Int iter = MakeInt(0);
    
    For(0, MAX_ITERATIONS, [&](Int& i) {
        Float zx2 = zx * zx;
        Float zy2 = zy * zy;
        
        // Check if escaped: |z|^2 > 4.0 (i.e., |z| > 2.0)
        If(zx2 + zy2 > 4.0f, [&]() {
            iter = i;
            Break();
        });
        
        // z = z^2 + c
        Float new_zy = 2.0f * zx * zy + JULIA_CY;
        zx = zx2 - zy2 + JULIA_CX;
        zy = new_zy;
        
        iter = i;
    });
    
    Return(iter);
};

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "Julia Set Fractal Renderer\n";
    std::cout << "Parameter c = " << JULIA_CX << " + " << JULIA_CY << "i\n";
    std::cout << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << " @ " << MAX_ITERATIONS << " iterations\n\n";
    
    // Create output image buffer
    Buffer<Vec4> image(IMAGE_WIDTH * IMAGE_HEIGHT, BufferMode::Write);
    
    std::cout << "Rendering...\n";
    
    // Calculate aspect ratio and scale
    float aspectRatio = static_cast<float>(IMAGE_WIDTH) / static_cast<float>(IMAGE_HEIGHT);
    float scaleX = ZOOM * aspectRatio;
    float scaleY = ZOOM;
    
    Kernel::Kernel2D kernel([&](Int& px, Int& py) {
        auto img = image.Bind();
        
        // Map pixel to complex plane for initial z
        // x ranges from [CENTER_X - scaleX, CENTER_X + scaleX]
        // y ranges from [CENTER_Y - scaleY, CENTER_Y + scaleY]
        Float u = (ToFloat(px) + 0.5f) / IMAGE_WIDTH;
        Float v = (ToFloat(py) + 0.5f) / IMAGE_HEIGHT;
        
        Float zx = CENTER_X + (u * 2.0f - 1.0f) * scaleX;
        Float zy = CENTER_Y + (v * 2.0f - 1.0f) * scaleY;
        
        // Compute Julia set iterations
        Int iter = Julia(zx, zy);
        
        // Get color
        Float3 col = GetColor(iter);
        
        // Store to image
        Int idx = py * IMAGE_WIDTH + px;
        img[idx] = MakeFloat4(col.x(), col.y(), col.z(), 1.0f);
    });
    
    kernel.Dispatch((IMAGE_WIDTH + 15) / 16, (IMAGE_HEIGHT + 15) / 16, true);
    
    // Download and convert to 8-bit RGB
    std::vector<Vec4> pixels;
    image.Download(pixels);
    
    std::vector<unsigned char> imageData(IMAGE_WIDTH * IMAGE_HEIGHT * 3);
    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            int srcIdx = (IMAGE_HEIGHT - 1 - y) * IMAGE_WIDTH + x;
            int dstIdx = (y * IMAGE_WIDTH + x) * 3;
            const auto& p = pixels[srcIdx];
            imageData[dstIdx + 0] = static_cast<unsigned char>(256 * std::clamp(p.x, 0.0f, 0.999f));
            imageData[dstIdx + 1] = static_cast<unsigned char>(256 * std::clamp(p.y, 0.0f, 0.999f));
            imageData[dstIdx + 2] = static_cast<unsigned char>(256 * std::clamp(p.z, 0.0f, 0.999f));
        }
    }
    
    if (stbi_write_png("julia_set.png", IMAGE_WIDTH, IMAGE_HEIGHT, 3, imageData.data(), IMAGE_WIDTH * 3)) {
        std::cout << "Saved to julia_set.png\n";
    } else {
        std::cout << "Failed to save image\n";
        return 1;
    }
    
    return 0;
}
