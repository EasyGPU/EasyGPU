/**
 * Mandelbrot:
 *      @Descripiton    :   Mandelbrot set fractal renderer
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
#include <IR/Value/ExprMatrix.h>

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
constexpr float CENTER_X = -0.5f;
constexpr float CENTER_Y = 0.0f;
constexpr float ZOOM = 1.5f;

// =============================================================================
// Color mapping function - Ultra Fractal style palette
// =============================================================================
Callable<Vec3(int)> GetColor = [](Int& iter) {
    Float3 color;
    If(iter == MAX_ITERATIONS, [&]() {
        // Inside set - deep black with subtle blue tint
        color = MakeFloat3(0.02f, 0.02f, 0.05f);
    }).Else([&]() {
        // Outside set - rich psychedelic gradient
        Float t = Expr<float>(iter) / float(MAX_ITERATIONS);
        
        // Use multiple sine waves for richer colors (Ultra Fractal style)
        // This creates bands of color that cycle through the spectrum
        Float freq = MakeFloat(6.28318f); // 2 * PI
        
        // Red channel: mix of two frequencies
        Float r = 0.5f + 0.5f * Sin(freq * t + 0.0f) * Cos(freq * t * 0.5f);
        
        // Green channel: phase shifted
        Float g = 0.5f + 0.5f * Sin(freq * t + 2.094f) * Cos(freq * t * 0.3f + 1.0f);
        
        // Blue channel: different phase
        Float b = 0.5f + 0.5f * Sin(freq * t + 4.188f) * Cos(freq * t * 0.7f + 2.0f);
        
        // Apply gamma correction for smoother appearance
        r = Pow(Clamp(r, 0.0f, 1.0f), 0.8f);
        g = Pow(Clamp(g, 0.0f, 1.0f), 0.8f);
        b = Pow(Clamp(b, 0.0f, 1.0f), 0.8f);
        
        // Boost saturation slightly
        Float intensity = MakeFloat(1.2f);
        r = Clamp(r * intensity, 0.0f, 1.0f);
        g = Clamp(g * intensity, 0.0f, 1.0f);
        b = Clamp(b * intensity, 0.0f, 1.0f);
        
        color = MakeFloat3(r, g, b);
    });
    
    Return(color);
};

// =============================================================================
// Mandelbrot iteration function
// =============================================================================
Callable<int(float, float)> Mandelbrot = [](Float& cx, Float& cy) {
    Float zx = MakeFloat(0.0f);
    Float zy = MakeFloat(0.0f);
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
        // z^2 = (zx + i*zy)^2 = zx^2 - zy^2 + 2*i*zx*zy
        zy = 2.0f * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        iter = i;
    });
    
    Return(iter);
};

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "Mandelbrot Set Fractal Renderer\n";
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
        
        // Map pixel to complex plane
        // x ranges from [CENTER_X - scaleX, CENTER_X + scaleX]
        // y ranges from [CENTER_Y - scaleY, CENTER_Y + scaleY]
        Float u = (Expr<float>(px) + 0.5f) / IMAGE_WIDTH;
        Float v = (Expr<float>(py) + 0.5f) / IMAGE_HEIGHT;
        
        Float cx = CENTER_X + (u * 2.0f - 1.0f) * scaleX;
        Float cy = CENTER_Y + (v * 2.0f - 1.0f) * scaleY;
        
        // Compute Mandelbrot iterations
        Int iter = Mandelbrot(cx, cy);
        
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
    
    if (stbi_write_png("mandelbrot.png", IMAGE_WIDTH, IMAGE_HEIGHT, 3, imageData.data(), IMAGE_WIDTH * 3)) {
        std::cout << "Saved to mandelbrot.png\n";
    } else {
        std::cout << "Failed to save image\n";
        return 1;
    }
    
    return 0;
}
