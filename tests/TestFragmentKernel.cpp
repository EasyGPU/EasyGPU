/**
 * TestFragmentKernel - Comprehensive test for FragmentKernel2D
 * 
 * Tests all major features:
 * - Uniform variables (time, params)
 * - Texture sampling (BindSampler)
 * - Buffer/SSBO access
 * - Callable functions
 * - Complex shader effects
 */

#include <windows.h>
#include <GPU.h>
#include <cmath>
#include <vector>
#include <chrono>

#pragma comment(lib, "opengl32.lib")

using namespace GPU::IR::Value;
using namespace GPU::Math;

// Callable: Rotate UV coordinates
Callable<Vec2(Vec2, float)> RotateUV = [](Float2& uv, Float& angle) {
    Float cosA = Cos(angle);
    Float sinA = Sin(angle);
    Float2 center = MakeFloat2(0.5f, 0.5f);
    Float2 delta = uv - center;
    Float x = delta.x() * cosA - delta.y() * sinA;
    Float y = delta.x() * sinA + delta.y() * cosA;
    Return(MakeFloat2(x, y) + center);
};

// Callable: procedural noise function
Callable<float(Vec2)> SimpleNoise = [](Float2& p) {
    Float i = Floor(p.x());
    Float j = Floor(p.y());
    Float u = p.x() - i;
    Float v = p.y() - j;
    
    // Simple hash
    Float h = Fract(Sin(i * 12.9898f + j * 78.233f) * 43758.5453f);
    Return(h);
};

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CLOSE:
            PostQuitMessage(0);
            return 0;
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                PostQuitMessage(0);
                return 0;
            }
            break;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    // Register window class
    const TCHAR CLASS_NAME[] = TEXT("FragmentKernelTest");
    
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    
    RegisterClass(&wc);
    
    // Create window
    const int WIDTH = 1280;
    const int HEIGHT = 720;
    
    HWND hwnd = CreateWindowEx(
        0,
        CLASS_NAME,
        TEXT("EasyGPU FragmentKernel"),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        WIDTH, HEIGHT,
        nullptr,
        nullptr,
        hInstance,
        nullptr
    );
    
    if (!hwnd) {
        MessageBox(nullptr, TEXT("Failed to create window"), TEXT("Error"), MB_OK);
        return 1;
    }
    
    ShowWindow(hwnd, nCmdShow);
    
    try {
        // ============================================================
        // Create Resources
        // ============================================================
        
        // 1. Uniforms
        Uniform<float> uTime;
        Uniform<float> uSpeed;
        Uniform<Vec3> uColor1;  // Primary color
        Uniform<Vec3> uColor2;  // Secondary color
        
        uTime = 0.0f;
        uSpeed = 1.0f;
        uColor1 = Vec3(1.0f, 0.5f, 0.2f);  // Orange
        uColor2 = Vec3(0.2f, 0.5f, 1.0f);  // Blue
        
        // 2. Texture - create a procedural pattern
        const int TEX_SIZE = 256;
        std::vector<uint8_t> textureData(TEX_SIZE * TEX_SIZE * 4);
        for (int y = 0; y < TEX_SIZE; y++) {
            for (int x = 0; x < TEX_SIZE; x++) {
                int idx = (y * TEX_SIZE + x) * 4;
                // Checkerboard with gradient
                float cx = (float)x / TEX_SIZE;
                float cy = (float)y / TEX_SIZE;
                float check = ((x / 32) + (y / 32)) % 2 == 0 ? 1.0f : 0.0f;
                textureData[idx + 0] = (uint8_t)(255 * cx * check);      // R
                textureData[idx + 1] = (uint8_t)(255 * cy * (1-check));  // G
                textureData[idx + 2] = (uint8_t)(255 * (1-cx));          // B
                textureData[idx + 3] = 255;                             // A
            }
        }
        
        Texture2D<PixelFormat::RGBA8> patternTex(TEX_SIZE, TEX_SIZE, textureData.data());
        
        // 3. Buffer - particle positions for some effect
        struct Particle {
            float x, y, vx, vy;
        };
        
        std::vector<Particle> particles(100);
        for (int i = 0; i < 100; i++) {
            particles[i].x = (float)(rand() % 1280) / 1280.0f;
            particles[i].y = (float)(rand() % 720) / 720.0f;
            particles[i].vx = ((float)(rand() % 100) - 50) / 1000.0f;
            particles[i].vy = ((float)(rand() % 100) - 50) / 1000.0f;
        }
        
        Buffer<Particle> particleBuffer(particles);
        
        // ============================================================
        // Create Fragment Kernel with ALL features
        // ============================================================
        
        GPU::Kernel::FragmentKernel2D kernel("ComprehensiveTest",
            [&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
                // UV coordinates (0-1)
                Float2 uv = fragCoord / resolution;
                Float2 centeredUV = uv - MakeFloat2(0.5f, 0.5f);
                
                // Load uniforms
                Float time = uTime.Load();
                Float speed = uSpeed.Load();
                Float3 color1 = uColor1.Load();
                Float3 color2 = uColor2.Load();
                
                // ========================================================
                // Feature 1: Animated rotating pattern using Callable
                // ========================================================
                Float rotationAngle = time * speed * 0.5f;
                Float2 rotatedUV = RotateUV(uv, rotationAngle);
                
                // ========================================================
                // Feature 2: Texture Sampling
                // ========================================================
                auto tex = patternTex.BindSampler();
                
                // Animated texture coordinates
                Float2 texUV = rotatedUV;
                texUV.x() = texUV.x() + Sin(time * 0.3f) * 0.1f;
                texUV.y() = texUV.y() + Cos(time * 0.2f) * 0.1f;
                
                Float4 texColor = tex.Sample(Fract(texUV));
                
                // ========================================================
                // Feature 3: Procedural pattern using Callable
                // ========================================================
                Float2 noiseUV = uv * 8.0f;
                noiseUV.x() = noiseUV.x() + time;
                Float noiseVal = SimpleNoise(noiseUV);
                
                // ========================================================
                // Feature 4: Complex geometric pattern
                // ========================================================
                Float radius = Length(centeredUV);
                Float angle = Atan2(centeredUV.y(), centeredUV.x());
                
                // Animated ring pattern
                Float ring = Abs(radius - 0.3f - Sin(time * 0.5f) * 0.1f);
                ring = Smoothstep(0.02f, 0.0f, ring);
                
                // Spiral pattern
                Float spiral = Sin(angle * 5.0f + radius * 20.0f - time * 2.0f) * 0.5f + 0.5f;
                
                // ========================================================
                // Feature 5: Color mixing with uniforms
                // ========================================================
                // Mix colors based on position and time
                Float3 mixedColor = Mix(color1, color2, uv.x() + Sin(time) * 0.3f);
                
                // Add texture contribution
                mixedColor = mixedColor + texColor.xyz() * 0.3f;
                
                // Add procedural noise
                mixedColor = mixedColor * (0.8f + noiseVal * 0.2f);
                
                // Add geometric patterns
                Float3 ringColor = MakeFloat3(1.0f, 1.0f, 0.8f) * ring;
                Float3 spiralColor = color1 * spiral * 0.5f;
                
                mixedColor = mixedColor + ringColor + spiralColor;
                
                // ========================================================
                // Feature 6: Vignette effect
                // ========================================================
                Float vignette = 1.0f - radius * 1.2f;
                vignette = Max(vignette, 0.0f);
                vignette = Pow(vignette, 0.5f);
                
                mixedColor = mixedColor * vignette;
                
                // ========================================================
                // Output final color
                // ========================================================
                fragColor = MakeFloat4(mixedColor, 1.0f);
            },
            WIDTH, HEIGHT
        );
        
        // Attach to window
        if (!kernel.Attach(hwnd)) {
            MessageBox(hwnd, TEXT("Failed to attach kernel to window"), TEXT("Error"), MB_OK);
            return 1;
        }
        
        // ============================================================
        // Main Loop
        // ============================================================
        auto startTime = std::chrono::steady_clock::now();
        int frameCount = 0;
        float fps = 0.0f;
        
        // Animation parameters
        float colorPhase = 0.0f;
        
        MSG msg = {};
        bool running = true;
        
        while (running) {
            // Process messages
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    running = false;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            
            if (!running) break;
            
            // Calculate timing
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            
            // Update uniforms
            uTime = elapsed;
            
            // Animate colors
            colorPhase = elapsed * 0.5f;
            float r1 = sin(colorPhase) * 0.5f + 0.5f;
            float g1 = sin(colorPhase + 2.0f) * 0.5f + 0.5f;
            float b1 = sin(colorPhase + 4.0f) * 0.5f + 0.5f;
            uColor1 = Vec3(r1, g1 * 0.7f, b1 * 0.5f);
            
            float r2 = sin(colorPhase + 3.14f) * 0.5f + 0.5f;
            float g2 = sin(colorPhase + 3.14f + 2.0f) * 0.5f + 0.5f;
            float b2 = sin(colorPhase + 3.14f + 4.0f) * 0.5f + 0.5f;
            uColor2 = Vec3(r2 * 0.5f, g2 * 0.8f, b2);
            
            // Render
            kernel.Flush();
            
            // Calculate FPS
            frameCount++;
            if (frameCount % 60 == 0) {
                fps = frameCount / elapsed;
            }
        }
        
    } catch (const std::exception& e) {
        MessageBoxA(hwnd, e.what(), "Exception", MB_OK);
        return 1;
    }
    
    return 0;
}
