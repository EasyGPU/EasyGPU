/**
 * VolumetricFog:
 *      @Descripiton    :   Volumetric cloud rendering with procedural noise
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
#include <Flow/ReturnFlow.h>
#include <Flow/BreakFlow.h>
#include <Utility/Vec.h>
#include <Utility/Math.h>
#include <Utility/Helpers.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Runtime;
using namespace GPU::Callables;
using namespace GPU::Flow;

// =============================================================================
// Configuration
// =============================================================================
constexpr int IMAGE_WIDTH = 1280;
constexpr int IMAGE_HEIGHT = 720;
constexpr int MAX_STEPS = 256;
constexpr float STEP_SIZE = 0.02f;
constexpr float DENSITY_SCALE = 8.0f;
constexpr float ABSORPTION = 0.8f;
constexpr float SCATTERING = 0.9f;
constexpr float AMBIENT = 0.1f;

// Camera settings
constexpr float FOV = 1.2f;
const Vec3 CAMERA_POS = Vec3(0.0f, 1.5f, 4.0f);
const Vec3 CAMERA_LOOK = Vec3(0.0f, -0.2f, -1.0f);
const Vec3 CAMERA_UP = Vec3(0.0f, 1.0f, 0.0f);

// Light settings - dramatic sunset lighting
const Vec3 LIGHT_POS = Vec3(4.0f, 2.0f, -2.0f);
const Vec3 LIGHT_COLOR = Vec3(1.0f, 0.7f, 0.4f);
constexpr float LIGHT_INTENSITY = 15.0f;

// Noise parameters for cloud shape
constexpr float NOISE_FREQ = 2.0f;
constexpr float NOISE_AMP = 1.0f;
constexpr int NOISE_OCTAVES = 4;

// =============================================================================
// Hash function for pseudo-random numbers
// =============================================================================
Callable<float(Vec3)> Hash = [](Float3& p) {
    Float3 q = p * MakeFloat3(0.1031f, 0.1030f, 0.0973f);
    Float h = Fract(q.x() + q.y() + q.z());
    h = h * 0.47f;
    Return(Fract(h * 93.31f));
};

// =============================================================================
// 3D Value Noise
// =============================================================================
Callable<float(Vec3)> ValueNoise = [](Float3& p) {
    Float3 i = Floor(p);
    Float3 f = Fract(p);
    f = f * f * (MakeFloat3(3.0f) - MakeFloat3(2.0f) * f);
    
    Float3 c = MakeFloat3(0.0f);
    Float n = MakeFloat(0.0f);
    
    For(0, 8, [&](Int& idx) {
        Float dx = Expr<float>(idx % 2);
        Float dy = Expr<float>((idx / 2) % 2);
        Float dz = Expr<float>(idx / 4);
        
        Float3 offset = MakeFloat3(dx, dy, dz);
        Float3 samplePos = i + offset;
        Float h = Hash(samplePos);
        
        Float weight = 
            (1.0f - Abs(f.x() - dx)) *
            (1.0f - Abs(f.y() - dy)) *
            (1.0f - Abs(f.z() - dz));
        
        n = n + h * weight;
    });
    
    Return(n);
};

// =============================================================================
// Fractal Brownian Motion - layered noise for cloud detail
// =============================================================================
Callable<float(Vec3)> Fbm = [](Float3& p) {
    Float value = MakeFloat(0.0f);
    Float amplitude = MakeFloat(NOISE_AMP);
    Float frequency = MakeFloat(NOISE_FREQ);
    
    For(0, NOISE_OCTAVES, [&](Int& i) {
        Float3 scaledP = p * frequency;
        value = value + amplitude * ValueNoise(scaledP);
        amplitude = amplitude * 0.5f;
        frequency = frequency * 2.0f;
    });
    
    Return(value);
};

// =============================================================================
// Cloud density function - procedural cloud shape
// =============================================================================
Callable<float(Vec3)> GetDensity = [](Float3& pos) {
    // Base cloud shape - ellipsoid
    Float3 center = MakeFloat3(0.0f, 0.5f, 0.0f);
    Float3 scale = MakeFloat3(1.5f, 0.8f, 1.5f);
    
    Float3 diff = pos - center;
    Float dist = Length(diff / scale);
    
    // Base shape falloff
    Float baseShape;
    If(dist < 1.0f, [&]() {
        baseShape = Pow(1.0f - dist, 2.0f);
    }).Else([&]() {
        baseShape = 0.0f;
    });
    
    // Apply procedural noise
    Float noise = Fbm(pos + MakeFloat3(1.5f, 2.3f, 0.7f));
    
    // Create wispy cloud structure
    Float density = baseShape * noise * DENSITY_SCALE;
    
    // Threshold for sharper edges
    density = Max(density - 0.5f, 0.0f) * 2.0f;
    
    Return(density);
};

// =============================================================================
// Light transmittance through cloud (Beer-Lambert law)
// =============================================================================
Callable<float(Vec3, Vec3)> LightTransmittance = [](Float3& from, Float3& to) {
    Float3 dir = Normalize(to - from);
    Float dist = Distance(to, from);
    
    Float transmittance = MakeFloat(1.0f);
    Float t = MakeFloat(0.1f); // Start slightly offset to avoid self-shadowing
    
    For(0, 32, [&](Int&) {
        If(t >= dist, [&]() {
            Break();
        });
        
        Float3 samplePos = from + dir * t;
        Float density = GetDensity(samplePos);
        
        // Beer-Lambert: T = exp(-density * absorption * step)
        transmittance = transmittance * Exp(-density * ABSORPTION * 0.05f);
        
        t = t + 0.05f;
    });
    
    Return(Max(transmittance, 0.001f));
};

// =============================================================================
// Henyey-Greenstein phase function for anisotropic scattering
// =============================================================================
Callable<float(float)> PhaseHG = [](Float& cosTheta) {
    // Asymmetry parameter - positive = forward scattering
    Float g = MakeFloat(0.3f);
    Float gg = g * g;
    Float denom = 1.0f + gg - 2.0f * g * cosTheta;
    Float result = (1.0f - gg) / Pow(denom, 1.5f);
    Return(result / (4.0f * 3.14159f));
};

// =============================================================================
// Background sky with atmospheric gradient
// =============================================================================
Callable<Vec3(Vec3)> SkyColor = [](Float3& rayDir) {
    Float y = rayDir.y();
    
    // Horizon color - warm orange
    Float3 horizonColor = MakeFloat3(1.0f, 0.5f, 0.2f);
    // Zenith color - deep blue
    Float3 zenithColor = MakeFloat3(0.1f, 0.3f, 0.6f);
    
    // Smooth interpolation based on height
    Float t = Clamp(y * 0.5f + 0.5f, 0.0f, 1.0f);
    t = Pow(t, 0.7f); // Non-linear for more dramatic horizon
    
    Float3 color = Mix(horizonColor, zenithColor, t);
    
    // Add sun disk
    Float3 sunDir = Normalize(MakeFloat3(LIGHT_POS));
    Float cosAngle = Dot(rayDir, sunDir);
    If(cosAngle > 0.995f, [&]() {
        Float sunIntensity = Pow((cosAngle - 0.995f) / 0.005f, 2.0f);
        color = color + MakeFloat3(1.0f, 0.9f, 0.7f) * sunIntensity * 5.0f;
    });
    
    Return(color);
};

// =============================================================================
// Main volumetric rendering
// =============================================================================
Callable<Vec3(Vec3, Vec3)> RenderVolume = [](Float3& rayOrigin, Float3& rayDir) {
    // Background color
    Float3 bgColor = SkyColor(rayDir);
    
    // Ray march bounds - find where ray enters/leaves cloud volume
    Float3 center = MakeFloat3(0.0f, 0.5f, 0.0f);
    Float radius = MakeFloat(2.5f);
    
    Float3 oc = rayOrigin - center;
    Float a = Dot(rayDir, rayDir);
    Float b = 2.0f * Dot(oc, rayDir);
    Float c = Dot(oc, oc) - radius * radius;
    Float discriminant = b * b - 4.0f * a * c;
    
    If(discriminant < 0.0f, [&]() {
        Return(bgColor);
    });
    
    Float sqrtDisc = Sqrt(discriminant);
    Float tNear = (-b - sqrtDisc) / (2.0f * a);
    Float tFar = (-b + sqrtDisc) / (2.0f * a);
    
    If(tNear < 0.0f, [&]() { tNear = 0.0f; });
    If(tFar < 0.0f, [&]() { Return(bgColor); });
    
    // Ray march through volume
    Float3 accumulatedLight = MakeFloat3(0.0f);
    Float transmittance = MakeFloat(1.0f);
    Float t = tNear;
    
    For(0, MAX_STEPS, [&](Int&) {
        If(t > tFar || transmittance < 0.001f, [&]() {
            Break();
        });
        
        Float3 pos = rayOrigin + rayDir * t;
        Float density = GetDensity(pos);
        
        If(density > 0.01f, [&]() {
            // Calculate lighting
            Float3 toLight = LIGHT_POS - pos;
            Float lightDist = Length(toLight);
            Float3 lightDir = Normalize(toLight);
            
            // Light transmittance to this point
            Float lightTrans = LightTransmittance(pos + lightDir * 0.05f, MakeFloat3(LIGHT_POS));
            
            // Phase function for anisotropic scattering
            Float cosTheta = Dot(rayDir, lightDir);
            Float phase = PhaseHG(cosTheta);
            
            // In-scattering
            Float3 sunLight = LIGHT_COLOR * LIGHT_INTENSITY * lightTrans * phase * SCATTERING;
            
            // Ambient light (sky color from above)
            Float3 ambient = MakeFloat3(0.3f, 0.4f, 0.6f) * AMBIENT;
            
            // Combined lighting
            Float3 lighting = sunLight + ambient;
            
            // Density contribution
            Float stepTrans = Exp(-density * ABSORPTION * STEP_SIZE);
            Float3 stepLight = lighting * density * STEP_SIZE * transmittance;
            
            accumulatedLight = accumulatedLight + stepLight;
            transmittance = transmittance * stepTrans;
        });
        
        t = t + STEP_SIZE;
    });
    
    // Composite with background
    Float3 result = accumulatedLight + bgColor * transmittance;
    Return(result);
};

// =============================================================================
// Tone mapping - ACES filmic curve
// =============================================================================
Callable<Vec3(Vec3)> ToneMapACES = [](Float3& color) {
    Float a = MakeFloat(2.51f);
    Float b = MakeFloat(0.03f);
    Float c = MakeFloat(2.43f);
    Float d = MakeFloat(0.59f);
    Float e = MakeFloat(0.14f);
    
    Float3 x = color;
    Float3 result = (x * (a * x + b)) / (x * (c * x + d) + e);
    Return(Clamp(result, 0.0f, 1.0f));
};

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "Volumetric Cloud Renderer\n";
    std::cout << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << "\n";
    std::cout << "Ray marching steps: " << MAX_STEPS << "\n\n";
    
    Buffer<Vec4> image(IMAGE_WIDTH * IMAGE_HEIGHT, BufferMode::Write);
    
    std::cout << "Rendering...\n";
    
    // Pre-compute camera basis
    Vec3 cameraForward = CAMERA_LOOK.Normalized();
    Vec3 cameraRight = cameraForward.Cross(CAMERA_UP).Normalized();
    Vec3 cameraUp = cameraRight.Cross(cameraForward);
    
    float aspectRatio = static_cast<float>(IMAGE_WIDTH) / IMAGE_HEIGHT;
    float scale = tan(FOV * 0.5f);
    
    Kernel::Kernel2D kernel([&](Int& px, Int& py) {
        auto img = image.Bind();
        
        // Generate ray direction
        Float u = (2.0f * (Expr<float>(px) + 0.5f) / IMAGE_WIDTH - 1.0f) * aspectRatio * scale;
        Float v = (1.0f - 2.0f * (Expr<float>(py) + 0.5f) / IMAGE_HEIGHT) * scale;
        
        Float3 rayDir = Normalize(
            MakeFloat3(cameraForward) +
            MakeFloat3(cameraRight) * u +
            MakeFloat3(cameraUp) * v
        );
        
        // Render
        Float3 color = RenderVolume(MakeFloat3(CAMERA_POS), rayDir);
        
        // Tone mapping
        color = ToneMapACES(color);
        
        // Gamma correction
        Float gamma = MakeFloat(1.0f / 2.2f);
        color.x() = Pow(color.x(), gamma);
        color.y() = Pow(color.y(), gamma);
        color.z() = Pow(color.z(), gamma);
        
        // Store
        Int idx = py * IMAGE_WIDTH + px;
        img[idx] = MakeFloat4(color.x(), color.y(), color.z(), 1.0f);
    });
    
    kernel.Dispatch((IMAGE_WIDTH + 15) / 16, (IMAGE_HEIGHT + 15) / 16, true);
    
    // Download and save
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
    
    if (stbi_write_png("volumetric_cloud.png", IMAGE_WIDTH, IMAGE_HEIGHT, 3, imageData.data(), IMAGE_WIDTH * 3)) {
        std::cout << "Saved to volumetric_cloud.png\n";
    } else {
        std::cout << "Failed to save image\n";
        return 1;
    }
    
    return 0;
}
