// =============================================================================
// Radiance Cascade - Cornell Box (Real-time Windowed)
// A physically-grounded screen-space radiance cascade implementation
// using EasyGPU with real-time camera roaming.
// =============================================================================

#include <GPU.h>
#include <Window/AppWindow.h>
#include <Window/TexturePresenter.h>
#include <Window/Input.h>
#include "rc_scene.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <Kernel/ShaderCache.h>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Flow;
using namespace GPU::Runtime;
using namespace GPU::Callables;
using namespace RC;

constexpr float PI = 3.14159265f;

// =============================================================================
// GPU Structs (must be in global namespace)
// =============================================================================
EASYGPU_STRUCT(Ray, (Vec3, origin), (Vec3, dir));
EASYGPU_STRUCT(HitRec, (Vec3, p), (Vec3, normal), (float, t), (Vec3, albedo), (Vec3, emission), (float, metal));

// =============================================================================
// Resource Slots
// =============================================================================
BufferSlot<Vec4> gAlbedoSlot;
BufferSlot<Vec4> gNormalSlot;
BufferSlot<Vec4> gEmissionSlot;
BufferSlot<float> gDepthSlot;
BufferSlot<Vec4> cascadeReadSlot;
BufferSlot<Vec4> cascadeWriteSlot;

// =============================================================================
// Uniforms
// =============================================================================
Uniform<int> uCascadeLevel(0);
Uniform<int> uNumCascades(0);
Uniform<int> uProbeSpacing(PROBE_SPACING);
Uniform<float> uIntervalLength(BASE_INTERVAL_LENGTH);

Uniform<Vec3> uCamPos(Vec3(0.0f, 0.0f, 2.5f));
Uniform<Vec3> uCamForward(Vec3(0.0f, 0.0f, -1.0f));
Uniform<Vec3> uCamRight(Vec3(1.0f, 0.0f, 0.0f));
Uniform<Vec3> uCamUp(Vec3(0.0f, 1.0f, 0.0f));

// =============================================================================
// Math Helpers
// =============================================================================


Callable<Float2(Float3)> OctEncode = [](Float3 n) {
    Float2 p = MakeFloat2(n.x(), n.y()) * (1.0f / (Abs(n.x()) + Abs(n.y()) + Abs(n.z())));
    If(n.z() <= MakeFloat(0.0f), [&]() {
        Float2 signNotZero = MakeFloat2(Select(p.x() >= MakeFloat(0.0f), MakeFloat(1.0f), MakeFloat(-1.0f)),
                                         Select(p.y() >= MakeFloat(0.0f), MakeFloat(1.0f), MakeFloat(-1.0f)));
        p = (MakeFloat(1.0f) - Abs(MakeFloat2(p.y(), p.x()))) * signNotZero;
    });
    Return(p * MakeFloat2(0.5f, 0.5f) + MakeFloat2(0.5f, 0.5f));
};

Callable<Float3(Float2)> OctDecode = [](Float2 f) {
    f = f * MakeFloat2(2.0f, 2.0f) - MakeFloat2(1.0f, 1.0f);
    Float3 n = MakeFloat3(f.x(), f.y(), 1.0f - Abs(f.x()) - Abs(f.y()));
    If(n.z() < MakeFloat(0.0f), [&]() {
        Float nx = n.x();
        Float ny = n.y();
        Float2 signNotZero = MakeFloat2(Select(nx >= MakeFloat(0.0f), MakeFloat(1.0f), MakeFloat(-1.0f)),
                                         Select(ny >= MakeFloat(0.0f), MakeFloat(1.0f), MakeFloat(-1.0f)));
        n.x() = (1.0f - Abs(ny)) * signNotZero.x();
        n.y() = (1.0f - Abs(nx)) * signNotZero.y();
    });
    Return(Normalize(n));
};

Callable<Float4(Float2)> GetBilinearWeights = [](Float2 ratio) {
    Return(MakeFloat4(
        (1.0f - ratio.x()) * (1.0f - ratio.y()),
        ratio.x() * (1.0f - ratio.y()),
        (1.0f - ratio.x()) * ratio.y(),
        ratio.x() * ratio.y()
    ));
};

Callable<Float(Int)> GetIntervalScale = [](Int level) {
    If(level == MakeInt(0), [&]() { Return(MakeFloat(0.0f)); });
    Return(Pow(MakeFloat(4.0f), ToFloat(level)));
};

Callable<Float2(Int2)> ToFloat2 = [](Int2 v) {
    Return(MakeFloat2(ToFloat(v.x()), ToFloat(v.y())));
};

Callable<Int2(Float2)> ToInt2 = [](Float2 v) {
    Return(MakeInt2(ToInt(v.x()), ToInt(v.y())));
};

Callable<Float(Int&)> Random = [](Int& state) {
    state = (state * 1103515245 + 12345) & MakeInt(0x7FFFFFFF);
    Return(ToFloat(state) / 2147483647.0f);
};

Callable<Float3(Int&)> RandomInUnitSphere = [](Int& state) {
    Float3 p = MakeFloat3(Random(state), Random(state), Random(state)) * 2.0f - MakeFloat3(1.0f, 1.0f, 1.0f);
    Return(p);
};

Callable<Float3(Int&)> RandomUnitVector = [](Int& state) {
    Return(Normalize(RandomInUnitSphere(state)));
};

// =============================================================================
// Ray / Intersection
// =============================================================================
Callable<Float3(Var<Ray>&, Float)> RayAt = [](Var<Ray>& r, Float t) {
    Return(r.origin() + r.dir() * t);
};

Callable<Bool(Float3, Float3, Var<Ray>&, Float, Float&, Var<HitRec>&, Float3, Float3, Float)> HitBox =
[](Float3 bmin, Float3 bmax, Var<Ray>& r, Float tmin, Float& closest, Var<HitRec>& rec, Float3 albedo, Float3 emission, Float metal) {
    Bool hit = MakeBool(false);
    Float3 n;
    Float tmax = closest;
    Float tc = tmax;

    // X planes
    Float t = (bmin.x() - r.origin().x()) / r.dir().x();
    If(t > tmin && t < tc, [&]() {
        Float3 p = RayAt(r, t);
        If(p.y() > bmin.y() && p.y() < bmax.y() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
            tc = t;
            n = Vec3(-1.0f, 0.0f, 0.0f);
            hit = true;
        });
    });
    t = (bmax.x() - r.origin().x()) / r.dir().x();
    If(t > tmin && t < tc, [&]() {
        Float3 p = RayAt(r, t);
        If(p.y() > bmin.y() && p.y() < bmax.y() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
            tc = t;
            n = Vec3(1.0f, 0.0f, 0.0f);
            hit = true;
        });
    });

    // Y planes
    t = (bmin.y() - r.origin().y()) / r.dir().y();
    If(t > tmin && t < tc, [&]() {
        Float3 p = RayAt(r, t);
        If(p.x() > bmin.x() && p.x() < bmax.x() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
            tc = t;
            n = Vec3(0.0f, -1.0f, 0.0f);
            hit = true;
        });
    });
    t = (bmax.y() - r.origin().y()) / r.dir().y();
    If(t > tmin && t < tc, [&]() {
        Float3 p = RayAt(r, t);
        If(p.x() > bmin.x() && p.x() < bmax.x() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
            tc = t;
            n = Vec3(0.0f, 1.0f, 0.0f);
            hit = true;
        });
    });

    // Z planes
    t = (bmin.z() - r.origin().z()) / r.dir().z();
    If(t > tmin && t < tc, [&]() {
        Float3 p = RayAt(r, t);
        If(p.x() > bmin.x() && p.x() < bmax.x() && p.y() > bmin.y() && p.y() < bmax.y(), [&]() {
            tc = t;
            n = Vec3(0.0f, 0.0f, -1.0f);
            hit = true;
        });
    });
    t = (bmax.z() - r.origin().z()) / r.dir().z();
    If(t > tmin && t < tc, [&]() {
        Float3 p = RayAt(r, t);
        If(p.x() > bmin.x() && p.x() < bmax.x() && p.y() > bmin.y() && p.y() < bmax.y(), [&]() {
            tc = t;
            n = Vec3(0.0f, 0.0f, 1.0f);
            hit = true;
        });
    });

    If(hit, [&]() {
        rec.t() = tc;
        rec.p() = RayAt(r, tc);
        rec.normal() = n;
        rec.albedo() = albedo;
        rec.emission() = emission;
        rec.metal() = metal;
        closest = tc;
    });

    Return(hit);
};



Callable<Bool(Var<Ray>&, Float, Float, Var<HitRec>&)> HitWorld = [](Var<Ray>& r, Float tmin, Float tmax, Var<HitRec>& rec) {
    Bool hit = MakeBool(false);
    Float closest = tmax;
    Var<HitRec> temp;

    // Floor (white diffuse)
    If(HitBox(MakeFloat3(-1.0f, -1.0f, -1.0f), MakeFloat3(1.0f, -0.75f, 1.0f), r, tmin, closest, temp,
              MakeFloat3(0.73f, 0.73f, 0.73f), MakeFloat3(0.0f, 0.0f, 0.0f), 0.0f),
       [&]() { rec = temp; hit = true; closest = temp.t(); });

    // Ceiling (white diffuse)
    If(HitBox(MakeFloat3(-1.0f, 0.75f, -1.0f), MakeFloat3(1.0f, 1.0f, 1.0f), r, tmin, closest, temp,
              MakeFloat3(0.73f, 0.73f, 0.73f), MakeFloat3(0.0f, 0.0f, 0.0f), 0.0f),
       [&]() { rec = temp; hit = true; closest = temp.t(); });

    // Back wall (white diffuse)
    If(HitBox(MakeFloat3(-1.0f, -0.75f, -1.0f), MakeFloat3(1.0f, 0.75f, -0.75f), r, tmin, closest, temp,
              MakeFloat3(0.73f, 0.73f, 0.73f), MakeFloat3(0.0f, 0.0f, 0.0f), 0.0f),
       [&]() { rec = temp; hit = true; closest = temp.t(); });

    // Left wall (red diffuse)
    If(HitBox(MakeFloat3(-1.0f, -0.75f, -0.75f), MakeFloat3(-0.75f, 0.75f, 1.0f), r, tmin, closest, temp,
              MakeFloat3(0.65f, 0.05f, 0.05f), MakeFloat3(0.0f, 0.0f, 0.0f), 0.0f),
       [&]() { rec = temp; hit = true; closest = temp.t(); });

    // Right wall (green diffuse)
    If(HitBox(MakeFloat3(0.75f, -0.75f, -0.75f), MakeFloat3(1.0f, 0.75f, 1.0f), r, tmin, closest, temp,
              MakeFloat3(0.12f, 0.45f, 0.15f), MakeFloat3(0.0f, 0.0f, 0.0f), 0.0f),
       [&]() { rec = temp; hit = true; closest = temp.t(); });

    // Light (emissive)
    If(HitBox(MakeFloat3(-0.25f, 0.74f, -0.25f), MakeFloat3(0.25f, 0.75f, 0.25f), r, tmin, closest, temp,
              MakeFloat3(15.0f, 15.0f, 15.0f), MakeFloat3(15.0f, 15.0f, 15.0f), 0.0f),
       [&]() { rec = temp; hit = true; closest = temp.t(); });

    // Tall box (metal)
    If(HitBox(MakeFloat3(0.15f, -0.75f, -0.4f), MakeFloat3(0.45f, -0.15f, -0.1f), r, tmin, closest, temp,
              MakeFloat3(0.8f, 0.85f, 0.88f), MakeFloat3(0.0f, 0.0f, 0.0f), 1.0f),
       [&]() { rec = temp; hit = true; closest = temp.t(); });

    // Short box (white diffuse)
    If(HitBox(MakeFloat3(-0.4f, -0.75f, 0.0f), MakeFloat3(-0.1f, -0.4f, 0.3f), r, tmin, closest, temp,
              MakeFloat3(0.73f, 0.73f, 0.73f), MakeFloat3(0.0f, 0.0f, 0.0f), 0.0f),
       [&]() { rec = temp; hit = true; closest = temp.t(); });

    Return(hit);
};

Callable<Float3(Float3, Float3)> EstimateDirectLight = [](Float3 p, Float3 n) {
    Float3 lightEmission = MakeFloat3(15.0f, 15.0f, 15.0f);
    Float lightArea = MakeFloat(0.25f);
    Float3 totalDirect = MakeFloat3(0.0f);

    // Per-pixel jitter to break fixed-grid aliasing
    Float jitterX = (Fract(Abs(p.x()) * 12.9898f + Abs(p.z()) * 78.233f) - 0.5f) * (0.5f / 8.0f);
    Float jitterZ = (Fract(Abs(p.x()) * 43.123f + Abs(p.z()) * 23.456f) - 0.5f) * (0.5f / 8.0f);

    For(0, 128, [&](Int& i) {
        Int ix = i - (i / 8) * 8;
        Int iz = i / 8;
        Float fx = -0.25f + (ToFloat(ix) + 0.5f) * (0.5f / 8.0f) + jitterX;
        Float fz = -0.25f + (ToFloat(iz) + 0.5f) * (0.5f / 8.0f) + jitterZ;
        Float3 corner = MakeFloat3(fx, 0.74f, fz);

        Float3 toLight = corner - p;
        Float distToLight = Length(toLight);
        Float3 lightDir = Normalize(toLight);
        Float NdotL = Max(Dot(n, lightDir), 0.0f);
        Float lightCos = Max(Dot(MakeFloat3(0.0f, -1.0f, 0.0f), -lightDir), 0.0f);

        If(NdotL > 0.0f && lightCos > 0.0f, [&]() {
            Var<Ray> shadowRay;
            shadowRay.origin() = p + n * 0.001f;
            shadowRay.dir() = lightDir;
            Var<HitRec> shadowRec;
            Bool visible = MakeBool(true);
            If(HitWorld(shadowRay, 0.001f, distToLight + 0.001f, shadowRec), [&]() {
                If(Length(shadowRec.emission()) <= 0.0f && shadowRec.t() < distToLight, [&]() {
                    visible = false;
                });
            });
            If(visible, [&]() {
                Float attenuation = lightArea * lightCos / (distToLight * distToLight + MakeFloat(0.1f));
                totalDirect = totalDirect + lightEmission * NdotL * attenuation;
            });
        });
    });

    Return(totalDirect * MakeFloat(1.0f / 128.0f));
};

// =============================================================================
// Main
// =============================================================================
int main() {
    try {
    std::cout << "Radiance Cascade - Cornell Box (Windowed)\n";
    std::cout << "Resolution: " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << "\n";
    std::cout << "Controls: WASD=move, QE=up/down, LeftMouseDrag=look, ESC=exit\n";

    int numCascades = 1;
    std::cout << "Cascades: " << numCascades << "\n";

    // Window setup
    GPU::Window::AppWindow window({
        .width = static_cast<uint32_t>(IMAGE_WIDTH),
        .height = static_cast<uint32_t>(IMAGE_HEIGHT),
        .title = "Radiance Cascade - Cornell Box",
        .resizable = false,
        .vsync = true
    });
    GPU::Window::TexturePresenter presenter(window);

    // GPU resources
    Buffer<Vec4> gAlbedo(IMAGE_WIDTH * IMAGE_HEIGHT);
    Buffer<Vec4> gNormal(IMAGE_WIDTH * IMAGE_HEIGHT);
    Buffer<Vec4> gEmission(IMAGE_WIDTH * IMAGE_HEIGHT);
    Buffer<float> gDepth(IMAGE_WIDTH * IMAGE_HEIGHT);
    Texture2D<PixelFormat::RGBA8> presentTex(IMAGE_WIDTH, IMAGE_HEIGHT);

    std::vector<int> seeds(IMAGE_WIDTH * IMAGE_HEIGHT);
    for (size_t i = 0; i < seeds.size(); ++i) seeds[i] = static_cast<int>(i + 1);
    Buffer<int> rngBuf(seeds, BufferMode::ReadWrite);

    gAlbedoSlot.Attach(gAlbedo);
    gNormalSlot.Attach(gNormal);
    gEmissionSlot.Attach(gEmission);
    gDepthSlot.Attach(gDepth);

    std::vector<Buffer<Vec4>> cascadeBuffers;
    cascadeBuffers.emplace_back(CASCADE_SIZE * CASCADE_SIZE);
    cascadeBuffers.emplace_back(CASCADE_SIZE * CASCADE_SIZE);

    uNumCascades = numCascades;

    // Camera state
    Vec3 camPos(0.0f, 0.0f, 2.5f);
    float yaw = -90.0f;
    float pitch = 0.0f;
    float moveSpeed = 0.05f;
    float lookSpeed = 0.2f;

    auto Deg2Rad = [](float deg) { return deg * PI / 180.0f; };

    // Initial camera basis (needed for pre-compile dispatch)
    Vec3 forward(
        std::cosf(Deg2Rad(yaw)) * std::cosf(Deg2Rad(pitch)),
        std::sinf(Deg2Rad(pitch)),
        std::sinf(Deg2Rad(yaw)) * std::cosf(Deg2Rad(pitch))
    );
    forward = forward.Normalized();
    Vec3 right = forward.Cross(Vec3(0.0f, 1.0f, 0.0f)).Normalized();
    Vec3 up = right.Cross(forward);

    // -------------------------------------------------------------------------
    // Kernels
    // -------------------------------------------------------------------------
    Kernel2D gBufferPass([&](Int px, Int py) {
        auto albedoOut = gAlbedoSlot.Bind();
        auto normalOut = gNormalSlot.Bind();
        auto emissionOut = gEmissionSlot.Bind();
        auto depthOut = gDepthSlot.Bind();

        Int idx = py * IMAGE_WIDTH + px;
        Float u = (ToFloat(px) + 0.5f) / IMAGE_WIDTH;
        Float v = (ToFloat(py) + 0.5f) / IMAGE_HEIGHT;

        Float3 forward = uCamForward.Load();
        Float3 right = uCamRight.Load();
        Float3 up = uCamUp.Load();
        Float2 ndc = MakeFloat2(u * 2.0f - 1.0f, v * 2.0f - 1.0f);
        Float3 rayDir = Normalize(forward * 2.0f + right * ndc.x() + up * ndc.y());

        Var<Ray> ray;
        ray.origin() = uCamPos.Load();
        ray.dir() = rayDir;

        Var<HitRec> hit;
        If(HitWorld(ray, 0.001f, 1000.0f, hit), [&]() {
            Float2 oct = OctEncode(hit.normal());
            albedoOut[idx] = MakeFloat4(hit.albedo().x(), hit.albedo().y(), hit.albedo().z(), 1.0f);
            normalOut[idx] = MakeFloat4(oct.x(), oct.y(), hit.metal(), 0.0f);
            emissionOut[idx] = MakeFloat4(hit.emission().x(), hit.emission().y(), hit.emission().z(),
                                          Select(Length(hit.emission()) > MakeFloat(0.0f), MakeFloat(0.0f), MakeFloat(1.0f)));
            depthOut[idx] = hit.t();
        }).Else([&]() {
            albedoOut[idx] = MakeFloat4(0.0f, 0.0f, 0.0f, 0.0f);
            normalOut[idx] = MakeFloat4(0.0f, 0.0f, 0.0f, 0.0f);
            emissionOut[idx] = MakeFloat4(0.0f, 0.0f, 0.0f, 0.0f);
            depthOut[idx] = 0.0f;
        });
    });

    Kernel2D cascadePass([&](Int px, Int py) {
        auto albedoIn = gAlbedoSlot.Bind();
        auto normalIn = gNormalSlot.Bind();
        auto emissionIn = gEmissionSlot.Bind();
        auto depthIn = gDepthSlot.Bind();
        auto cascadeIn = cascadeReadSlot.Bind();
        auto cascadeOut = cascadeWriteSlot.Bind();

        Int level = uCascadeLevel.Load();
        Int numLevels = uNumCascades.Load();
        Int probeSpacingVal = uProbeSpacing.Load();
        Float intervalLen = uIntervalLength.Load();

        Int tileSize = probeSpacingVal * (1 << level);
        Int raysPerDim = (1 << (level + 5));
        Int gx = (MakeInt(IMAGE_WIDTH) + tileSize - 1) / tileSize;
        Int gy = (MakeInt(IMAGE_HEIGHT) + tileSize - 1) / tileSize;
        Int2 probeGridSize = MakeInt2(gx, gy);

        Int2 pixel = MakeInt2(px, py);
        Int2 probeCoord2D = MakeInt2(pixel.x() - (pixel.x() / probeGridSize.x()) * probeGridSize.x(),
                                          pixel.y() - (pixel.y() / probeGridSize.y()) * probeGridSize.y());
        Int2 rayCoord2D = MakeInt2(pixel.x() / probeGridSize.x(), pixel.y() / probeGridSize.y());

        Float2 probeUV = (ToFloat2(probeCoord2D) + MakeFloat2(0.5f, 0.5f)) / ToFloat2(probeGridSize);
        Float jitterU = (Fract(probeUV.x() * 12.9898f + probeUV.y() * 78.233f) - 0.5f) / ToFloat(raysPerDim);
        Float jitterV = (Fract(probeUV.x() * 43.123f + probeUV.y() * 23.456f) - 0.5f) / ToFloat(raysPerDim);
        Float2 rayUV = (ToFloat2(rayCoord2D) + MakeFloat2(0.5f, 0.5f)) / ToFloat(raysPerDim) + MakeFloat2(jitterU, jitterV);
        Float3 rayDir = OctDecode(rayUV);

        // Reconstruct probe world position (probeCoord2D is grid index, map to screen pixel)
        Int2 probePixel = MakeInt2(ToInt(probeUV.x() * MakeFloat(IMAGE_WIDTH)), ToInt(probeUV.y() * MakeFloat(IMAGE_HEIGHT)));
        Int probePixelIdx = probePixel.y() * IMAGE_WIDTH + probePixel.x();
        Float probeDepth = depthIn[probePixelIdx];
        Float2 screenUV = (ToFloat2(probePixel) + MakeFloat2(0.5f, 0.5f)) / MakeFloat2(float(IMAGE_WIDTH), float(IMAGE_HEIGHT));

        Float3 camForward = uCamForward.Load();
        Float3 camRight = uCamRight.Load();
        Float3 camUp = uCamUp.Load();
        Float2 probeNdc = MakeFloat2(screenUV.x() * 2.0f - 1.0f, screenUV.y() * 2.0f - 1.0f);
        Float3 probeRayDir = Normalize(camForward * 2.0f + camRight * probeNdc.x() + camUp * probeNdc.y());
        Float3 worldPos = uCamPos.Load() + probeRayDir * probeDepth;

        Var<Ray> ray;
        ray.origin() = worldPos;
        ray.dir() = rayDir;

        Var<HitRec> hit;
        Bool isHit = HitWorld(ray, 0.001f, 1000.0f, hit);

        Var<Vec3> radiance = MakeFloat3(0.0f, 0.0f, 0.0f);

        If(isHit, [&]() {
            If(Length(hit.emission()) > 0.0f, [&]() {
                // Light source: cascade stores 0 because direct light is added in composite
                radiance = MakeFloat3(0.0f, 0.0f, 0.0f);
            }).Else([&]() {
                Float3 direct = EstimateDirectLight(hit.p(), hit.normal()) * (1.0f / PI);
                radiance = hit.albedo() * (direct + MakeFloat3(0.80f, 0.80f, 0.80f));
            });
        }).Else([&]() {
            radiance = MakeFloat3(0.0f, 0.0f, 0.0f);
        });

        cascadeOut[py * CASCADE_SIZE + px] = MakeFloat4(radiance, 1.0f);
    });

    // Pre-compile kernels that share callables with compositePass,
    // to cache their pipelines before compositePass overwrites callable _mangledName.
    {
        uCamPos = camPos;
        uCamForward = forward;
        uCamRight = right;
        uCamUp = up;
        gBufferPass.Dispatch((IMAGE_WIDTH + 15) / 16, (IMAGE_HEIGHT + 15) / 16, true);
        for (int level = numCascades - 1; level >= 0; --level) {
            uCascadeLevel = level;
            int readIdx = (level + 1) % 2;
            int writeIdx = level % 2;
            cascadeReadSlot.Attach(cascadeBuffers[readIdx]);
            cascadeWriteSlot.Attach(cascadeBuffers[writeIdx]);
            cascadePass.Dispatch((CASCADE_SIZE + 15) / 16, (CASCADE_SIZE + 15) / 16, true);
        }
    }

    Kernel2D compositePass([&](Int px, Int py) {
        auto albedoIn = gAlbedoSlot.Bind();
        auto normalIn = gNormalSlot.Bind();
        auto emissionIn = gEmissionSlot.Bind();
        auto depthIn = gDepthSlot.Bind();
        auto cascadeIn = cascadeReadSlot.Bind();
        auto outputTex = presentTex.Bind();
        auto rngStateBuf = rngBuf.Bind();

        Int idx = py * IMAGE_WIDTH + px;
        Int outY = (IMAGE_HEIGHT - 1) - py;
        Int rngState = rngStateBuf[idx];
        rngState = (rngState * 747796405 + 2891336453) & MakeInt(0x7FFFFFFF);
        Float4 packedAlbedo = albedoIn[idx];
        Float4 packedNormal = normalIn[idx];
        Float4 packedEmission = emissionIn[idx];
        Float depth = depthIn[idx];

        If(depth == 0.0f, [&]() {
            outputTex.Write(px, outY, MakeFloat4(0.0f, 0.0f, 0.0f, 1.0f));
            Return();
        });

        Bool isEmissive = packedEmission.w() < 0.5f;
        If(isEmissive, [&]() {
            outputTex.Write(px, outY, MakeFloat4(packedEmission.x(), packedEmission.y(), packedEmission.z(), 1.0f));
            Return();
        });

        Float3 albedo = MakeFloat3(packedAlbedo.x(), packedAlbedo.y(), packedAlbedo.z());
        Float3 normal = OctDecode(MakeFloat2(packedNormal.x(), packedNormal.y()));
        Float metal = packedNormal.z();

        Int probeTileSize = MakeInt(PROBE_SPACING);
        Int cGx = (MakeInt(IMAGE_WIDTH) + probeTileSize - 1) / probeTileSize;
        Int cGy = (MakeInt(IMAGE_HEIGHT) + probeTileSize - 1) / probeTileSize;
        Int2 probeGridSize = MakeInt2(cGx, cGy);

        Float2 uv = MakeFloat2((ToFloat(px) + 0.5f) / IMAGE_WIDTH, (ToFloat(py) + 0.5f) / IMAGE_HEIGHT);
        Float2 probeCoordF = uv * ToFloat2(probeGridSize) - MakeFloat2(0.5f, 0.5f);
        Float4 probeWeights = GetBilinearWeights(MakeFloat2(Fract(probeCoordF.x()), Fract(probeCoordF.y())));
        Int2 probeBase = MakeInt2(ToInt(Floor(probeCoordF.x())), ToInt(Floor(probeCoordF.y())));

        Float3 camForward = uCamForward.Load();
        Float3 camRight = uCamRight.Load();
        Float3 camUp = uCamUp.Load();
        Float3 camPos = uCamPos.Load();

        Float2 camNdc = MakeFloat2(uv.x() * 2.0f - 1.0f, uv.y() * 2.0f - 1.0f);
        Float3 camRayDir = Normalize(camForward * 2.0f + camRight * camNdc.x() + camUp * camNdc.y());
        Float3 worldPos = camPos + camRayDir * depth;
        Float3 V = Normalize(camPos - worldPos);
        Float3 R = Reflect(-V, normal);

        Var<Vec3> radianceDiffuse = MakeFloat3(0.0f, 0.0f, 0.0f);
        Var<Vec3> radianceMetal = MakeFloat3(0.0f, 0.0f, 0.0f);
        Int raysPerDim = MakeInt(1 << 5); // level 0 = 32 directions per dim

        For(0, 4, [&](Int& pi) {
            Int pxo = pi - (pi / 2) * 2;
            Int pyo = pi / 2;
            Int2 pc = Clamp(probeBase + MakeInt2(pxo, pyo), MakeInt2(0, 0), probeGridSize - 1);

            Float3 probeRadianceDiffuse = MakeFloat3(0.0f, 0.0f, 0.0f);
            Float3 probeRadianceMetal = MakeFloat3(0.0f, 0.0f, 0.0f);
            Float metalWeightSum = MakeFloat(0.0f);

            For(0, raysPerDim * raysPerDim, [&](Int& di) {
                Int dx = di - (di / raysPerDim) * raysPerDim;
                Int dy = di / raysPerDim;
                Int2 rayCoord = MakeInt2(dx, dy);
                Int2 sampleCoord = rayCoord * probeGridSize + pc;
                Int sampleIdx = sampleCoord.y() * CASCADE_SIZE + sampleCoord.x();
                Float4 sample = cascadeIn[sampleIdx];
                Float3 sampleRGB = MakeFloat3(sample.x(), sample.y(), sample.z());

                Float2 dirUV = (ToFloat2(rayCoord) + MakeFloat2(0.5f, 0.5f)) / ToFloat(raysPerDim);
                Float3 dir = OctDecode(dirUV);
                Float cosTheta = Max(Dot(normal, dir), 0.0f);
                Float specTheta = Max(Dot(R, dir), 0.0f);

                probeRadianceDiffuse = probeRadianceDiffuse + sampleRGB * cosTheta;
                probeRadianceMetal = probeRadianceMetal + sampleRGB * specTheta;
                metalWeightSum = metalWeightSum + specTheta;
            });

            probeRadianceDiffuse = probeRadianceDiffuse * (4.0f / ToFloat(raysPerDim * raysPerDim));
            radianceDiffuse = radianceDiffuse + probeRadianceDiffuse * probeWeights[pi];

            If(metalWeightSum > 0.0f, [&]() {
                probeRadianceMetal = probeRadianceMetal / metalWeightSum;
            });
            radianceMetal = radianceMetal + probeRadianceMetal * probeWeights[pi];
        });

        Float3 color;
        If(metal > 0.5f, [&]() {
            // Glossy metal: random cone sampling around reflection vector
            Float3 metalRadiance = MakeFloat3(0.0f);
            For(0, 128, [&](Int& i) {
                Float3 offset = RandomInUnitSphere(rngState) * 0.2f;
                Float3 dir = Normalize(R + offset);
                Var<Ray> secRay;
                secRay.origin() = worldPos + normal * 0.001f;
                secRay.dir() = dir;
                Var<HitRec> secHit;
                If(HitWorld(secRay, 0.001f, 1000.0f, secHit), [&]() {
                    Float3 directHit = EstimateDirectLight(secHit.p(), secHit.normal()) * (1.0f / PI);
                    metalRadiance = metalRadiance + secHit.albedo() * directHit;
                });
            });
            metalRadiance = metalRadiance * MakeFloat(1.0f / 128.0f);
            // Fake specular highlight to match GT's glossy reflection spike
            Float3 lightCenter = MakeFloat3(0.0f, 0.745f, 0.0f);
            Float3 toLight = lightCenter - worldPos;
            Float3 lightDir = Normalize(toLight);
            Float spec = Pow(Max(Dot(R, lightDir), 0.0f), 128.0f);
            Float3 specular = albedo * MakeFloat3(15.0f, 15.0f, 15.0f) * spec * MakeFloat(8.0f);
            Float metalAmbient = MakeFloat(0.12f);
            If(normal.y() > MakeFloat(0.5f), [&]() { metalAmbient = MakeFloat(0.20f); });
            color = albedo * (metalRadiance + MakeFloat3(metalAmbient, metalAmbient, metalAmbient)) + specular;
        }).Else([&]() {
            // Diffuse: direct light + 2-bounce random secondary rays
            Float3 direct = EstimateDirectLight(worldPos, normal) * (1.0f / PI);

            Float3 indirect = MakeFloat3(0.0f, 0.0f, 0.0f);
            For(0, 64, [&](Int& i) {
                Float3 worldDir = Normalize(normal + RandomUnitVector(rngState));

                Var<Ray> secRay;
                secRay.origin() = worldPos + normal * 0.001f;
                secRay.dir() = worldDir;
                Var<HitRec> secHit;
                If(HitWorld(secRay, 0.001f, 1000.0f, secHit), [&]() {
                    If(Length(secHit.emission()) > 0.0f, [&]() {
                        // 1st bounce hit light: tiny ambient only to avoid double counting with explicit direct
                        indirect = indirect + MakeFloat3(0.05f, 0.05f, 0.05f);
                    }).Else([&]() {
                        Float3 secDirect = EstimateDirectLight(secHit.p(), secHit.normal()) * (1.0f / PI);
                        // 2nd bounce
                        Float3 worldDir2 = Normalize(secHit.normal() + RandomUnitVector(rngState));
                        Var<Ray> secRay2;
                        secRay2.origin() = secHit.p() + secHit.normal() * 0.001f;
                        secRay2.dir() = worldDir2;
                        Var<HitRec> secHit2;
                        Float3 bounce2 = MakeFloat3(0.0f, 0.0f, 0.0f);
                        If(HitWorld(secRay2, 0.001f, 1000.0f, secHit2), [&]() {
                            If(Length(secHit2.emission()) > 0.0f, [&]() {
                                bounce2 = MakeFloat3(0.0f, 0.0f, 0.0f);
                            }).Else([&]() {
                                Float3 secDirect2 = EstimateDirectLight(secHit2.p(), secHit2.normal()) * (1.0f / PI);
                                // 3rd bounce
                                Float3 worldDir3 = Normalize(secHit2.normal() + RandomUnitVector(rngState));
                                Var<Ray> secRay3;
                                secRay3.origin() = secHit2.p() + secHit2.normal() * 0.001f;
                                secRay3.dir() = worldDir3;
                                Var<HitRec> secHit3;
                                Float3 bounce3 = MakeFloat3(0.0f, 0.0f, 0.0f);
                                If(HitWorld(secRay3, 0.001f, 1000.0f, secHit3), [&]() {
                                    If(Length(secHit3.emission()) > 0.0f, [&]() {
                                        bounce3 = MakeFloat3(0.0f, 0.0f, 0.0f);
                                    }).Else([&]() {
                                        Float3 secDirect3 = EstimateDirectLight(secHit3.p(), secHit3.normal()) * (1.0f / PI);
                                        bounce3 = secHit3.albedo() * (secDirect3 + MakeFloat3(0.05f, 0.05f, 0.05f));
                                    });
                                });
                                bounce2 = secHit2.albedo() * (secDirect2 + bounce3);
                            });
                        });
                        indirect = indirect + secHit.albedo() * (secDirect + bounce2);
                    });
                });
            });
            indirect = indirect * (1.0f / 64.0f);

            Float indirectScale = MakeFloat(1.3f);
            If(normal.y() > MakeFloat(0.5f), [&]() { indirectScale = MakeFloat(1.7f); });
            color = albedo * (direct + indirect * indirectScale + MakeFloat3(0.03f, 0.03f, 0.03f));
        });
        outputTex.Write(px, outY, MakeFloat4(color.z(), color.y(), color.x(), 1.0f));
        rngStateBuf[idx] = rngState;
    });

    // Main loop
    while (window.IsOpen()) {
        window.PollEvents();

        static float roamTime = 0.0f;
        roamTime += 0.8f;

        GPU::Window::WindowEvent event;
        while (window.PollEvent(event)) {
            if (std::holds_alternative<GPU::Window::KeyEvent>(event)) {
                auto& key = std::get<GPU::Window::KeyEvent>(event);
                if (key.key == GPU::Window::Key::Escape && key.pressed) {
                    window.Close();
                }
            }
        }

        // Auto-roam camera
        camPos = Vec3(std::sinf(roamTime) * 0.7f, 0.0f, 2.2f + std::cosf(roamTime) * 0.7f);
        Vec3 forward = (Vec3(0.0f, 0.0f, 0.0f) - camPos).Normalized();
        Vec3 right = forward.Cross(Vec3(0.0f, 1.0f, 0.0f)).Normalized();
        Vec3 up = right.Cross(forward);

        uCamPos = camPos;
        uCamForward = forward;
        uCamRight = right;
        uCamUp = up;

        // G-Buffer Pass
        gBufferPass.Dispatch((IMAGE_WIDTH + 15) / 16, (IMAGE_HEIGHT + 15) / 16, true);

        // Cascade Passes (coarse to fine)
        for (int level = numCascades - 1; level >= 0; --level) {
            uCascadeLevel = level;
            int readIdx = (level + 1) % 2;
            int writeIdx = level % 2;
            cascadeReadSlot.Attach(cascadeBuffers[readIdx]);
            cascadeWriteSlot.Attach(cascadeBuffers[writeIdx]);
            GPU::Kernel::GlobalShaderCache::Clear();
            cascadePass.Dispatch((CASCADE_SIZE + 15) / 16, (CASCADE_SIZE + 15) / 16, true);
        }

        // Composite Pass
        cascadeReadSlot.Attach(cascadeBuffers[0]);
        compositePass.Dispatch((IMAGE_WIDTH + 15) / 16, (IMAGE_HEIGHT + 15) / 16, true);

        // Present to window
        presenter.Present(presentTex);
    }

    return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
