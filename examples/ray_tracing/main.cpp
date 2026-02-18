/**
 * RayTracing:
 *      @Descripiton    :   Monte Carlo path tracing renderer for Cornell Box
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#include <Kernel/Kernel.h>
#include <IR/Value/Var.h>
#include <IR/Value/BufferRef.h>
#include <Runtime/Buffer.h>
#include <Flow/ForFlow.h>
#include <Flow/IfFlow.h>
#include <Flow/BreakFlow.h>
#include <Flow/ReturnFlow.h>
#include <Callable/Callable.h>
#include <Utility/Vec.h>
#include <Utility/Math.h>
#include <Utility/Helpers.h>
#include <Utility/Meta/StructMeta.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>

#include <iostream>
#include <vector>
#include <cmath>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Flow;
using namespace GPU::Runtime;
using namespace GPU::Callables;

// =============================================================================
// Configuration
// =============================================================================
constexpr int IMAGE_WIDTH = 512;
constexpr int IMAGE_HEIGHT = 512;
constexpr int SAMPLES_PER_PIXEL = 2560 * 4;
constexpr int MAX_DEPTH = 8;

// =============================================================================
// GPU Struct Definitions
// =============================================================================
EASYGPU_STRUCT(Ray,
    (Vec3, origin),
    (Vec3, dir)
);

EASYGPU_STRUCT(Material,
    (Vec3, albedo),
    (int, type)  // 0=diffuse, 1=metal, 2=light
);

EASYGPU_STRUCT(HitRec,
    (Vec3, p),
    (Vec3, normal),
    (float, t),
    (Material, mat)
);


Callable<float(int&)> Random = [](Int& state) {
    state = (state * 747796405 + 2891336453) & 0x7FFFFFFF;
    Int word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    Int result = (word >> 22) ^ word;
    result = Abs(result);

    Return(ToFloat(result) / 2147483647.0f);
    };
Callable<Vec3(int&)> RandomInUnitSphere = [](Int& state) {
    Float3 p;
    For(0, 50, [&](Int&) {
        p = MakeFloat3(Random(state), Random(state), Random(state)) * 2.0f - MakeFloat3(1.0f, 1.0f, 1.0f);
        });

    Return(p);
    };
Callable<Vec3(int&)> RandomUnitVector = [](Int& state) {
    Return(Normalize(RandomInUnitSphere(state)));
    };

// =============================================================================
// Ray Helpers
// =============================================================================
Callable<Vec3(Ray&, float)> RayAt = [](Var<Ray>& r, Float t) {
    Return(r.origin() + r.dir() * t);
    };

Callable<void(Ray&, Vec3&, Vec3&)> SetRay = [](Var<Ray>& r, Float3& o, Float3& d) {
    r.origin() = o;
    r.dir() = d;
    };


// =============================================================================
// Box Intersection
// =============================================================================
Callable<bool(Vec3, Vec3, Ray&, float, float&, HitRec&, Material&)> HitBox = [](Var<Vec3> bmin, Float3 bmax,
    Var<Ray>& r, Float tmin, Float& closest,
    Var<HitRec>& rec, Var<Material>& mat) {
        Bool hit = MakeBool(false);
        Float3 n;
        Float tmax = closest;
        Float tc = tmax;

        // X planes
        Float t = (bmin.x() - r.origin().x()) / r.dir().x();
        If(t > tmin && t < tc, [&]() {
            Float3 p = RayAt(r, t);
            If(p.y() > bmin.y() && p.y() < bmax.y() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
                tc = Expr<float>(t); n = Vec3(-1.0f, 0.0f, 0.0f); hit = true;
                });
            });
        t = (bmax.x() - r.origin().x()) / r.dir().x();
        If(t > tmin && t < tc, [&]() {
            Float3 p = RayAt(r, t);
            If(p.y() > bmin.y() && p.y() < bmax.y() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
                tc = Expr<float>(t); n = Vec3(1.0f, 0.0f, 0.0f); hit = true;
                });
            });

        // Y planes
        t = (bmin.y() - r.origin().y()) / r.dir().y();
        If(t > tmin && t < tc, [&]() {
            Float3 p = RayAt(r, t);
            If(p.x() > bmin.x() && p.x() < bmax.x() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
                tc = Expr<float>(t); n = Vec3(0.0f, -1.0f, 0.0f); hit = true;
                });
            });
        t = (bmax.y() - r.origin().y()) / r.dir().y();
        If(t > tmin && t < tc, [&]() {
            Float3 p = RayAt(r, t);
            If(p.x() > bmin.x() && p.x() < bmax.x() && p.z() > bmin.z() && p.z() < bmax.z(), [&]() {
                tc = Expr<float>(t); n = Vec3(0.0f, 1.0f, 0.0f); hit = true;
                });
            });

        // Z planes
        t = (bmin.z() - r.origin().z()) / r.dir().z();
        If(t > tmin && t < tc, [&]() {
            Float3 p = RayAt(r, t);
            If(p.x() > bmin.x() && p.x() < bmax.x() && p.y() > bmin.y() && p.y() < bmax.y(), [&]() {
                tc = Expr<float>(t); n = Vec3(0.0f, 0.0f, -1.0f); hit = true;
                });
            });
        t = (bmax.z() - r.origin().z()) / r.dir().z();
        If(t > tmin && t < tc, [&]() {
            Float3 p = RayAt(r, t);
            If(p.x() > bmin.x() && p.x() < bmax.x() && p.y() > bmin.y() && p.y() < bmax.y(), [&]() {
                tc = Expr<float>(t); n = Vec3(0.0f, 0.0f, 1.0f); hit = true;
                });
            });

        If(hit, [&]() {
            rec.t() = Expr<float>(tc);
            rec.p() = RayAt(r, tc);
            rec.normal() = n;
            rec.mat() = mat;
            closest = Expr<float>(tc);
            });

        Return(hit);
    };

// =============================================================================
// Scene - Cornell Box
// =============================================================================
Callable<bool(Ray&, float, float, HitRec&, int&)> HitWorld = [](Var<Ray>& r, Float tmin, Float tmax,
    Var<HitRec>& rec, Int& rng) {
        Bool hit = MakeBool(false);
        Float closest = tmax;
        Var<HitRec> temp;

        // Floor (white, diffuse)
        Var<Material> whiteDiff;
        whiteDiff.albedo() = MakeFloat3(0.73f, 0.73f, 0.73f);
        whiteDiff.type() = 0;
        If(HitBox(MakeFloat3(-1.0f, -1.0f, -1.0f), MakeFloat3(1.0f, -0.75f, 1.0f),
            r, tmin, closest, temp, whiteDiff), [&]() { hit = true; rec = temp; });

        // Ceiling (white, diffuse)
        If(HitBox(MakeFloat3(-1.0f, 0.75f, -1.0f), MakeFloat3(1.0f, 1.0f, 1.0f),
            r, tmin, closest, temp, whiteDiff), [&]() { hit = true; rec = temp; });

        // Back (white, diffuse)
        If(HitBox(MakeFloat3(-1.0f, -0.75f, -1.0f), MakeFloat3(1.0f, 0.75f, -0.75f),
            r, tmin, closest, temp, whiteDiff), [&]() { hit = true; rec = temp; });

        // Left (red, diffuse)
        Var<Material> redDiff;
        redDiff.albedo() = MakeFloat3(0.65f, 0.05f, 0.05f);
        redDiff.type() = 0;
        If(HitBox(MakeFloat3(-1.0f, -0.75f, -0.75f), MakeFloat3(-0.75f, 0.75f, 1.0f),
            r, tmin, closest, temp, redDiff), [&]() { hit = true; rec = temp; });

        // Right (green, diffuse)
        Var<Material> greenDiff;
        greenDiff.albedo() = MakeFloat3(0.12f, 0.45f, 0.15f);
        greenDiff.type() = 0;
        If(HitBox(MakeFloat3(0.75f, -0.75f, -0.75f), MakeFloat3(1.0f, 0.75f, 1.0f),
            r, tmin, closest, temp, greenDiff), [&]() { hit = true; rec = temp; });

        // Light (emissive)
        Var<Material> lightMat;
        lightMat.albedo() = MakeFloat3(15.0f, 15.0f, 15.0f);
        lightMat.type() = 2;
        If(HitBox(MakeFloat3(-0.25f, 0.74f, -0.25f), MakeFloat3(0.25f, 0.75f, 0.25f),
            r, tmin, closest, temp, lightMat), [&]() { hit = true; rec = temp; });

        // Tall box (metal)
        Var<Material> metalMat;
        metalMat.albedo() = MakeFloat3(0.8f, 0.85f, 0.88f);
        metalMat.type() = 1;
        If(HitBox(MakeFloat3(0.15f, -0.75f, -0.4f), MakeFloat3(0.45f, -0.15f, -0.1f),
            r, tmin, closest, temp, metalMat), [&]() { hit = true; rec = temp; });

        // Short box (diffuse)
        If(HitBox(MakeFloat3(-0.4f, -0.75f, 0.0f), MakeFloat3(-0.1f, -0.4f, 0.3f),
            r, tmin, closest, temp, whiteDiff), [&]() { hit = true; rec = temp; });

        Return(hit);
    };

// =============================================================================
// Material Scatter
// =============================================================================
Callable<Vec3(HitRec&, Ray&, bool&, int&)> Scatter = [](Var<HitRec>& rec, Var<Ray>& rIn, Bool& scattered, Int& rng) {
    Int matType = rec.mat().type();

    If(matType == 2, [&]() {
        scattered = false;
        Return(rec.mat().albedo());
        });

    If(matType == 1, [&]() {
        Float3 refl = Reflect(Normalize(rIn.dir()), rec.normal());
        scattered = true;

        Return(refl + 0.2f * RandomInUnitSphere(rng));
        });

    // Diffuse
    Float3 target = rec.normal() + RandomUnitVector(rng);
    scattered = true;
    Return(target);
    };

// =============================================================================
// Path Tracing
// =============================================================================
Callable<Vec3(Ray&, int&)> Trace = [](Var<Ray>& r, Int& rng) {
    Float3 color = MakeFloat3(1.0f);
    Var<Ray> cur;
    SetRay(cur, r.origin(), r.dir());

    For(0, MAX_DEPTH, [&](Int&) {
        Var<HitRec> rec;
        If(!HitWorld(cur, 0.001f, 1000.0f, rec, rng), [&]() {
            color = Vec3(0.0f);
            Break();
            }).Else([&]() {
                If(rec.mat().type() == 2, [&]() {
                    color = color * rec.mat().albedo();
                    Break();
                    }).Else([&]() {
                        Bool scat = MakeBool(false);
                        Float3 dir = Scatter(rec, cur, scat, rng);

                        If(scat, [&]() {
                            color = color * rec.mat().albedo();
                            Float3 newDir = Normalize(dir);
                            SetRay(cur, rec.p(), newDir);
                            }).Else([&]() {
                                color = Vec3(0.0f);
                                Break();
                                });
                        });
                });
        });

    Return(color);
    };

// =============================================================================
// Camera
// =============================================================================

Callable<void(float, float, Ray&)> CameraRay = [](Float u, Float v, Var<Ray>& ray) {
    Float3 origin = MakeFloat3(0.0f, 0.0f, 2.5f);
    Float3 llc = MakeFloat3(-1.0f, -1.0f, 0.5f);
    Float3 h = MakeFloat3(2.0f, 0.0f, 0.0f);
    Float3 vert = MakeFloat3(0.0f, 2.0f, 0.0f);
    Float3 dir = Normalize(llc + u * h + v * vert - origin);
    SetRay(ray, origin, dir);
    };

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "Cornell Box Path Tracer\n";
    std::cout << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << " @ " << SAMPLES_PER_PIXEL << "spp\n\n";

    Buffer<Vec4> image(IMAGE_WIDTH * IMAGE_HEIGHT, BufferMode::Write);

    std::vector<int> seeds(IMAGE_WIDTH * IMAGE_HEIGHT);
    for (size_t i = 0; i < seeds.size(); ++i) seeds[i] = static_cast<int>(i + 1);
    Buffer<int> rngBuf(seeds, BufferMode::ReadWrite);

    std::cout << "Rendering...\n";

    Kernel::Kernel2D kernel([&](Int& px, Int& py) {
        auto img = image.Bind();
        auto state = rngBuf.Bind();

        Int idx = py * IMAGE_WIDTH + px;
        Int rngState = state[idx];
        Float3 col = MakeFloat3(0.0f);

        For(0, SAMPLES_PER_PIXEL, [&](Int&) {
            Float u = (Expr<float>(px) + Random(rngState)) / IMAGE_WIDTH;
            Float v = (Expr<float>(py) + Random(rngState)) / IMAGE_HEIGHT;
            Var<Ray> r;
            CameraRay(u, v, r);
            col = col + Trace(r, rngState);
            });

        Float scale = MakeFloat(1.0f / SAMPLES_PER_PIXEL);
        img[idx] = MakeFloat4(
            Sqrt(Clamp(col.x() * scale, 0.0f, 1.0f)),
            Sqrt(Clamp(col.y() * scale, 0.0f, 1.0f)),
            Sqrt(Clamp(col.z() * scale, 0.0f, 1.0f)),
            1.f
        );
        state[idx] = Expr<int>(rngState);
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

    if (stbi_write_png("cornell_box.png", IMAGE_WIDTH, IMAGE_HEIGHT, 3, imageData.data(), IMAGE_WIDTH * 3)) {
        std::cout << "Saved to cornell_box.png\n";
    }
    else {
        std::cout << "Failed to save image\n";
        return 1;
    }

    return 0;
}