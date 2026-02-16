/**
 * SDFRenderer:
 *      @Descripiton    :   Signed Distance Field path tracing renderer
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/15/2026
 *
 *  Credit: https://github.com/taichi-dev/taichi/blob/master/examples/rendering/sdf_renderer.py
 *  Ported from LuisaCompute test_sdf_renderer.cpp
 */
#include <Kernel/Kernel.h>
#include <IR/Value/Var.h>
#include <IR/Value/ExprVector.h>
#include <Runtime/Buffer.h>
#include <Flow/ForFlow.h>
#include <Flow/IfFlow.h>
#include <Flow/BreakFlow.h>
#include <Flow/ReturnFlow.h>
#include <Callable/Callable.h>
#include <Utility/Vec.h>
#include <Utility/Math.h>
#include <Utility/Helpers.h>

#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image_write.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Flow;
using namespace GPU::Runtime;
using namespace GPU::Callables;

// =============================================================================
// Configuration
// =============================================================================
constexpr int MAX_RAY_DEPTH = 6;
constexpr float EPS = 1e-4f;
constexpr float INF = 1e10f;
constexpr float FOV = 0.23f;
constexpr float DIST_LIMIT = 100.0f;
constexpr int IMAGE_WIDTH = 1280;
constexpr int IMAGE_HEIGHT = 720;
constexpr int TOTAL_SPP = 1240;

const float LIGHT_RADIUS_F = 2.0f;

// =============================================================================
// Helper functions
// =============================================================================

// Intersect with rectangular light
Callable<float(Vec3, Vec3)> intersect_light = [](Float3 pos, Float3 d) {
    Float3 light_pos = MakeFloat3(-1.5f, 0.6f, 0.3f);
    Float3 light_normal = MakeFloat3(1.0f, 0.0f, 0.0f);
    Float cos_w = Dot(-d, light_normal);
    Float dist = Dot(d, light_pos - pos);
    Float D = dist / cos_w;
    Float3 hit_point = pos + D * d;
    Float dist_to_center = Distance(light_pos, hit_point);

    Bool valid = (cos_w > 0.0f) && (dist > 0.0f) && (dist_to_center < LIGHT_RADIUS_F);

    Float result;
    If(valid, [&]() {
        result = D;
        }).Else([&]() {
            result = INF;
            });
        Return(result);
    };

// Tea hash function for random number generation
Callable<int(int, int)> tea = [](Int v0, Int v1) {
    Int s0 = MakeInt(0);
    Int sum = v0;
    Int sum2 = v1;

    For(0, 4, [&](Int&) {
        s0 = s0 + MakeInt(0x9e3779b9);
        sum = sum + (((sum2 << 4) + 0xa341316c) ^ (sum2 + s0) ^ ((sum2 >> 5) + 0xc8013ea4));
        sum2 = sum2 + (((sum << 4) + 0xad90777d) ^ (sum + s0) ^ ((sum >> 5) + 0x7e95761e));
        });

    Return(sum);
    };

// Random number generator
Callable<float(int&)> rand_f = [](Int& state) {
    constexpr int lcg_a = 1664525;
    constexpr int lcg_c = 1013904223;
    state = lcg_a * state + lcg_c;
    Float result = (Expr<float>(state & 0x7FFFFFFF) / 2147483648.0f);
    Return(result);
    };

// Generate outgoing direction (cosine weighted hemisphere sampling)
Callable<Vec3(Vec3, int&)> out_dir = [](Float3 n, Int& seed) {
    Float3 u;
    If(Abs(n.y()) < 1.0f - EPS, [&]() {
        u = Normalize(Cross(n, MakeFloat3(0.0f, 1.0f, 0.0f)));
        }).Else([&]() {
            u = MakeFloat3(1.0f, 0.0f, 0.0f);
            });
        Float3 v = Cross(n, u);

        Float phi = 2.0f * std::numbers::pi_v<float> *rand_f(seed);
        Float ay = Sqrt(rand_f(seed));
        Float ax = Sqrt(1.0f - ay * ay);

        Return(ax * (Cos(phi) * u + Sin(phi) * v) + ay * n);
    };

// Helper to cast float to int for modulo
Callable<int(float)> FloatToInt = [](Float f) {
    Int result = ToInt(f);
    If(f < 0.0f, [&]() {
        result = result - MakeInt(1);
        });
    Return(result);
    };

// Create nested pattern
Callable<float(float)> make_nested = [](Float f) {
    constexpr float freq = 40.0f;
    f = f * freq;

    Float result;
    If(f < 0.0f, [&]() {
        Int f_int = FloatToInt(f);
        Float fract_val = Fract(f);
        If(f_int % 2 == 0, [&]() {
            result = 1.0f - fract_val;
            }).Else([&]() {
                result = fract_val;
                });
        }).Else([&]() {
            result = f;
            });

        Return((result - 0.2f) * (1.0f / freq));
    };

// Signed Distance Function
Callable<float(Vec3)> sdf = [](Float3 o) {
    Float wall = Min(o.y() + 0.1f, o.z() + 0.4f);
    Float sphere = Distance(o, MakeFloat3(0.0f, 0.35f, 0.0f)) - 0.36f;

    Float3 q = Abs(o - MakeFloat3(0.8f, 0.3f, 0.0f)) - 0.3f;
    Float box = Length(Max(q, 0.0f)) + Min(Max(Max(q.x(), q.y()), q.z()), 0.0f);

    Float3 O = o - MakeFloat3(-0.8f, 0.3f, 0.0f);
    Var<Vec2> d = MakeFloat2(Length(MakeFloat2(O.x(), O.z())) - 0.3f, Abs(O.y()) - 0.3f);
    Float cylinder = Min(Max(d.x(), d.y()), 0.0f) + Length(Max(d, 0.0f));

    Float geometry = make_nested(Min(Min(sphere, box), cylinder));
    Float g = Max(geometry, -(0.32f - (o.y() * 0.6f + o.z() * 0.8f)));

    Return(Min(wall, g));
    };

// Ray marching
Callable<float(Vec3, Vec3)> ray_march = [](Float3 p, Float3 d) {
    Float dist = MakeFloat(0.0f);

    For(0, 100, [&](Int&) {
        Float s = sdf(p + dist * d);
        If(s <= 1e-6f || dist >= INF, [&]() {
            Break();
            });
        dist = dist + s;
        });

    Return(Min(dist, INF));
    };

// Compute SDF normal
Callable<Vec3(Vec3)> sdf_normal = [](Float3 p) {
    constexpr float delta = 1e-3f;
    Float3 n;
    Float sdf_center = sdf(p);

    Float3 inc_x = p;
    inc_x.x() = inc_x.x() + delta;
    n.x() = (1.0f / delta) * (sdf(inc_x) - sdf_center);

    Float3 inc_y = p;
    inc_y.y() = inc_y.y() + delta;
    n.y() = (1.0f / delta) * (sdf(inc_y) - sdf_center);

    Float3 inc_z = p;
    inc_z.z() = inc_z.z() + delta;
    n.z() = (1.0f / delta) * (sdf(inc_z) - sdf_center);

    Return(Normalize(n));
    };

// Find next hit
Callable<void(float&, Vec3&, Vec3&, Vec3, Vec3)> next_hit = [](
    Float& closest, Float3& normal, Float3& c, Float3 pos, Float3 d) {

        closest = INF;
        normal = MakeFloat3(0.0f, 0.0f, 0.0f);
        c = MakeFloat3(0.0f, 0.0f, 0.0f);

        Float ray_march_dist = ray_march(pos, d);
        If(ray_march_dist < Min(DIST_LIMIT, closest), [&]() {
            closest = ray_march_dist;
            Float3 hit_pos = pos + d * closest;
            normal = sdf_normal(hit_pos);
            Int t = FloatToInt((hit_pos.x() + 10.0f) * 1.1f + 0.5f) % 3;

            Float3 base_color = MakeFloat3(0.4f, 0.4f, 0.4f);
            Float3 pattern = MakeFloat3(0.3f, 0.2f, 0.3f);

            If(t == 0, [&]() {
                c = base_color + pattern * 1.0f;
                }).Elif(t == 1, [&]() {
                    c = base_color + MakeFloat3(pattern.y(), pattern.x(), pattern.z()) * 1.0f;
                    }).Else([&]() {
                        c = base_color + MakeFloat3(pattern.z(), pattern.y(), pattern.x()) * 1.0f;
                        });
            });
    };

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "SDF Path Tracer\n";
    std::cout << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << " @ " << TOTAL_SPP << "spp\n\n";

    // Create buffers
    Buffer<int> seed_buffer(IMAGE_WIDTH * IMAGE_HEIGHT, BufferMode::ReadWrite);
    Buffer<Vec4> accum_buffer(IMAGE_WIDTH * IMAGE_HEIGHT, BufferMode::ReadWrite);

    // Initialize seeds
    std::vector<int> seeds(IMAGE_WIDTH * IMAGE_HEIGHT);
    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            int idx = y * IMAGE_WIDTH + x;
            int v0 = x;
            int v1 = y;
            int s0 = 0;
            for (int n = 0; n < 4; n++) {
                s0 += 0x9e3779b9;
                v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
                v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
            }
            seeds[idx] = v0;
        }
    }
    seed_buffer.Upload(seeds);

    // Initialize accum buffer to zero
    std::vector<Vec4> zeros(IMAGE_WIDTH * IMAGE_HEIGHT, Vec4(0.0f));
    accum_buffer.Upload(zeros);

    std::cout << "Rendering...\n";

    // Single kernel dispatch - all samples computed in GPU
    Kernel::Kernel2D render_kernel([&](Int& px, Int& py) {
        auto seeds = seed_buffer.Bind();
        auto accums = accum_buffer.Bind();

        Int idx = py * IMAGE_WIDTH + px;
        Int seed = seeds[idx];
        Var<Vec4> accum = accums[idx];

        float aspect_ratio = static_cast<float>(IMAGE_WIDTH) / IMAGE_HEIGHT;

        // Accumulate multiple samples per pixel
        For(0, TOTAL_SPP, [&](Int& frame) {
            Float3 pos = MakeFloat3(0.0f, 0.32f, 3.7f);

            // Jittered UV
            Float ux = rand_f(seed);
            Float uy = rand_f(seed);
            Float u = (Expr<float>(px) + ux);
            Float v = (Expr<float>(py) + uy);

            // Ray direction
            Float du = 2.0f * FOV * u / IMAGE_HEIGHT - FOV * aspect_ratio - 1e-5f;
            Float dv = 2.0f * FOV * v / IMAGE_HEIGHT - FOV - 1e-5f;
            Float3 d = Normalize(MakeFloat3(du, dv, -1.0f));

            // Path tracing
            Float3 throughput = MakeFloat3(1.0f, 1.0f, 1.0f);
            Float hit_light = MakeFloat(0.0f);

            For(0, MAX_RAY_DEPTH, [&](Int&) {
                Float closest;
                Float3 normal = Float3(0.0f, 0.0f, 0.0f);
                Float3 c = Float3(0.0f, 0.0f, 0.0f);
                next_hit(closest, normal, c, pos, d);

                Float dist_to_light = intersect_light(pos, d);
                If(dist_to_light < closest, [&]() {
                    hit_light = 1.0f;
                    Break();
                    });

                If(Length(normal) == 0.0f, [&]() {
                    Break();
                    });

                Float3 hit_pos = pos + closest * d;
                d = out_dir(normal, seed);
                pos = hit_pos + 1e-4f * d;
                throughput = throughput * c;
                });

            // Accumulate color
            Float3 current = accum.xyz();
            Float3 sample_color = throughput * hit_light;
            Float3 new_color = Mix(current, sample_color, 1.0f / (Expr<float>(frame) + 1.0f));
            accum = MakeFloat4(new_color.x(), new_color.y(), new_color.z(), 1.0f);
            });

        accums[idx] = accum;
        seeds[idx] = seed;
        });

    render_kernel.Dispatch((IMAGE_WIDTH + 15) / 16, (IMAGE_HEIGHT + 15) / 16, true);

    std::cout << "Rendering complete!\n";

    // Download and convert to LDR
    std::vector<Vec4> hdr_pixels;
    accum_buffer.Download(hdr_pixels);

    std::vector<unsigned char> imageData(IMAGE_WIDTH * IMAGE_HEIGHT * 4);
    float exposure_scale = 2.0f;

    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        for (int x = 0; x < IMAGE_WIDTH; ++x) {
            int srcIdx = (IMAGE_HEIGHT - 1 - y) * IMAGE_WIDTH + x;
            int dstIdx = (y * IMAGE_WIDTH + x) * 4;

            Vec3 hdr = Vec3(hdr_pixels[srcIdx].x, hdr_pixels[srcIdx].y, hdr_pixels[srcIdx].z);
            hdr = hdr * exposure_scale;

            auto to_srgb = [](float c) {
                if (c <= 0.00031308f) return c * 12.92f;
                return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
                };

            float r = std::clamp(to_srgb(hdr.x), 0.0f, 1.0f);
            float g = std::clamp(to_srgb(hdr.y), 0.0f, 1.0f);
            float b = std::clamp(to_srgb(hdr.z), 0.0f, 1.0f);

            imageData[dstIdx + 0] = static_cast<unsigned char>(255 * r);
            imageData[dstIdx + 1] = static_cast<unsigned char>(255 * g);
            imageData[dstIdx + 2] = static_cast<unsigned char>(255 * b);
            imageData[dstIdx + 3] = 255;
        }
    }

    if (stbi_write_png("sdf_renderer.png", IMAGE_WIDTH, IMAGE_HEIGHT, 4, imageData.data(), IMAGE_WIDTH * 4)) {
        std::cout << "Saved to sdf_renderer.png\n";
    }
    else {
        std::cout << "Failed to save image\n";
        return 1;
    }

    return 0;
}
