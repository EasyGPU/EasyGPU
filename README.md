 <div align="center">

<img src="docs/image/logo.png" alt="EasyGPU Logo" width="180">

# EasyGPU

C++20 Embedded DSL for GPU Compute, Rasterization, Autograd & Neural Networks

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-orange.svg)](https://en.cppreference.com/w/cpp/20)
[![Feather C# Frontend](https://img.shields.io/badge/C%23%20Frontend-Feather-512BD4.svg)](https://github.com/FeatherCompute/Feather)
[![OpenGL](https://img.shields.io/badge/OpenGL-4.3+-green.svg)](https://www.opengl.org/)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.1+-red.svg)](https://www.vulkan.org/)
[![Dear ImGui](https://img.shields.io/badge/Dear%20ImGui-Integrated-blueviolet.svg)](https://github.com/ocornut/imgui)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![Autograd](https://img.shields.io/badge/Autograd-Reverse--Mode-blue.svg)](docs/autodiff.md)

[Getting Started](docs/getting-started.md) · [Tutorial](docs/tutorial.md) · [Feather C# Frontend](https://github.com/FeatherCompute/Feather) · [Graphics Pipeline](docs/graphics-pipeline.md) · [API Reference](docs/api-reference.md)

**Feather** brings EasyGPU to C# and .NET developers.

</div>

---

## Table of Contents

- [Overview](#overview)
- [C# Frontend](#c-frontend--feather)
- [Concept](#concept)
- [Features](#features)
  - [Automatic Differentiation](#automatic-differentiation--gpu-gradients-zero-hand-written-math)
  - [Neural Network Training](#neural-network-training--tensor--optimizer)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Ecosystem](#ecosystem)
- [Best Practices](#best-practices)
- [Documentation](#documentation)
- [Building](#building)
- [License](#license)

---

## Overview

EasyGPU is an embedded domain-specific language (eDSL) for GPU programming that allows writing compute kernels in standard C++20. No shader language knowledge required.

EasyGPU now ships with **reverse-mode automatic differentiation** — write your forward pass once and get GPU gradients for free. No hand-written derivatives, no dual-codebase synchronization. Train models, fit curves, and solve inverse problems entirely on the GPU.

The same DSL runs on both an OpenGL compute backend and a Vulkan compute backend, so you can target a lightweight OpenGL path or a modern Vulkan path without changing a line of kernel code.

```cpp
#include <GPU.h>

int main() {
    std::vector<float> data(1024, 2.0f);
    Buffer<float> input(data);
    Buffer<float> output(1024);

    Kernel1D square([](Int i) {
        auto in = input.Bind();
        auto out = output.Bind();
        out[i] = in[i] * in[i];
    });

    square.Dispatch(16, true);
    return 0;
}
```

### Who Is This For

**For beginners learning GPU programming:**
- Write GPU kernels using familiar C++ syntax instead of learning GLSL/HLSL
- No graphics programming background required — works with arrays, not just triangles
- Full IDE support: autocomplete, type checking, compile-time error detection
- 10 lines of code for your first working GPU kernel

**For ML practitioners and researchers:**
- Reverse-mode autograd that runs entirely on the GPU — no Python required
- Record the forward pass in C++, get combined forward+backward GLSL automatically
- 30+ differentiable intrinsics (sin, cos, exp, log, pow, sqrt...), vector ops, control flow
- Train tiny models, fit curves, and compute gradients without leaving C++

**For C# and .NET developers:**
- Use [Feather](https://github.com/FeatherCompute/Feather), the C# frontend for EasyGPU
- Keep the same GPU-first programming model while writing host-side code in C#
- Target EasyGPU without maintaining hand-written shader strings

**For experienced developers:**
- Zero vendor lock-in (OpenGL 4.3+ or Vulkan 1.1+, cross-platform)
- Explicit dependency model: bundled GLAD for OpenGL; Vulkan SDK, glslang, and SPIRV-Tools for Vulkan
- Clean C++20 interface without heavy template metaprogramming
- First-class Vulkan backend for compute workloads, profiling, textures, uniforms, and descriptor-backed resource binding

### Requirements

- C++20 compatible compiler (GCC 11+, Clang 14+, MSVC 2022+, Apple Clang 14+)
- OpenGL 4.3+ or Vulkan 1.1+
- CMake 3.21+
- **Windows:** No additional dependencies
- **Linux:** X11 development libraries (`libx11-dev` on Ubuntu/Debian)
- **macOS:** Vulkan backend via MoltenVK; OpenGL backend is intentionally disabled
- **Vulkan builds:** Vulkan SDK with `glslang` / `SPIRV-Tools` libraries available to CMake

---

## C# Frontend — Feather

<p align="center">
  <a href="https://github.com/FeatherCompute/Feather">
    <img src="docs/image/feather.svg" alt="Feather C# Frontend" width="420">
  </a>
</p>

**[Feather](https://github.com/FeatherCompute/Feather)** is the C# frontend for EasyGPU. It brings the same embedded GPU programming model to .NET developers: write compute code from C#, let the frontend generate GPU work, and keep shader plumbing out of application logic.

| Frontend | Language | Use Case |
|:---------|:---------|:---------|
| **EasyGPU** | C++20 | Native C++ compute, graphics, autograd, and NN training |
| **Feather** | C# / .NET | C# applications that want EasyGPU-style GPU compute without hand-written shader code |

[Feather on GitHub →](https://github.com/FeatherCompute/Feather)

---

## Concept

### The Problem

Traditional GPU programming requires maintaining two separate codebases:

```cpp
// CPU: C++
std::vector<float> data = {1, 2, 3, 4, 5};

// GPU: GLSL (separate language)
const char* shader = R"(
    #version 430 core
    layout(local_size_x = 256) in;
    layout(std430, binding = 0) buffer Data { float values[]; };
    void main() {
        uint idx = gl_GlobalInvocationID.x;
        values[idx] = values[idx] * values[idx];
    }
)";
```

Issues: language fragmentation, no IDE support, runtime error detection, string-based data passing.

### The Approach

EasyGPU unifies both sides in C++:

```cpp
// CPU and GPU: C++
std::vector<float> data = {1, 2, 3, 4, 5};
Buffer<float> input(data);
Buffer<float> output(data.size());

Kernel1D square([](Int i) {
    auto in = input.Bind();
    auto out = output.Bind();
    out[i] = in[i] * in[i];
});

square.Dispatch(16, true);
```

### Implementation

1. User writes C++ kernels using EasyGPU types
2. Library constructs an Intermediate Representation (IR)
3. IR is compiled to GLSL compute shaders
4. OpenGL or Vulkan executes on GPU

---

## Features

### Cross-Platform Support

EasyGPU keeps the kernel DSL backend-independent. Actual feature availability depends on the selected backend, driver, and platform; see [Support Status](docs/support-status.md).

| Platform | Compute Kernels | Graphics Pipeline | Backend |
|:---------|:---------------|:------------------|:--------|
| **Windows** | OpenGL / Vulkan | Vulkan | OpenGL is continuously built |
| **Linux** | OpenGL / Vulkan | Vulkan | OpenGL is continuously built |
| **macOS** | Vulkan via MoltenVK | Vulkan via MoltenVK | Locally validated; GPU timestamp profiling uses a synchronized fallback |

```cpp
// The kernel DSL source is shared across supported backends and platforms
Kernel1D transform([](Int i) {
    data[i] = Sqrt(data[i] * 2.0f);
});
transform.Dispatch(16, true);
```

### Dual Backends

EasyGPU supports two compute backends, switchable at CMake configure time with zero kernel code changes:

- **Vulkan** *(default)* — Modern compute backend with explicit resource binding, push-constant uniforms, storage textures, sampled textures, profiler queries, and stronger long-term scalability
- **OpenGL** — Minimal setup, excellent for teaching, rapid iteration, and existing GL applications

```cmake
cmake -S . -B build -DEASYGPU_BACKEND=Vulkan   # default
cmake -S . -B build_gl -DEASYGPU_BACKEND=OpenGL
```

For new projects, Vulkan is the recommended default. OpenGL remains available for users who need a lighter dependency footprint or are targeting existing GL applications.

### Graphics Pipeline — Rasterization in C++ DSL

Write vertex and fragment shaders as C++ lambdas. The framework compiles them to GLSL/SPIR-V and renders via Vulkan dynamic rendering (`VK_KHR_dynamic_rendering`). The preview API includes depth testing, push-constant uniforms, SSBO vertex data, and `Varying<T>` interpolation.

```cpp
Varying<Vec3> vColor;

GraphicsPipeline pipeline(
    // ── Vertex Shader ──
    [&](Float4 &gl_Position) {
        Int  vid  = VertexIndex();
        auto vert = vertexBuffer.Bind()[vid];
        auto u    = ubo.Load();
        gl_Position = u.mvp() * MakeFloat4(vert.pos(), 1.0f);

        Float3 N(Normalize(vert.normal()));
        Float  diff = Max(Dot(N, MakeFloat3(0.4f, 0.6f, 0.7f)), 0.15f);
        vColor = Float3(MakeFloat3(diff, diff * 0.5f, diff * 0.3f));
    },
    // ── Fragment Shader ──
    [&](Float4 &fragColor) {
        Float3 c = vColor;  // interpolated varying
        fragColor = MakeFloat4(c.x(), c.y(), c.z(), 1.0f);
    });

// Render with depth testing
DepthBuffer db(W, H);
pipeline.Draw(renderTarget, db, vertCount, true);
```

**Key features:**
- `GraphicsPipeline` — complete VS+FS DSL class, matching `Kernel1D` API conventions
- `FragmentShader` — simplified fullscreen pass (hardcoded VS + user FS)
- `Varying<T>` — vertex→fragment interpolated variables
- `DepthBuffer` — RAII depth buffer for occlusion testing
- `VertexIndex()` / `FragmentCoord()` — built-in shader variable helpers

See the full guide: [docs/graphics-pipeline.md](docs/graphics-pipeline.md)

### Automatic Differentiation — GPU Gradients, Zero Hand-Written Math

EasyGPU's reverse-mode autograd records every operation during the forward pass and generates the backward pass automatically. The forward and backward passes are merged into a single GPU shader — one dispatch computes both your loss and its gradients. No dual-codebase synchronization, no hand-derived adjoints, no context switching between C++ and GLSL.

**Three API levels for different needs:**

| API | Use Case | GPU Execution |
|:----|:---------|:--------------|
| `AdjointInspector1D` | Inspect generated backward GLSL | Offline only |
| `AdjointKernel1D` | Get combined forward+backward shader | Compile, no built-in training |
| `ADKernel1D` | End-to-end GPU training | Dispatch, GPU gradient buffers, optimizer handoff |

**Quick example — linear regression in 20 lines:**

```cpp
#include <GPU.h>

// y = w*x + b,  fit to noisy data
ADKernel1D model([&](Int &id) {
    auto x_ref = buf_x.Bind();
    auto y_ref = buf_y.Bind();

    Float w, b;
    w = Param(W_ref[0]);    // trainable weight
    b = Param(W_ref[1]);    // trainable bias

    Float x = x_ref[id];
    Float y_pred = w * x + b;
    Float diff = y_pred - y_ref[id];
    Float loss = diff * diff;

    Loss(loss);
}, N);

// One call = forward + backward + gradient download
model.Backward(groups, true);
auto grad_w = model.Gradient(0);  // ∂loss/∂w
auto grad_b = model.Gradient(1);  // ∂loss/∂b
```

**What makes it work:**

1. **Gradient tape** — Every `+`, `*`, `Exp()`, `Max()`, `Sin()`... is recorded during the forward pass
2. **Adjoint generation** — The tape is walked backwards, applying the chain rule to emit adjoint GLSL
3. **Combined shader** — Forward and backward code are merged into a single compute shader; gradients flow directly into SSBOs
4. **GPU gradient buffers** — built-in optimizers consume gradients on device; `ADKernel1D::Gradient(p)` remains available for CPU inspection

**30+ differentiable operations, out of the box:**

| Category | Operations |
|:---------|:-----------|
| Arithmetic | `+`, `-`, `*`, `/` |
| Transcendental | `Sin`, `Cos`, `Tan`, `Exp`, `Log`, `Pow`, `Sqrt`, `Abs`, `Floor`, `Ceil`, `Round` |
| Trigonometry | `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Tanh` |
| Vector | `Dot`, `Cross`, `Length`, `Normalize`, `Distance`, `Reflect`, `Refract` |
| Activation | `Max` (ReLU), `Clamp`, `Smoothstep`, `Step`, `Sign` |
| Control flow | `If`/`Else`, `For` loops — gradients flow through branches correctly |

**Callables are differentiable too:**

```cpp
Callable<Float(Float)> Sigmoid = [](Float &x) {
    Return(MakeFloat(1.0f) / (MakeFloat(1.0f) + Exp(-x)));
};

// Use inside AD kernel — gradient flows through the Callable automatically
Float activation = Sigmoid(logits);
```

**Real example — GPT Name Generator & Poetry Transformer:**

A character-level GPT (TransformerBlock + CausalSelfAttention) that trains entirely on GPU through the EasyGPU AD engine. Two demos ship with the project:

- **Name Generator** (`ad_gpt_demo`) — 1-layer transformer (16-dim, 4 heads, ~7K params) on Karpathy's `names.txt` dataset (~32K names). Learns to generate novel names character by character. Full training loop: Tensor, Adam optimizer, gradient buffer sharing, CPU inference.

- **Poetry Generator** (`ad_gpt_poet_demo`) — Same architecture on an embedded Shakespeare sonnet corpus. 16-dim embeddings, 4 heads, vocab size 36. Continuous-text language modeling with checkpoint save/load.

Zero hand-written derivatives. The AD engine generates all backward-pass GLSL from the forward DSL, merges both passes into a single compute shader, and writes per-parameter gradients directly to shared SSBOs. See [`examples/ad_gpt_demo/`](examples/ad_gpt_demo/main.cpp) and [`examples/ad_gpt_poet_demo/`](examples/ad_gpt_poet_demo/main.cpp).

[Learn AD from scratch →](docs/autodiff.md) · [AD API Reference →](docs/api-reference.md#automatic-differentiation)

### Neural Network Training — Tensor + Optimizer

Building on the AD engine, EasyGPU provides a full NN training stack: compile-time-shaped tensors, built-in optimizers, and composable layers — all running through the same AD kernel.

**Three pillars of NN training in EasyGPU:**

| Component | Purpose | API |
|:----------|:--------|:----|
| `Tensor<T, Dims...>` | Multi-dimensional weight containers with CPU/GPU sync | `.Bind()`, `.RegisterAsParam()`, `.Upload()`, `.Download()` |
| `Adam` / `SGD` / `RMSprop` | Optimizers with weight decay, gradient clipping | `.AddTensor(t)`, `.Step(kernel)` |
| Layers | Reusable building blocks (Linear, Attention, Transformer...) | `.Setup()`, `.Forward(...)` |

**Full GPT training in under 70 lines:**

```cpp
#include <GPU.h>
#include <NN/NN.h>
using namespace GPU::NN;

// Model components
TokenEmbedding<float, 27, 16>     tokEmb;
PositionalEmbedding<float, 16, 16> posEmb;
TransformerBlock<float, 16, 16, 4> transformer(batchSize);
Tensor<float, 27, 16>             lmHead;

// Adam optimizer — registers all parameters in 6 lines
Adam adam(0.001f, 0.85f, 0.99f);
adam.AddTensor(tokEmb.Weight());
adam.AddTensor(posEmb.Weight());
adam.AddTensor(transformer.Attention().Weights());
adam.AddTensor(transformer.FC1());
adam.AddTensor(transformer.FC2());
adam.AddTensor(lmHead);

// AD kernel with 4192 parameters
ADKernel1D kernel([&](Var<int> &batchIdx) {
    // ... forward pass: embeddings → attention → MLP → logits → loss
    AD::Loss(totalLoss);
}, batchSize);

// Training loop — 3 lines per step
for (int step = 0; step < 5000; step++) {
    kernel.Forward(groups, true);
    kernel.Backward(groups, true);
    adam.Step(kernel);  // reduce gradients and update weights on GPU
}
```

**Key design decisions:**
- **Tensor `RegisterAsParam()`** registers whole tensor buffers, so large weights use one buffer-level adjoint instead of thousands of scalar tape entries
- **Adam `Step()`** consumes AD gradient buffers directly on GPU; the combined path runs a parallel reduction dispatch followed by an update dispatch
- **Gradient buffer sharing** packs scalar parameters from one source buffer into an interleaved gradient SSBO, while tensor parameters use a compact per-buffer gradient layout
- **Layers are purely structural** — no virtual dispatch, no runtime graph. `Setup()` registers parameters, `Forward()` emits DSL code. The compiler inlines everything.

See [`ad_gpt_demo`](examples/ad_gpt_demo/main.cpp) for a full name-generating transformer and [`ad_gpt_poet_demo`](examples/ad_gpt_poet_demo/main.cpp) for poetry generation.

[NN API Reference →](docs/api-reference.md#neural-network)

### Unified Language

Standard C++ syntax for GPU code. IDE features (autocomplete, refactoring, static analysis) work out of the box.

```cpp
Kernel1D sum([](Int i) {
    c[i] = a[i] + b[i];
});
```

### Interoperability — Works with Your Favorite Framework

EasyGPU integrates seamlessly with any OpenGL-based windowing framework. You control the window lifecycle; EasyGPU handles the GPU compute.

| Framework | Use Case | Demo |
|-----------|----------|------|
| **EasyX** | Teaching / Rapid prototyping | <img src="docs/image/easyx.png" width="320"> |
| **GLFW** | Cross-platform applications | <img src="docs/image/GLFW-Show.png" width="320"> |

*Original shader: [Seascape](https://www.shadertoy.com/view/Ms2SD1) on Shadertoy*

Key benefits:
- **Zero windowing interference** — Bring your own window, EasyGPU only touches the GPU
- **Native OpenGL interop** — Render compute results directly to window textures
- **Non-intrusive design** — Adopt incrementally in existing projects

### Control Flow

Structured control flow with C++-like semantics:

```cpp
If(x > 0, [&]() { 
    result = Sqrt(x); 
}).Else([&]() { 
    result = 0; 
});

For(0, 100, [&](Int& i) {
    If(i % 2 == 0) { Continue(); }
    Process(i);
});
```

### Shared Memory & Atomics

High-performance workgroup-level cooperation with shared memory and atomic operations:

```cpp
// Parallel reduction using shared memory
SharedMemory<float, 256> shared;

Kernel1D reduce([](Int i) {
    // Each thread contributes one value
    Expr<float> myValue = input[i];
    
    // Reduce across all threads in workgroup
    Expr<float> sum = WorkgroupReduce(shared, myValue);
    
    // Thread 0 writes result
    Int localId = LocalThreadId();
    If(localId == 0, [&]() {
        output[WorkgroupId()] = sum;
    });
}, 256);
```

Atomic operations for thread-safe counters and histograms:

```cpp
Kernel1D histogram([](Int i) {
    auto hist = histogram.Bind();
    Int bin = ComputeBin(input[i]);
    
    // Atomic increment
    ExprBase::NotUse(AtomicAdd(hist[bin], MakeInt(1)));
});
```

**Features:**
- **SharedMemory** — Fast workgroup-local storage (~1-10 cycles vs ~100s for global memory)
- **Atomic Operations** — AtomicAdd, AtomicMin, AtomicMax, AtomicAnd, AtomicOr, AtomicXor, AtomicExchange, AtomicCompSwap
- **Parallel Primitives** — WorkgroupReduce, WorkgroupScanInclusive, WorkgroupScanExclusive
- **Barriers** — WorkgroupBarrier, MemoryBarrier, FullBarrier for synchronization

See [Parallel Primitives Guide](docs/parallel-primitives.md) for detailed documentation.

### Memory Management

Automatic buffer alignment and struct layout:

```cpp
EASYGPU_STRUCT(Particle,
    (Float3, position),
    (Float3, velocity),
    (float, mass)
);

Buffer<Particle> particles(1000);
```

> **Important:** `EASYGPU_STRUCT` must be defined in the **global namespace**. Defining it inside any namespace will cause compilation errors.
> 
> ```cpp
> // Correct: global namespace
> EASYGPU_STRUCT(Particle, ...);
> 
> // Wrong: inside namespace
> namespace MyProject {
>     EASYGPU_STRUCT(Particle, ...);  // ERROR
> }
> ```

### Reusable Functions

```cpp
Callable<Float(Float)> square = [](Float x) {
    Return(x * x);
};

result = square(input);
```

**Generic Callables with Templates:**

```cpp
// Works with any GPU types (Int, Float, Float2, etc.)
template <class T1, class T2>
Callable<Float(T1, T2)> weightedSum = [&](T1 a, T2 b) {
    Return(ToFloat(a) * 0.7f + ToFloat(b) * 0.3f);
};

// Mix Int and Float seamlessly
Float result = weightedSum<Int, Float>(MakeInt(100), MakeFloat(0.5f));
```

**Header-Defined Callables (Multi-File Projects):**

When defining Callables in header files, add `inline` to prevent linker errors:

```cpp
// In header file (.h)
inline Callable<Float(Float, Float)> IntensityToColor = [](Float intensity, Float scale) {
    Return(intensity * scale);
};
```

### Introspection

```cpp
// Inspect generated GLSL
std::cout << kernel.GetGeneratedGLSL() << std::endl;

// Profile execution
KernelProfiler::PrintReport(kernel);
```

### Built-in Cross-Platform Window

EasyGPU includes a cross-platform window component for interactive GPU compute visualization. Built on top of [GLFW](https://www.glfw.org/) and [Dear ImGui](https://github.com/ocornut/imgui), it gives compute demos a real application shell: native windows, keyboard/mouse input, GPU texture presentation, and immediate-mode controls in the same render loop.

**Dear ImGui is first-class.** A `UIContext` can be layered over `TexturePresenter`, so sliders, color editors, checkboxes, stats panels, and debug controls can sit directly on top of live EasyGPU output.

<p align="center">
  <img src="docs/image/imgui.png" alt="EasyGPU ImGui Lab" width="720">
</p>

```cpp
#include <GPU.h>
#include <imgui.h>

int main() {
    using namespace GPU;
    using namespace GPU::Window;

    // Create a lightweight window
    AppWindow window({
        .width = 1024,
        .height = 768,
        .title = "EasyGPU Real-time Compute"
    });
    
    // Create GPU texture and presenter
    Texture2D<PixelFormat::RGBA8> texture(1024, 768);
    TexturePresenter presenter(window);
    UIContext ui(window);
    
    // Render kernel
    Kernel2D render([&](Int x, Int y) {
        auto tex = texture.Bind();
        tex.Write(x, y, MakeFloat4(ToFloat(x) / 1024, ToFloat(y) / 768, 0.0f, 1.0f));
    });
    
    // Real-time loop
    while (window.IsOpen()) {
        window.PollEvents();
        render.Dispatch(64, 48);
        ui.Render([&] {
            ImGui::Begin("Controls");
            ImGui::Text("Live EasyGPU texture");
            ImGui::End();
        });
        presenter.Present(texture);  // Display GPU result
    }
}
```

**Key Features:**
- **GLFW platform layer** — Native windows and input on Windows, Linux, and macOS
- **Dear ImGui overlay** — Immediate-mode debug panels and interactive controls
- **Event-driven input** — Keyboard, mouse, and resize events
- **Dual presentation paths** — CPU `PixelBuffer` for software rendering, `TexturePresenter` for EasyGPU textures
- **Vulkan swapchain path** — Direct texture presentation with ImGui overlay on Vulkan
- **OpenGL window path** — Fullscreen RGBA texture upload and ImGui overlay on Windows/Linux OpenGL
- **Optional at build time** — Control via `EASYGPU_BUILD_WINDOW` CMake option

[Learn more about Window API →](docs/window.md)

### Async Data Transfer

Pixel Buffer Objects (PBO) for non-blocking CPU/GPU transfers:

```cpp
Texture2D<PixelFormat::RGBA8> video(1920, 1080);
video.InitUploadPBOPool(2);  // Double buffering

// Upload without blocking - essential for real-time video
video.UploadAsync(frameData.data());
kernel.Dispatch(120, 68, true);  // GPU processes while CPU continues
```

### Shader Binary Cache

Automatic in-memory caching of compiled GPU programs for faster kernel execution:

```cpp
Kernel1D kernel([](Int i) {
    data[i] = data[i] * 2.0f;
});

// First dispatch: compiles and caches (~15ms)
kernel.Dispatch(16, true);

// Subsequent dispatches: uses cached binary (~0.5ms)
kernel.Dispatch(16, true);
```

**Key Features:**
- **Zero configuration** — Works automatically, no code changes needed
- **Cross-backend** — Supports both OpenGL and Vulkan
- **In-memory only** — No disk I/O, cache cleared on exit
- **Thread-safe** — Safe for multi-threaded applications

[Learn more about Shader Cache →](docs/shader-cache.md)

### SPIR-V Optimization Inspection

On the Vulkan backend, EasyGPU can expose the exact shader shape after the full optimization toolchain:

```text
C++ DSL → GLSL → SPIR-V → SPIRV-Tools opt → SPIRV-Cross GLSL
```

This feature is **Vulkan-backend only**. OpenGL builds accept the same APIs for source compatibility, but `SetOptimizationLevel(...)` is a silent no-op and optimized GLSL inspection returns an empty string. See [SPIR-V Optimization](docs/spirv-optimization.md) for the full API reference.

Use `GetCode()` for the original generated GLSL and `GetOptimizedGLSL()` for the optimized, SPIRV-Cross decompiled GLSL on Vulkan. The complete reproducible example lives in `examples/spirv_opt_inspection/main.cpp`:

```cpp
std::vector<Vec3> hdr(64, Vec3(1.0f, 0.5f, 0.25f));
Buffer<Vec3> hdrInput(hdr);
Buffer<Vec3> ldrOutput(hdr.size());

Kernel1D kernel("ToneMapInspection", [&](Int i) {
    auto src = hdrInput.Bind();
    auto dst = ldrOutput.Bind();

    Float3 color        = src[i];
    Float  exposure     = MakeFloat(1.25f);
    Float3 whiteBalance = MakeFloat3(1.03f, 0.98f, 0.92f);
    Float3 balanced     = color * whiteBalance;
    Float3 exposed      = balanced * exposure;

    Float  lumaA      = Dot(exposed, MakeFloat3(0.2126f, 0.7152f, 0.0722f));
    Float  lumaB      = Dot(exposed, MakeFloat3(0.2126f, 0.7152f, 0.0722f));
    Float3 acesTop    = exposed * (exposed * 2.51f + MakeFloat(0.03f));
    Float3 acesBottom = exposed * (exposed * 2.43f + MakeFloat(0.59f)) + MakeFloat(0.14f);
    Float3 acesMapped = Clamp(acesTop / acesBottom, 0.0f, 1.0f);
    Float  vignette   = Clamp(1.0f - Abs(lumaA - 0.5f) * 0.08f, 0.92f, 1.0f);
    Float3 normalized = Normalize(exposed + 0.001f);
    Float  blend      = Clamp(lumaA / (1.0f + lumaA), 0.0f, 1.0f) * 0.15f;
    Float  dead       = (lumaB * 0.0f) + Dot(normalized, normalized) * 0.0f;

    If(MakeBool(false), [&] {
        dst[i] = MakeFloat3(dead, dead, dead);
    }).Else([&] {
        Float3 graded = Mix(acesMapped, normalized, blend) * vignette;
        dst[i] = Clamp(graded, 0.0f, 1.0f);
    });
}, 256);

std::string before = kernel.GetCode();
std::string after  = kernel.GetOptimizedGLSL();
```

The default optimization preset is `Backend::ShaderOptimizationLevel::Aggressive`. SPIRV-Tools exposes `-O` and `-Os` as its general built-in recipes; EasyGPU maps `Aggressive` to the strongest general performance recipe available, `RegisterPerformancePasses()` / `spirv-opt -O`. You can override it per kernel:

```cpp
kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::None);       // no SPIRV-Tools opt passes
kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Aggressive); // default, strongest general -O preset
kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Size);       // optimize binary size, like spirv-opt -Os
```

`Backend::ShaderOptimizationLevel::Performance` remains available as a compatibility alias for `Aggressive`.

AD kernels use the same Vulkan shader path. Forward, internal clear, and combined forward+backward shaders are optimized by default on Vulkan, and the combined AD shader can also be inspected:

```cpp
ADKernel1D trainStep([&](Int i) {
    auto x = samples.Bind();
    auto w = weights.Bind();
    Float xVal = x[i];
    Float wVal = w[i];
    Float y = xVal * wVal;
    Float loss = y * y;
    AD::Param(wVal);
    AD::Loss(loss);
}, sampleCount);

trainStep.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Size);
std::string optimizedBackward = trainStep.GetOptimizedCombinedGLSL();
```

Reproduce the output below:

```bash
cmake -S . -B build -DEASYGPU_ENABLE_SPIRV_OPT=ON -DEASYGPU_ENABLE_SPIRV_CROSS_GLSL=ON
cmake --build build --target spirv_opt_inspection --parallel
./build/spirv_opt_inspection
```

**Before - real `GetCode()` excerpt, preserving the DSL's debug-friendly shape:**

```glsl
void main() {
    int v1;
    (v1)=((int(gl_GlobalInvocationID.x)));
    float v2;
    (v2)=(float(1.25));
    vec3 v3;
    (v3)=(vec3(float(1.03),float(0.98),float(0.92)));
    vec3 v4;
    (v4)=((buf0[v1])*(v3));
    vec3 v5;
    (v5)=((v4)*(v2));
    float v6;
    (v6)=(dot(v5,vec3(float(0.2126),float(0.7152),float(0.0722))));
    float v7;
    (v7)=(dot(v5,vec3(float(0.2126),float(0.7152),float(0.0722))));
    vec3 v8;
    (v8)=((v5)*(((v5)*(float(2.51)))+(float(0.03))));
    vec3 v9;
    (v9)=(((v5)*(((v5)*(float(2.43)))+(float(0.59))))+(float(0.14)));
    vec3 v10;
    (v10)=(clamp((v8)/(v9),float(0),float(1)));
    float v11;
    (v11)=(clamp((float(1))-((abs((v6)-(float(0.5))))*(float(0.08))),float(0.92),float(1)));
    vec3 v12;
    (v12)=(normalize((v5)+(float(0.001))));
    float v13;
    (v13)=((clamp((v6)/((v6)+(float(1))),float(0),float(1)))*(float(0.15)));
    float v14;
    (v14)=(((v7)*(float(0)))+((dot(v12,v12))*(float(0))));
    if (false) {
        (buf1[v1])=(vec3(v14,v14,v14));
    } else {
        vec3 v15;
        (v15)=((mix(v10,v12,v13))*(v11));
        (buf1[v1])=(clamp(v15,float(0),float(1)));
    }
}
```

**After - real `GetOptimizedGLSL()` excerpt after SPIRV-Tools opt and SPIRV-Cross:**

```glsl
void main()
{
    int _17 = int(gl_GlobalInvocationID.x);
    vec3 _44 = (_33.buf0[_17] * vec3(1.0299999713897705078125, 0.980000019073486328125, 0.920000016689300537109375)) * 1.25;
    float _51 = dot(_44, vec3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
    _123.buf1[_17] = clamp(mix(clamp((_44 * ((_44 * 2.5099999904632568359375) + vec3(0.02999999932944774627685546875))) / ((_44 * ((_44 * 2.4300000667572021484375) + vec3(0.589999973773956298828125))) + vec3(0.14000000059604644775390625)), vec3(0.0), vec3(1.0)), normalize(_44 + vec3(0.001000000047497451305389404296875)), vec3(clamp(_51 / (_51 + 1.0), 0.0, 1.0) * 0.1500000059604644775390625)) * clamp(1.0 - (abs(_51 - 0.5) * 0.07999999821186065673828125), 0.920000016689300537109375, 1.0), vec3(0.0), vec3(1.0));
}
```

This is real output captured by running `./build/spirv_opt_inspection`. The optimized dump makes compiler effects visible: the constant debug branch disappears, zero-multiplied dead work is removed, the duplicate luminance dot product is collapsed, debug temporaries are folded, and the final tone-mapping store becomes one compact expression much closer to what the Vulkan pipeline consumes. The feature is controlled by `EASYGPU_ENABLE_SPIRV_OPT` and `EASYGPU_ENABLE_SPIRV_CROSS_GLSL`.

### Performance Notes — Exclusive OpenGL Context Mode

EasyGPU operates in **exclusive mode** by default, assuming it has sole ownership of the OpenGL context within the current thread. This design choice maximizes performance by:

- **State caching**: Programs, buffers, and textures are only rebound when actually changed
- **Eliminating redundant `glMakeCurrent`**: Context is made current once during `Attach()` and stays current
- **No defensive `glGet` calls**: Trusting cached state avoids expensive driver synchronization

**Implications:**
- Do not interleave raw OpenGL calls with EasyGPU operations in the same context
- If you must use raw OpenGL, either:
  - Use a separate OpenGL context, or
  - Call `GPU::Runtime::GetStateCache().Invalidate()` before returning to EasyGPU

**FragmentKernel Lifecycle:** *(Deprecated — prefer [GraphicsPipeline](docs/graphics-pipeline.md) for new code)*
```cpp
FragmentKernel2D kernel(...);
kernel.Attach(hwnd);  // Context becomes current here

while (running) {
    // No need to MakeCurrent - context stays current
    kernel.Flush();     // Minimal state changes thanks to caching
}
// Context cleanup happens automatically
```

---

## Quick Start

### Installation

**CMake FetchContent:**

```cmake
include(FetchContent)
FetchContent_Declare(
    easygpu
    GIT_REPOSITORY https://github.com/easygpu/EasyGPU.git
    GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(easygpu)
target_link_libraries(your_target PRIVATE EasyGPU::EasyGPU)
```

To build EasyGPU itself with the Vulkan backend:

```cmake
set(EASYGPU_BACKEND Vulkan CACHE STRING "" FORCE)
```

Or from the command line:

```bash
cmake -S . -B build -DEASYGPU_BACKEND=Vulkan
```

If you want OpenGL explicitly:

```bash
cmake -S . -B build -DEASYGPU_BACKEND=OpenGL
```

**Installed package:**

```bash
cmake --install build --prefix /your/prefix
```

```cmake
find_package(EasyGPU CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE EasyGPU::EasyGPU)
```

EasyGPU is a compiled library; copying only `include/` is not a supported installation method.

The installed package exports:

- `EasyGPU::EasyGPU` — Core compute and graphics library
- `EasyGPU::Window` — Optional window component when built with `EASYGPU_BUILD_WINDOW=ON`

The package configuration resolves the selected backend dependencies for downstream projects. A package-consumer fixture in `tests/package-consumer` verifies that an installed package can be discovered, compiled, linked, and executed.

### Using EasyGPU in Your Own CMake Project

If you embed EasyGPU via `FetchContent`, you can select the backend before `FetchContent_MakeAvailable`:

```cmake
include(FetchContent)

set(EASYGPU_BACKEND Vulkan CACHE STRING "EasyGPU backend" FORCE)
set(EASYGPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(EASYGPU_BUILD_FRAGMENT_TESTER OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    easygpu
    GIT_REPOSITORY https://github.com/easygpu/EasyGPU.git
    GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(easygpu)

target_link_libraries(your_target PRIVATE EasyGPU::EasyGPU)
```

To switch back to OpenGL in your own project, change only one line:

```cmake
set(EASYGPU_BACKEND OpenGL CACHE STRING "EasyGPU backend" FORCE)
```

No kernel rewrite is required.

### Reliability and Validation

- Every `tests/Test*.cpp` source is automatically registered with CTest, except the optional Windows EasyX fragment tester.
- Release tests keep assertions enabled and use deterministic test compilation settings.
- Buffer and texture slots detect access after the attached resource has been destroyed.
- Shader and pipeline creation use explicit lifetime management and exception-safe cleanup.
- Vulkan profiling uses native timestamp queries where reliable. MoltenVK uses a synchronized CPU timing fallback to avoid device-loss failures.
- Vulkan pipeline-cache data accelerates pipeline creation but never replaces the required live shader module.

Run the complete configured test suite with:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

See [Support Status](docs/support-status.md) for capability maturity and qualification expectations.

### First Program

```cpp
#include <GPU.h>
#include <iostream>
#include <vector>

int main() {
    std::vector<float> numbers = {1, 2, 3, 4, 5};
    
    Buffer<float> gpu_input(numbers);
    Buffer<float> gpu_output(numbers.size());
    
    Kernel1D double_values([&](Int i) {
        auto input = gpu_input.Bind();
        auto output = gpu_output.Bind();
        output[i] = input[i] * 2.0f;
    });
    
    double_values.Dispatch(1, true);
    gpu_output.Download(numbers);
    
    for (float n : numbers) {
        std::cout << n << " ";
    }
    
    return 0;
}
```

> ⚠️ **Important: Variable Initialization**
> 
> Always use `Unref()` when initializing variables from buffer elements:
> ```cpp
> // ✅ CORRECT: Creates a new independent variable
> Int val = Unref(input[i]);
> val = 5;  // Only modifies val
> 
> // ❌ DANGEROUS: May create a reference to input[i]
> Int val = input[i];
> val = 5;  // May unexpectedly modify input[i]!
> ```
> See [Unref Documentation](docs/unref.md) for details.

Build:

```bash
g++ -std=c++20 hello_gpu.cpp -lEasyGPU -lGL -o hello_gpu
./hello_gpu
```

---

## Examples

### Compute Examples

| Level | Example | Topics |
|:------|:--------|:-------|
| Beginner | [hello_world](examples/hello_world/main.cpp) | Buffers, kernels |
| Beginner | [mandelbrot](examples/mandelbrot/main.cpp) | 2D kernels, math |
| Intermediate | [julia_set](examples/julia_set/main.cpp) | Complex numbers |
| Intermediate | [ray_tracing](examples/ray_tracing/main.cpp) | Structs, RNG, basic ray tracing |
| Advanced | [sdf_renderer](examples/sdf_renderer/main.cpp) | Callables, SDF, path tracing |
| Advanced | [parallel_reduction](examples/parallel_reduction/main.cpp) | Shared memory, parallel reduce |
| Advanced | [histogram](examples/histogram/main.cpp) | Atomic operations, shared memory |

### Automatic Differentiation Examples

| Level | Example | Topics |
|:------|:--------|:-------|
| Beginner | [ad_linear_regression](examples/ad_linear_regression/main.cpp) | AD basics, Param/Loss, gradient tape |
| Intermediate | [ad_transformer](examples/ad_transformer/main.cpp) | Self-attention, softmax AD, multi-parameter |
| Advanced | [ad_gpt_demo](examples/ad_gpt_demo/main.cpp) | GPT name generator: Tensor, Adam, CausalSelfAttention, CPU inference |
| Advanced | [ad_gpt_poet_demo](examples/ad_gpt_poet_demo/main.cpp) | GPT poetry: embedded sonnets, checkpoint save/load |

### Parallel Primitives Examples

| Level | Example | Topics |
|:------|:--------|:-------|
| Intermediate | [workgroup_reduce](examples/workgroup_reduce/main.cpp) | WorkgroupReduce, sum/max |
| Intermediate | [prefix_sum](examples/prefix_sum/main.cpp) | WorkgroupScanInclusive, WorkgroupScanExclusive |
| Advanced | [matrix_transpose](examples/matrix_transpose/main.cpp) | Shared memory tiling |
| Advanced | [parallel_sort](examples/parallel_sort/main.cpp) | Bitonic sort with shared memory |

### Window Examples

| Level | Example | Topics |
|:------|:--------|:-------|
| Beginner | [window_hello](examples/window_hello/main.cpp) | Window creation, event handling |
| Beginner | [window_pixels](examples/window_pixels/main.cpp) | CPU pixel buffer, animation |
| Intermediate | [window_compute](examples/window_compute/main.cpp) | Real-time GPU compute visualization |
| Intermediate | [window_imgui_lab](examples/window_imgui_lab/main.cpp) | EasyGPU texture + Window + Dear ImGui controls |

### Mandelbrot Set

```cpp
Kernel2D mandelbrot([&](Int px, Int py) {
    Float x = CENTER_X + (Float(px) / WIDTH - 0.5f) * ZOOM;
    Float y = CENTER_Y + (Float(py) / HEIGHT - 0.5f) * ZOOM;
    
    Float zx = 0, zy = 0;
    Int iter = 0;
    
    For(0, MAX_ITER, [&](Int i) {
        If(zx*zx + zy*zy > 4.0f) {
            iter = i;
            Break();
        };
        Float new_zx = zx*zx - zy*zy + x;
        zy = 2.0f*zx*zy + y;
        zx = new_zx;
    });
    
    image[py * WIDTH + px] = ColorFromIteration(iter);
});

mandelbrot.Dispatch(WIDTH/16, HEIGHT/16);
```

<img src="docs/image/mandelbrot.png" width="400">

[View full example →](examples/mandelbrot/main.cpp)

### Ray Tracing

<img src="docs/image/cornell_box.png" width="600">

Basic Monte Carlo ray tracer demonstrating struct handling and random number generation.

[View full example →](examples/ray_tracing/main.cpp)

### SDF Path Tracer

<img src="docs/image/sdf_renderer.png" width="600">

Signed distance field path tracer with support for complex lighting and materials. Demonstrates advanced Callable usage and reusable kernel functions.

[View full example →](examples/sdf_renderer/main.cpp)

---

## Ecosystem

### Feather — C# Frontend for EasyGPU

[![Feather](https://img.shields.io/badge/EasyGPU-Feather-512BD4.svg)](https://github.com/FeatherCompute/Feather)

**[Feather](https://github.com/FeatherCompute/Feather)** is the C# / .NET frontend for EasyGPU. It gives C# developers a native-feeling entry point into the EasyGPU programming model while keeping the generated GPU execution path behind a clean frontend API.

[Feather on GitHub →](https://github.com/FeatherCompute/Feather)

### HashEncoder — Instant-NGP Style Hash Grid Encoding

[![HashEncoder](https://img.shields.io/badge/EasyGPU-HashEncoder-blue.svg)](https://github.com/EasyGPU/HashEncoder)

**[HashEncoder](https://github.com/EasyGPU/HashEncoder)** is an optional companion library for multi-resolution hash grid encoding, built on top of EasyGPU. It implements the core technique from NVIDIA's Instant Neural Graphics Primitives (Instant-NGP) — learnable feature grids indexed by spatial hash functions — entirely in C++ with EasyGPU's DSL.

```cpp
#include <HashEncoder/HashEncoder.h>

// 16 levels, 2 features per level, hash table size 2^16, 3D input
GPU::HashEncoder::HashGridEncoder<float, 16, 2, 65536, 3> encoder;

// Encode 3D coordinates into learned features on GPU
encoder.Encode(positions, features, count);
```

| Feature | Description |
|:--------|:------------|
| Multi-resolution grids | Configurable levels, features per level, and hash table size |
| GPU-native | Encoding runs entirely on EasyGPU compute kernels |
| Trainable | Backed by EasyGPU buffers — plug into AD training loops |
| Instant-NGP compatible | Same hash encoding algorithm as the original paper |

[HashEncoder on GitHub →](https://github.com/EasyGPU/HashEncoder)

---

## Best Practices

### Variable Initialization

**Always use `Unref()`** when creating GPU variables from buffer elements:

```cpp
auto buf = buffer.Bind();

// ✅ CORRECT: Explicitly create a new independent variable
Int val = Unref(buf[i]);
Float f = Unref(buf[i]);
val = 5;  // Only modifies val, NOT buf[i]

// ❌ DANGEROUS: Direct initialization may create a reference
Int val = buf[i];  // val may become an alias to buf[i]!
val = 5;  // May unexpectedly modify buf[i] in the generated shader
```

**Why this matters:** Due to move constructor optimizations, `Int val = buf[i]` selects the move constructor, transferring ownership of the underlying variable name. This causes `val` to reference `buffer[i]` directly in the generated GLSL. Use `Unref()` to force the copy constructor and create truly independent variables. See [Unref Documentation](docs/unref.md) for details.

### Uniform Variables

**`Uniform.Load()` returns an independent copy by default**, so you can safely modify the result:

```cpp
Uniform<float> uScale(2.0f);

Kernel1D kernel([&](Int i) {
    auto buf = buffer.Bind();
    
    // ✅ Load() returns an independent copy, safe to modify
    Float scale = uScale.Load();
    scale = scale * 2.0f;  // Modifies only the local 'scale' variable
    buf[i] = buf[i] * scale;
});
```

**For read-only access** (to avoid the small overhead of copying), use `LoadRef()`:

```cpp
// Read-only usage - no modification needed
Float scale = uScale.LoadRef();  // Returns reference to uniform
buf[i] = buf[i] * scale;         // Safe for reading
// scale = 5.0f;                 // ❌ DON'T do this - "assignment to uniform" error
```

### Var-Var Assignment

**Var-Var assignment now works correctly** and generates proper IR automatically:

```cpp
Int A;
Int B = MakeInt(10);

// ✅ CORRECT: Direct Var-Var assignment now generates correct IR
A = B;

// Also works with explicit conversion if needed
A = Expr<int>(B);
```

### Handling Side-Effects

**Use `ExprBase::NotUse()`** for expressions with side-effects that aren't captured by operators:

```cpp
Callable<void(Int&)> A = [](Int &a) { a = 20; };

// ✅ CORRECT: Void-returning Callables automatically preserve side-effects
A(b);

Callable<Float(Float, Float&)> B = [](Float x, Float& out) {
    out = x * 2;
    Return(x + 1);
};

// ❌ WRONG: Non-void return with ignored result may lose side-effect on 'out'
Float y;
B(MakeFloat(5.0f), y);

// ✅ CORRECT: Explicitly mark the expression as "not used" to preserve side-effect
Float z;
ExprBase::NotUse(B(MakeFloat(5.0f), z));
```

> **Important:** Only `Callable<void>` automatically handles side-effects. For `Callable<T>` where `T` is not `void`, if you ignore the return value but need the side-effects (e.g., modifications to reference parameters), you **must** wrap the call with `ExprBase::NotUse()`.

## Documentation

- [Getting Started](docs/getting-started.md)
- [Tutorial](docs/tutorial.md)
- [API Reference](docs/api-reference.md)
- [Automatic Differentiation](docs/autodiff.md) — Complete AD guide: tape recording, adjoint generation, GPU training, NN integration
- [API Reference](docs/api-reference.md#neural-network) — Neural Network API: Tensor, Optimizer (Adam/SGD/RMSprop), Layers, Loss, Checkpoint
- [Texture3D Guide](docs/texture3d.md) — Volumetric textures and 3D compute
- [Window Component](docs/window.md) — Cross-platform window for interactive visualization
- [Window + ImGui](docs/window.md#dear-imgui-overlay) — Interactive controls over live EasyGPU textures
- [Graphics Pipeline](docs/graphics-pipeline.md) — Vertex + Fragment shader DSL, Varying\<T\>, depth testing, OBJ rendering
- [Shader Cache](docs/shader-cache.md) — Automatic kernel compilation caching
- [Support Status](docs/support-status.md) — Capability maturity, platform notes, and verification policy
- [Common Patterns](docs/patterns.md)
- [Unref - Independent Variable Copies](docs/unref.md)
- [FAQ](docs/faq.md)

---

## Building

### Dependencies

| Dependency | Required | Size | Purpose |
|:-----------|:---------|:-----|:--------|
| OpenGL 4.3+ | OpenGL builds | System | OpenGL compute backend |
| Vulkan 1.1+ SDK | Vulkan builds | System | Vulkan compute backend (MoltenVK on macOS) |
| GLAD | OpenGL builds | ~500KB (bundled) | OpenGL loader |
| GLFW | Window builds | Bundled | Cross-platform windows and input |
| Dear ImGui | Window builds | Bundled | Immediate-mode UI overlay |
| X11 (Linux) | Linux window builds | System | GLFW windowing backend |
| stb_image | No | ~50KB (examples only) | Image I/O |

### Build Commands

**Windows (MSVC):**
```powershell
git clone --recursive https://github.com/easygpu/EasyGPU.git
cd EasyGPU

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Windows (MSVC, Vulkan backend):**
```powershell
git clone --recursive https://github.com/easygpu/EasyGPU.git
cd EasyGPU

cmake -B build_vulkan -DEASYGPU_BACKEND=Vulkan -DEASYGPU_BUILD_FRAGMENT_TESTER=OFF
cmake --build build_vulkan --config Release
```

**Linux (GCC/Clang):**
```bash
git clone --recursive https://github.com/easygpu/EasyGPU.git
cd EasyGPU

# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake libgl1-mesa-dev libx11-dev

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run tests
cd build && ctest
```

**Linux (Vulkan backend):**
```bash
cmake -B build_vulkan -DEASYGPU_BACKEND=Vulkan -DEASYGPU_BUILD_FRAGMENT_TESTER=OFF
cmake --build build_vulkan -j
```

**macOS (Vulkan backend):**
```bash
git clone https://github.com/easygpu/EasyGPU.git
cd EasyGPU

# macOS requires Vulkan backend (OpenGL is Linux/Windows only)
# Install Vulkan SDK from https://vulkan.lunarg.com/
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEASYGPU_BACKEND=Vulkan
cmake --build build -j

# Run tests
cd build && ctest
```

### CMake Options

| Option | Default | Description |
|:-------|:--------|:------------|
| `EASYGPU_BACKEND` | `Vulkan` | Backend API: `OpenGL` or `Vulkan` |
| `EASYGPU_BUILD_EXAMPLES` | Top-level only | Build examples |
| `EASYGPU_BUILD_TESTS` | Top-level only | Build tests |
| `EASYGPU_BUILD_FRAGMENT_TESTER` | `OFF` | Build the Windows FragmentKernel tester |
| `EASYGPU_BUILD_WINDOW` | `ON` | Build the window component |
| `EASYGPU_BUILD_WINDOW_EXAMPLES` | Top-level only | Build window examples |

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

- [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) — DSL design
- [Taichi](https://github.com/taichi-dev/taichi) — Algorithms
- [GLAD](https://glad.dav1d.de/) — OpenGL loader
- [stb](https://github.com/nothings/stb) — Image utilities
- [GLFW](https://www.glfw.org/) — Cross-platform windows and input
- [Dear ImGui](https://github.com/ocornut/imgui) — Immediate-mode GUI

---

<div align="center">

[Back to Top](#easygpu)

</div>
