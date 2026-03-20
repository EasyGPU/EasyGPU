<div align="center">

<img src="docs/image/logo.png" alt="EasyGPU Logo" width="180">

# EasyGPU

Lightweight C++20 Embedded DSL for GPU Compute

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-orange.svg)](https://en.cppreference.com/w/cpp/20)
[![OpenGL](https://img.shields.io/badge/OpenGL-4.3+-green.svg)](https://www.opengl.org/)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.1+-red.svg)](https://www.vulkan.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()

[Getting Started](docs/getting-started.md) · [Tutorial](docs/tutorial.md) · [Examples](#examples) · [API Reference](docs/api-reference.md)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Concept](#concept)
- [Features](#features)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Documentation](#documentation)
- [Building](#building)
- [License](#license)

---

## Overview

EasyGPU is an embedded domain-specific language (eDSL) for GPU programming that allows writing compute kernels in standard C++20. No shader language knowledge required.

Today, that simplicity is no longer tied to a single graphics API. EasyGPU now ships with both an OpenGL compute backend and a Vulkan compute backend, so the same kernel code can target a lightweight OpenGL path or a modern Vulkan path without changing the DSL.

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

**For experienced developers:**
- Zero vendor lock-in (OpenGL 4.3+ or Vulkan 1.1+, cross-platform)
- Minimal dependencies (only GLAD, ~500KB)
- Clean C++20 interface without heavy template metaprogramming
- First-class Vulkan backend for compute workloads, profiling, textures, uniforms, and descriptor-backed resource binding

### Requirements

- C++20 compatible compiler (GCC 11+, Clang 14+, MSVC 2022+)
- OpenGL 4.3+ or Vulkan 1.1+
- CMake 3.21+ (optional)
- **Windows:** No additional dependencies
- **Linux:** X11 development libraries (`libx11-dev` on Ubuntu/Debian)
- **Vulkan builds:** Vulkan SDK with `glslang` / `SPIRV-Tools` libraries available to CMake

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

Runs natively on Windows and Linux with zero code changes. The same kernel code works identically across platforms.

| Platform | Compute Kernels | Fragment Kernels | Backend |
|:---------|:---------------|:-----------------|:--------|
| **Windows** | ✅ Full support | ✅ Full support | WGL |
| **Linux** | ✅ Full support | — | GLX |

```cpp
// This code runs identically on Windows and Linux
Kernel1D transform([](Int i) {
    data[i] = Sqrt(data[i] * 2.0f);
});
transform.Dispatch(16, true);
```

### Dual Backends

EasyGPU now supports two compute backends:

- **OpenGL** — Minimal setup, excellent for teaching, rapid iteration, and existing GL applications
- **Vulkan** — Modern compute backend with explicit resource binding, push-constant uniforms, storage textures, sampled textures, profiler queries, and stronger long-term scalability

This is one of EasyGPU's biggest practical advantages: you keep the same C++ DSL, the same buffer and texture abstractions, and the same kernel code while switching the backend at CMake configure time.

```cmake
cmake -S . -B build_gl -DEASYGPU_BACKEND=OpenGL
cmake -S . -B build_vk -DEASYGPU_BACKEND=Vulkan
```

For projects that want a simple on-ramp, OpenGL remains a great default. For projects that want a modern compute stack, Vulkan is now a first-class path rather than an experimental branch.

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

### Introspection

```cpp
// Inspect generated GLSL
std::cout << kernel.GetGeneratedGLSL() << std::endl;

// Profile execution
KernelProfiler::PrintReport(kernel);
```

### Built-in Cross-Platform Window

EasyGPU now includes a lightweight, cross-platform window component for interactive GPU compute visualization. No need to integrate external frameworks like GLFW or SDL.

```cpp
#include <GPU.h>

int main() {
    // Create a window
    Window window({
        .width = 1024,
        .height = 768,
        .title = "EasyGPU Real-time Compute"
    });
    
    // Create GPU texture and presenter
    Texture2D<PixelFormat::RGBA8> texture(1024, 768);
    TexturePresenter presenter(window);
    
    // Render kernel
    Kernel2D render([&](Int x, Int y) {
        auto tex = texture.Bind();
        tex[Int2(x, y)] = Vec4(Float(x)/1024, Float(y)/768, 0.0f, 1.0f);
    });
    
    // Real-time loop
    while (window.IsOpen()) {
        window.PollEvents();
        render.Dispatch(64, 48);
        presenter.Present(texture);  // Display GPU result
    }
}
```

**Key Features:**
- Simple, modern C++20 API
- Cross-platform: Windows (Win32) and Linux (X11)
- Event-driven input (keyboard, mouse, resize)
- CPU `PixelBuffer` for software rendering
- `TexturePresenter` for direct GPU texture display
- Optional at build time (`EASYGPU_BUILD_WINDOW`)

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

**FragmentKernel Lifecycle:**
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
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(easygpu)
target_link_libraries(your_target EasyGPU)
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

**Manual:** Copy `include/` to your project and link against OpenGL.

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
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(easygpu)

target_link_libraries(your_target PRIVATE EasyGPU)
```

To switch back to OpenGL in your own project, change only one line:

```cmake
set(EASYGPU_BACKEND OpenGL CACHE STRING "EasyGPU backend" FORCE)
```

No kernel rewrite is required.

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

### Window Examples

| Level | Example | Topics |
|:------|:--------|:-------|
| Beginner | [window_hello](examples/window_hello/main.cpp) | Window creation, event handling |
| Beginner | [window_pixels](examples/window_pixels/main.cpp) | CPU pixel buffer, animation |
| Intermediate | [window_compute](examples/window_compute/main.cpp) | Real-time GPU compute visualization |

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
- [Window Component](docs/window.md) — Cross-platform window for interactive visualization
- [Common Patterns](docs/patterns.md)
- [Unref - Independent Variable Copies](docs/unref.md)
- [FAQ](docs/faq.md)

---

## Building

### Dependencies

| Dependency | Required | Size | Purpose |
|:-----------|:---------|:-----|:--------|
| OpenGL 4.3+ | Yes | System | OpenGL compute backend |
| Vulkan 1.1+ SDK | Vulkan builds | System | Vulkan compute backend |
| X11 (Linux) | Yes | System | Windowing system |
| GLAD | Yes | ~500KB (bundled) | OpenGL loader |
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

### CMake Options

| Option | Default | Description |
|:-------|:--------|:------------|
| `EASYGPU_BACKEND` | `OpenGL` | Backend API: `OpenGL` or `Vulkan` |
| `EASYGPU_BUILD_EXAMPLES` | `ON` | Build examples |
| `EASYGPU_BUILD_TESTS` | `ON` | Build tests |
| `EASYGPU_BUILD_FRAGMENT_TESTER` | `OFF` | Build the Windows FragmentKernel tester |
| `EASYGPU_BUILD_WINDOW` | `ON` | Build the window component |
| `EASYGPU_BUILD_WINDOW_EXAMPLES` | `ON` | Build window examples |

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

- [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) — DSL design
- [Taichi](https://github.com/taichi-dev/taichi) — Algorithms
- [GLAD](https://glad.dav1d.de/) — OpenGL loader
- [stb](https://github.com/nothings/stb) — Image utilities

---

<div align="center">

[Back to Top](#easygpu)

</div>
