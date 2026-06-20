# API Reference

Complete reference for all EasyGPU classes and functions.

## Backend Selection

EasyGPU supports both a Vulkan compute backend (default) and an OpenGL compute backend.

Select the backend at CMake configure time:

```bash
cmake -S . -B build -DEASYGPU_BACKEND=Vulkan      # default
cmake -S . -B build_gl -DEASYGPU_BACKEND=OpenGL
```

In an embedding project that uses `FetchContent`, set the cache variable before `FetchContent_MakeAvailable`:

```cmake
set(EASYGPU_BACKEND Vulkan CACHE STRING "EasyGPU backend" FORCE)
```

The DSL API is the same on both backends. Buffer, texture, sampler, and uniform usage does not change at the call site.

## Table of Contents

- [Core Types](#core-types)
- [Kernels](#kernels)
- [Vulkan SPIR-V Optimization](#vulkan-spir-v-optimization)
- [Fragment Kernels](#fragment-kernels)
- [Buffers](#buffers)
- [Uniforms](#uniforms)
- [UniformBuffer](#uniformbuffer)
- [Graphics Pipeline](#graphics-pipeline)
- [Inspector Validation](#inspector-validation)
- [Variables and Expressions](#variables-and-expressions)
- [Unref](#unref)
- [Select (Ternary Operator)](#select-ternary-operator)
- [Control Flow](#control-flow)
- [Math Functions](#math-functions)
- [Vector Types](#vector-types)
- [Matrix Types](#matrix-types)
- [Callable](#callable)
- [Automatic Differentiation](#automatic-differentiation)
- [Neural Network](#neural-network)
- [Structs](#structs)
- [Textures](#textures)
- [Texture Samplers](#texture-samplers)
- [Mipmaps](mipmaps.md)
- [Shared Memory](#shared-memory)
- [Atomic Operations](#atomic-operations)
- [Active Compaction](#active-compaction)
- [Parallel Primitives](#parallel-primitives)
- [Thread Index Utilities](#thread-index-utilities)
- [PBO Async Transfer](#pbo-async-transfer)
- [Error Handling](#error-handling)
- [Benchmark Suite](#benchmark-suite)
- [OpenGL State Cache](#opengl-state-cache)

---

## Unref

Creates an independent copy of a GPU variable, ensuring value semantics instead of reference semantics.

```cpp
template <typename T>
[[nodiscard]] Var<T> Unref(const Var<T>& var);

template <typename T>
[[nodiscard]] Var<T> Unref(Var<T>&& var);
```

**Purpose:**
When initializing a `Var` from a buffer element (`buf[i]`), the default behavior uses move semantics, which may create an alias to the buffer element instead of an independent copy. `Unref()` forces the copy constructor to create a truly independent variable.

**Parameters:**
- `var` - The source variable, typically from buffer access

**Returns:**
A new `Var<T>` with its own storage in the generated GLSL

**Example:**
```cpp
Kernel1D kernel([](Int i) {
    auto buf = buffer.Bind();
    
    // Without Unref - creates alias
    Int alias = buf[i];
    alias = 5;  // May modify buf[i]!
    
    // With Unref - creates independent copy
    Int copy = Unref(buf[i]);
    copy = 5;   // Only modifies copy, NOT buf[i]
});
```

**When to Use:**
- Always when storing buffer elements to named variables
- When you need to modify a copy without affecting the original
- Before passing buffer elements to Callables that modify their arguments

**See Also:**
- [Unref Documentation](unref.md) - Complete guide

---

## Core Types

### Type Aliases

```cpp
using Int   = Var<int>;      // 32-bit signed integer
using Float = Var<float>;    // 32-bit float
using Bool  = Var<bool>;     // Boolean

using Kernel1D = Kernel::Kernel1D;
using Kernel2D = Kernel::Kernel2D;
using Kernel3D = Kernel::Kernel3D;
```

---

## Kernels

### Kernel1D

1D compute kernel for parallel array processing.

```cpp
// Constructor
Kernel1D kernel(
    const std::function<void(Var<int>&)>& func,  // Kernel function
    int workSizeX = 256                           // Threads per work group
);

// With name
Kernel1D kernel(
    const std::string& name,
    const std::function<void(Var<int>&)>& func,
    int workSizeX = 256
);
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Dispatch(int groupX, bool sync = false)` | Execute kernel. `sync=true` waits for GPU completion and automatically inserts memory barriers for writable buffers/textures. |
| `SetName(const std::string& name)` | Set kernel name |
| `GetName() const` | Get kernel name |
| `GetCode()` | Get generated GLSL code |
| `SetOptimizationLevel(Backend::ShaderOptimizationLevel level)` | Select Vulkan SPIR-V optimization preset; no-op on OpenGL |
| `GetOptimizationLevel() const` | Get stored optimization preset |
| `GetOptimizedGLSL()` | Vulkan only: return optimized, SPIRV-Cross decompiled GLSL; returns empty string on OpenGL |

**Example:**

```cpp
Kernel1D kernel([](Int i) {
    data[i] = data[i] * 2;
}, 256);

kernel.Dispatch(100, true);  // 100 groups, wait for completion
```

### Kernel2D

2D compute kernel for image/grid processing.

```cpp
Kernel2D kernel(
    const std::function<void(Var<int>&, Var<int>&)>& func,  // (x, y)
    int workSizeX = 16,
    int workSizeY = 16
);
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Dispatch(int groupX, int groupY, bool sync = false)` | Execute kernel. `sync=true` waits for GPU completion and automatically inserts memory barriers for writable buffers/textures. |

### Kernel3D

3D compute kernel for volume processing.

```cpp
Kernel3D kernel(
    const std::function<void(Var<int>&, Var<int>&, Var<int>&)>& func,  // (x, y, z)
    int workSizeX = 8,
    int workSizeY = 8,
    int workSizeZ = 4
);
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Dispatch(int groupX, int groupY, int groupZ, bool sync = false)` | Execute kernel. `sync=true` waits for GPU completion and automatically inserts memory barriers for writable buffers/textures. |

### Inspector Kernels

For debugging - compiles but doesn't execute.

```cpp
InspectorKernel1D inspector([](Int i) { ... });
inspector.PrintCode();                    // Print GLSL
std::string code = inspector.GetCode();   // Get GLSL
bool ok = inspector.Compile();            // Test compilation
```

## Vulkan SPIR-V Optimization

SPIR-V optimization is **Vulkan-backend only**. OpenGL accepts the API for source compatibility and handles it silently: setting an optimization level is a no-op for OpenGL shader compilation, and optimized GLSL inspection returns an empty string.

```cpp
kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Aggressive);
std::string optimized = kernel.GetOptimizedGLSL();
```

| Level | Vulkan behavior | OpenGL behavior |
|:--|:--|:--|
| `Backend::ShaderOptimizationLevel::None` | Skip SPIRV-Tools opt passes | No-op |
| `Backend::ShaderOptimizationLevel::Aggressive` | Default; strongest general SPIRV-Tools performance recipe (`spirv-opt -O`) | No-op |
| `Backend::ShaderOptimizationLevel::Performance` | Compatibility alias for `Aggressive` | No-op |
| `Backend::ShaderOptimizationLevel::Size` | SPIRV-Tools size recipe (`spirv-opt -Os`) | No-op |

The same methods are available on `Kernel1D/2D/3D` and `InspectorKernel1D/2D/3D`. For AD kernels, see [ADKernel1D](#adkernel1d) and [SPIR-V Optimization](spirv-optimization.md).

### Kernel Barriers

Synchronization within work groups:

```cpp
Kernel1D::WorkgroupBarrier();  // Synchronize threads in work group
Kernel1D::MemoryBarrier();     // Ensure memory writes are visible
Kernel1D::FullBarrier();       // Both barriers combined
```

---

## Fragment Kernels

> **Deprecated.** Fragment kernels (`FragmentKernel2D`) are superseded by the
> [Graphics Pipeline](#graphics-pipeline) DSL, which supports vertex+fragment
> shaders, `Varying<T>` interpolation, depth testing, push-constant uniforms,
> SSBO vertex data, `VertexIndex()`, and cross-platform rendering (Vulkan on
> Windows, Linux, and macOS). See the [migration section](graphics-pipeline.md).
> FragmentKernel2D is retained for backward compatibility but will not receive
> new features.

> **Platform Note:** Fragment kernels are currently only available on **Windows**. The Graphics Pipeline (above) is recommended for cross-platform work — it runs on Windows, Linux, and macOS via Vulkan.

### FragmentKernel2D

2D fragment kernel for real-time pixel rendering.

```cpp
FragmentKernel2D kernel(
    const std::string& name,
    const std::function<void(Var<Vec4>&)>& func,  // Kernel function with fragColor output
    uint32_t width,                               // Viewport width
    uint32_t height                               // Viewport height
);
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Attach(HWND hwnd)` | Attach to a window for rendering |
| `Detach()` | Detach from window |
| `Flush()` | Render frame and swap buffers |
| `SetResolution(uint32_t w, uint32_t h)` | Change viewport resolution |
| `GetShaderSource()` | Get generated GLSL code |

**Basic Example:**

```cpp
#include <GPU.h>
#include <windows.h>

// Create window
HWND hwnd = CreateWindowEx(...);

FragmentKernel2D kernel("Gradient",
    [&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
        // fragCoord: fragment coordinates in pixels (0,0 at bottom-left)
        // resolution: viewport resolution in pixels
        
        Float2 uv = fragCoord / resolution;
        Float3 color = MakeFloat3(uv.x(), uv.y(), 0.5f);
        fragColor = MakeFloat4(color, 1.0f);
    },
    1280, 720
);

kernel.Attach(hwnd);

// Render loop
while (running) {
    kernel.Flush();  // Direct to screen, no Download() needed!
}
```

**With Uniforms and Animation:**

```cpp
Uniform<float> uTime;
Uniform<Vec2> uMouse;

FragmentKernel2D kernel("Animated",
    [&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
        Float time = uTime.Load();
        Float2 mouse = uMouse.Load();
        
        Float2 uv = fragCoord / resolution;
        
        // Animated plasma effect
        Float v = Sin(uv.x() * 10.0f + time) + 
                  Sin(uv.y() * 10.0f + time);
        Float3 color = MakeFloat3(v * 0.5f + 0.5f, 0.2f, 0.8f);
        
        fragColor = MakeFloat4(color, 1.0f);
    },
    1280, 720
);

kernel.Attach(hwnd);

while (running) {
    uTime = clock() / 1000.0f;
    uMouse = Vec2(mouseX, mouseY);
    kernel.Flush();
}
```

**With Texture Sampling:**

```cpp
Texture2D<PixelFormat::RGBA8> image(512, 512);
// ... upload image data ...

FragmentKernel2D kernel("Textured",
    [&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
        auto tex = image.BindSampler();  // Use BindSampler, not Bind
        
        // Sample at normalized coordinates
        Float2 uv = fragCoord / resolution;
        
        fragColor = tex.Sample(uv);  // Sample texture
    },
    512, 512
);
```

### Compute vs Fragment Kernels

| Feature | Compute Kernel | Fragment Kernel |
|:--------|:---------------|:----------------|
| **Execution** | `Dispatch()` | `Flush()` after `Attach()` |
| **Output** | Buffers/Textures | Direct to window |
| **Coordinates** | `gl_GlobalInvocationID` | `gl_FragCoord` |
| **Texture Read** | `tex.Read(x, y)` | `tex.Sample(uv)` |
| **Texture Write** | `tex.Write(x, y, color)` | Not supported |
| **CPU Readback** | `Download()` required | Not needed |
| **Best For** | Data processing | Real-time rendering |

### Lambda Parameters

The fragment kernel lambda receives three parameters automatically:

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `fragCoord` | `Float2` | Fragment coordinates in pixels (0,0 at bottom-left) |
| `resolution` | `Float2` | Viewport resolution in pixels (width, height) |
| `fragColor` | `Var<Vec4>&` | Output color variable (write final color here) |

```cpp
FragmentKernel2D kernel("Effect",
    [&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
        // fragCoord: (0,0) to (width-1, height-1)
        // resolution: (width, height)
        
        Float2 uv = fragCoord / resolution;  // Normalized (0-1)
        
        // Your shader logic here
        
        fragColor = MakeFloat4(color, 1.0f);  // Write output
    },
    width, height
);
```

---

## Buffers

### Buffer<T>

GPU buffer for data storage and transfer.

```cpp
template<typename T>
class Buffer;
```

**Constructors:**

| Constructor | Description |
|:------------|:------------|
| `Buffer(size_t count, BufferMode mode = BufferMode::ReadWrite)` | Allocate |
| `Buffer(const std::vector<T>& data, BufferMode mode = BufferMode::ReadWrite)` | Upload from vector |
| `Buffer(Buffer&& other)` | Move constructor |

**BufferMode:**

| Mode | Description |
|:-----|:------------|
| `BufferMode::Read` | Read-only on GPU |
| `BufferMode::Write` | Write-only on GPU |
| `BufferMode::ReadWrite` | Read-write on GPU (default) |

**Methods:**

| Method | Description |
|:-------|:------------|
| `Bind()` | Bind to current kernel (returns BufferRef) |
| `Upload(const T* data, size_t count)` | Upload data to GPU (synchronous) |
| `Upload(const std::vector<T>& data)` | Upload from vector (synchronous) |
| `Download(T* outData, size_t count)` | Download data from GPU (synchronous) |
| `Download(std::vector<T>& outData)` | Download to vector (synchronous) |
| `GetCount() const` | Get element count |
| `GetElementSize() const` | Get element size in bytes |
| `GetBufferSize() const` | Get total size in bytes |
| `GetHandle() const` | Get OpenGL buffer ID |

> **Performance Note:** `Upload()` and `Download()` perform **synchronous** CPU-GPU memory copies. For multi-pass algorithms, keep intermediate results in GPU buffers and avoid unnecessary `Download()` calls between passes. Reuse `Buffer` instances rather than recreating them each frame.

**Example:**

```cpp
Buffer<float> buf1(1024);                          // Allocate
Buffer<float> buf2(data);                          // Upload
Buffer<float> buf3(1024, BufferMode::Write);       // Write-only

// In kernel
Kernel1D kernel([](Int i) {
    auto b = buf1.Bind();
    b[i] = b[i] * 2;
});

// After kernel
buf1.Download(data);
```

---

## Uniforms

### Uniform<T>

Uniform variables for passing constants from CPU to GPU. Unlike captured values which are embedded directly into the generated GLSL code, uniforms are dynamically uploaded to the GPU at dispatch time, allowing you to change values between kernel executions without recompiling.

```cpp
template<typename T>
class Uniform;
```

**Supported Types:**

| C++ Type | GLSL Type | Description |
|:---------|:----------|:------------|
| `float` | `float` | 32-bit floating point |
| `int` | `int` | 32-bit signed integer |
| `bool` | `bool` | Boolean value |
| `Math::Vec2` | `vec2` | 2-component float vector |
| `Math::Vec3` | `vec3` | 3-component float vector |
| `Math::Vec4` | `vec4` | 4-component float vector |
| `Math::IVec2` | `ivec2` | 2-component int vector |
| `Math::IVec3` | `ivec3` | 3-component int vector |
| `Math::IVec4` | `ivec4` | 4-component int vector |
| `Math::Mat2` | `mat2` | 2x2 float matrix |
| `Math::Mat3` | `mat3` | 3x3 float matrix |
| `Math::Mat4` | `mat4` | 4x4 float matrix |
| `Math::Mat2x3` | `mat2x3` | 2 columns, 3 rows |
| `Math::Mat2x4` | `mat2x4` | 2 columns, 4 rows |
| `Math::Mat3x2` | `mat3x2` | 3 columns, 2 rows |
| `Math::Mat3x4` | `mat3x4` | 3 columns, 4 rows |
| `Math::Mat4x2` | `mat4x2` | 4 columns, 2 rows |
| `Math::Mat4x3` | `mat4x3` | 4 columns, 3 rows |

**Constructors:**

| Constructor | Description |
|:------------|:------------|
| `Uniform()` | Default constructor - creates uninitialized uniform |
| `Uniform(T value)` | Constructor with initial value |
| `Uniform(const Uniform& other)` | Copy constructor |

**Methods:**

| Method | Description |
|:-------|:------------|
| `Load()` | Load the uniform in kernel context, returns `Var<T>` |
| `GetValue() const` | Get the current CPU-side value |
| `SetValue(T value)` | Set the CPU-side value |
| `operator=(T value)` | Assign value from literal |
| `operator=(const Uniform& other)` | Assign from another uniform |
| `operator T() const` | Implicit conversion to value type |

**Example:**

```cpp
// Create uniforms
Uniform<int> offset;
Uniform<float> scale(2.5f);

// Set values on CPU
offset = 100;
scale.SetValue(3.0f);

// Use in kernel
Buffer<float> data(1024);
Kernel1D kernel([&](Int i) {
    auto buf = data.Bind();
    auto off = offset.Load();    // Load uniform as Var<int>
    auto s = scale.Load();       // Load uniform as Var<float>
    buf[i] = (buf[i] + ToFloat(off)) * s;
});

// Dispatch with current uniform values
kernel.Dispatch(4, true);

// Change uniform values and dispatch again
offset = 200;
scale = 1.5f;
kernel.Dispatch(4, true);  // Uses new values without recompilation
```

**Multiple Uniforms:**

```cpp
Uniform<int> threshold;
Uniform<float> factor1;
Uniform<float> factor2;

threshold = 50;
factor1 = 0.5f;
factor2 = 2.0f;

Kernel1D kernel([&](Int i) {
    auto buf = buffer.Bind();
    auto t = threshold.Load();
    auto f1 = factor1.Load();
    auto f2 = factor2.Load();
    
    If(buf[i] > ToFloat(t), [&]() {
        buf[i] = buf[i] * f1;
    }).Else([&]() {
        buf[i] = buf[i] * f2;
    });
});
```

**Uniform<bool> for Conditional Logic:**

```cpp
Uniform<bool> enableFeature;
enableFeature = true;

Kernel1D kernel([&](Int i) {
    auto buf = buffer.Bind();
    auto enabled = enableFeature.Load();
    
    If(enabled, [&]() {
        buf[i] = Process(buf[i]);
    });
});

// Toggle feature off
enableFeature = false;
kernel.Dispatch(4, true);
```

---

## UniformBuffer

### UniformBuffer&lt;T&gt;

Read-only structured GPU buffer for passing large structs to kernels. Unlike `Uniform<T>`, which uses push constants on Vulkan, `UniformBuffer<T>` uses a read-only std430 storage buffer and is not constrained by the push-constant limit.

```cpp
template<typename T>
class UniformBuffer;
```

**When to use UniformBuffer vs Uniform:**

| Feature | `Uniform<T>` | `UniformBuffer<T>` |
|:--------|:-------------|:-------------------|
| Transport | Push constant / `glProgramUniform` | Read-only storage buffer |
| GLSL layout | `uniform float u0;` | `layout(std430, binding=N) readonly buffer ...` |
| Size limit | Device push-constant limit on Vulkan | Device storage-buffer limit |
| Layout standard | std430 | std430 |
| Best for | Small params (float, Vec3, Mat4) | Large structs, multiple fields |

**Prerequisite — Register struct with `EASYGPU_STRUCT`:**

```cpp
EASYGPU_STRUCT(MyConfig,
    (GPU::Math::Vec3, lightDir),
    (float, exposure),
    (GPU::Math::Mat4, shadowMatrix)
);
```

**Constructors:**

| Constructor | Description |
|:------------|:------------|
| `UniformBuffer()` | Default constructor |
| `UniformBuffer(const T& value)` | Constructor with initial value |

**Methods:**

| Method | Description |
|:-------|:------------|
| `Load()` | Load the structured buffer in kernel context, returns `Var<T>` |
| `GetValue() const` | Get the current CPU-side value |
| `SetValue(const T& value)` | Set the CPU-side value and upload to GPU |
| `operator=(const T& value)` | Assign value from struct literal |
| `GetHandle() const` | Get the backend buffer handle |

**Example:**

```cpp
// Define config struct
EASYGPU_STRUCT(RenderConfig, (GPU::Math::Vec3, lightDir), (float, exposure));

// Create structured uniform buffer
UniformBuffer<RenderConfig> config;
RenderConfig cfg;
cfg.lightDir = Vec3(0.5f, 1.0f, 0.3f);
cfg.exposure = 2.0f;
config = cfg;

// Use in kernel
Buffer<float> output(256);
Kernel1D kernel([&](Int i) {
    auto buf = output.Bind();
    auto c   = config.Load();

    buf[0]   = c.lightDir().x();
    buf[1]   = c.lightDir().y();
    buf[2]   = c.lightDir().z();
    buf[3]   = c.exposure();
});

kernel.Dispatch(1, true);
```

**Multiple Dispatches with Value Update:**

```cpp
UniformBuffer<RenderConfig> config;

// First dispatch
RenderConfig cfg1;
cfg1.lightDir  = Vec3(1.0f, 1.0f, 1.0f);
cfg1.exposure  = 1.0f;
config         = cfg1;
kernel.Dispatch(1, true);

// Second dispatch with different values — no recompilation needed
RenderConfig cfg2;
cfg2.lightDir  = Vec3(0.0f, 0.0f, 1.0f);
cfg2.exposure  = 4.0f;
config         = cfg2;
kernel.Dispatch(1, true);
```

**Generated GLSL:**

```glsl
layout(set=0, std430, binding=0) readonly buffer ubo_0_t {
    RenderConfig ubo_0[];
};

void main() {
    // c.lightDir().x() → ubo_0[0].lightDir.x
    // c.exposure()     → ubo_0[0].exposure
}
```

---

## Graphics Pipeline

EasyGPU provides a complete rasterization pipeline — vertex shader + fragment shader — as a C++ embedded DSL. You write both stages as C++ lambdas. The framework compiles them to GLSL/SPIR-V and executes via Vulkan (`VK_KHR_dynamic_rendering`).

For a tutorial-style guide, see [Graphics Pipeline](graphics-pipeline.md).

### GraphicsPipeline (DSL)

The main user-facing class. Construction follows the same pattern as `Kernel1D`/`Kernel2D`: lambda-based DSL, optional profiling name, lazy compilation.

```cpp
// Fullscreen triangle (no explicit vertex input — use VertexIndex())
GraphicsPipeline pipeline(
    [&](Float4 &gl_Position) {
        Int  vid = VertexIndex();
        Float x  = ToFloat((vid & 1) << 2) - 1.0f;
        Float y  = ToFloat((vid & 2) << 1) - 1.0f;
        gl_Position = MakeFloat4(x, y, 0.0f, 1.0f);
    },
    [&](Float4 &fragColor) {
        fragColor = MakeFloat4(1.0f, 0.0f, 0.0f, 1.0f);
    });

pipeline.Draw(renderTarget, 3, true);                    // no depth
pipeline.Draw(renderTarget, depthBuffer, 3, true);       // with depth
pipeline.Draw({
    GraphicsPipeline::RenderTarget(gbuffer0),
    GraphicsPipeline::RenderTarget(gbuffer1)
}, depthBuffer, 3, true);                                // MRT
```

**Methods**

| Method | Description |
|:-------|:------------|
| `Draw(Texture2D&, uint32_t vertexCount, bool sync)` | Non-indexed draw without depth |
| `Draw(Texture2D&, DepthBuffer&, uint32_t vertexCount, bool sync)` | Non-indexed draw with depth |
| `Draw({RenderTarget(...)...}, uint32_t vertexCount, bool sync)` | Non-indexed draw to multiple render targets |
| `Draw({RenderTarget(...)...}, DepthBuffer&, uint32_t vertexCount, bool sync)` | MRT draw with depth |
| `SetVertexBuffer(BufferHandle, uint32_t stride)` | Bind a vertex buffer |
| `SetIndexBuffer(BufferHandle)` | Bind an index buffer |
| `SetIndexCount(uint32_t count)` | Set index count for indexed draws |
| `SetName(name)` / `GetName()` | Profiling name |
| `GetShaderSource()` | Return generated GLSL for debugging |

### FragmentShader (DSL)

Simplified fullscreen pass — hardcoded fullscreen-triangle VS, user-provided FS lambda:

```cpp
FragmentShader shader([](Float2 &fragCoord, Float4 &fragColor) {
    fragColor = MakeFloat4(1.0f, 0.0f, 0.0f, 1.0f);
}, width, height);

shader.Render(renderTarget, sync);
```

### Varying\<T\>

Declared outside both VS and FS lambdas, captured by reference. VS writes, FS reads the rasterizer-interpolated value.

```cpp
Varying<Vec3> vColor;

GraphicsPipeline pipeline(
    [&](Float4 &gl_Position) {
        // VS writes
        vColor = Float3(MakeFloat3(r, g, b));
    },
    [&](Float4 &fragColor) {
        // FS reads (interpolated)
        Float3 c = vColor;
        fragColor = MakeFloat4(c.x(), c.y(), c.z(), 1.0f);
    });
```

Supported types: `float`, `int`, `Vec2`, `Vec3`, `Vec4`, `IVec2`, `IVec3`, `IVec4`, `Mat3`, `Mat4`, registered structs.

### DepthBuffer

RAII depth buffer:

```cpp
DepthBuffer db(width, height);
// ...
pipeline.Draw(rt, db, vertCount, true);
```

### Built-in Shader Variables

Free functions in `GPU::Kernel`:

| Function | GLSL Built-in | Stage |
|:---------|:-------------|:------|
| `VertexIndex()` | `gl_VertexIndex` | Vertex |
| `FragmentCoord()` | `gl_FragCoord` | Fragment |

### Type Aliases

For cleaner DSL code, use the aliases from `GPU.h` / `<Utility/Helpers.h>`:

| Alias | Full Type |
|:------|:----------|
| `Float` | `IR::Value::Var<float>` |
| `Float2` | `IR::Value::Var<Math::Vec2>` |
| `Float3` | `IR::Value::Var<Math::Vec3>` |
| `Float4` | `IR::Value::Var<Math::Vec4>` |
| `Int` | `IR::Value::Var<int>` |

### Backend API (Low-Level)

For raw GLSL usage or custom pipeline construction. The DSL classes above are built on this API.

**Capability Check**

```cpp
auto* backend = GPU::Runtime::Context::GetBackend();
if (!backend->GetCaps().supportsGraphics) { /* not available */ }
```

**GraphicsPipelineDesc**

```cpp
struct GraphicsPipelineDesc {
    ShaderHandle      vertexShader           = INVALID_SHADER_HANDLE;
    ShaderHandle      fragmentShader         = INVALID_SHADER_HANDLE;
    PrimitiveTopology topology               = PrimitiveTopology::TriangleList;
    PixelFormat       colorAttachmentFormat  = PixelFormat::RGBA8;
    std::vector<PixelFormat> colorAttachmentFormats;
    bool              depthTestEnable        = false;
    bool              depthWriteEnable       = true;
    std::vector<VertexLayoutEntry> vertexLayout;
    std::vector<ResourceLayoutEntry> resources;
    uint32_t          pushConstantSize       = 0;
};
```

**RenderPassBeginDesc**

```cpp
struct RenderPassBeginDesc {
    TextureHandle colorAttachment  = INVALID_TEXTURE_HANDLE;
    std::vector<TextureHandle> colorAttachments;
    TextureHandle depthAttachment  = INVALID_TEXTURE_HANDLE;
    float         clearColor[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
    float         clearDepth       = 1.0f;
    bool          clearColorFlag   = true;
    bool          clearDepthFlag   = true;
};
```

**PrimitiveTopology**

| Value | Description |
|:------|:------------|
| `PointList` | Individual points |
| `LineList` / `LineStrip` | Line segments |
| `TriangleList` | Independent triangles (default) |
| `TriangleStrip` / `TriangleFan` | Connected triangles |

**Backend Methods**

| Method | Description |
|:-------|:------------|
| `CreateGraphicsPipeline(desc)` | Create a graphics pipeline |
| `BeginRendering(desc)` / `EndRendering()` | Dynamic render pass |
| `SetViewport(x, y, w, h)` / `SetScissor(x, y, w, h)` | Viewport and scissor |
| `BindVertexBuffer(h, stride)` / `BindIndexBuffer(h)` | Bind vertex/index data |
| `Draw(vc, ic, fv, fi)` | Non-indexed draw |
| `DrawIndexed(ic, ic, fi, vo, fi)` | Indexed draw |
| `CreateDepthBuffer(w, h)` / `DestroyDepthBuffer(h)` | Depth buffer management |
| `GenerateMipmaps(texture)` | Generate all allocated mip levels from level zero |

**Error Handling**

- `Draw()` outside `BeginRendering()`/`EndRendering()` throws `std::runtime_error`
- `BeginRendering()` while already in a pass throws `std::runtime_error`
- Push constant size exceeding device limit throws at pipeline creation

**Vulkan Implementation Notes**

- Uses `VK_KHR_dynamic_rendering` (loaded via `vkGetDeviceProcAddr`)
- Queue family selected with `VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT`
- Depth format: `VK_FORMAT_D32_SFLOAT`
- Compute operations unaffected when graphics is unavailable

---

## Inspector Validation

The `InspectorKernel` family provides GLSL code inspection and offline validation without requiring a working GPU.

### Validate()

Performs pure-software GLSL syntax validation using `glslang`. Returns `true` if the generated shader code is valid GLSL.

```cpp
class InspectorKernel1D {
public:
    /** @brief Validate the generated GLSL without GPU compilation. */
    bool Validate(std::string& errorMessage) const;
};
```

**Example:**

```cpp
InspectorKernel1D kernel([&](Var<int>& id) {
    auto buf = buffer.Bind();
    buf[id]  = buf[id] * 2.0f;
});

std::string error;
if (kernel.Validate(error)) {
    std::cout << "GLSL is valid" << std::endl;
} else {
    std::cout << "GLSL validation failed: " << error << std::endl;
}
```

**CI Integration:**

For headless CI environments without a GPU, use `Validate()` instead of `Compile()`:

```cpp
InspectorKernel1D kernel(myKernelFunction);
std::string error;

// Works in CI without GPU
if (!kernel.Validate(error)) {
    std::cerr << "Kernel validation failed: " << error << std::endl;
    return 1;
}
```

---

## Variables and Expressions

### Var<T>

Mutable GPU variable.

```cpp
template<typename T>
class Var;

// Type aliases
using Int   = Var<int>;
using Float = Var<float>;
using Bool  = Var<bool>;
```

**Construction:**

```cpp
Int i;                          // Uninitialized
Int i = MakeInt(5);            // From literal
Int i = otherVar;              // Copy
```

> ⚠️ **CRITICAL: `Var` Initialization May Accidentally Create a Reference**
> 
> When initializing a `Var` from a buffer element, **always** use `Unref()` to ensure value semantics:
> 
> ```cpp
> auto buf = buffer.Bind();
> 
> // ✅ CORRECT: Explicitly create a new variable with a copy of the value
> Int val = Unref(buf[i]);
> val = 5;  // Only modifies val, NOT buf[i]
> 
> // ❌ DANGEROUS: Direct initialization may create a reference
> Int val = buf[i];
> val = 5;  // May unexpectedly modify buf[i] in the generated GLSL!
> ```
> 
> **Why this happens:**
> - `buf[i]` returns a temporary `Var<T>` (rvalue)
> - `Int val = buf[i]` selects the **move constructor** `VarBase(VarBase&&)`
> - The move transfers ownership of the underlying variable name (e.g., `"buffer[i]"`)
> - Result: `val` becomes an alias to `buffer[i]` in the generated shader
> 
> **Always use `Unref()`** to force creation of a new independent variable:
> ```cpp
> Int    val = Unref(buf[i]);
> Float  f   = Unref(buf[i]);
> Float3 v   = Unref(buf[i]);
> ```
> 
> See [Unref Documentation](unref.md) for complete details.

**Assignment:**

```cpp
Var<int> a = MakeInt(5);
Var<int> b = MakeInt(10);
a = b;        // Copy value
a = b + 5;    // Arithmetic result
```

### VarArray<Type, N>

Fixed-size array for GPU-local storage. Unlike `Buffer<T>` which resides in global GPU memory, `VarArray` creates a local array within the kernel (similar to `float arr[N]` in GLSL).

```cpp
template<ScalarType Type, int N>
class VarArray;
```

**Construction:**

```cpp
// Empty array (uninitialized)
VarArray<float, 10> localFloats;

// Initialized from CPU array
std::array<int, 5> cpuData = {1, 2, 3, 4, 5};
VarArray<int, 5> localInts(cpuData);
```

**Element Access:**

```cpp
VarArray<float, 10> arr;

// Index with literal
arr[0] = 5.0f;
Float val = arr[3];

// Index with Var<int>
Int idx = MakeInt(5);
arr[idx] = arr[idx] + 1.0f;

// Index with Expr<int>
For(0, 10, [&](Int& i) {
    arr[i] = arr[i] * 2.0f;  // Dynamic indexing
});
```

**Use Cases:**
- Local scratch space within a thread
- Small lookup tables
- Stencil computation buffers
- Sorting small arrays locally

**Comparison: Buffer vs VarArray**

| Feature | Buffer<T> | VarArray<Type, N> |
|:--------|:----------|:------------------|
| Memory | Global GPU memory | Local/thread-private memory |
| Lifetime | Survives kernel exit | Created/destroyed per thread |
| Size | Large (millions of elements) | Small (typically < 1000) |
| Access | All threads can access | Only owning thread can access |
| Persistence | Data persists between kernels | Data lost after kernel |
| Binding | Requires `.Bind()` | Created directly in kernel |

### Expr<T>

Immutable GPU expression (read-only).

```cpp
template<typename T>
class Expr;
```

Used for values that cannot be assigned to:

```cpp
Expr<float> e = a + b;  // Expression result
e = 5.0f;               // Error: Expr is read-only
```

### Constructors (Make)

**Important:** `Make` APIs wrap C++ literals into GPU `Var` types **without type conversion**. They are NOT the same as `Cast` APIs.

```cpp
// Helper functions to create GPU values from C++ literals
MakeInt(int value)           -> Var<int>      // Wrap int literal
MakeFloat(float value)       -> Var<float>    // Wrap float literal
MakeBool(bool value)         -> Var<bool>     // Wrap bool literal

MakeFloat3(float, float, float)  -> Var<Vec3>   // Wrap 3 floats
MakeFloat4(float, float, float, float) -> Var<Vec4>  // Wrap 4 floats
MakeInt2(int, int)           -> Var<IVec2>    // Wrap 2 ints
MakeInt3(int, int, int)      -> Var<IVec3>    // Wrap 3 ints
```

**Key difference:**
```cpp
// Make: No conversion, just wrapping
Float f = MakeFloat(3.14f);   // OK: float literal -> Var<float>
Float f = MakeFloat(42);      // ERROR: int literal, use MakeFloat(42.0f) or ToFloat(MakeInt(42))

// Cast: Type conversion
Int i = MakeInt(3);
Float f = ToFloat(i);         // OK: Var<int> -> Var<float> with conversion

**Implicit Conversions:**

```cpp
Var<int> accepts: Var<int>, int, Expr<int>
Expr<int> accepts: Expr<int>, Var<int>, int
```

---

## Select (Ternary Operator)

The ternary conditional operator - selects between two expressions based on a boolean condition. This is the expression-level equivalent of if-else, returning a value that can be used in larger expressions.

```cpp
template <ScalarType T>
[[nodiscard]] Expr<T> Select(Expr<bool> condition, Expr<T> trueExpr, Expr<T> falseExpr);

template <ScalarType T>
[[nodiscard]] Expr<T> Select(Expr<bool> condition, const Var<T>& trueExpr, const Var<T>& falseExpr);

template <ScalarType T>
[[nodiscard]] Expr<T> Select(Expr<bool> condition, Expr<T> trueExpr, const Var<T>& falseExpr);

template <ScalarType T>
[[nodiscard]] Expr<T> Select(Expr<bool> condition, const Var<T>& trueExpr, Expr<T> falseExpr);
```

**Purpose:**
Unlike `If` which is a statement for control flow, `Select` is an expression that returns a value. It generates the GLSL ternary operator `condition ? trueExpr : falseExpr`.

**Parameters:**
- `condition` - Boolean expression that determines which branch to evaluate
- `trueExpr` - Expression evaluated when condition is true
- `falseExpr` - Expression evaluated when condition is false

**Returns:**
An `Expr<T>` representing the selected value.

**Type Support:**
- Scalar types: `float`, `int`, `bool`
- Vector types: `Vec2`, `Vec3`, `Vec4`, `IVec2`, `IVec3`, `IVec4`
- Matrix types: `Mat2`, `Mat3`, `Mat4`, etc.
- Custom structs registered with `EASYGPU_STRUCT`

**Example - Basic Usage:**

```cpp
Kernel1D kernel([](Int i) {
    auto buf = buffer.Bind();
    Float x = buf[i];
    
    // Absolute value using Select
    Float absX = Select(x < 0.0f, -x, x);
    
    // Max of two values
    Float y = buf[i + 1];
    Float maxVal = Select(x > y, x, y);
    
    // Clamp to [0, 1] range using nested Select
    Float clamped = Select(x < 0.0f, 0.0f,
                          Select(x > 1.0f, 1.0f, x));
    
    buf[i] = absX;
});
```

**Example - With Vector Types:**

```cpp
Kernel1D kernel([](Int i) {
    auto colors = palette.Bind();
    auto output = output.Bind();
    
    Vec3 colorA = colors[i * 2];
    Vec3 colorB = colors[i * 2 + 1];
    Bool useA = colors[i * 2].x() > 0.5f;
    
    // Select between two colors
    Vec3 selected = Select(useA, colorA, colorB);
    
    // Conditional blend
    Vec3 result = Select(useA, colorA * 0.8f + colorB * 0.2f, 
                                colorB * 0.8f + colorA * 0.2f);
    
    output[i] = result;
});
```

**Example - Nested Ternary:**

```cpp
Kernel1D grade_calculator([](Int i) {
    auto scores = scoresBuf.Bind();
    auto grades = gradesBuf.Bind();
    
    Int score = scores[i];
    
    // Grade mapping: A=4, B=3, C=2, D=1, F=0
    Int grade = Select(score >= 90, 4,
                      Select(score >= 80, 3,
                            Select(score >= 70, 2,
                                  Select(score >= 60, 1, 0))));
    
    grades[i] = grade;
});
```

**Example - In Expressions:**

```cpp
Kernel1D process([](Int i) {
    auto buf = buffer.Bind();
    Float a = buf[i];
    Float b = buf[i + 1];
    
    // Select can be used within larger expressions
    Float result = Select(a > b, a, b) * 2.0f + 1.0f;
    
    // Multiple selects in one expression
    Float mixed = Select(a > 0.0f, a, 0.0f) + 
                  Select(b > 0.0f, b, 0.0f);
    
    buf[i] = result;
});
```

**Select vs If:**

| Feature | Select | If |
|:--------|:-------|:---|
| **Type** | Expression (returns value) | Statement (control flow) |
| **Use Case** | Value selection | Side effects, multiple statements |
| **Chaining** | Can nest and compose | Method chaining with `.Elif().Else()` |
| **GLSL** | `cond ? a : b` | `if (cond) { ... }` |
| **Performance** | Both branches may be evaluated | Only one branch executes |

**Common Patterns:**

```cpp
// Absolute value
Float absX = Select(x < 0.0f, -x, x);

// Min/Max
Float maxVal = Select(a > b, a, b);
Float minVal = Select(a < b, a, b);

// Clamp (ensure value is in range)
Float clamped = Select(x < minVal, minVal,
                      Select(x > maxVal, maxVal, x));

// Sign function
Float sign = Select(x > 0.0f, 1.0f,
                   Select(x < 0.0f, -1.0f, 0.0f));

// Step function
Float step = Select(x >= threshold, 1.0f, 0.0f);

// Conditional assignment without if-statement
vec = Select(shouldBlend, blended, original);
```

**Notes:**

**⚠️ Performance Warning - Warp Divergence:**

`Select` generates the GLSL ternary operator `?:` which **evaluates both branches before selecting the result**. This has important performance implications:

1. **Both branches execute**: Unlike CPU `?:` or `If` statements, both `trueExpr` and `falseExpr` are fully evaluated. This means expensive operations in either branch always run.

2. **Warp Divergence**: When threads in the same warp (work group) take different paths, the GPU must serialize execution:
   ```cpp
   // BAD: High divergence - threads in warp take different paths
   Kernel1D bad([](Int i) {
       Float result = Select(i % 2 == 0, ExpensiveEven(i), ExpensiveOdd(i));
   });
   
   // BETTER: Group threads to reduce divergence
   Kernel1D better([](Int i) {
       // Process all even indices first, then odd
       If(i < N/2, [&]() {
           Float result = ExpensiveEven(i * 2);
       }).Else([&]() {
           Float result = ExpensiveOdd((i - N/2) * 2 + 1);
       });
   });
   ```

3. **When to use `If` instead**: For expensive computations or when divergence is a concern:
   ```cpp
   // Expensive operations - use If to avoid evaluating both
   Float result;
   If(condition, [&]() {
       result = ExpensiveComputationA();
   }).Else([&]() {
       result = ExpensiveComputationB();
   });
   ```

**Summary:**
- Both `trueExpr` and `falseExpr` are always evaluated (no short-circuiting)
- All arguments must be of compatible types; use explicit constructors if needed
- `Var<T>` arguments are automatically converted to `Expr<T>`

**See Also:**
- [Control Flow](#control-flow) - For statement-level conditionals
- [Common Patterns](patterns.md) - Select patterns like clamp, min/max, etc.

---

## Control Flow

### If

Conditional execution.

```cpp
IfChain If(Expr<bool> condition, const std::function<void()>& body);
```

**Chaining:**

```cpp
If(condition1, [&]() {
    // if body
}).Elif(condition2, [&]() {
    // else if body
}).Else([&]() {
    // else body
});
```

### For

For loop with integer index.

```cpp
// Default step = 1
void For(Expr<int> start, Expr<int> end, 
         const std::function<void(Var<int>&)>& body);

// Explicit step
void For(Expr<int> start, Expr<int> end, Expr<int> step,
         const std::function<void(Var<int>&)>& body);
```

**Example:**

```cpp
For(0, 100, [&](Int& i) {
    // i ranges from 0 to 99
    data[i] = data[i] * 2;
});

// With step
For(0, 100, 2, [&](Int& i) {
    // i = 0, 2, 4, ..., 98
});
```

### While

While loop.

```cpp
void While(Expr<bool> condition, const std::function<void()>& body);
```

**Example:**

```cpp
Float x = MakeFloat(1.0f);
While(x < 100.0f, [&]() {
    x = x * 1.1f;
});
```

### DoWhile

Do-while loop.

```cpp
void DoWhile(const std::function<void()>& body, Expr<bool> condition);
```

### Break and Continue

```cpp
void Break();     // Exit current loop
void Continue();  // Skip to next iteration
```

**Example:**

```cpp
For(0, 100, [&](Int& i) {
    If(i % 2 == 0, [&]() {
        Continue();  // Skip even numbers
    });
    
    If(data[i] > threshold, [&]() {
        Break();  // Exit loop early
    });
});
```

### Return

Return from Callable.

```cpp
template<typename T>
void Return(Expr<T> value);
```

**Example:**

```cpp
Callable<Float(Float)> Square = [](Float& x) {
    Return(x * x);
};
```

---

## Math Functions

### Arithmetic

```cpp
// Built-in operators: +, -, *, /, %, - (negation)
// Built-in comparisons: ==, !=, <, >, <=, >=
// Built-in logical: &&, ||, !

Expr<T> Abs(Expr<T> x);           // Absolute value
Expr<T> Sign(Expr<T> x);          // Sign (-1, 0, or 1)
Expr<T> Min(Expr<T> a, Expr<T> b); // Minimum
Expr<T> Max(Expr<T> a, Expr<T> b); // Maximum
Expr<T> Clamp(Expr<T> x, Expr<T> min, Expr<T> max); // Clamp to range

// CopySign - Returns value with magnitude of x and sign of y
Expr<float> CopySign(Expr<float> x, Expr<float> y);
Expr<float> CopySign(Expr<float> x, float y);
Expr<float> CopySign(float x, Expr<float> y);
Expr<Vec2> CopySign(Expr<Vec2> x, Expr<Vec2> y);
Expr<Vec2> CopySign(Expr<Vec2> x, Expr<float> y);  // Broadcast scalar sign
Expr<Vec2> CopySign(Expr<Vec2> x, float y);        // Broadcast scalar sign
Expr<Vec3> CopySign(Expr<Vec3> x, Expr<Vec3> y);
Expr<Vec3> CopySign(Expr<Vec3> x, Expr<float> y);  // Broadcast scalar sign
Expr<Vec3> CopySign(Expr<Vec3> x, float y);        // Broadcast scalar sign
Expr<Vec4> CopySign(Expr<Vec4> x, Expr<Vec4> y);
Expr<Vec4> CopySign(Expr<Vec4> x, Expr<float> y);  // Broadcast scalar sign
Expr<Vec4> CopySign(Expr<Vec4> x, float y);        // Broadcast scalar sign
```

**CopySign Examples:**

```cpp
// Basic usage - transfer sign from one value to another
Float x = MakeFloat(5.0f);   // positive magnitude
Float y = MakeFloat(-3.0f);  // negative sign
Float result = CopySign(x, y);  // Returns -5.0f

// With vectors
Vec3 v = MakeFloat3(1.0f, 2.0f, 3.0f);
Vec3 signSource = MakeFloat3(-1.0f, 1.0f, -1.0f);
Vec3 result = CopySign(v, signSource);  // (-1.0f, 2.0f, -3.0f)

// Broadcast scalar sign to all vector components
Vec3 v2 = MakeFloat3(1.0f, 2.0f, 3.0f);
Vec3 negative = CopySign(v2, -1.0f);  // All components become negative
```

### Power and Roots

```cpp
Expr<float> Sqrt(Expr<float> x);      // Square root
Expr<float> Pow(Expr<float> x, Expr<float> y);  // x^y
Expr<float> Exp(Expr<float> x);       // e^x
Expr<float> Log(Expr<float> x);       // Natural log
Expr<float> Log2(Expr<float> x);      // Base-2 log
```

### Trigonometry

```cpp
Expr<float> Sin(Expr<float> x);   // Sine (radians)
Expr<float> Cos(Expr<float> x);   // Cosine (radians)
Expr<float> Tan(Expr<float> x);   // Tangent (radians)
Expr<float> Asin(Expr<float> x);  // Arcsine
Expr<float> Acos(Expr<float> x);  // Arccosine
Expr<float> Atan(Expr<float> x);  // Arctangent
Expr<float> Atan2(Expr<float> y, Expr<float> x);  // Arctangent(y/x)
```

### Type Conversion (Cast)

**Important:** Do not confuse `Cast` APIs with `Make` APIs. Cast APIs perform type **conversion** between `Var` types.

```cpp
Expr<float> ToFloat(Expr<int> x);   // Convert int to float (widening conversion)
Expr<int> ToInt(Expr<float> x);     // Convert float to int (truncate toward zero)
Expr<int> Round(Expr<float> x);     // Round to nearest int
Expr<int> Floor(Expr<float> x);     // Floor (round down)
Expr<int> Ceil(Expr<float> x);      // Ceiling (round up)
```

**Cast vs Make:**

| API | Purpose | Has Conversion Semantics |
|:----|:--------|:-------------------------|
| `ToFloat(Var<int>)` | Convert `Var<int>` to `Var<float>` | Yes (int �?float conversion) |
| `MakeFloat(1.0f)` | Create `Var<float>` from literal | No (just wraps the value) |
| `ToInt(Var<float>)` | Convert `Var<float>` to `Var<int>` | Yes (truncation) |
| `MakeInt(5)` | Create `Var<int>` from literal | No (just wraps the value) |

### Screen-Space Derivatives

```cpp
Expr<T> Ddx(Expr<T> value);  // GLSL dFdx()
Expr<T> Ddy(Expr<T> value);  // GLSL dFdy()
```

Screen-space derivatives are valid only in fragment shaders. They are commonly used with `TextureSampler2D::SampleGrad()` to control mip selection after discontinuous UV operations.

### Vector Math

```cpp
Expr<float> Dot(Expr<Vec3> a, Expr<Vec3> b);       // Dot product
Expr<Vec3> Cross(Expr<Vec3> a, Expr<Vec3> b);      // Cross product
Expr<float> Length(Expr<Vec3> v);                  // Vector length
Expr<float> Length2(Expr<Vec3> v);                 // Squared length
Expr<Vec3> Normalize(Expr<Vec3> v);                // Normalize vector
Expr<Vec3> Reflect(Expr<Vec3> v, Expr<Vec3> n);    // Reflect vector
Expr<Vec3> Refract(Expr<Vec3> v, Expr<Vec3> n, Expr<float> eta);  // Refract
```

---

## Vector Types

### CPU Types (Host)

```cpp
struct Vec2 { float x, y; };
struct Vec3 { float x, y, z; };
struct Vec4 { float x, y, z, w; };

struct IVec2 { int x, y; };
struct IVec3 { int x, y, z; };
struct IVec4 { int x, y, z, w; };
```

**CPU Operations:**

```cpp
Vec3 a(1, 2, 3);
Vec3 b = a + Vec3(4, 5, 6);  // (5, 7, 9)
float d = a.Dot(b);
Vec3 n = a.Normalized();
```

### GPU Types (Device)

In kernels, use `Var<Vec3>`, `Var<Vec2>`, etc.

```cpp
// Construction
Float3 v = MakeFloat3(1.0f, 2.0f, 3.0f);
Float3 v = MakeFloat3(1.0f);  // (1, 1, 1)

// Component access
Float x = v.x();
Float y = v.y();
Float z = v.z();

// Swizzling
Float2 xy = v.xy();
Float2 yz = v.yz();

// Assignment
v.x() = 5.0f;
```

**GPU Operations:**

```cpp
Float3 a = MakeFloat3(1, 2, 3);
Float3 b = MakeFloat3(4, 5, 6);

Float3 c = a + b;      // Addition
Float3 c = a - b;      // Subtraction
Float3 c = a * 2.0f;   // Scalar multiplication
Float3 c = a / 2.0f;   // Scalar division
Float3 c = a * b;      // Component-wise multiplication

Float d = Dot(a, b);   // Dot product
Float3 c = Cross(a, b); // Cross product
Float len = Length(a);  // Length
Float3 n = Normalize(a); // Normalization
```

---

## Matrix Types

### CPU Types (Host)

```cpp
struct Mat2;   // 2x2 matrix
struct Mat3;   // 3x3 matrix
struct Mat4;   // 4x4 matrix
struct Mat2x3; // 2 columns, 3 rows
struct Mat2x4; // 2 columns, 4 rows
struct Mat3x2; // 3 columns, 2 rows
struct Mat3x4; // 3 columns, 4 rows
struct Mat4x2; // 4 columns, 2 rows
struct Mat4x3; // 4 columns, 3 rows
```

**CPU Construction:**

```cpp
// From columns
Mat4 m(
    Vec4(1, 0, 0, 0),  // Column 0
    Vec4(0, 1, 0, 0),  // Column 1
    Vec4(0, 0, 1, 0),  // Column 2
    Vec4(0, 0, 0, 1)   // Column 3
);

// Transform matrices
Mat4 translation = Mat4::Translate(Vec3(1, 2, 3));
Mat4 rotation = Mat4::Rotate(45.0f * 3.14159f / 180.0f, Vec3(0, 1, 0));
Mat4 scale = Mat4::Scale(Vec3(2, 2, 2));
Mat4 perspective = Mat4::Perspective(60.0f, 16.0f/9.0f, 0.1f, 100.0f);
Mat4 ortho = Mat4::Ortho(-1, 1, -1, 1, 0.1f, 100.0f);
```

**CPU Operations:**

```cpp
Mat4 a, b;
Mat4 c = a * b;        // Matrix multiplication
Vec4 v = a * Vec4(1, 2, 3, 1);  // Matrix-vector multiplication
Mat4 inv = a.Inverse();
Mat4 trans = a.Transpose();
```

### GPU Types (Device)

```cpp
// In kernels
Var<Mat4> m;
Float4 v = m * MakeFloat4(1, 2, 3, 1);
```

---

## Callable

Define reusable functions.

```cpp
template<typename Signature>
class Callable;

// Recommended: Use GPU types in signature
Callable<Float(Float, Float)> Add = [](Float& a, Float& b) {
    Return(a + b);
};

// Also supported: C++ scalar types (auto-converted to GPU types)
Callable<float(float, float)> Add2 = [](Float& a, Float& b) {
    Return(a + b);
};
```

### Multi-File Projects and Linkage

**Important:** In multi-file projects (when including Callable definitions in headers), you **must** add the `inline` keyword to avoid "multiple definition" linker errors:

```cpp
// PhysicsKernel.h - Header file
#pragma once
#include <GPU.h>

// ❌ WRONG: Will cause "multiple definition" link errors
// when this header is included in multiple .cpp files
Callable<Float(Float, Float)> IntensityToColor = [](Float intensity, Float scale) {
    Return(intensity * scale);
};

// ✅ CORRECT: Add inline keyword for header-defined Callables
inline Callable<Float(Float, Float)> IntensityToColor = [](Float intensity, Float scale) {
    Return(intensity * scale);
};
```

**When to use `inline`:**

| Scenario | `inline` Required | Reason |
|:---------|:------------------|:-------|
| Callable in header (.h) | **Yes** | Prevents multiple definition errors |
| Callable in single .cpp file | No | Only one definition exists |
| Callable as class static member | **Yes** | Same as header inclusion |
| Callable in anonymous namespace | No | Internal linkage |

**Why this happens:**
Callable objects are defined as variables (not functions), so they follow C++ variable linkage rules. Without `inline`, each translation unit that includes the header gets its own definition, causing linker errors.

**Template Callables and `inline`:**

Template Callables also need `inline` when defined in headers:

```cpp
// MathUtils.h
#pragma once
#include <GPU.h>

// ✅ CORRECT: Add inline for template Callables in headers
template <class T>
inline Callable<T(T, T)> GenericClamp = [&](T value, T minVal, T maxVal) {
    If(value < minVal, [&]() { Return(minVal); });
    If(value > maxVal, [&]() { Return(maxVal); });
    Return(value);
};
```

**Type Mapping:**
- `Float` �?`float`
- `Int` �?`int`  
- `Float2` �?`Math::Vec2`
- `Float3` �?`Math::Vec3`
- `Float4` �?`Math::Vec4`
- etc.

**Features:**
- Can be called from any kernel
- Supports reference parameters for output
- Can capture host values (constants)
- Supports recursion (limited)

**Reference Parameters:**

```cpp
Callable<void(Float, Float&)> GetComponents = [](Float& v, Float& x, Float& y) {
    x = v;
    y = v * 2;
};

// Usage in kernel
Float x, y;
GetComponents(value, x, y);
```

**Texture Parameters:**

Callables can accept textures and samplers as parameters, enabling reusable image processing functions:

```cpp
// Callable that samples from a texture
Callable<Vec4(TextureRef<PixelFormat::RGBA8>, int, int)> ReadPixel =
    [](TextureRef<PixelFormat::RGBA8> img, Int x, Int y) {
        Return(img.Read(x, y));
    };

// Callable that writes to a texture
Callable<void(TextureRef<PixelFormat::RGBA8>, int, int, Vec4)> WritePixel =
    [](TextureRef<PixelFormat::RGBA8> img, Int x, Int y, Vec4 color) {
        img.Write(x, y, color);
    };

// Usage in kernel
Texture2D<PixelFormat::RGBA8> tex(256, 256);
Kernel1D kernel([&](Int i) {
    auto img = tex.Bind();
    
    Int x = i % 256;
    Int y = i / 256;
    
    // Read using callable
    Vec4 color = ReadPixel(img, x, y);
    
    // Process and write back
    WritePixel(img, x, y, Vec4(1.0f) - color);
});
```

**Sampler Parameters:**

For fragment kernels or when UV sampling is needed:

```cpp
Callable<Vec4(TextureSampler2D<PixelFormat::RGBA8>, float, float)> SampleUV =
    [](TextureSampler2D<PixelFormat::RGBA8> sampler, Float u, Float v) {
        Return(sampler.Sample(u, v));
    };

// In fragment kernel
FragmentKernel2D kernel("Textured",
    [&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
        auto sampler = image.BindSampler();
        
        Float2 uv = fragCoord / resolution;
        fragColor = SampleUV(sampler, uv.x(), uv.y());
    },
    512, 512
);
```

**Supported Texture Types:**

| C++ Type | GLSL Type | Description |
|:---------|:----------|:------------|
| `TextureRef<PixelFormat::RGBA8>` | `image2D` | Read-write RGBA8 texture |
| `TextureRef<PixelFormat::RGBA32F>` | `image2D` | Read-write RGBA32F texture |
| `TextureRef<PixelFormat::R32F>` | `image2D` | Read-write single-channel float |
| `TextureRef<PixelFormat::RGBA32I>` | `iimage2D` | Read-write signed integer texture |
| `TextureRef<PixelFormat::RGBA32UI>` | `uimage2D` | Read-write unsigned integer texture |
| `TextureSampler2D<PixelFormat::RGBA8>` | `sampler2D` | Sampled RGBA8 texture |
| `TextureSampler2D<PixelFormat::RGBA32I>` | `isampler2D` | Sampled signed integer texture |

**Important Notes:**

1. **TextureRef vs TextureSampler2D**: Use `TextureRef` for compute kernels (with `Bind()`), and `TextureSampler2D` for fragment kernels (with `BindSampler()`).

2. **Pixel Format Must Match**: The format in the Callable signature must match the format of the texture being passed:
   ```cpp
   // Correct
   Texture2D<PixelFormat::RGBA8> tex(...);
   Callable<void(TextureRef<PixelFormat::RGBA8>)> func = ...;
   
   // Incorrect - format mismatch
   Callable<void(TextureRef<PixelFormat::R32F>)> func = ...;  // Error!
   ```

3. **No Copy Overhead**: Textures are passed by name reference in GLSL (not by value), so there's no performance overhead.

4. **Thread Safety**: Each thread can safely read/write different coordinates. Use atomic operations or barriers for coordinating access to the same pixel.

**Template Metaprogramming with Callable:**

You can use C++ templates to create generic Callables that work with multiple GPU types:

```cpp
// Generic callable that works with any two GPU types convertible to Float
template <class T1, class T2>
Callable<Float(T1, T2)> WeightedSum = [&](T1 X1, T2 X2) {
    Return(ToFloat(X1) * 0.7f + ToFloat(X2) * 0.3f);
};

// Usage with different GPU types
Kernel1D kernel([](Int i) {
    auto buf = data.Bind();
    
    // Works with (Float, Float)
    Float a = MakeFloat(1.0f);
    Float b = MakeFloat(2.0f);
    Float result1 = WeightedSum<Float, Float>(a, b);
    
    // Works with (Int, Float) - Int is automatically converted
    Int x = MakeInt(10);
    Float y = MakeFloat(3.14f);
    Float result2 = WeightedSum<Int, Float>(x, y);
    
    // Works with (Float2, Float2) - component-wise conversion
    Float2 uv = MakeFloat2(0.5f, 0.5f);
    Float2 coord = MakeFloat2(1.0f, 0.0f);
    Float2 blended = WeightedSum<Float2, Float2>(uv, coord);
});
```

> **Important:** Template parameters `T1` and `T2` **must be GPU types** (`Int`, `Float`, `Float2`, etc.), not C++ literal types (`int`, `float`, `Vec2`). Use `ToFloat()` or `ToInt()` for type conversion within the callable.

**Generic Math Utilities:**

```cpp
// Clamp any numeric GPU type to a range
template <class T>
Callable<T(T, T, T)> GenericClamp = [&](T value, T minVal, T maxVal) {
    If(value < minVal, [&]() { Return(minVal); });
    If(value > maxVal, [&]() { Return(maxVal); });
    Return(value);
};

// Usage
Float f = MakeFloat(5.0f);
Float clampedF = GenericClamp<Float>(f, MakeFloat(0.0f), MakeFloat(1.0f));

Int i = MakeInt(100);
Int clampedI = GenericClamp<Int>(i, MakeInt(0), MakeInt(50));
```

**Side-Effect Handling:**

`Callable<void>` automatically handles side-effects when called as a statement:

```cpp
Callable<void(Int&)> A = [](Int &a) { a = 20; };
Int b;
A(b);  // Side-effect automatically preserved
```

For `Callable<T>` where `T` is not `void`, if you ignore the return value but need side-effects (e.g., from reference parameters), use `ExprBase::NotUse()`:

```cpp
Callable<Float(Float, Float&)> B = [](Float x, Float& out) {
    out = x * 2;
    Return(x + 1);
};

Float result;
ExprBase::NotUse(B(MakeFloat(5.0f), result));  // Explicitly preserve side-effect
```

---

## Automatic Differentiation

Reverse-mode automatic differentiation. Records operations during the forward pass and generates adjoint (gradient) GLSL code. See [Automatic Differentiation](autodiff.md) for the full guide.

### Class Overview

| Class | GPU Required | Purpose |
|:------|:-------------|:--------|
| `AdjointInspector1D/2D/3D` | No | Inspect forward + backward GLSL, debug gradients |
| `AdjointKernel1D/2D/3D` | Yes | Combined forward+backward shader, single dispatch |
| `ADKernel1D` | Yes | Separate Forward/Backward calls, GPU gradient buffers, optional gradient download |

---

### AdjointInspector1D

Offline inspection and validation. Builds a forward kernel, records the tape, and generates backward GLSL — all without a GPU.

```cpp
template <typename Func>
class AdjointInspector1D {
public:
    /** @brief Construct the inspector. Func signature: void(Var<int>& id, AdjointContext& ctx) */
    AdjointInspector1D(Func&& func, int workSizeX = 256);

    /** @brief Get the forward-pass GLSL source. */
    std::string GetForwardCode() const;

    /** @brief Get the backward-pass GLSL source. */
    std::string GetBackwardCode() const;

    /** @brief Get a text summary of all tape entries. */
    std::string GetTapeSummary() const;

    /** @brief Print the tape summary to stdout. */
    void PrintTape();

    /** @brief Check if backward code was generated. */
    bool HasBackwardCode() const;

    /** @brief Access the underlying gradient tape. */
    const GradientTape& Tape() const;

    /** @brief Access the adjoint table (forward var → adjoint var mapping). */
    const AdjointTable& Adjoints() const;
};
```

**Example:**

```cpp
AD::AdjointInspector1D inspector([](Var<int>& i, auto& ctx) {
    Var<float> w; w = 2.0f;
    Var<float> x; x = 3.0f;
    Var<float> y = w * x;
    Var<float> loss = y * y;

    ctx.RegisterParameter(w);
    ctx.MarkLoss(loss);
});

std::string forward  = inspector.GetForwardCode();
std::string backward = inspector.GetBackwardCode();
inspector.PrintTape();  // Debug: see every recorded operation
```

### AdjointInspector2D

```cpp
template <typename Func>
class AdjointInspector2D {
public:
    /** @brief Func signature: void(Var<int>& idX, Var<int>& idY, AdjointContext& ctx) */
    AdjointInspector2D(Func&& func, int workSizeX = 16, int workSizeY = 16);

    std::string GetForwardCode() const;
    std::string GetBackwardCode() const;
    std::string GetTapeSummary() const;
    void        PrintTape();
    bool        HasBackwardCode() const;
};
```

### AdjointInspector3D

```cpp
template <typename Func>
class AdjointInspector3D {
public:
    /** @brief Func signature: void(Var<int>& idX, Var<int>& idY, Var<int>& idZ, AdjointContext& ctx) */
    AdjointInspector3D(Func&& func, int workSizeX = 8, int workSizeY = 8, int workSizeZ = 4);

    std::string GetForwardCode() const;
    std::string GetBackwardCode() const;
    std::string GetTapeSummary() const;
    void        PrintTape();
    bool        HasBackwardCode() const;
};
```

---

### AdjointContext

Passed to the kernel lambda. Provides methods to register parameters and mark the loss.

```cpp
class AdjointContext {
public:
    // ── Recommended: pass Var<T> directly, type is deduced ──
    template <typename T> void RegisterParameter(const Var<T>& var);
    template <typename T> void MarkLoss(const Var<T>& var);

    // ── String-based overloads (advanced / dynamic-name use) ──
    void RegisterParameter(const std::string& name, const std::string& glslType);
    template <typename T> void RegisterParameter(const std::string& name);
    void MarkLoss(const std::string& name, const std::string& glslType);
    template <typename T> void MarkLoss(const std::string& name);

    // ── Access underlying tape ──
    GradientTape& Tape();
};
```

| Method | Use |
|:-------|:----|
| `ctx.RegisterParameter(var)` | Register `Var<T>` as trainable parameter (recommended) |
| `ctx.RegisterParameter("v3", "float")` | Register by explicit name + type string |
| `ctx.MarkLoss(var)` | Mark `Var<T>` as scalar loss (recommended) |
| `ctx.MarkLoss("v5", "float")` | Mark loss by explicit name + type string |

**Example:**

```cpp
Var<float> w; w = 2.0f;
Var<Vec3> v;  v = MakeFloat3(1, 2, 3);

ctx.RegisterParameter(w);    // float parameter
ctx.RegisterParameter(v);    // vec3 parameter
ctx.MarkLoss(loss);          // float loss
```

---

### Free Functions: AD::Param / AD::Loss

Used with `ADKernel1D`. Must be called inside the kernel lambda during construction.

```cpp
/** @brief Mark a Var<T> as a trainable parameter. Returns the parameter index (0, 1, ...). */
template <typename T>
int AD::Param(const Var<T>& var);

/** @brief Mark a whole BufferRef<T> as a trainable parameter group. */
template <typename T>
int AD::ParamBuffer(const BufferRef<T>& buffer, size_t elementCount);

/** @brief Mark a Var<T> as the scalar loss. */
template <typename T>
void AD::Loss(const Var<T>& var);
```

**Example:**

```cpp
AD::ADKernel1D kernel([](Var<int>& i) {
    auto x = buf_x[i];
    auto w = buf_w[i];
    auto b = buf_b[i];

    auto y_pred = w * x + b;
    auto diff   = y_pred - y_true;
    auto loss   = diff * diff;

    int iw = AD::Param(w);   // returns 0
    int ib = AD::Param(b);   // returns 1
    AD::Loss(loss);
}, N);
```

---

### ADKernel1D

GPU-executable training kernel. Wraps a `Kernel1D` and generates a combined forward+backward shader with gradient buffer management. Gradient and adjoint buffers are cleared on the GPU before each backward dispatch.

```cpp
class ADKernel1D {
public:
    /** @brief Construct the AD kernel.
     *  @param func         The computation lambda: void(Var<int>& id)
     *  @param elementCount Total number of GPU threads
     *  @param groupSize    Work group size (default 256)
     */
    template <typename Func>
    ADKernel1D(Func&& func, size_t elementCount, int groupSize = 256);


    // ── Execution ──────────────────────────────────────────────

    /** @brief Dispatch forward pass only (user's computation). */
    void Forward(int groupCount, bool sync = false);

    /** @brief Dispatch combined forward+backward pass. Computes loss and writes gradients.
     *  @param sync If true, wait for GPU completion before returning.
     */
    void Backward(int groupCount, bool sync = false);

    /** Vulkan-only SPIR-V optimization controls. OpenGL accepts them silently. */
    void SetOptimizationLevel(Backend::ShaderOptimizationLevel level);
    Backend::ShaderOptimizationLevel GetOptimizationLevel() const;
    std::string GetOptimizedForwardGLSL();
    std::string GetOptimizedCombinedGLSL();


    // ── Gradient download ─────────────────────────────────────

    /** @brief Download gradient for a parameter by index (matching AD::Param() call order).
     *  @return std::vector<float> with elementCount entries.
     */
    std::vector<float> Gradient(int paramIndex) const;

    /** @brief Download gradient for a parameter by its GLSL variable name. */
    std::vector<float> Gradient(const std::string& paramVarName) const;

    /** @brief Batch-download all parameter gradients for inspection or custom CPU optimizers. */
    std::vector<std::vector<float>> DownloadAllGradients() const;


    // ── Debugging ─────────────────────────────────────────────

    /** @brief Get the forward-only GLSL source. */
    std::string ForwardCode() const;

    /** @brief Get the combined forward+backward GLSL source. */
    std::string CombinedCode() const;

    /** @brief Access the gradient tape. */
    const GradientTape& Tape() const;

    /** @brief Number of registered parameters. */
    size_t ParameterCount() const;
};
```

**Full training example:**

```cpp
Buffer<float> buf_x(xData);
Buffer<float> buf_y(yData);
Buffer<float> buf_w(N);  // weight parameter
Buffer<float> buf_b(N);  // bias parameter

AD::ADKernel1D kernel([](Var<int>& i) {
    auto x      = buf_x[i];
    auto y_true = buf_y[i];
    auto w      = buf_w[i];
    auto b      = buf_b[i];

    auto y_pred = w * x + b;
    auto loss   = (y_pred - y_true) * (y_pred - y_true);

    AD::Param(w);
    AD::Param(b);
    AD::Loss(loss);
}, N);

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    kernel.Backward(4, true);           // Forward + backward in one dispatch
    auto grad_w = kernel.Gradient(0);   // ∂loss/∂w
    auto grad_b = kernel.Gradient(1);   // ∂loss/∂b
    // ... SGD update on CPU ...
}
```

For NN training, prefer `kernel.Backward(groups, false)` followed by `Adam::Step(kernel)`, `SGD::Step(kernel)`, or `RMSprop::Step(kernel)`. The built-in optimizers consume the gradient buffers directly on the GPU, so the CPU download path above is mostly useful for debugging and custom experiments.

> ⚠️ **Gradient buffer sharing**: Multiple parameters from the same source buffer (e.g., `buf_W[0]`, `buf_W[1]`) share a single gradient SSBO with an interleaved layout. `Gradient(index)` automatically extracts the correct slice. This keeps shader storage block usage within `GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS`.

---

### AdjointKernel1D / 2D / 3D

GPU-executable combined forward+backward kernels. Similar to `AdjointInspector` but produces a single merged shader for GPU dispatch.

```cpp
template <typename Func>
class AdjointKernel1D {
public:
    /** @brief Func signature: void(Var<int>& id, AdjointContext& ctx) */
    AdjointKernel1D(Func&& func, int workSizeX = 256);

    std::string GetForwardCode() const;
    std::string GetCombinedCode() const;
    std::string GetBackwardBodyCode() const;
    const GradientTape& Tape() const;
    const AdjointBody& Body() const;
    const std::vector<GradBuffer>& GradBuffers() const;
    bool HasCombinedCode() const;
};

template <typename Func>
class AdjointKernel2D {
public:
    AdjointKernel2D(Func&& func, int workSizeX = 16, int workSizeY = 16);
    // ... same methods as AdjointKernel1D
};

template <typename Func>
class AdjointKernel3D {
public:
    AdjointKernel3D(Func&& func, int workSizeX = 8, int workSizeY = 8, int workSizeZ = 4);
    // ... same methods as AdjointKernel1D
};
```

---

### GradientTape (Low-Level)

The Wengert list that records every differentiable operation during the forward pass.

```cpp
class GradientTape {
public:
    // ── Recording ──────────────────────────────────────────
    void Record(const Node& node, bool isStatement);
    void RecordRemapped(const TapeEntry& entry);

    // ── Parameter & loss management ────────────────────────
    void RegisterParameter(const std::string& name, const std::string& glslType);
    bool IsParameter(const std::string& name) const;
    void MarkLoss(const std::string& name, const std::string& glslType);
    const std::optional<TapeVar>& LossVar() const;

    // ── Variable queries ───────────────────────────────────
    bool IsActive(const std::string& name) const;
    const std::string* GetVarType(const std::string& name) const;
    size_t ParameterCount() const;
    const auto& Parameters() const;  // → unordered_map<string, string>

    // ── Control flow markers ───────────────────────────────
    void BeginIfBranch(const std::string& condExpr);
    void BeginElifBranch(const std::string& condExpr);
    void BeginElseBranch();
    void EndIfChain();
    void BeginForLoop(const std::string& varName, const std::string& start,
                      const std::string& end, const std::string& step);
    void EndForLoop();

    // ── Sub-tape support (Callable body recording) ─────────
    void PushSubTape();
    int  PopSubTape();       // returns sub-tape index
    size_t SubTapeCount() const;
    const GradientTape& SubTape(int index) const;

    // ── Access ─────────────────────────────────────────────
    size_t Size() const;
    const TapeEntry& operator[](int32_t i) const;
    const auto& Entries() const;  // → vector<TapeEntry>
    static bool IsActive();       // tape active on the Builder?
};
```

---

### AdjointGenerator (Low-Level)

Walks a tape in reverse and generates adjoint GLSL code.

```cpp
class AdjointGenerator {
public:
    /** @brief Generate complete backward-pass GLSL (with #version, layout, main). */
    std::string Generate(const GradientTape& tape, bool writeBackParams = true);

    /** @brief Generate adjoint body parts for merging into an existing shader. */
    AdjointBody GenerateBody(const GradientTape& tape, bool writeBackParams = true);

    /** @brief Get the adjoint table after generation. */
    const AdjointTable& GetAdjointTable() const;
};
```

---

### Supporting Types

#### AdjointBody

```cpp
struct AdjointBody {
    std::vector<std::pair<std::string, std::string>> declarations;  // (adjName, glslType)
    std::vector<std::string> lines;                                  // adjoint accumulation statements
    std::vector<std::pair<std::string, std::string>> writebacks;     // (paramName, adjName)
    std::string callableAdjointFunctions;                            // adjoint function definitions
};
```

#### AdjointTable

Maps forward variable names to adjoint (gradient) variable names.

```cpp
class AdjointTable {
public:
    std::string GetOrCreate(const std::string& varName, const std::string& glslType);
    std::string Get(const std::string& varName) const;
    bool Has(const std::string& varName) const;
    std::vector<std::pair<std::string, std::string>> AllDeclarations() const;
    void Clear();
    static std::string MakeAdjointName(const std::string& varName);  // "v5" → "d_v5"
};
```

#### TapeEntry

A single recorded operation on the tape.

```cpp
struct TapeVar {
    std::string name;        // GLSL variable name (e.g., "v5", "buf0[v3]")
    std::string glslType;    // GLSL type (e.g., "float", "vec3")
    bool        isParameter;
};

enum class TapeOpKind : uint8_t {
    BinaryOp,          // v = a + b, a * b, a / b, a - b
    UnaryOp,           // v = -a
    Intrinsic1,        // v = sin(x), sqrt(x), exp(x) ...
    Intrinsic2,        // v = pow(a,b), atan2(y,x) ...
    Intrinsic3,        // v = clamp(x,lo,hi), mix(a,b,t) ...
    Ternary,           // v = cond ? a : b
    CompoundAssign,    // v += a, v *= a ...
    Call,              // v = callable_func(args...)
    Return,            // return v
    ControlFlowBegin,  // Entering if / for block
    ControlFlowEnd,    // Leaving if / for block
    Loss,              // Scalar loss marker
};

enum class ControlFlowKind : uint8_t {
    IfBranch,    // if(condition)
    ElifBranch,  // else if(condition)
    ElseBranch,  // else
    ForLoop,     // for(start, end, step)
};

struct TapeEntry {
    int32_t                    id;
    TapeOpKind                 kind;
    TapeVar                    output;
    std::vector<TapeVar>       inputs;

    // Operation-specific metadata
    OperationCode              binaryOp;        // for BinaryOp/UnaryOp
    CompoundAssignmentCode     compoundOp;      // for CompoundAssign
    std::string                intrinsicName;   // for Intrinsic1/2/3
    std::string                callableFuncName; // for Call
    int                        callableIndex;    // for Call: sub-tape index

    // Control flow metadata (ControlFlowBegin only)
    ControlFlowKind            controlFlowKind;
    std::string                conditionVarName;
    std::string                forVarName;
    std::string                forStart;
    std::string                forEnd;
    std::string                forStep;
};
```

#### GradBuffer

Tracks a gradient buffer associated with a registered parameter (used by `AdjointKernel1D/2D/3D`).

```cpp
struct GradBuffer {
    std::string         paramName;
    std::string         glslType;
    uint32_t            binding;
    uint32_t            count;
    BufferHandle        handle;
    bool                allocated;
};
```

---

### Supported Differentiable Operations

| Category | Operations | Gradient Rule |
|:---------|:-----------|:--------------|
| Arithmetic | `+`, `-`, `*`, `/`, `-x` | Standard calculus rules |
| Compound | `+=`, `-=` | Accumulated adjoint |
| Trig | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` | Standard trig derivatives |
| Hyperbolic | `sinh`, `cosh`, `tanh` | Standard hyperbolic derivatives |
| Exp/Log | `exp`, `log`, `exp2`, `log2` | Standard exp/log derivatives |
| Power/Root | `pow`, `sqrt`, `inversesqrt` | Standard power/root derivatives |
| Other | `abs`, `min`, `max`, `clamp`, `mix`, `smoothstep` | Subgradient where applicable |
| Vector | `dot`, `cross`, `length`, `distance`, `normalize`, `reflect`, `refract` | Vector calculus rules |
| Control Flow | `If`/`Else`, `For` | Reversed in backward pass |
| Callable | User-defined functions | Sub-tape recording + inline |

**Zero-gradient operations** (skipped by the tape): `floor`, `ceil`, `trunc`, `round`, `sign`, `step`, `faceforward`, `floatBitsToInt`, `intBitsToFloat`, `floatBitsToUint`, `uintBitsToFloat`.

---

## Neural Network

The NN module (`include/NN/NN.h`) provides Tensor, Optimizer, Layers, and Loss functions that integrate with the AD engine. All classes live in `namespace GPU::NN`.

Include the umbrella header:

```cpp
#include <NN/NN.h>
using namespace GPU::NN;
```

### Class Overview

| Class | Purpose |
|:------|:--------|
| `Tensor<T, Dims...>` | Multi-dimensional weight container, GPU sync, batch param registration |
| `TensorRef<T, Dims...>` | DSL-side tensor handle (returned by `Tensor::Bind()`) |
| `Adam` | GPU Adam optimizer with bias correction, weight decay, gradient clipping |
| `SGD` | GPU SGD with momentum, weight decay, gradient clipping |
| `RMSprop` | GPU RMSprop with moving average, weight decay, gradient clipping |
| `Linear<T, In, Out>` | Fully-connected layer with Xavier init |
| `ReLU<T>` / `Sigmoid<T>` / `Tanh<T>` | Activation layers |
| `Sequential<T, Layers...>` | Compile-time layer pipeline |
| `FusedMLP2<T, In, Hidden, Out>` | Two-layer MLP emitted as one shader block |
| `FusedMLP2Trainer<T, In, Hidden, Out>` | Specialized fused MLP training path for 16/32/64-wide networks |
| `MSELoss` / `CrossEntropyLoss` | Loss functions for scalar and multi-class outputs |
| `TokenEmbedding<T, V, E>` | Learned token embedding lookup |
| `PositionalEmbedding<T, B, E>` | Learned positional embedding |
| `RMSNorm<T, Dim>` | Root-mean-square normalization (GPT-style) |
| `CausalSelfAttention<T, E, H>` | Multi-head causal self-attention |
| `TransformerBlock<T, B, E, H>` | Pre-norm transformer block (attention + fused FFN) |
| `SaveWeights` / `LoadWeights` | Checkpoint save/load to binary files |

---

### Tensor

`Tensor<T, Dims...>` wraps a `Buffer<T>` with multi-dimensional indexing, CPU/GPU synchronization, and batch parameter registration. The shape is fixed at compile time.

```cpp
template <typename T, size_t... Dims>
class Tensor {
public:
    /// Default constructor — zero-initialized.
    Tensor();

    /// Construct from flat vector data.
    explicit Tensor(const std::vector<T>& data, BufferMode mode = ReadWrite);

    /// Construct from vector (rvalue, move).
    explicit Tensor(std::vector<T>&& data, BufferMode mode = ReadWrite);

    // ── CPU access ──────────────────────────────────────

    /// Raw pointer to CPU data.
    T* Data();
    const T* Data() const;

    /// Multi-dimensional CPU indexing: W(i, j), tok(id, dim)
    template <typename... Indices>
    T& operator()(Indices... indices);

    /// Total number of elements (= Dims[0] * Dims[1] * ...).
    static constexpr size_t Size();

    // ── GPU synchronization ─────────────────────────────

    /// Upload CPU data to GPU buffer.
    void Upload();

    /// Download GPU data to CPU buffer.
    void Download();

    // ── DSL integration ─────────────────────────────────

    /// Bind for use inside a kernel lambda. Returns TensorRef<T, Dims...>.
    auto Bind();

    /// Access the underlying Buffer<T> (for manual GPU I/O).
    Buffer<T>& GetBuffer();
    const Buffer<T>& GetBuffer() const;

    // ── Move semantics ──────────────────────────────────

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Non-copyable (single owner of GPU buffer)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
};
```

**Example:**

```cpp
// Create a weight matrix
std::vector<float> data(128 * 64);
// ... fill with Xavier init ...
Tensor<float, 128, 64> W(data);

// CPU access
float val = W(3, 7);
W(0, 1) = 0.5f;

// Upload to GPU
W.Upload();
```

---

### TensorRef

`TensorRef<T, Dims...>` is the DSL-side handle returned by `Tensor::Bind()`. It provides multi-dimensional indexing into kernel values and batch parameter registration.

```cpp
template <typename T, size_t... Dims>
class TensorRef {
public:
    /// Multi-dimensional indexing. Returns Var<T> for scalar access,
    /// Expr<int> for index arithmetic.
    template <typename... Indices>
    auto operator()(Indices... indices);

    /// Register the whole tensor buffer as one AD parameter group.
    void RegisterAsParam();

    /// Legacy scalar registration hook. Prefer RegisterAsParam() for tensors.
    template <typename F>
    void ForEachParam(F&& f);
};
```

**Example:**

```cpp
// Inside kernel lambda
auto W = weightTensor.Bind();
auto b = biasTensor.Bind();

// Multi-dimensional indexing
Var<float> w = W(i, j);               // weightTensor[i * stride_j + j]
Var<float> bias = b(k);

// Batch parameter registration — one buffer-level parameter group per tensor
W.RegisterAsParam();
b.RegisterAsParam();
```

**Index computation:** `TensorRef::operator()` computes the flat offset at compile time using `StrideAt<I, Dims...>`. For `Tensor<float, 128, 64>`:
- `W(i, j)` → `W[i * 64 + j]` (row-major layout)
- `W(k)` → `W[k]` (flat indexing)

---

### Optimizers

All optimizers live in `GPU::NN`. They manage per-parameter state (first/second moments for Adam, velocity for SGD, squared average for RMSprop) and update registered tensors directly on the GPU in `Step()`.

The optimizer state is flattened into one GPU buffer per state vector (`m`/`v`, velocity, or square average). When the backend binding limit allows, registered tensor parameters are handled by combined GPU dispatches. Adam uses a parallel reduction dispatch to average AD gradients, then an update dispatch. If the model uses too many tensor buffers for the combined path, EasyGPU falls back to one GPU dispatch per tensor.

#### Adam

```cpp
class Adam {
public:
    /// @param lr    Learning rate (default 0.001)
    /// @param beta1 First moment decay (default 0.9)
    /// @param beta2 Second moment decay (default 0.999)
    /// @param eps   Numerical stability (default 1e-8)
    Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

    /// Enable L2 weight decay.
    void SetWeightDecay(float wd);

    /// Enable gradient clipping by value.
    void SetGradClip(float clip);

    /// Register a raw weight array as trainable.
    void AddParameter(float* data, size_t size, Buffer<float>* buf = nullptr);

    /// Register all elements of a Tensor as trainable parameters.
    template <size_t... Dims>
    void AddTensor(Tensor<float, Dims...>& tensor);

    /// Execute one optimization step on the GPU.
    /// Reads AD gradient buffers, reduces per-thread gradients,
    /// applies Adam update, and writes weights in place.
    void Step(ADKernel1D& kernel);

    int GetStep() const;
    size_t ParameterCount() const;
};
```

**Update rule:** For each scalar parameter (with per-thread gradient averaging):

```
m[t]   = beta1 * m[t-1] + (1 - beta1) * g
v[t]   = beta2 * v[t-1] + (1 - beta2) * g²
m_hat  = m[t] / (1 - beta1^step)
v_hat  = v[t] / (1 - beta2^step)
weight -= lr * m_hat / (sqrt(v_hat) + eps)
```

**Example:**

```cpp
Adam adam(0.001f, 0.9f, 0.999f);
adam.SetWeightDecay(0.0001f);
adam.SetGradClip(1.0f);

adam.AddTensor(fc1.Weight());
adam.AddTensor(fc1.Bias());
adam.AddTensor(fc2.Weight());
adam.AddTensor(fc2.Bias());

// Training loop
for (int step = 0; step < 1000; step++) {
    kernel.Backward(groups, false);
    adam.Step(kernel);  // GPU aggregate + update
}
```

#### SGD

```cpp
class SGD {
public:
    /// @param lr       Learning rate (default 0.01)
    /// @param momentum Momentum coefficient (0 = vanilla SGD)
    SGD(float lr = 0.01f, float momentum = 0.0f);

    void SetWeightDecay(float wd);
    void SetGradClip(float clip);
    void AddParameter(float* data, size_t size, Buffer<float>* buf = nullptr);
    template <size_t... Dims> void AddTensor(Tensor<float, Dims...>& tensor);
    void Step(ADKernel1D& kernel);
    int GetStep() const;
    size_t ParameterCount() const;
};
```

**Update rule:** `v = momentum * v + g; weight -= lr * v`

#### RMSprop

```cpp
class RMSprop {
public:
    /// @param lr   Learning rate (default 0.001)
    /// @param beta Moving average decay (default 0.9)
    /// @param eps  Numerical stability (default 1e-8)
    RMSprop(float lr = 0.001f, float beta = 0.9f, float eps = 1e-8f);

    void SetWeightDecay(float wd);
    void SetGradClip(float clip);
    void AddParameter(float* data, size_t size, Buffer<float>* buf = nullptr);
    template <size_t... Dims> void AddTensor(Tensor<float, Dims...>& tensor);
    void Step(ADKernel1D& kernel);
    int GetStep() const;
    size_t ParameterCount() const;
};
```

**Update rule:** `sq = beta * sq + (1 - beta) * g²; weight -= lr * g / sqrt(sq + eps)`

---

### Layers

All layers follow the same pattern: construct outside the kernel, call `Setup()` inside the kernel to register parameters, call `Forward()` to emit DSL code.

#### Linear

Fully-connected layer: `y = xW + b`. Weights initialized with Xavier uniform, biases initialized to zero.

```cpp
template <typename T, size_t InFeatures, size_t OutFeatures>
class Linear {
public:
    /// Construct with Xavier-initialized weights and zero biases.
    Linear(unsigned initSeed = 42);

    /// Register weights and biases as AD parameters.
    void Setup();

    /// Forward pass. Reads input from inBuf starting at offset, writes to outBuf.
    void Forward(const BufferRef<T>& inBuf, const Var<int>& threadId,
                 const BufferRef<T>& outBuf, int inOffset = 0, int outOffset = 0);

    /// Forward pass with explicit offset expressions.
    void Forward(const BufferRef<T>& inBuf, const Expr<int>& inOff,
                 const BufferRef<T>& outBuf, const Expr<int>& outOff);

    Tensor<T, InFeatures, OutFeatures>& Weight();
    Tensor<T, OutFeatures>& Bias();
    static constexpr size_t WeightSize = InFeatures * OutFeatures;
    static constexpr size_t BiasSize   = OutFeatures;
};
```

**Example:**

```cpp
Linear<float, 784, 128> fc1(42);  // 784→128 with Xavier init

// In kernel:
fc1.Setup();
fc1.Forward(input, threadId, hidden);
```

#### Activation Layers

```cpp
template <typename T, size_t Dim = 0>
class ReLU {
public:
    /// @param dim If > 0, activates exactly dim elements per sample. If 0, Dim is used.
    explicit ReLU(size_t dim = Dim);

    void Setup();  // no-op (no parameters)
    void Forward(const BufferRef<T>& buf, const Var<int>& threadId,
                 const BufferRef<T>& out, int offset = 0);
    void Forward(const BufferRef<T>& buf, const Expr<int>& baseOff,
                 const BufferRef<T>& out, const Expr<int>& outOff);
};
```

`Sigmoid<T, Dim>` and `Tanh<T, Dim>` have the same API as `ReLU`.

**Example:**

```cpp
ReLU<float, 128> relu;
relu.Setup();
relu.Forward(hidden, threadId, activated);
```

#### Sequential

Compile-time layer pipeline. Composes multiple layers and forwards through them in order.

```cpp
template <typename T, typename... Layers>
class Sequential {
public:
    Sequential(Layers&... layers);

    /// Register all parameters from all layers.
    void Setup();

    /// Forward through all layers. First layer reads from inBuf,
    /// intermediate results flow through internal buffers, final output
    /// written to outBuf.
    void Forward(const BufferRef<T>& inBuf, const Var<int>& threadId,
                 const BufferRef<T>& outBuf, size_t dim);
};
```

**Example:**

```cpp
Linear<float, 784, 128> fc1;
ReLU<float, 128> relu;
Linear<float, 128, 10> fc2;

Sequential<float, Linear<float, 784, 128>, ReLU<float, 128>, Linear<float, 128, 10>>
    mlp(fc1, relu, fc2);

// In kernel — one Setup + one Forward
mlp.Setup();
mlp.Forward(input, threadId, output, 10);
```

#### FusedMLP2

Two-layer MLP block: `y = W2(activation(W1x + b1)) + b2`. Unlike `Sequential<Linear, Activation, Linear>`, `FusedMLP2` emits the hidden layer as shader-local values inside one block of DSL code, avoiding intermediate buffer traffic.

```cpp
enum class FusedActivation {
    ReLU,
    Tanh,
    None
};

template <typename T, size_t InFeatures, size_t HiddenFeatures, size_t OutFeatures>
class FusedMLP2 {
public:
    FusedMLP2(unsigned initSeed = 42,
              FusedActivation activation = FusedActivation::ReLU);

    void Reset(unsigned initSeed = 42);
    void Setup(bool registerParams = true);

    void Forward(const BufferRef<T>& input,
                 const Var<int>& threadId,
                 const BufferRef<T>& output);

    Tensor<T, HiddenFeatures, InFeatures>& W1();
    Tensor<T, HiddenFeatures>& B1();
    Tensor<T, OutFeatures, HiddenFeatures>& W2();
    Tensor<T, OutFeatures>& B2();
    static constexpr size_t ParamCount();
};
```

Use this when the MLP dimensions are small enough to keep the hidden activation in shader locals. For GPT-style blocks, `TransformerBlock` already uses an integrated FFN path.

#### FusedMLP2Trainer

Specialized two-layer MLP trainer for small dense networks with widths `16`, `32`, or `64`. This is a raw GLSL path, not a generic DSL layer: it generates statically-unrolled shaders for forward inference, MSE loss, backward gradient accumulation, and Adam update.

```cpp
template <typename T, size_t InFeatures, size_t HiddenFeatures, size_t OutFeatures>
class FusedMLP2Trainer {
public:
    /// Widths must be 16, 32, or 64. T must be float.
    explicit FusedMLP2Trainer(unsigned seed = 42);

    Runtime::Buffer<float>& W1();
    Runtime::Buffer<float>& B1();
    Runtime::Buffer<float>& W2();
    Runtime::Buffer<float>& B2();
    Runtime::Buffer<float>& LossBuffer();

    void Reset(unsigned seed = 42);
    void SetWeights(const std::vector<float>& w1,
                    const std::vector<float>& b1,
                    const std::vector<float>& w2,
                    const std::vector<float>& b2);

    void Forward(Runtime::Buffer<float>& input,
                 Runtime::Buffer<float>& output,
                 size_t batch,
                 bool sync = false);

    void TrainMSE(Runtime::Buffer<float>& input,
                  Runtime::Buffer<float>& target,
                  size_t batch,
                  float lr = 0.001f,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float eps = 1e-8f,
                  bool sync = false);

    float DownloadLoss();
    static constexpr size_t ParameterCount();

    static std::string ForwardShaderSource(size_t batch);
    static std::string TrainingShaderSource(size_t batch);
    static std::string UpdateShaderSource();
};
```

`TrainMSE()` uses three compute dispatches: clear gradient/loss buffers, run fused forward+loss+backward, then update all parameters with Adam. The generated training shader stages weights in shared memory and keeps activations/adjoints in scalar locals. Floating-point gradient accumulation uses an integer `atomicCompSwap` CAS loop, so it does not require vendor-specific float atomic extensions.

```cpp
using Trainer = FusedMLP2Trainer<float, 16, 32, 16>;

Trainer mlp(42);
Buffer<float> input(batch * 16, BufferMode::Read);
Buffer<float> target(batch * 16, BufferMode::Read);

for (int step = 0; step < 1000; step++) {
    mlp.TrainMSE(input, target, batch, 1e-3f);
}
```

Use this path when the network shape is known and small enough to benefit from static unrolling. For arbitrary layer stacks, keep using `Sequential` + AD + GPU optimizers.

---

### Loss Functions

Loss functions produce a `Var<float>` that you pass to `AD::Loss()`.

#### MSELoss

```cpp
/// Mean Squared Error from buffers. Sums over outputDim elements per sample.
Var<float> MSELoss(const BufferRef<float>& predBuf,
                    const BufferRef<float>& targetBuf,
                    const Var<int>& threadId,
                    int outputDim);

/// Scalar MSE: loss = (pred - target)²
Var<float> MSELoss(const Var<float>& pred, const Var<float>& target);
```

#### CrossEntropyLoss

```cpp
/// Cross-entropy loss for multi-class classification (logits, not log-probabilities).
/// Performs max-reduction → exp → sum → log → subtraction → negation.
/// The targetId indexes into logits starting at logitsBase.
Var<float> CrossEntropyLoss(const BufferRef<float>& logitsBuf,
                              int vocabSize,
                              const Var<int>& targetId,
                              const Expr<int>& logitsBase);
```

**Example:**

```cpp
// Inside kernel — GPT-style next-token prediction
Var<float> totalLoss = CrossEntropyLoss(data, vocabSize, targetId, logitsBase);
AD::Loss(totalLoss);
```

---

### Embeddings

#### TokenEmbedding

Learned embedding lookup by token ID. Shape: `[VocabSize, EmbedDim]`.

```cpp
template <typename T, size_t VocabSize, size_t EmbedDim>
class TokenEmbedding {
public:
    TokenEmbedding(unsigned initSeed = 42);

    void Setup();  // registers weight as AD parameters
    void Forward(const Expr<int>& tokenId, const BufferRef<T>& out, const Expr<int>& outOffset);

    Tensor<T, VocabSize, EmbedDim>& Weight();
    static constexpr size_t TotalSize = VocabSize * EmbedDim;
};
```

#### PositionalEmbedding

Learned positional embedding. Shape: `[BlockSize, EmbedDim]`.

```cpp
template <typename T, size_t BlockSize, size_t EmbedDim>
class PositionalEmbedding {
public:
    PositionalEmbedding(unsigned initSeed = 42);

    void Setup();
    void Forward(const Expr<int>& pos, const BufferRef<T>& out, const Expr<int>& outOffset);

    Tensor<T, BlockSize, EmbedDim>& Weight();
    static constexpr size_t TotalSize = BlockSize * EmbedDim;
};
```

**Example:**

```cpp
TokenEmbedding<float, 27, 16>     tokEmb(42);
PositionalEmbedding<float, 16, 16> posEmb(123);

// In kernel:
tokEmb.Setup();
posEmb.Setup();
// x[pos] = tokEmb[tokenId] + posEmb[pos]
tokEmb.Forward(tokenId, data, pos * embedDim);
posEmb.Forward(pos, data, pos * embedDim);
```

---

### Normalization

#### RMSNorm

Root-mean-square normalization (GPT-2 style). No learnable parameters — purely structural.

```cpp
template <typename T, size_t Dim>
class RMSNorm {
public:
    void Setup();  // no-op
    void Forward(const BufferRef<T>& inBuf, const Var<int>& threadId,
                 const BufferRef<T>& outBuf, int offset = 0);
    void Forward(const BufferRef<T>& inBuf, const Expr<int>& inOff,
                 const BufferRef<T>& outBuf, const Expr<int>& outOff);
};
```

**Computation:** For input `x[0..dim-1]`: `rms = sqrt(mean(x²) + eps); out[i] = x[i] / rms`

---

### Attention

#### CausalSelfAttention

Multi-head causal self-attention (GPT-2 style). Weights packed into a single `[4, EmbedDim, EmbedDim]` tensor (Q, K, V, O) to minimize SSBO binding slots. Uses three-pass safe softmax (max reduction → exp-sum → weighted sum). No biases.

```cpp
template <typename T, size_t EmbedDim, size_t NumHeads>
class CausalSelfAttention {
    static constexpr size_t HeadDim = EmbedDim / NumHeads;  // must divide evenly

public:
    CausalSelfAttention(unsigned initSeed = 42);

    void Setup();  // registers weight tensor as AD parameters

    /// Forward pass for a single position in a sequence.
    /// @param scratch  Single scratch buffer (holds K, V, AttnOut, MLP regions)
    /// @param xOff     Offset of input (normalized residual) within scratch
    /// @param kOff     Offset of K region
    /// @param vOff     Offset of V region
    /// @param aOff     Offset of AttnOut region
    /// @param pos      Current position (0..blockSize-1)
    /// @param offset   Base offset for this batch
    void Forward(const BufferRef<T>& scratch,
                 const Expr<int>& xOff, const Expr<int>& kOff,
                 const Expr<int>& vOff, const Expr<int>& aOff,
                 const Expr<int>& pos,  const Expr<int>& offset);

    Tensor<T, 4, EmbedDim, EmbedDim>& Weights();
    static constexpr size_t ParamCount = 4 * EmbedDim * EmbedDim;
};
```

---

### Transformer

#### TransformerBlock

Pre-norm transformer block (GPT-2 architecture): RMSNorm → Attention → +residual → RMSNorm → MLP(ReLU) → +residual. Uses a single scratch buffer internally to minimize SSBO bindings and keeps the FFN path inside the Transformer block kernel.

```cpp
template <typename T, size_t BlockSize, size_t EmbedDim, size_t NumHeads>
class TransformerBlock {
    static constexpr size_t MLPDim = 4 * EmbedDim;

public:
    /// @param batchSize Number of samples per batch (for scratch buffer sizing)
    TransformerBlock(size_t batchSize, unsigned seed = 42);

    void Setup();  // registers attention, fc1, fc2 parameters

    /// Forward pass for a single position.
    /// @param data     Main data buffer (input + output)
    /// @param pos      Current position (0..blockSize-1)
    /// @param seqBase  Base offset for this batch in the data buffer
    void Forward(const BufferRef<T>& data, const Expr<int>& pos, const Expr<int>& seqBase);

    CausalSelfAttention<T, EmbedDim, NumHeads>& Attention();
    Tensor<T, MLPDim, EmbedDim>& FC1();  // MLP first linear layer
    Tensor<T, EmbedDim, MLPDim>& FC2();  // MLP second linear layer
    static constexpr size_t ParamCount = 4 * EmbedDim * EmbedDim + EmbedDim * MLPDim * 2;
};
```

**Architecture:**
```
x = x + Attention(RMSNorm(x))    // attention with residual
x = x + FC2(ReLU(FC1(RMSNorm(x))))  // MLP with residual
```

---

### Checkpoint

Save and load model weights to/from binary files.

```cpp
/// Save one or more tensors to a binary checkpoint file.
/// File format: [numTensors: uint32] [size0: uint64] [data0...] [size1: uint64] [data1...] ...
template <typename... Tensors>
void SaveWeights(const std::string& path, Tensors&... tensors);

/// Load one or more tensors from a binary checkpoint file.
/// Throws std::runtime_error if sizes don't match.
template <typename... Tensors>
void LoadWeights(const std::string& path, Tensors&... tensors);
```

**Example:**

```cpp
// Save
SaveWeights("model.bin", fc1.Weight(), fc1.Bias(), fc2.Weight(), fc2.Bias());

// Load
LoadWeights("model.bin", fc1.Weight(), fc1.Bias(), fc2.Weight(), fc2.Bias());
```

---

## SideEffectToken

Internal token class used by `Callable<void>` to ensure side-effects are recorded.

```cpp
// Returned by Callable<void>::operator()
// Automatically commits side-effects at statement end
Callable<void(int&)> A = [](Int &a) { a = 20; };
A(b);  // Creates and destroys SideEffectToken, committing the effect
```

**Note:** This is an implementation detail. Users do not interact with `SideEffectToken` directly.

---

## Structs

### EASYGPU_STRUCT Macro

Define GPU-compatible structs.

```cpp
EASYGPU_STRUCT(Name,
    (Type1, field1),
    (Type2, field2),
    ...
);
```

**Supported Types:**
- `float`, `int`, `bool`
- `Vec2`, `Vec3`, `Vec4`
- `IVec2`, `IVec3`, `IVec4`
- `Mat2`, `Mat3`, `Mat4`, etc.
- Other registered structs

**Example:**

```cpp
EASYGPU_STRUCT(Particle,
    (Float3, position),
    (Float3, velocity),
    (float, mass)
);

// Use in buffer
Buffer<Particle> particles(1000);

// Access in kernel
Kernel1D update([](Int i) {
    auto p = particles.Bind();
    
    // Read
    Float3 pos = p[i].position();
    
    // Write
    p[i].position() = pos + velocity * dt;
});
```

### Nested Structs

```cpp
EASYGPU_STRUCT(Material,
    (Float3, albedo),
    (Float, roughness)
);

EASYGPU_STRUCT(Triangle,
    (Vec3, v0),
    (Vec3, v1),
    (Vec3, v2),
    (Material, mat)
);
```

---

## Resource Slots

Slots enable dynamic resource switching at runtime without kernel recompilation.

### BufferSlot

Dynamic buffer binding for ping-pong and multi-pass algorithms.

```cpp
BufferSlot<float> dataSlot;

Kernel1D kernel([&](Int i) {
    auto data = dataSlot.Bind();
    data[i] = data[i] * 2.0f;
});

Buffer<float> bufA(1024), bufB(1024);

dataSlot.Attach(bufA);
kernel.Dispatch(4, true);  // Process bufA

dataSlot.Attach(bufB);
kernel.Dispatch(4, true);  // Process bufB - same kernel!
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Attach(Buffer<T>& buffer)` | Attach a buffer to this slot |
| `Detach()` | Detach current buffer |
| `IsAttached()` | Check if a buffer is attached |
| `GetAttached()` | Get pointer to attached buffer |
| `Bind()` | Bind slot in kernel (returns `BufferRef<T>`) |

### TextureSlot

Dynamic texture binding for image processing pipelines.

```cpp
TextureSlot<RGBA8> imageSlot;

Kernel2D kernel([&](Int x, Int y) {
    auto img = imageSlot.Bind();
    Float4 color = img.Read(x, y);
    img.Write(x, y, color * 0.5f);
});

TextureRGBA8 texA(1024, 1024), texB(1024, 1024);

imageSlot.Attach(texA);
kernel.Dispatch(64, 64, true);  // Process texA

imageSlot.Attach(texB);
kernel.Dispatch(64, 64, true);  // Process texB - same kernel!
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Attach(Texture2D<Format>& texture)` | Attach a texture to this slot |
| `Detach()` | Detach current texture |
| `IsAttached()` | Check if a texture is attached |
| `GetAttached()` | Get pointer to attached texture |
| `Bind()` | Bind slot in kernel (returns `TextureRef<Format>`) |
| `GetDimensions(width, height)` | Get dimensions of attached texture |

### Why Use Slots?

| Without Slots | With Slots |
|:--------------|:-----------|
| Recompile kernel for each resource | Compile once, switch at runtime |
| Resource fixed at definition time | Dynamic switching at dispatch time |
| Code duplication for similar operations | Single kernel, multiple resources |

---

## Textures

### Texture2D

2D texture for image data.

```cpp
Texture2D<PixelFormat::RGBA8> texture(width, height);
```

**PixelFormat:**

| Format | Description |
|:-------|:------------|
| `PixelFormat::R8` | Single channel, 8-bit |
| `PixelFormat::RG8` | Two channels, 8-bit each |
| `PixelFormat::RGBA8` | Four channels, 8-bit each |
| `PixelFormat::R32F` | Single channel, 32-bit float |
| `PixelFormat::RG32F` | Two channels, 32-bit float |
| `PixelFormat::RGBA32F` | Four channels, 32-bit float |
| `PixelFormat::R16F` | Single channel, 16-bit float |
| `PixelFormat::RG16F` | Two channels, 16-bit float |
| `PixelFormat::RGBA16F` | Four channels, 16-bit float |
| `PixelFormat::R32I` | Single channel, 32-bit signed int |
| `PixelFormat::RG32I` | Two channels, 32-bit signed int |
| `PixelFormat::RGBA32I` | Four channels, 32-bit signed int |
| `PixelFormat::R32UI` | Single channel, 32-bit unsigned int |
| `PixelFormat::RG32UI` | Two channels, 32-bit unsigned int |
| `PixelFormat::RGBA32UI` | Four channels, 32-bit unsigned int |

**Constructors:**

```cpp
Texture2D<PixelFormat::RGBA8> tex(width, height);              // Empty texture
Texture2D<PixelFormat::RGBA8> tex(width, height, data);        // With initial data
Texture2D<PixelFormat::RGBA8> tex(width, height, MipmapMode::Generate);
Texture2D<PixelFormat::RGBA8> tex(width, height, data, MipmapMode::Generate);
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Upload(const void* data)` | Upload pixel data to GPU (synchronous) |
| `UploadSubRegion(x, y, w, h, data)` | Upload partial data |
| `GenerateMipmaps()` | Regenerate all mip levels from level zero |
| `Download(void* outData)` | Download pixel data from GPU (synchronous) |
| `Download(std::vector<T>& outData)` | Download to vector |
| `Bind()` | Bind to current kernel (returns TextureRef) |
| `GetWidth()` | Get texture width |
| `GetHeight()` | Get texture height |
| `GetMipLevels()` | Get the allocated mip-level count |
| `GetHandle()` | Get OpenGL texture ID |
| `GetSizeInBytes()` | Get total size in bytes |

**PBO Async Methods:**

| Method | Description |
|:-------|:------------|
| `InitUploadPBOPool(bufferCount)` | Initialize PBO pool for async upload (typically 2-3) |
| `InitDownloadPBOPool(bufferCount)` | Initialize PBO pool for async download |
| `UploadAsync(data)` | Asynchronous upload using PBO (non-blocking) |
| `UploadAsyncStream(data, timeoutMs)` | Async upload with blocking wait for idle PBO |
| `DownloadAsync()` | Start asynchronous download |
| `GetDownloadData(outData)` | Get data from completed async download |
| `Sync()` | Wait for all async operations to complete |
| `IsIdle()` | Check if all async operations are complete |

**Usage in Kernel:**

```cpp
Texture2D<PixelFormat::RGBA8> texture(1024, 1024);

Kernel2D kernel([&](Int x, Int y) {
    auto img = texture.Bind();
    
    // Read pixel
    Float4 color = img.Read(x, y);
    
    // Write pixel
    img.Write(x, y, color * 0.5f);
});

kernel.Dispatch(64, 64, true);
```

**Type Aliases:**

```cpp
using TextureRGBA8   = Texture2D<PixelFormat::RGBA8>;
using TextureRGBA32F = Texture2D<PixelFormat::RGBA32F>;
using TextureR32F    = Texture2D<PixelFormat::R32F>;
using TextureRG32F   = Texture2D<PixelFormat::RG32F>;
using TextureR8      = Texture2D<PixelFormat::R8>;
using image2d<Format> = IR::Value::TextureRef<Format>;  // Inside kernel
```

---
---

### Texture3D

3D texture for volumetric data.

```cpp
Texture3D<PixelFormat::RGBA8> volume(width, height, depth);
Texture3D<PixelFormat::RGBA8> volume(width, height, depth, data);
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Upload(const void* data)` | Upload full volume data |
| `UploadSubRegion(x, y, z, w, h, d, data)` | Upload partial volume |
| `Download(void* outData)` | Download full volume data |
| `Download(std::vector<T>& outData)` | Download to vector |
| `Bind()` | Bind to current kernel (returns TextureRef3D) |
| `BindSampler()` | Bind as sampler (returns TextureSampler3D) |
| `GetWidth()` | Get texture width |
| `GetHeight()` | Get texture height |
| `GetDepth()` | Get texture depth |
| `GetSizeInBytes()` | Get total size in bytes |

**Type Aliases:**

```cpp
using Texture3DRGBA8   = Texture3D<PixelFormat::RGBA8>;
using Texture3DRGBA32F = Texture3D<PixelFormat::RGBA32F>;
using Texture3DR32F    = Texture3D<PixelFormat::R32F>;
using Texture3DRG32F   = Texture3D<PixelFormat::RG32F>;
using Texture3DR8      = Texture3D<PixelFormat::R8>;
using image3d<Format>  = IR::Value::TextureRef3D<Format>;
```

---

### TextureRef (2D)
### TextureRef (2D)

Reference to a 2D texture inside a kernel, returned by `Texture2D::Bind()`.

**Read Methods:**

```cpp
// All combinations of Var<int>, Expr<int>, and literal int
Float4 color = img.Read(x, y);
```

**Write Methods:**

```cpp
// All combinations of Var<int>, Expr<int>, literal int for coordinates
// and Var<Vec4>, Expr<Vec4> for color
img.Write(x, y, color);
```

---
---

### TextureRef3D (3D)

Reference to a 3D texture inside a kernel, returned by `Texture3D::Bind()`.

**Read Methods:**

```cpp
// All combinations of Var<int>, Expr<int>, and literal int
Float4 color = vol.Read(x, y, z);
```

**Write Methods:**

```cpp
// All combinations of Var<int>, Expr<int>, literal int for coordinates
// and Var<Vec4>, Expr<Vec4> for color
vol.Write(x, y, z, color);
```

---
## Texture Samplers

Texture samplers provide filtered texture sampling for fragment kernels. Unlike `TextureRef` which uses `imageLoad`/`imageStore` for compute kernels, samplers use `texture()` for hardware-accelerated filtered sampling in fragment shaders.

### TextureSampler2D

Returned by `Texture2D::BindSampler()` for use in fragment kernels.

```cpp
Texture2D<PixelFormat::RGBA8> texture(1024, 1024);

FragmentKernel2D kernel([&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
    auto tex = texture.BindSampler();  // Returns TextureSampler2D
    
    Float2 uv = fragCoord / resolution;
    
    // Sample with hardware filtering
    Float4 color = tex.Sample(uv);
    fragColor = color;
});
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Sample(uv)` | Sample texture at normalized UV coordinates (0-1) |
| `SampleLevel(uv, level)` | Sample an explicit mip level with `textureLod()` |
| `SampleGrad(uv, ddx, ddy)` | Sample using explicit screen-space gradients with `textureGrad()` |
| `GetSize()` | Get texture size as `Vec2` |
| `GetTextureWidth()` | Get width in pixels |
| `GetTextureHeight()` | Get height in pixels |

**Sampling with Different Wrapping:**

```cpp
FragmentKernel2D kernel([&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
    auto tex = texture.BindSampler();
    
    Float2 uv = fragCoord / resolution;
    
    // Repeat/tile the texture
    Float2 tiledUV = Fract(uv * MakeFloat2(3.0f, 3.0f));
    Float4 color = tex.Sample(tiledUV);
    
    fragColor = color;
});
```

### Compute vs Fragment Texture Access

| Operation | Compute Kernel | Fragment Kernel |
|:----------|:---------------|:----------------|
| Binding | `tex.Bind()` | `tex.BindSampler()` |
| Returns | `TextureRef` | `TextureSampler2D` |
| Read | `tex.Read(x, y)` | `tex.Sample(uv)` |
| Write | `tex.Write(x, y, color)` | Not supported |
| Coordinates | Integer pixel | Normalized UV (0-1) |
| Filtering | Nearest only | Bilinear/trilinear |
| GLSL | `imageLoad/Store` | `texture()` |

### Mipmaps

Create a mipmapped `Texture2D` with `MipmapMode::Generate`:

```cpp
Texture2D<PixelFormat::RGBA8> texture(1024, 1024, MipmapMode::Generate);
texture.Upload(pixels.data());  // Uploads level zero and regenerates the mip chain
```

Mipmapped textures allocate the complete chain down to `1x1`. `Upload()` and `UploadSubRegion()` automatically regenerate the chain. Use `GenerateMipmaps()` for explicit regeneration.

For discontinuous UV operations such as `Fract()`, preserve gradients from the unwrapped UV:

```cpp
Float2 tiled = Fract(sourceUV);
Float4 color = sampler.SampleGrad(tiled, Ddx(sourceUV), Ddy(sourceUV));
```

`Ddx()` and `Ddy()` map to `dFdx()` and `dFdy()` and are valid only in fragment shaders.

See [Mipmaps](mipmaps.md) for atlas sampling, backend behavior, and limitations.

### Complete Fragment Kernel with Textures

```cpp
#include <GPU.h>
#include <windows.h>

int main() {
    HWND hwnd = CreateWindowEx(...);  // Your window
    
    // Create and fill texture
    Texture2D<PixelFormat::RGBA8> image(512, 512);
    std::vector<uint8_t> pixels(512 * 512 * 4);
    // ... fill pixels ...
    image.Upload(pixels.data());
    
    Uniform<float> uTime;
    
    FragmentKernel2D kernel("TextureDemo",
        [&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
            auto tex = image.BindSampler();
            
            Float time = uTime.Load();
            
            // Animated UV coordinates
            Float2 uv = fragCoord / resolution;
            
            // Rotate UVs
            Float angle = time * 0.5f;
            Float cosA = Cos(angle);
            Float sinA = Sin(angle);
            Float2 center = MakeFloat2(0.5f, 0.5f);
            Float2 delta = uv - center;
            Float x = delta.x() * cosA - delta.y() * sinA;
            Float y = delta.x() * sinA + delta.y() * cosA;
            Float2 rotatedUV = MakeFloat2(x, y) + center;
            
            // Sample rotated texture
            fragColor = tex.Sample(rotatedUV);
        },
        512, 512
    );
    
    kernel.Attach(hwnd);
    
    while (running) {
        uTime = clock() / 1000.0f;
        kernel.Flush();
    }
    
    return 0;
}
```

---

## PBO Async Transfer

Pixel Buffer Objects (PBOs) enable asynchronous CPU/GPU data transfers, allowing CPU and GPU to work in parallel. This is essential for real-time applications like video streaming and interactive simulations.

### Overview

```
CPU Memory              GPU Memory
     �?                      �?
     �? Synchronous Upload   �?
     �?─────────────────────>�? CPU waits for GPU
     �?                      �?
     �? Async with PBO       �?
     �?─────────────────────>�? CPU continues immediately
     �?(non-blocking)        �? GPU copies in background
```

### Basic Async Upload

```cpp
Texture2D<PixelFormat::RGBA8> texture(1920, 1080);

// Initialize PBO pool with 2 buffers (double buffering)
texture.InitUploadPBOPool(2);

// Upload without blocking
std::vector<uint8_t> frame(1920 * 1080 * 4);
// ... fill frame data ...

texture.UploadAsync(frame.data());  // Returns immediately
// CPU can continue processing while GPU uploads

kernel.Dispatch(120, 68, true);
```

### Streaming Pattern

For continuous streaming (e.g., video playback), use `UploadAsyncStream` which blocks if no PBO is available:

```cpp
Texture2D<PixelFormat::RGBA8> videoFrame(1920, 1080);
videoFrame.InitUploadPBOPool(3);  // Triple buffering

Kernel2D processFrame([&](Int x, Int y) {
    auto frame = videoFrame.Bind();
    Float4 color = frame.Read(x, y);
    // Apply filter...
    frame.Write(x, y, filtered);
}, 16, 16);

// Stream loop
for (const auto& frameData : videoFrames) {
    // Upload blocks if all PBOs busy (waits up to 1000ms)
    videoFrame.UploadAsyncStream(frameData.data(), 1000);
    
    // Process while next frame uploads
    processFrame.Dispatch(120, 68, true);
}

// Wait for final upload
videoFrame.Sync();
```

### Async Download

Download GPU-computed results without blocking:

```cpp
Texture2D<PixelFormat::RGBA8> result(1024, 1024);
result.InitDownloadPBOPool(2);

// Render to texture
renderKernel.Dispatch(64, 64, true);

// Start async download (returns immediately)
result.DownloadAsync();

// Do other work while GPU prepares data...
otherKernel.Dispatch(32, 32, true);

// Try to get data
std::vector<uint8_t> pixels(1024 * 1024 * 4);
if (result.GetDownloadData(pixels.data())) {
    // Data ready
    SaveToFile(pixels);
} else {
    // Still pending, sync and retry
    result.Sync();
    result.GetDownloadData(pixels.data());
}
```

### Synchronization

| Method | Use When |
|:-------|:---------|
| `Sync()` | Need all operations complete before next step |
| `IsIdle()` | Polling to check if operations finished |
| Non-blocking | `UploadAsync` returns `false` if no PBO available |

```cpp
// Poll for completion
while (!texture.IsIdle()) {
    // Do other CPU work
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// Or block until complete
texture.Sync();
```

### Buffer Count Guidelines

| Count | Pattern | Use Case |
|:------|:--------|:---------|
| 1 | Single buffering | Simple async, may stall |
| 2 | Double buffering | Most common, good balance |
| 3+ | Triple buffering | High-latency tolerance, max throughput |

```cpp
// Double buffering: CPU fills one PBO while GPU uploads from another
texture.InitUploadPBOPool(2);

// Triple buffering: More tolerance for timing variations
texture.InitUploadPBOPool(3);
```

### Error Handling

```cpp
// UploadAsync returns false if no PBO available
if (!texture.UploadAsync(data)) {
    // Option 1: Sync and retry
    texture.Sync();
    texture.UploadAsync(data);  // Now succeeds
    
    // Option 2: Use streaming version (blocks until ready)
    // texture.UploadAsyncStream(data, timeoutMs);
}
```

---

## Error Handling

### Exception Hierarchy

All EasyGPU-specific exceptions derive from `GPU::Runtime::Exception`, which itself extends `std::runtime_error`.  The `what()` string is formatted at construction and cached, making it safe to call from `noexcept` contexts.

| Exception Class | Component | When Thrown |
|:----------------|:----------|:------------|
| `GPU::Runtime::Exception` | (varies) | Root class for all library errors |
| `GPU::Runtime::InternalIRException` | `IR` | Internal invariant violation during IR construction (indicates a library bug) |
| `GPU::Runtime::BuilderContextException` | `Builder` | IR node built without a valid Builder context (e.g. GPU variable outside a kernel) |
| `GPU::Runtime::ResourceExhaustionException` | `Resource` | Buffer, texture, or pipeline allocation failure (out-of-memory) |

Catch the root class to handle any library error uniformly:

```cpp
#include <Runtime/Exception.h>

try {
    kernel.Dispatch(100, true);
} catch (const GPU::Runtime::Exception &e) {
    std::cerr << "Component: " << e.Component() << "\n";
    std::cerr << "Message:   " << e.RawMessage() << "\n";
    std::cerr << "Full:      " << e.what() << std::endl;
}
```

### ShaderException

Thrown on GPU shader compilation and linking errors.

```cpp
try {
    kernel.Dispatch(100, true);
} catch (const ShaderException& e) {
    std::cerr << "Shader error: " << e.what() << std::endl;
}
```

### Common Errors

| Error | Solution |
|:------|:---------|
| `Buffer::Bind() called outside of Kernel` | Move Bind() inside kernel lambda |
| `No active OpenGL context` | Check OpenGL support and drivers |
| `GLSL compilation failed` | Use `GetCode()` to debug generated code |
| `[GPU::IR] Failed to create variable node` | Ensure GPU variables are constructed inside a kernel definition |

---

## Benchmark Suite

The benchmark framework provides a lightweight way to measure GPU kernel and operation performance using wall-clock time.  For GPU-side timer queries (GL_TIME_ELAPSED / Vulkan timestamps), use `Kernel::KernelProfiler` directly.

Include `<Benchmark/Benchmark.h>` or use the lazy header `<GPU.h>`.

### BenchmarkConfig

Controls iteration counts and synchronisation behaviour.

```cpp
struct BenchmarkConfig {
    int  warmupIterations   = 5;   // Dispatches executed but not measured
    int  measuredIterations = 20;  // Dispatches whose timing is recorded
    bool syncAfterEach      = true; // glFinish / vkQueueWaitIdle after each dispatch
};
```

### BenchmarkResult

Statistical summary computed from per-iteration timings.

```cpp
struct BenchmarkResult {
    std::string        name;
    int                warmupCount;
    int                measuredCount;
    double             minMs, maxMs, avgMs, medianMs, stddevMs, totalMs;
    std::vector<double> individualTimesMs;
};
```

### BenchmarkRunner

Ad-hoc runner for timing individual operations.

```cpp
GPU::Benchmark::BenchmarkRunner runner;

runner.RunAndRecord("vector_add", [&]() {
    kernel.Dispatch(64, true);
});

runner.RunAndRecord("vector_mul", [&]() {
    kernelMul.Dispatch(64, true);
}, BenchmarkConfig(10, 50));  // Custom config

runner.PrintResults();
std::string report = runner.GetFormattedResults(); // For logging
runner.Clear();
```

### BenchmarkSuite

Organised collection executed as a group.

```cpp
GPU::Benchmark::BenchmarkSuite suite("MySuite");

suite.Add("kernel_a", [&]() { kernelA.Dispatch(64, true); });
suite.Add("kernel_b", [&]() { kernelB.Dispatch(64, true); });
suite.Add("kernel_c", [&]() { kernelC.Dispatch(64, true); });

suite.Run(BenchmarkConfig(5, 50));  // 5 warmup, 50 measured
suite.PrintResults();

// Query individual results
const auto &results = suite.GetResults();
for (const auto &r : results) {
    std::cout << r.name << ": " << r.avgMs << " ms (median: " << r.medianMs << " ms)\n";
}
```

---

## OpenGL State Cache

EasyGPU uses an internal `GLStateCache` to minimize redundant OpenGL state changes. This is an implementation detail for most users, but understanding it helps optimize performance and debug state-related issues.

### Design Philosophy

The state cache operates in **exclusive mode** (default):
- Assumes EasyGPU is the sole controller of OpenGL state in the current context
- No defensive `glGet` calls to verify state - trusts the cache
- Minimal state changes between consecutive operations

### Cached State

The following OpenGL state is cached per context:

| State | Cache Behavior |
|:------|:---------------|
| Current shader program | Only `glUseProgram()` if program ID changed |
| SSBO bindings | Only `glBindBufferBase()` if buffer changed per binding point |
| Image texture bindings | Only `glBindImageTexture()` if texture changed |
| Sampler texture bindings | Only `glBindTexture()` if texture changed per unit |
| Active texture unit | Only `glActiveTexture()` if unit changed |
| VAO binding | Only `glBindVertexArray()` if VAO changed |

### Manual Invalidation

If you perform raw OpenGL operations that modify state, invalidate the cache:

```cpp
#include <Runtime/GLStateCache.h>

// Perform raw OpenGL operations
GPU::Runtime::GetStateCache().Invalidate();

// Now raw GL calls won't conflict with EasyGPU's cached state
glUseProgram(myRawProgram);
glDrawArrays(...);

// Re-invalidate before returning to EasyGPU
GPU::Runtime::GetStateCache().Invalidate();

// EasyGPU will now re-bind all state on next Dispatch()
kernel.Dispatch(16);
```

### Invalidate Guard (RAII)

For scoped invalidation:

```cpp
{
    GPU::Runtime::StateCacheInvalidateGuard guard;
    
    // Raw GL operations here
    glUseProgram(program);
    glDrawArrays(...);
    
} // Cache invalidated on destruction

// Next EasyGPU operation will re-bind state
kernel.Dispatch(16);
```

### Context Switching

The cache is automatically invalidated when the OpenGL context changes:

```cpp
// MakeCurrent automatically invalidates cache
GPU::Runtime::Context::GetInstance().MakeCurrent();

// State will be re-bound on next kernel operation
```

### Performance Impact

State caching provides significant performance benefits:

| Operation | Without Cache | With Cache | Improvement |
|:----------|:--------------|:-----------|:------------|
| Consecutive Dispatch() (same kernel) | ~50-100μs overhead | ~5-10μs overhead | **5-10x** |
| FragmentKernel Flush() | Full state setup | Minimal changes | **3-5x** |
| Multi-pass rendering | Re-bind everything | Bind only changes | **4-8x** |

**Best Practice:** Avoid interleaving raw OpenGL with EasyGPU operations. Group raw GL operations and wrap with `Invalidate()` or use a separate context.

---

### TextureSampler3D

Returned by `Texture3D::BindSampler()` for filtered 3D sampling in fragment kernels.

```cpp
Texture3D<PixelFormat::RGBA8> volume(64, 64, 64);

FragmentKernel2D kernel([&](Float2 fragCoord, Float2 resolution, Var<Vec4>& fragColor) {
    auto tex = volume.BindSampler();
    Float3 uvw = MakeFloat3(fragCoord.x() / resolution.x(), fragCoord.y() / resolution.y(), 0.5f);
    fragColor = tex.Sample(uvw);
});
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Sample(uvw)` | Sample 3D texture at normalized UVW coordinates |
| `GetSize()` | Get texture size as `Vec3` |
| `GetTextureWidth()` | Get width in pixels |
| `GetTextureHeight()` | Get height in pixels |
| `GetTextureDepth()` | Get depth in pixels |

---

## Shared Memory

Workgroup-local memory for fast thread cooperation.

### SharedMemory<T, N>

```cpp
template <ScalarType Type, int N>
class SharedMemory;
```

**Declaration:**
```cpp
SharedMemory<float, 256> shared;  // 256 floats per workgroup
```

**Element Access:**
```cpp
shared[index]           // index: Var<int>, Expr<int>, or int
shared.GetName()        // Returns GLSL variable name
shared.GetSize()        // Returns N (compile-time constant)
```

**Usage:**
```cpp
Kernel1D kernel([](Int i) {
    SharedMemory<float, 256> shared;
    Int localId = LocalThreadId();  // Clean API!
    
    // Write to shared memory
    shared[localId] = input[i];
    
    // Synchronize before reading
    Kernel1D::WorkgroupBarrier();
    
    // Read from other threads
    float neighbor = shared[(localId + 1) % 256];
});
```

**Generated GLSL:**
```glsl
shared float v1[256];
```

---

## Atomic Operations

Read-modify-write operations with guaranteed atomicity.

### Integer Atomics

```cpp
// Add - returns old value
[[nodiscard]] Expr<int> AtomicAdd(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicAdd(const Expr<int>& target, int value);

// Subtract (GLSL 4.6+)
[[nodiscard]] Expr<int> AtomicSub(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicSub(const Expr<int>& target, int value);

// Min/Max
[[nodiscard]] Expr<int> AtomicMin(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicMin(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicMax(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicMax(const Expr<int>& target, int value);

// Bitwise
[[nodiscard]] Expr<int> AtomicAnd(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicAnd(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicOr(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicOr(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicXor(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicXor(const Expr<int>& target, int value);

// Exchange and Compare-And-Swap
[[nodiscard]] Expr<int> AtomicExchange(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicExchange(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicCompSwap(const Expr<int>& target, const Expr<int>& compare, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicCompSwap(const Expr<int>& target, int compare, int value);
```

### Floating-Point Atomics

```cpp
// Note: float atomics have limited hardware support
[[nodiscard]] Expr<float> AtomicAdd(const Expr<float>& target, const Expr<float>& value);
[[nodiscard]] Expr<float> AtomicAdd(const Expr<float>& target, float value);
[[nodiscard]] Expr<float> AtomicMin(const Expr<float>& target, const Expr<float>& value);
[[nodiscard]] Expr<float> AtomicMin(const Expr<float>& target, float value);
[[nodiscard]] Expr<float> AtomicMax(const Expr<float>& target, const Expr<float>& value);
[[nodiscard]] Expr<float> AtomicMax(const Expr<float>& target, float value);
[[nodiscard]] Expr<float> AtomicExchange(const Expr<float>& target, const Expr<float>& value);
[[nodiscard]] Expr<float> AtomicExchange(const Expr<float>& target, float value);
```

**Usage:**
```cpp
Kernel1D histogram([](Int i) {
    auto data = input.Bind();
    auto hist = histogram.Bind();
    
    Int bin = data[i];
    ExprBase::NotUse(AtomicAdd(hist[bin], MakeInt(1)));
});
```

---

## Active Compaction

GPU utility for building dense active-index lists from sparse integer masks. Useful for sparse ray, particle, pixel, and cell workloads.

```cpp
#include <Utility/ActiveCompaction.h>

namespace GPU::Utility {

class ActiveCompaction {
public:
    explicit ActiveCompaction(size_t maxElements);

    Runtime::Buffer<int>& CountBuffer();
    Runtime::Buffer<int>& IndicesBuffer();
    const Runtime::Buffer<int>& CountBuffer() const;
    const Runtime::Buffer<int>& IndicesBuffer() const;

    size_t MaxElements() const;

    /// Compact activeMask[0..elementCount) into IndicesBuffer().
    /// activeMask[i] != 0 means element i is active.
    void Compact(Runtime::Buffer<int>& activeMask,
                 size_t elementCount,
                 bool sync = false);

    int DownloadCount();
    std::vector<int> DownloadIndices(size_t count);
};

} // namespace GPU::Utility
```

**Example:**

```cpp
Buffer<int> activeMask(N, BufferMode::Read);
GPU::Utility::ActiveCompaction compactor(N);

compactor.Compact(activeMask, N);

int activeCount = compactor.DownloadCount();
auto activeIndices = compactor.DownloadIndices(activeCount);
```

`Compact()` runs two small GPU passes: clear the counter, then append active indices with an atomic counter. Follow-up kernels can bind `IndicesBuffer()` and process only active elements. If a backend does not support indirect dispatch, dispatch a conservative upper bound and read `CountBuffer()` inside the shader, or occasionally download the count to choose a tighter CPU-side dispatch.

---

## Parallel Primitives

Built-in parallel algorithms using shared memory.

### WorkgroupReduce

Reduce values across all threads in a workgroup.

```cpp
// With custom operation
template <typename T, int N, typename Op>
[[nodiscard]] Expr<T> WorkgroupReduce(SharedMemory<T, N>& shared, const Expr<T>& value, Op op);

// Default: Add operation
template <typename T, int N>
[[nodiscard]] Expr<T> WorkgroupReduce(SharedMemory<T, N>& shared, const Expr<T>& value);
```

**Operations:**
```cpp
Parallel::AddOp   // a + b
Parallel::MulOp   // a * b
Parallel::MinOp   // min(a, b)
Parallel::MaxOp   // max(a, b)
```

**Example:**
```cpp
SharedMemory<float, 256> shared;

Kernel1D reduce([](Int i) {
    SharedMemory<float, 256> shared;
    
    // Each thread contributes a value
    Expr<float> myValue = input[i];
    
    // Get sum of all values in workgroup
    Expr<float> sum = WorkgroupReduce(shared, myValue);
    
    // Only thread 0 writes result
    Int localId = LocalThreadId();
    If(localId == 0, [&]() {
        output[WorkgroupId()] = sum;
    });
}, 256);
```

### WorkgroupScanInclusive

Inclusive prefix sum (scan).

```cpp
template <typename T, int N, typename Op>
[[nodiscard]] Var<T> WorkgroupScanInclusive(SharedMemory<T, N>& shared, const Expr<T>& value, Op op);

template <typename T, int N>
[[nodiscard]] Var<T> WorkgroupScanInclusive(SharedMemory<T, N>& shared, const Expr<T>& value);
```

**Example:**
```cpp
// Input:  [1, 2, 3, 4, 5]
// Output: [1, 3, 6, 10, 15]  (cumulative sum including self)

Kernel1D scan([](Int i) {
    SharedMemory<float, 256> shared;
    
    Var<float> scanned = WorkgroupScanInclusive(shared, input[i]);
    output[i] = scanned;
}, 256);
```

### WorkgroupScanExclusive

Exclusive prefix sum (scan).

```cpp
template <typename T, int N, typename Op>
[[nodiscard]] Var<T> WorkgroupScanExclusive(SharedMemory<T, N>& shared, const Expr<T>& value, T identity, Op op);

template <typename T, int N>
[[nodiscard]] Var<T> WorkgroupScanExclusive(SharedMemory<T, N>& shared, const Expr<T>& value, T identity = T{});
```

**Example:**
```cpp
// Input:  [1, 2, 3, 4, 5]
// Output: [0, 1, 3, 6, 10]  (cumulative sum of previous elements)
//                            first element = identity (0)

Kernel1D exclusiveScan([](Int i) {
    SharedMemory<float, 256> shared;
    
    Var<float> scanned = WorkgroupScanExclusive(shared, input[i], 0.0f);
    output[i] = scanned;
}, 256);
```

---

## Barriers

Synchronization primitives for workgroup coordination.

### WorkgroupBarrier

Synchronize all threads within a workgroup.

```cpp
class KernelBase {
public:
    // Wait for all threads in workgroup to reach this point
    static void WorkgroupBarrier();
    
    // Ensure memory writes are visible to subsequent operations
    static void MemoryBarrier();
    
    // Combined: memory + execution barrier
    static void FullBarrier();
};
```

**Usage:**
```cpp
Kernel1D kernel([](Int i) {
    SharedMemory<float, 256> shared;
    Int localId = LocalThreadId();
    
    // Write to shared memory
    shared[localId] = input[i];
    
    // Must barrier before reading other threads' data
    Kernel1D::WorkgroupBarrier();
    
    // Now safe to read
    float neighbor = shared[(localId + 1) % 256];
});
```

---

## Thread Index Utilities

Convenient access to GPU thread hierarchy information.

### Functions

```cpp
// Local Thread ID (within workgroup)
Var<int> LocalThreadId();      // 1D - same as LocalThreadIdX()
Var<int> LocalThreadIdX();     // X dimension
Var<int> LocalThreadIdY();     // Y dimension
Var<int> LocalThreadIdZ();     // Z dimension
auto LocalThreadId2D();        // Returns { x(), y() }

// Workgroup ID
Var<int> WorkgroupId();        // 1D - same as WorkgroupIdX()
Var<int> WorkgroupIdX();       // X dimension
Var<int> WorkgroupIdY();       // Y dimension
Var<int> WorkgroupIdZ();       // Z dimension
auto WorkgroupId2D();          // Returns { x(), y() }

// Global Thread ID
Var<int> GlobalThreadIdX();    // X dimension
Var<int> GlobalThreadIdY();    // Y dimension
Var<int> GlobalThreadIdZ();    // Z dimension
auto GlobalThreadId2D();       // Returns { x(), y() }

// Workgroup Size
Var<int> WorkgroupSize();      // 1D - same as WorkgroupSizeX()
Var<int> WorkgroupSizeX();     // X dimension
Var<int> WorkgroupSizeY();     // Y dimension
Var<int> WorkgroupSizeZ();     // Z dimension
```

### Usage Example

```cpp
Kernel1D kernel([](Int i) {
    SharedMemory<float, 256> shared;
    
    // Get local thread ID (0-255)
    Var<int> localId = LocalThreadId();
    
    // Write to shared memory
    shared[localId] = input[i];
    
    // Barrier
    Kernel1D::WorkgroupBarrier();
    
    // Read neighbor's value
    float neighbor = shared[(localId + 1) % 256];
});
```

### 2D Kernel Example

```cpp
Kernel2D kernel([](Int x, Int y) {
    SharedMemory<float, 16 * 16> shared;
    
    // Get 2D local thread ID
    auto localId = LocalThreadId2D();
    Int localX = localId.x();
    Int localY = localId.y();
    
    // Flatten to 1D index
    Int localIdx = localY * 16 + localX;
    shared[localIdx] = input[y * width + x];
    
    Kernel2D::WorkgroupBarrier();
    
    // Access neighbor
    Float neighbor = shared[(localY * 16 + (localX + 1)) % 256];
});
```

### Comparison: Old vs New API

| Purpose | Old (Hacky) | New (Clean) |
|:--------|:------------|:------------|
| Local ID | `Var<int>("(int(gl_LocalInvocationID.x))")` | `LocalThreadId()` |
| Workgroup ID | `Var<int>("(int(gl_WorkGroupID.x))")` | `WorkgroupId()` |
| Global ID | `Var<int>("(int(gl_GlobalInvocationID.x))")` | `GlobalThreadIdX()` |
| Workgroup Size | `Var<int>("(int(gl_WorkGroupSize.x))")` | `WorkgroupSize()` |

---

## See Also

- [Parallel Primitives Guide](parallel-primitives.md) - Detailed usage patterns and examples
- [Automatic Differentiation](autodiff.md) - Compute gradients of GPU kernels
- [Tutorial](tutorial.md) - Learn GPU programming basics
- [Common Patterns](patterns.md) - Solutions to common tasks
