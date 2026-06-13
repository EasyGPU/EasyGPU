# Graphics Pipeline

EasyGPU provides a complete cross-platform rasterization pipeline with a C++ embedded DSL. You write vertex and fragment shaders as C++ lambdas; the framework compiles them to GLSL/SPIR-V and executes them via Vulkan.

**Table of Contents**

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [GraphicsPipeline](#graphicspipeline)
  - [FragmentShader (Fullscreen Pass)](#fragmentshader-fullscreen-pass)
  - [Varying\<T\>](#varyingt)
  - [DepthBuffer](#depthbuffer)
- [Built-in Shader Variables](#built-in-shader-variables)
- [Uniforms and Push Constants](#uniforms-and-push-constants)
- [OBJ Model Rendering](#obj-model-rendering)
- [Textured Sponza Rendering](#textured-sponza-rendering)
- [Camera and Interaction](#camera-and-interaction)
- [Window Integration](#window-integration)
- [Backend Reference](#backend-reference)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```cpp
#include <GPU.h>
#include <Window/AppWindow.h>
#include <Window/TexturePresenter.h>

using namespace GPU;
using namespace GPU::Math;
using namespace GPU::Runtime;
using namespace GPU::Kernel;

int main() {
    const uint32_t W = 1024, H = 768;

    // Create resources
    Texture2D<PixelFormat::RGBA8> rt(W, H);
    DepthBuffer                   db(W, H);

    // Declare varying — data passed from vertex to fragment shader
    Varying<Vec3> vColor;

    // Build a graphics pipeline with vertex + fragment shader DSL
    GraphicsPipeline pipeline(
        // ── Vertex Shader ──
        [&](Float4 &gl_Position) {
            Int vid = VertexIndex();            // built-in: gl_VertexIndex

            // Fullscreen triangle without vertex buffer
            Float x = ToFloat((vid & 1) << 2) - 1.0f;
            Float y = ToFloat((vid & 2) << 1) - 1.0f;
            gl_Position = MakeFloat4(x, y, 0.0f, 1.0f);

            vColor = Float3(MakeFloat3(1.0f, 0.5f, 0.2f));   // write varying
        },
        // ── Fragment Shader ──
        [&](Float4 &fragColor) {
            Float3 c = vColor;                                 // read varying (interpolated)
            fragColor = MakeFloat4(c.x(), c.y(), c.z(), 1.0f);
        });

    // Render
    pipeline.Draw(rt, db, 3, true);   // 3 vertices = fullscreen triangle, sync

    // Present to window
    GPU::Window::AppWindow window({.width = W, .height = H, .title = "EasyGPU"});
    GPU::Window::TexturePresenter presenter(window);
    while (window.IsOpen()) {
        window.PollEvents();
        presenter.Present(rt);
    }
}
```

---

## Core Concepts

### GraphicsPipeline

`GraphicsPipeline` is the main class for rasterization. It follows the same conventions as `Kernel1D`/`Kernel2D`: lambda-based DSL construction, optional name, lazy compilation on first dispatch, and `GetShaderSource()` for debugging.

**Construction**

```cpp
// With vertex input (for custom geometry via SSBO)
GraphicsPipeline pipeline(vertexFunc, fragmentFunc);

// Named variant
GraphicsPipeline pipeline("MyPipeline", vertexFunc, fragmentFunc);
```

**Vertex Shader Signature**

```cpp
// Fullscreen triangle (no explicit vertex input — uses VertexIndex())
std::function<void(Float4 &gl_Position)>

// With SSBO vertex data
std::function<void(Float4 &gl_Position)>  // read from Bound buffer via VertexIndex()
```

**Fragment Shader Signature**

```cpp
std::function<void(Float4 &fragColor)>
```

**Rendering**

```cpp
// Without depth buffer
pipeline.Draw(RenderTarget, vertexCount, sync);

// With depth buffer
pipeline.Draw(RenderTarget, depthBuffer, vertexCount, sync);
```

**Debugging**

```cpp
std::string glsl = pipeline.GetShaderSource();  // Full VS + FS source
```

### FragmentShader (Fullscreen Pass)

`FragmentShader` is a simplified wrapper around `GraphicsPipeline` with a hardcoded fullscreen-triangle vertex shader. It is the easiest way to write a fullscreen post-processing pass.

```cpp
FragmentShader shader([](Float2 &fragCoord, Float4 &fragColor) {
    fragColor = MakeFloat4(1.0f, 0.0f, 0.0f, 1.0f);   // solid red
}, width, height);

shader.Render(renderTarget, sync);
```

### Varying\<T\>

A `Varying<T>` is declared **outside** both vertex and fragment shader lambdas and captured by reference. In the vertex shader you **write** to it; in the fragment shader you **read** the rasterizer-interpolated value.

```cpp
Varying<Vec3> vWorldPos;
Varying<Vec2> vUV;
Varying<Vec3> vNormal;

GraphicsPipeline pipeline(
    // VS: write varyings
    [&](Float4 &gl_Position) {
        // ... compute worldPos, uv, normal ...
        vWorldPos = Float3(MakeFloat3(wx, wy, wz));
        vUV       = Float2(MakeFloat2(u, v));
        vNormal   = vt.normal();
    },
    // FS: read varyings (automatically interpolated)
    [&](Float4 &fragColor) {
        Float3 N  = Float3(Normalize(vNormal));
        Float2 uv = vUV;
        // ...
    });
```

**Supported Types**

`Varying<T>` accepts any `ScalarType` — `float`, `int`, `Vec2`, `Vec3`, `Vec4`, `IVec2`, `IVec3`, `IVec4`, `Mat3`, `Mat4`, and registered struct types.

### DepthBuffer

RAII depth buffer for occlusion testing. Create it once, pass it to every `Draw` call that needs depth testing.

```cpp
DepthBuffer db(1024, 768);
pipeline.Draw(rt, db, vertexCount, true);
```

Depth testing uses `VK_COMPARE_OP_LESS` with a clear value of `1.0f`. The Vulkan depth attachment format is `VK_FORMAT_D32_SFLOAT` for improved precision across large scenes.

---

## Built-in Shader Variables

EasyGPU provides free functions in `GPU::Kernel` for common GLSL built-in variables:

| Function | GLSL Equivalent | Available In |
|:---------|:----------------|:-------------|
| `VertexIndex()` | `gl_VertexIndex` | Vertex Shader |
| `FragmentCoord()` | `gl_FragCoord` | Fragment Shader |

```cpp
Int    vid = VertexIndex();     // current vertex index
Float4 fc  = FragmentCoord();   // pixel coordinate (x, y, z, 1/w)
```

These are available inside any `GraphicsPipeline` or `FragmentShader` lambda.

---

## Uniforms and Push Constants

The `Uniform<T>` class provides push-constant data to shaders. On Vulkan, uniforms are packed into a single push-constant block (≤ 128 bytes on most GPUs).

```cpp
EASYGPU_STRUCT(SceneUniforms, (Mat4, mvp), (Vec3, lightDir), (float, time));
Uniform<SceneUniforms> ubo;

// Set on CPU
SceneUniforms data;
data.mvp      = proj * view * model;
data.lightDir = Vec3(0.4f, 0.6f, 0.7f);
data.time     = 1.5f;
ubo = data;

// Read in vertex/fragment shader
auto u = ubo.Load();
gl_Position = u.mvp() * MakeFloat4(pos, 1.0f);
Float t     = u.time();
```

**Push Constant Size Limits**

Vulkan guarantees at least 128 bytes of push constant space. Keep your uniform struct ≤ 128 bytes. Prefer `Mat4` (64 bytes) and `Vec4` (16 bytes) over `Vec3` (12 bytes + 4 padding in std430).

---

## OBJ Model Rendering

The library's OBJ example demonstrates a complete model viewer with:

- **SSBO vertex storage:** vertices are flattened (non-indexed) and stored in a `Buffer<GpuVertex>` SSBO, read by `gl_VertexIndex` in the vertex shader
- **Gouraud shading:** per-vertex lighting computed in the vertex shader, interpolated via `Varying<Vec3>`
- **Depth testing:** correct occlusion via `DepthBuffer`
- **Free camera:** WASD movement + mouse look

```cpp
EASYGPU_STRUCT(GpuVertex, (Vec3, pos), (Vec3, normal));
EASYGPU_STRUCT(SceneUBO,   (Mat4, mvp));

// Load OBJ, flatten vertices
ObjMesh mesh;
mesh.Load("sponza.obj");
std::vector<GpuVertex> verts;
mesh.Flatten(verts);

Buffer<GpuVertex> vb(verts);
Uniform<SceneUBO> ubo;
Varying<Vec3> vColor;

GraphicsPipeline pipeline(
    [&](Float4 &gl_Position) {
        Int  vid  = VertexIndex();
        auto vert = vb.Bind()[vid];
        auto u    = ubo.Load();
        gl_Position = u.mvp() * MakeFloat4(vert.pos(), 1.0f);

        Float3 N(Normalize(vert.normal()));
        Float  diff = Max(Dot(N, MakeFloat3(0.4f, 0.6f, 0.7f)), 0.15f);
        vColor = Float3(MakeFloat3(diff * 0.9f, diff * 0.5f, diff * 0.3f));
    },
    [&](Float4 &fragColor) {
        Float3 c = vColor;
        fragColor = MakeFloat4(c.x(), c.y(), c.z(), 1.0f);
    });

// Render loop
while (window.IsOpen()) {
    UpdateCamera();
    ubo = { proj * view };
    pipeline.Draw(rt, db, vertCount, true);
    presenter.Present(rt);
}
```

---

## Textured Sponza Rendering

The `examples/sponza_renderer` example demonstrates a multi-material scene using a texture atlas:

- Loads OBJ geometry, MTL materials, and diffuse textures
- Packs diffuse textures into a single atlas with edge-repeated gutters
- Preserves unwrapped UVs through rasterization
- Uses a generated mip chain for stable minification
- Uses `Ddx()` / `Ddy()` with `SampleGrad()` to avoid incorrect atlas LOD selection at `Fract()` boundaries
- Uses a `D32_SFLOAT` depth buffer and a tightened projection range for stable depth testing

```cpp
Texture2D<PixelFormat::RGBA8> atlas(
    ATLAS_SIZE,
    ATLAS_SIZE,
    MipmapMode::Generate
);

// Fragment shader
Float2 tiled = Fract(sourceUV);
Float2 dx = Ddx(sourceUV) * atlasScale;
Float2 dy = Ddy(sourceUV) * atlasScale;
Float4 color = atlas.BindSampler().SampleGrad(atlasUV, dx, dy);
```

The Sponza model and texture files are external assets and are not included in the EasyGPU repository. Pass the local asset directory when launching the example:

```bash
./build/sponza_renderer /path/to/Sponza
```

See [Mipmaps](mipmaps.md) for the complete mipmap and explicit-gradient sampling API.

---

## Camera and Interaction

The OBJ example includes a first-person camera with mouse look and WASD movement:

```
WASD    — Move forward / back / left / right
Mouse   — Look around (yaw + pitch)
ESC     — Exit
```

**Camera Implementation Sketch**

```cpp
float camYaw = 0.0f, camPitch = 0.0f;
Vec3  camPos(0.0f, 0.3f, 0.0f);

// Mouse look
auto [mx, my] = window.MousePosition();
camYaw   -= (mx - lastMX) * sensitivity;
camPitch -= (my - lastMY) * sensitivity;

// Movement
Vec3 fwd(sin(camYaw), 0, -cos(camYaw));
if (window.IsKeyDown(Key::W)) camPos += fwd * speed;

// Build view matrix
Mat4 view = CameraView(camPos, camYaw, camPitch);
```

---

## Window Integration

Graphics pipeline output integrates seamlessly with the existing `AppWindow` / `TexturePresenter` API:

```cpp
Texture2D<PixelFormat::RGBA8> rt(W, H);
DepthBuffer                   db(W, H);

AppWindow window({.width = W, .height = H, .title = "My App"});
TexturePresenter presenter(window);

while (window.IsOpen()) {
    window.PollEvents();

    // Handle input, update uniforms...

    pipeline.Draw(rt, db, vertCount, true);   // render to texture
    presenter.Present(rt);                     // display in window
}
```

The `TexturePresenter` downloads the GPU texture to a CPU-side `PixelBuffer` and uploads it to the window framebuffer via MiniFB.

---

## Backend Reference

The low-level backend API is documented in [`api-reference.md`](api-reference.md#graphics-pipeline). Key backend types:

| Type | Purpose |
|:-----|:--------|
| `GraphicsPipelineDesc` | Pipeline creation descriptor (VS + FS + topology + depth + vertex layout) |
| `RenderPassBeginDesc` | Render pass descriptor (color + depth attachments, clear values) |
| `VertexLayoutEntry` | Per-attribute vertex format (location, format, offset) |
| `PrimitiveTopology` | `TriangleList`, `LineList`, `PointList`, etc. |

**Backend Methods**

| Method | Description |
|:-------|:------------|
| `CreateGraphicsPipeline(desc)` | Create a graphics pipeline |
| `BeginRendering(desc)` / `EndRendering()` | Dynamic render pass |
| `BindVertexBuffer(handle, stride)` / `BindIndexBuffer(handle)` | Bind vertex/index data |
| `Draw(vc, ic, fv, fi)` / `DrawIndexed(ic, ic, fi, vo, fi)` | Issue draw calls |
| `SetViewport(x, y, w, h)` / `SetScissor(x, y, w, h)` | Viewport and scissor |
| `CreateDepthBuffer(w, h)` / `DestroyDepthBuffer(h)` | Depth buffer management |

---

## Troubleshooting

**Nothing renders (black screen)**

1. **SSBO out-of-bounds:** With non-indexed draws, every vertex invokes the VS with `gl_VertexIndex` from `0` to `vertexCount - 1`. If your SSBO has fewer entries than `vertexCount`, the shader reads zeroes. Flatten your vertex data: one entry per drawn vertex.

2. **Push constant overflow:** The push constant block must fit within `maxPushConstantsSize` (≥ 128 bytes guaranteed). Use `Mat4` (64 bytes) and `Vec4` (16 bytes) instead of `Vec3` to avoid wasted padding.

3. **Projection matrix:** EasyGPU uses Vulkan NDC conventions — `z` in [0, 1], `y` increasing upward in NDC (the viewport flips to screen space). The perspective division trigger (`-1`) must be at `m[3][2]` (column 3, row 2), not elsewhere. Use `PerspVk` from the OBJ example as a reference.

**Depth testing doesn't work (transparent-looking geometry)**

Ensure the pipeline was created with depth enabled. The `GraphicsPipeline` DSL class enables depth testing by default. If using the raw backend API, set `GraphicsPipelineDesc::depthTestEnable = true` and `depthWriteEnable = true`.

**"Pipeline push constant size exceeds device limit"**

Your `Uniform<T>` struct is too large. Maximum push constant size varies by GPU (128 to 256 bytes). Reduce the struct size or switch to a UBO (Uniform Buffer Object) via `UniformBuffer<T>`.
