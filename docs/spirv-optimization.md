# SPIR-V Optimization

EasyGPU's SPIR-V optimization controls are **Vulkan-backend only**. They affect Vulkan shader module creation and Vulkan-only optimized GLSL inspection.

OpenGL builds accept the same public APIs for source compatibility, but handle them silently:

- `SetOptimizationLevel(...)` is a no-op for OpenGL.
- `GetOptimizationLevel()` still returns the stored EasyGPU preference.
- `GetOptimizedGLSL()`, `GetOptimizedForwardGLSL()`, and `GetOptimizedCombinedGLSL()` return an empty string on OpenGL.
- OpenGL shader compilation and execution continue through the normal GLSL path.

## Optimization Levels

```cpp
enum class ShaderOptimizationLevel {
    None,
    Size,
    Aggressive,
    Performance = Aggressive
};
```

`Aggressive` is the default. SPIRV-Tools exposes two built-in general optimization recipes: `-O` and `-Os`. EasyGPU maps `Aggressive` to the strongest general performance recipe available from SPIRV-Tools, `RegisterPerformancePasses()` / `spirv-opt -O`. `Performance` is kept as a compatibility alias for `Aggressive`.

| Level | Vulkan behavior | OpenGL behavior |
|:--|:--|:--|
| `None` | Compile GLSL to SPIR-V and skip SPIRV-Tools opt passes | No-op |
| `Aggressive` | Default; run SPIRV-Tools performance passes (`-O`) | No-op |
| `Performance` | Alias for `Aggressive` | No-op |
| `Size` | Run SPIRV-Tools size passes (`-Os`) | No-op |

## Kernel API

```cpp
Kernel1D kernel([&](Int i) {
    auto src = input.Bind();
    auto dst = output.Bind();
    Float x = src[i];
    dst[i] = x * x + 1.0f;
});

kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Aggressive);
std::string optimized = kernel.GetOptimizedGLSL();
```

The same methods are available on `Kernel1D`, `Kernel2D`, `Kernel3D`, and the inspector kernels.

`GetOptimizedGLSL()` returns SPIRV-Cross decompiled GLSL after Vulkan's GLSL to SPIR-V compilation and SPIRV-Tools optimization. It is an inspection API, not the source passed into OpenGL.

## ADKernel API

`ADKernel1D` uses the same Vulkan shader path for its forward kernel, internal clear kernel, and combined forward+backward kernel.

```cpp
ADKernel1D trainStep([&](Int i) {
    auto xs = samples.Bind();
    auto ws = weights.Bind();

    Float x = xs[i];
    Float w = ws[i];
    Float y = x * w;
    Float loss = y * y;

    AD::Param(w);
    AD::Loss(loss);
}, sampleCount);

trainStep.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Aggressive);

std::string optimizedForward  = trainStep.GetOptimizedForwardGLSL();
std::string optimizedBackward = trainStep.GetOptimizedCombinedGLSL();
```

On Vulkan these strings contain the optimized, SPIRV-Cross decompiled GLSL. On OpenGL they are empty strings.

## Build Options

SPIR-V optimization and GLSL inspection are enabled by default for Vulkan builds:

```bash
cmake -S . -B build \
  -DEASYGPU_BACKEND=Vulkan \
  -DEASYGPU_ENABLE_SPIRV_OPT=ON \
  -DEASYGPU_ENABLE_SPIRV_CROSS_GLSL=ON
```

`EASYGPU_ENABLE_SPIRV_OPT=OFF` keeps the public API but disables SPIRV-Tools optimization at build time. `EASYGPU_ENABLE_SPIRV_CROSS_GLSL=OFF` disables optimized GLSL inspection.

## Reproducible Demo

The repository includes a runnable showcase:

```bash
cmake --build build --target spirv_opt_inspection --parallel
./build/spirv_opt_inspection
```

It prints the original EasyGPU-generated GLSL from `GetCode()` and the Vulkan optimized GLSL from `GetOptimizedGLSL()`, using a tone-mapping kernel with constant branches, duplicate luminance work, and dead arithmetic so the optimizer effect is visible.
