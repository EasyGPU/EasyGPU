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
    Ultra,
    Extreme,
    Performance = Aggressive
};
```

`Aggressive` is the default. SPIRV-Tools exposes two built-in general optimization recipes: `-O` and `-Os`. EasyGPU maps `Aggressive` to the strongest general performance recipe available from SPIRV-Tools, `RegisterPerformancePasses()` / `spirv-opt -O`. `Performance` is kept as a compatibility alias for `Aggressive`.

`Ultra` starts with SPIRV-Tools' maintained `RegisterPerformancePasses()` recipe instead of copying its current pass list. It then runs a conservative, target-independent tail: loop invariant code motion (LICM), strength reduction, local and global redundancy elimination, code sinking, simplification, preserve-aware dead-code elimination, CFG cleanup, and ID compaction. These passes preserve shader precision and keep risky loop restructuring out of the production-oriented preset.

`Extreme` is an explicitly experimental preset. It additionally enables loop unswitching, peeling, fission and fusion, relaxed-precision FP16 conversion, decoration flattening, and AMD extension-to-core replacement. These transformations can increase code size or register pressure, and FP16 conversion can change numerical results. Use it only with workload-specific correctness and performance measurements.

All optimizing presets honor `ShaderDesc::preserveInterface`. EasyGPU also validates the final optimized module for Vulkan 1.1 before creating a shader module; an invalid result fails compilation with a diagnostic instead of reaching the driver.

| Level | Vulkan behavior | OpenGL behavior |
|:--|:--|:--|
| `None` | Compile GLSL to SPIR-V and skip SPIRV-Tools opt passes | No-op |
| `Aggressive` | Default; run the SPIRV-Tools performance recipe (`-O`) | No-op |
| `Ultra` | Maintained `-O` recipe plus conservative scalar/loop/code-motion passes and final cleanup | No-op |
| `Extreme` | Ultra plus speculative loop restructuring and relaxed-precision conversion | No-op |
| `Performance` | Alias for `Aggressive` | No-op |
| `Size` | Run SPIRV-Tools size passes (`-Os`) | No-op |

### Ultra Additional Passes

| Pass | Purpose |
|:---|:---|
| **LICM** | Hoist loop-invariant instructions to a loop preheader |
| **StrengthReduction** | Replace eligible integer operations with equivalent cheaper operations, such as power-of-two multiplication with a shift |
| **LocalRedundancyElimination** | Remove repeated value computations within a basic block |
| **RedundancyElimination** | Re-run global value numbering after the added transformations |
| **CodeSinking** | Move instructions into more deeply nested constructs, closer to their uses |
| **Simplification + DCE + CFG cleanup** | Remove artifacts exposed by the added passes while respecting `preserveInterface` |
| **CompactIds** | Renumber result IDs to a compact range after all semantic transformations |

These are SPIR-V-level, target-independent transformations. A smaller module or fewer SPIR-V instructions does not by itself prove lower GPU execution time; the Vulkan driver still performs target-specific lowering and scheduling.

### Extreme-Only Passes

| Pass | Why only in Extreme | Benefit |
|:---|:---|:---|
| **LoopUnswitch / LoopPeeling** | Can duplicate loop bodies and grow the module | Expose loop-invariant branches and boundary iterations to later passes |
| **LoopFission / LoopFusion** | Can improve or worsen estimated live-value pressure | Restructure compatible loops using SPIRV-Tools' SSA-level estimate; the threshold is not a physical GPU register count |
| **ConvertRelaxedToHalf** | Can change numerical results | Convert eligible `RelaxedPrecision` operations and values to 16-bit types |
| **FlattenDecoration** | Primarily structural | Replace grouped decorations with equivalent direct decorations |
| **AmdExtToKhr** | Only relevant when AMD extension instructions are present | Replace supported AMD extension instructions with equivalent core/KHR forms |

## Kernel API

```cpp
Kernel1D kernel([&](Int i) {
    auto src = input.Bind();
    auto dst = output.Bind();
    Float x = src[i];
    dst[i] = x * x + 1.0f;
});

kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Ultra);
std::string optimized = kernel.GetOptimizedGLSL();
```

The same methods are available on `Kernel1D`, `Kernel2D`, `Kernel3D`, and the inspector kernels.

## ADKernel API

`ADKernel1D` uses the same Vulkan shader path. Identical `SetOptimizationLevel` / `GetOptimized*GLSL` methods.

## Build Options

```bash
cmake -S . -B build \
  -DEASYGPU_BACKEND=Vulkan \
  -DEASYGPU_ENABLE_SPIRV_OPT=ON \
  -DEASYGPU_ENABLE_SPIRV_CROSS_GLSL=ON
```

## Reproducible Demo

```bash
cmake --build build --target spirv_opt_inspection --parallel
./build/spirv_opt_inspection
```

Prints Mandelbrot kernel GLSL at all four optimization levels (Raw, Aggressive, Ultra, Extreme), along with line count, byte size, and median cold/warm host timings. Each cold sample forces a persistent-cache miss; each warm sample reads validated optimized SPIR-V from disk. These timings cover compilation and inspection, not GPU execution.

See [Shader Caching](shader-cache.md) for cache keys, storage locations, validation, and runtime statistics.
