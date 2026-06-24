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

`Ultra` builds on the complete `-O` 44-pass pipeline and augments it with 8 GPU-specific passes: loop invariant code motion (LICM), loop unswitching, loop peeling, strength reduction, local redundancy elimination, code sinking, loop fission, plus an extended cleanup phase (constant unification, dead constant elimination, duplicate removal, dead variable elimination, unused interface variable removal, and capability trimming).

`Extreme` layers speculative optimizations on top of Ultra: FP16 relaxed precision conversion (2× ALU throughput on modern GPUs), loop fusion, decoration flattening, AMD extension-to-core replacement, and canonical ID assignment. Extreme uses `CanonicalizeIds` instead of `CompactIds` — individual shader binaries may be slightly larger, but ID ranges are structured to improve dictionary-based compression when multiple shader modules coexist in the same pipeline cache. Extreme is intended for benchmarking and production release builds on known hardware.

| Level | Vulkan behavior | OpenGL behavior |
|:--|:--|:--|
| `None` | Compile GLSL to SPIR-V and skip SPIRV-Tools opt passes | No-op |
| `Aggressive` | Default; run SPIRV-Tools performance passes (`-O`, ~44 passes) | No-op |
| `Ultra` | `-O` + GPU-specific passes (LICM, LoopUnswitch, LoopPeeling, StrengthReduction, CodeSinking, LoopFission + cleanup, ~55 passes) | No-op |
| `Extreme` | Ultra + FP16 conversion, LoopFusion, FlattenDecoration, AmdExtToKhr, CanonicalizeIds (~60 passes) | No-op |
| `Performance` | Alias for `Aggressive` | No-op |
| `Size` | Run SPIRV-Tools size passes (`-Os`) | No-op |

### Ultra GPU-Specific Passes

| New Pass | Why not in `-O` | GPU Compute Benefit |
|:---|:---|:---|
| **LICM** (`loop-invariant-code-motion`) | Compile-time cost | Hoists uniform/buffer loads out of inner loops — every compute shader benefits |
| **LoopUnswitch** | Compile-time cost | Moves invariant branch conditions outside loops → eliminates warp-divergent branching |
| **LoopPeeling** | Added complexity | Peels boundary iterations to expose constant patterns → enables better unrolling |
| **StrengthReduction** | Minimal CPU gain | `* 1024` → `<< 10` — measurable at shader-wide scale |
| **LocalRedundancyElimination** | GVN covers cross-block | Per-block CSE is O(1) cheap; catches intra-block duplicates GVN misses |
| **CodeSinking** | Not in recipe | Moves computations closer to use → shorter live ranges → fewer registers → higher occupancy |
| **LoopFission** (threshold=64) | Niche use case | Splits high-register-pressure loops to avoid costly spills to memory |
| **UnifyConst + EliminateDeadConst** | Not in recipe | Deduplicates and removes dead constants left after aggressive optimization |
| **RemoveDuplicates** | Not in recipe | Deduplicates types, decorations, and capabilities |
| **TrimCapabilities** | Not in recipe | Removes unused capabilities/extensions → enables more aggressive driver optimization |

### Extreme-Only Passes

| Pass | Why only in Extreme | Benefit |
|:---|:---|:---|
| **ConvertRelaxedToHalf** | Not safe for all workloads | Converts RelaxedPrecision FP32→FP16 — 2× ALU throughput on NVIDIA Pascal+, AMD Vega+, Apple M-series |
| **LoopFusion** | Can increase register pressure | Merges adjacent loops with same bounds → reduces dispatch overhead |
| **FlattenDecoration** | Minor gain | Simplifies OpDecorationGroup → individual OpDecorate for cleaner subsequent passes |
| **AmdExtToKhr** | AMD-specific | Replaces AMD vendor extensions with core SPIR-V instructions |
| **CanonicalizeIds** | Cross-shader only | Renumbers IDs for better compression when multiple shader modules coexist |

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

Prints Mandelbrot kernel GLSL at all four optimization levels (Raw, Aggressive, Ultra, Extreme) with line/size stats.
