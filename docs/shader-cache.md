# Shader Caching

EasyGPU uses separate cache layers for shader compilation and driver pipeline creation. They solve different startup costs and have different compatibility rules.

## Cache Layers

| Layer | Lifetime | Stored data | Purpose |
|:--|:--|:--|:--|
| Kernel context | Process | Live shader and pipeline handles | Reuse an already-created pipeline when the same kernel instance dispatches again |
| Global pipeline cache | Process | Backend pipeline binary data | Feed reusable driver data back into pipeline creation |
| Vulkan SPIR-V cache | Disk | Validated optimized SPIR-V modules | Skip glslang and SPIRV-Tools across processes |

The persistent Vulkan cache is enabled by `EASYGPU_ENABLE_SHADER_CACHE`. A lookup occurs before GLSL parsing. A hit therefore avoids both glslang code generation and the selected SPIRV-Tools optimization recipe. Vulkan pipeline creation still runs because pipelines are device- and driver-specific.

## Persistent Cache Key

Each SPIR-V entry is content-addressed with SHA-256 over:

- the complete GLSL source bytes;
- shader stage;
- optimization level;
- `ShaderDesc::preserveInterface`;
- whether SPIRV-Tools optimization was compiled in;
- Vulkan and SPIR-V target versions;
- glslang version;
- SPIRV-Tools version and commit details;
- EasyGPU cache schema version.

Changing any input produces a different file name. Toolchain upgrades and optimizer recipe changes cannot silently reuse an incompatible module.

## Validation And Writes

Cached modules are bounded to 64 MiB and validated for Vulkan 1.1 before use. Truncated, malformed, or invalid files are deleted and treated as misses. Cache writes use a temporary file in the destination directory followed by an atomic rename. Concurrent processes compiling the same shader may duplicate compilation, but they cannot expose a partially written final cache entry.

Cache I/O is an optimization. An unavailable or read-only cache directory does not make shader compilation fail.

## Cache Location

The default persistent cache directories are:

| Platform | Directory |
|:--|:--|
| Windows | `%LOCALAPPDATA%/EasyGPU/shader-cache/spirv-v1` |
| macOS | `~/Library/Caches/EasyGPU/spirv-v1` |
| Linux | `$XDG_CACHE_HOME/easygpu/spirv-v1`, or `~/.cache/easygpu/spirv-v1` |

Set a runtime directory for development, CI, or application-managed cleanup:

```bash
export EASYGPU_SHADER_CACHE_DIR=/path/to/cache
```

The cache then uses `/path/to/cache/spirv-v1`. A build-time default can also be configured:

```bash
cmake -S . -B build -DEASYGPU_SHADER_CACHE_DIR=/path/to/cache
```

The runtime environment variable takes precedence over the CMake default.

## Compilation Statistics

Backends expose cache and compilation counters:

```cpp
auto *backend = Runtime::Context::GetBackend();
backend->ResetShaderCompilationStats();

// Compile or inspect a shader here.

const auto stats = backend->GetShaderCompilationStats();
std::cout << "SPIR-V cache hits: " << stats.diskCacheHits << '\n';
std::cout << "SPIR-V cache misses: " << stats.diskCacheMisses << '\n';
std::cout << "Frontend compilations: " << stats.frontendCompilations << '\n';
std::cout << "Last frontend time: " << stats.lastFrontendMilliseconds << " ms\n";
std::cout << "Last optimizer time: " << stats.lastOptimizationMilliseconds << " ms\n";
```

On a valid disk-cache hit, `lastDiskCacheHit` is true, both last-phase durations are zero, and `frontendCompilations` does not increase.

## In-Memory Pipeline Cache

`GPU::Kernel::GlobalShaderCache` remains the process-local store for backend pipeline binary data:

```cpp
#include <Kernel/ShaderCache.h>

using namespace GPU::Kernel;

GlobalShaderCache::Clear();
auto &cache = GlobalShaderCache::Get();
size_t entries = 0;
size_t bytes = 0;
cache.GetStats(entries, bytes);
```

`GlobalShaderCache::Clear()` does not delete persistent SPIR-V files. Remove the configured disk directory when explicit invalidation or size management is required.

## CMake Options

Persistent shader caching is enabled by default:

```bash
cmake -S . -B build -DEASYGPU_ENABLE_SHADER_CACHE=ON
```

Set it to `OFF` to disable persistent SPIR-V reads and writes. Live per-kernel pipeline reuse remains active.

## Limits

- There is no automatic disk-size eviction yet. Applications can point the cache at a managed directory and remove old schema directories.
- Persistent optimized SPIR-V currently applies to the Vulkan backend. OpenGL continues to use its process-local program and pipeline mechanisms.
- Vulkan pipeline data is still process-local in EasyGPU. It has stricter device and driver compatibility requirements than SPIR-V and must be persisted separately.
- Cache-hit improvements affect startup and compilation latency, not GPU execution time after a pipeline has been created.
