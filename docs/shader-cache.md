# Shader Caching

EasyGPU uses separate cache layers for shader compilation and driver pipeline creation. They solve different startup costs and have different compatibility rules.

## Cache Layers

| Layer | Lifetime | Stored data | Purpose |
|:--|:--|:--|:--|
| Kernel context | Process | Live shader and pipeline handles | Reuse an already-created pipeline when the same kernel instance dispatches again |
| Global binary map | Process | Backend pipeline binary snapshots | Reuse driver data between equivalent kernel contexts in one process |
| Vulkan SPIR-V cache | Disk | Validated optimized SPIR-V modules | Skip glslang and SPIRV-Tools across processes |
| Vulkan pipeline cache | Disk | Device- and driver-specific pipeline data | Feed prior driver compilation results into compute and graphics pipeline creation |

Both persistent Vulkan layers are enabled by `EASYGPU_ENABLE_SHADER_CACHE`. The SPIR-V lookup occurs before GLSL parsing, so a hit avoids glslang and the selected SPIRV-Tools recipe. Pipeline creation still calls Vulkan, but a compatible driver cache can reduce the work performed inside `vkCreateComputePipelines` and `vkCreateGraphicsPipelines`.

## SPIR-V Cache Key

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

## SPIR-V Validation And Writes

Cached modules are bounded to 64 MiB and validated for Vulkan 1.1 before use. Truncated, malformed, or invalid files are deleted and treated as misses. Cache writes use a temporary file in the destination directory followed by an atomic rename. Concurrent processes compiling the same shader may duplicate compilation, but they cannot expose a partially written final cache entry.

Cache I/O is an optimization. An unavailable or read-only cache directory does not make shader compilation fail.

## Vulkan Pipeline Cache

EasyGPU loads one device-level `VkPipelineCache` during Vulkan initialization and supplies it to both compute and graphics pipeline creation. A pipeline cache file is selected using a SHA-256 key containing the cache schema, Vulkan vendor and device IDs, driver version, API version, and the full `pipelineCacheUUID`.

The standard `VkPipelineCacheHeaderVersionOne` is checked again before data reaches the driver. Its header size, version, vendor ID, device ID, and all `VK_UUID_SIZE` UUID bytes must match the selected physical device. Files are bounded to 256 MiB; truncated, oversized, or incompatible entries are deleted and treated as misses.

After successful pipeline creation, EasyGPU persists newly available driver data; backend shutdown performs a final dirty-cache flush. Writers take a cross-process file lock, merge any newer compatible disk cache with `vkMergePipelineCaches`, and atomically replace the final file. This avoids partial files and prevents concurrent processes from silently discarding one another's pipeline data. Cache failures never make pipeline creation or shutdown fail.

## Cache Location

The default persistent cache directories are:

| Platform | Cache root |
|:--|:--|
| Windows | `%LOCALAPPDATA%/EasyGPU/shader-cache` |
| macOS | `~/Library/Caches/EasyGPU` |
| Linux | `$XDG_CACHE_HOME/easygpu`, or `~/.cache/easygpu` |

Set a runtime directory for development, CI, or application-managed cleanup:

```bash
export EASYGPU_SHADER_CACHE_DIR=/path/to/cache
```

SPIR-V entries are stored under `<root>/spirv-v1`. Device pipeline data is stored under `<root>/vulkan-pipeline-v1`, with one hashed `.bin` name per compatible device and driver identity. A build-time root can also be configured:

```bash
cmake -S . -B build -DEASYGPU_SHADER_CACHE_DIR=/path/to/cache
```

The runtime environment variable takes precedence over the CMake default.

## Cache Statistics

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

const auto pipelineStats = backend->GetPipelineCacheStats();
std::cout << "Pipeline cache hits: " << pipelineStats.diskCacheHits << '\n';
std::cout << "Pipeline cache writes: " << pipelineStats.diskCacheWrites << '\n';
std::cout << "Pipeline cache loaded bytes: " << pipelineStats.loadedBytes << '\n';
std::cout << "Pipeline cache saved bytes: " << pipelineStats.savedBytes << '\n';
```

On a valid SPIR-V hit, `ShaderCompilationStats::lastDiskCacheHit` is true, both last-phase durations are zero, and `frontendCompilations` does not increase. Pipeline statistics are per backend lifetime; a valid initial driver blob increments `diskCacheHits`, while rejected headers increment `invalidDiskEntries` and `diskCacheMisses`.

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

`GlobalShaderCache::Clear()` does not delete either persistent Vulkan layer. Remove the configured cache root when explicit invalidation or size management is required.

## CMake Options

Persistent shader caching is enabled by default:

```bash
cmake -S . -B build -DEASYGPU_ENABLE_SHADER_CACHE=ON
```

Set it to `OFF` to disable persistent SPIR-V and Vulkan pipeline reads and writes. Live per-kernel pipeline reuse and Vulkan's process-local pipeline cache remain active.

## Limits

- There is no automatic disk-size eviction yet. Applications can point the cache at a managed directory and remove old schema directories.
- Persistent optimized SPIR-V currently applies to the Vulkan backend. OpenGL continues to use its process-local program and pipeline mechanisms.
- Pipeline cache data is opaque driver output. A validated load can accelerate pipeline creation, but the Vulkan implementation decides whether a particular pipeline is a real cache hit.
- Pipeline data is flushed after successful creation and again during normal backend shutdown when dirty. An abrupt termination during a write cannot expose a partial final cache file.
- Cache-hit improvements affect startup and compilation latency, not GPU execution time after a pipeline has been created.
