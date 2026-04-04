# Shader Binary Cache

Runtime caching of compiled GPU programs for faster kernel execution.

## Overview

EasyGPU automatically caches compiled shader binaries in memory to avoid repeated GLSL compilation overhead within a single application session. This provides significant performance improvements when the same kernels are dispatched multiple times.

## How It Works

```
First Dispatch          Subsequent Dispatches
     │                         │
     ▼                         ▼
┌─────────┐              ┌──────────┐
│ Compile │              │  Lookup  │
│  GLSL   │              │   Cache  │
└────┬────┘              └────┬─────┘
     │                         │
     ▼                         ▼
┌─────────┐              ┌──────────┐
│  Store  │              │ Reuse    │
│  Cache  │              │ Binary   │
└─────────┘              └──────────┘
```

## Features

- **Automatic caching** — No code changes required; kernels are cached automatically
- **In-memory only** — No disk writes, cache exists only for the application lifetime
- **Cross-backend** — Works with both OpenGL and Vulkan backends
- **Thread-safe** — Safe to use from multiple threads
- **Zero overhead** — Cache lookup is negligible compared to shader compilation

## Usage

No explicit action is required. The cache operates transparently:

```cpp
#include <GPU.h>

int main() {
    Buffer<float> data(1024);
    
    // First call: compiles GLSL, executes, caches binary
    Kernel1D kernel([](Int i) {
        auto buf = data.Bind();
        buf[i] = buf[i] * 2.0f;
    });
    kernel.Dispatch(4, true);  // ~15ms (includes compilation)
    
    // Same kernel instance: uses cached binary
    kernel.Dispatch(4, true);  // ~0.5ms (reuses cached program)
    
    return 0;
}
```

## Manual Cache Control

For advanced use cases, you can access the global cache directly:

```cpp
#include <Kernel/ShaderCache.h>

using namespace GPU::Kernel;

// Clear all cached binaries
GlobalShaderCache::Clear();

// Check if cache is active
if (GlobalShaderCache::IsEnabled()) {
    // ...
}

// Get cache statistics
ShaderCache& cache = GlobalShaderCache::Get();
size_t entries, bytes;
cache.GetStats(entries, bytes);
std::cout << "Cached shaders: " << entries 
          << " (" << bytes << " bytes)" << std::endl;
```

## Backend Support

| Backend | Cache Support | Cache Method |
|:--------|:-------------:|:-------------|
| **OpenGL** | ✅ Yes | `glGetProgramBinary` / `glProgramBinary` |
| **Vulkan** | ✅ Yes | `VkPipelineCache` |

### Checking Backend Capabilities

```cpp
auto* backend = Runtime::Context::GetBackend();
if (backend->SupportsPipelineCache()) {
    uint32_t format = backend->GetPipelineCacheFormat();
    std::cout << "Cache format: 0x" << std::hex << format << std::endl;
}
```

## Implementation Details

### Cache Key

Cache entries are keyed by:
- **SHA256 hash** of the complete GLSL source code
- **Backend type** (OpenGL/Vulkan)
- **Driver-specific format identifier**

### Cache Storage

The cache stores:
- Compiled GPU binary (driver-specific format)
- Format identifier (for validation)
- Timestamp (for potential LRU eviction)

### Lifetime

- **Created**: When a kernel is first compiled
- **Reused**: On subsequent dispatches of the same kernel
- **Destroyed**: When the application exits

## Performance

Typical performance improvements on warm cache:

| Operation | Cold | Warm | Speedup |
|:----------|:----:|:----:|:-------:|
| Simple kernel | ~15ms | ~0.5ms | **30x** |
| Complex kernel | ~50ms | ~1ms | **50x** |

> **Note:** Performance varies by GPU driver and kernel complexity.

## CMake Options

Shader cache is enabled by default. To disable:

```cmake
set(EASYGPU_ENABLE_SHADER_CACHE OFF CACHE BOOL "" FORCE)
```

## Limitations

- Cache is **not persisted** to disk — restarting the application clears the cache
- Each kernel context maintains its own cache entry
- Maximum cache size is limited only by available system memory
