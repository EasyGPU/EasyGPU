# Frequently Asked Questions

## General

### What is EasyGPU?

EasyGPU is an embedded domain-specific language (eDSL) that lets you write GPU compute kernels using standard C++ syntax. It compiles to GLSL compute shaders and executes via OpenGL.

### Why not just use CUDA or Vulkan?

| | EasyGPU | CUDA | Vulkan Compute |
|:--|:--|:--|:--|
| Setup | Minutes | Hours | Days |
| Dependencies | ~500KB GLAD | 2GB+ SDK | Complex setup |
| Code verbosity | Low | Medium | High |
| Cross-platform | Yes (OpenGL) | NVIDIA only | Yes |
| Learning curve | Low | Medium | High |

EasyGPU trades some performance for extreme ease of use and minimal setup.

### Is it production-ready?

EasyGPU is suitable for:
- Learning GPU programming
- Prototyping algorithms
- Small to medium compute workloads
- Educational purposes

For maximum performance in production, consider CUDA, ROCm, or optimized Vulkan.

---

## Installation & Build

### CMake can't find EasyGPU

```cmake
# Make sure to use FetchContent correctly
include(FetchContent)
FetchContent_Declare(
    easygpu
    GIT_REPOSITORY https://github.com/easygpu/EasyGPU.git
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(easygpu)
target_link_libraries(your_target EasyGPU)
```

### OpenGL context creation fails

EasyGPU auto-initializes the OpenGL context on first GPU operation. If this fails:

1. Check OpenGL 4.3+ support: `glxinfo | grep "OpenGL version"`
2. Update GPU drivers
3. On headless servers, use EGL or OSMesa for off-screen rendering

### Linker errors (undefined references)

Make sure to link OpenGL:

```cmake
target_link_libraries(your_target EasyGPU OpenGL::GL)
```

On Linux: `target_link_libraries(your_target EasyGPU GL)`

---

## Usage

### "Buffer::Bind() called outside of Kernel"

**Problem:**
```cpp
auto ref = buffer.Bind();  // ERROR: Outside kernel

Kernel1D kernel([](Int i) {
    data[i] = ref[i];  // Using pre-bound ref
});
```

**Solution:**
```cpp
Kernel1D kernel([](Int i) {
    auto ref = buffer.Bind();  // Bind inside kernel
    data[i] = ref[i];
});
```

### Kernels run but produce wrong results

Common causes:

1. **Work group size mismatch:**
```cpp
// Default work size is 256 for 1D kernels
// Ensure your dispatch covers all elements
kernel.Dispatch((N + 255) / 256, true);  // Ceiling division
```

2. **Buffer not bound:**
```cpp
// Must call Bind() inside kernel
Kernel1D kernel([](Int i) {
    auto buf = buffer.Bind();  // Don't forget this!
    buf[i] = value;
});
```

3. **Missing sync:**
```cpp
kernel.Dispatch(groups, true);  // true = wait for completion
// Without sync, download may happen before kernel finishes
```

### How do I pass constants to kernels?

**Method 1: Capture by value (recommended for static constants)**
```cpp
float constant_value = 3.14f;

Kernel1D kernel([&, constant_value](Int i) {  // Capture by value
    data[i] = data[i] * constant_value;
});
```

**Method 2: Use MakeFloat/MakeInt**
```cpp
Kernel1D kernel([](Int i) {
    Float pi = MakeFloat(3.14159f);
    data[i] = data[i] * pi;
});
```

**Method 3: Uniforms (for dynamic values that change between dispatches)**
```cpp
Uniform<float> scale;
scale = 2.0f;

Kernel1D kernel([&](Int i) {
    auto s = scale.Load();  // Load uniform value
    data[i] = data[i] * s;
});

kernel.Dispatch(groups, true);  // Uses scale = 2.0
scale = 3.0f;                    // Change value
kernel.Dispatch(groups, true);  // Uses scale = 3.0, no recompilation
```

**Which to use:**
- **Capture by value**: Value is constant for the kernel's lifetime
- **Uniforms**: Value changes between dispatches, avoids recompilation

### Can I use std::vector inside kernels?

No. Kernels execute on the GPU and can only use GPU types:
- `Buffer<T>` for global GPU data
- `VarArray<Type, N>` for small local arrays
- `Var<T>`, `Expr<T>` for values
- `Callable` for functions

Use `VarArray<float, N>` for small fixed-size arrays within kernels:
```cpp
Kernel1D kernel([](Int i) {
    VarArray<float, 10> arr;  // Local array of 10 floats
    For(0, 10, [&](Int& j) {
        arr[j] = MakeFloat(j);
    });
});
```

### What's the difference between Make and To functions?

**`Make*` functions** wrap C++ literals into GPU `Var` types. They do **NOT** perform type conversion.

```cpp
// Correct: wrapping literals
Float f = MakeFloat(3.14f);   // Wrap float literal
Int i = MakeInt(42);          // Wrap int literal
Float3 v = MakeFloat3(1, 2, 3); // Wrap 3 floats

// Wrong: type mismatch
Float f = MakeFloat(42);      // ERROR: 42 is int literal
// Fix: MakeFloat(42.0f) or ToFloat(MakeInt(42))
```

**`To*` functions** convert between `Var` types. They **DO** perform type conversion.

```cpp
Int i = MakeInt(42);
Float f = ToFloat(i);         // Convert Var<int> to Var<float>

Float pi = MakeFloat(3.14f);
Int approx = ToInt(pi);       // Convert to int, truncates to 3
```

**Summary:**

| API | Purpose | Example |
|:----|:--------|:--------|
| `MakeFloat(3.14f)` | Wrap float literal into Var<float> | `Float f = MakeFloat(3.14f);` |
| `ToFloat(var)` | Convert Var<int> to Var<float> | `Float f = ToFloat(int_var);` |
| `MakeInt(5)` | Wrap int literal into Var<int> | `Int i = MakeInt(5);` |
| `ToInt(var)` | Convert Var<float> to Var<int> | `Int i = ToInt(float_var);` |

---

## Performance

### Why is my kernel slower than expected?

1. **Dispatch overhead:** Small kernels have fixed overhead. Process more elements per thread:
```cpp
// Instead of 1:1 thread:element mapping
constexpr int ELEMENTS_PER_THREAD = 16;

Kernel1D kernel([](Int i) {
    int start = i * ELEMENTS_PER_THREAD;
    for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
        data[start + j] = Process(data[start + j]);
    }
});
```

2. **Memory access patterns:** Ensure coalesced memory access:
```cpp
// Good: Sequential access
for (int i = tid; i < N; i += stride) {
    data[i] = ...;
}

// Bad: Strided access
for (int i = tid * 100; i < N; i += stride * 100) {
    data[i] = ...;
}
```

3. **Divergence:** Minimize branching within warps:
```cpp
// Bad: Different threads take different branches
If(thread_id % 2 == 0, [&]() { ... }).Else([&]() { ... });

// Better: Process in separate passes
```

### How do I profile kernels?

```cpp
#include <Kernel/KernelProfiler.h>

Kernel1D kernel([](Int i) { ... });
kernel.SetName("MyKernel");

kernel.Dispatch(100, true);

// Print timing report
KernelProfiler::PrintReport(kernel);
```

---

## Debugging

### How do I see the generated GLSL?

```cpp
Kernel1D kernel([](Int i) { ... });

// Method 1: Print to stdout
std::cout << kernel.GetCode() << std::endl;

// Method 2: InspectorKernel (doesn't execute)
InspectorKernel1D inspector([](Int i) { ... });
inspector.PrintCode();
```

### GLSL compilation errors

The error message will include the generated code. Look for:
- Syntax errors in your kernel
- Type mismatches
- Missing Bind() calls

Example error:
```
Shader compilation failed:
0:15(2): error: syntax error, unexpected NEW_IDENTIFIER
```

Check line 15 of the generated GLSL (printed with `GetCode()`).

### "No active OpenGL context"

The context auto-initializes on first GPU operation. If you see this:

1. Check OpenGL support
2. Ensure you're not calling GPU operations from multiple threads without proper context handling

---

## Advanced

### Can I use templates in kernels?

No. Kernels are compiled to GLSL at runtime, which doesn't support C++ templates. Use function overloading or macros instead:

```cpp
// Instead of template
template<typename T>
void Process(T& value) { ... }

// Use Callable overloads
Callable<void(float&)> ProcessFloat = [](Float& v) { ... };
Callable<void(int&)> ProcessInt = [](Int& v) { ... };
```

### Can I call kernels from kernels?

No. Kernels cannot call other kernels. Use `Callable` for reusable functions:

```cpp
// Callable can be called from kernels
Callable<float(float)> Square = [](Float& x) {
    Return(x * x);
};

Kernel1D kernel([](Int i) {
    result[i] = Square(data[i]);
});
```

### How do I use PBO for async transfer?

PBO (Pixel Buffer Object) enables asynchronous CPU/GPU data transfer:

```cpp
Texture2D<PixelFormat::RGBA8> texture(1920, 1080);
texture.InitUploadPBOPool(2);  // Double buffering

// Non-blocking upload
texture.UploadAsync(frameData.data());
// CPU continues immediately while GPU copies data

// Check if complete
if (texture.IsIdle()) { /* ready for next upload */ }

// Or wait explicitly
texture.Sync();
```

**When to use:**
- Video streaming (continuous frame upload)
- Real-time applications (maintain 60 FPS)
- Large texture processing (overlap transfer and compute)

See [Patterns](patterns.md#async-data-transfer) for more examples.

### How do I handle expressions with side-effects?

The EasyGPU DSL cannot automatically handle side-effects from expressions. If a function or callable produces a side-effect (like modifying a reference parameter) and its return value is not used by any operator, the expression may be incorrectly optimized away and not translated into IR.

**Problem:**
```cpp
Callable<void(int&)> A = [](Int &a) {
    a = 20;  // Side-effect: modifies the input parameter
};

Kernel1D kernel([](Int i) {
    Int b = MakeInt(0);
    A(b);  // ❌ WRONG: Side-effect won't work!
           // This call may be optimized away and not generate IR
    // b is still 0 here, not 20
});
```

**Solution:**
Use `ExprBase::NotUse()` to explicitly mark expressions with side-effects that must be preserved:

```cpp
Kernel1D kernel([](Int i) {
    Int b = MakeInt(0);
    ExprBase::NotUse(A(b));  // ✅ CORRECT: Side-effect is preserved
    // b is now 20 as expected
});
```

**When to use `ExprBase::NotUse`:**
- When calling a `Callable` that modifies its parameters (reference parameters)
- When the return value of an expression is not used by any other operation
- When you need to ensure an expression is evaluated for its side-effects only

**Note:** `ExprBase::NotUse` takes an `ExprBase` (the base class of all expressions). Most function calls in the DSL return typed expressions (`Expr<T>`), which can be implicitly converted to `ExprBase`.

---

### How do I handle large datasets?

For datasets larger than GPU memory, process in chunks:

```cpp
constexpr size_t CHUNK_SIZE = 1000000;

for (size_t offset = 0; offset < total_size; offset += CHUNK_SIZE) {
    size_t chunk = std::min(CHUNK_SIZE, total_size - offset);
    
    // Upload chunk
    std::vector<float> chunk_data(data.begin() + offset, 
                                   data.begin() + offset + chunk);
    Buffer<float> gpu_chunk(chunk_data);
    
    // Process
    Kernel1D kernel([&, offset](Int i) {
        auto buf = gpu_chunk.Bind();
        buf[i] = Process(buf[i]);
    });
    kernel.Dispatch((chunk + 255) / 256, true);
    
    // Download
    gpu_chunk.Download(chunk_data);
    std::copy(chunk_data.begin(), chunk_data.end(), 
              result.begin() + offset);
}
```

### Thread safety

EasyGPU contexts are per-thread. Do not share `Buffer` or `Kernel` objects between threads without synchronization.

Each thread should:
1. Create its own buffers
2. Define its own kernels
3. Not access another thread's GPU resources

---

## Comparison with Other Libraries

### EasyGPU vs Taichi

| | EasyGPU | Taichi |
|:--|:--|:--|
| Language | C++ | Python/C++ |
| JIT compilation | No | Yes |
| Differentiable | No | Yes |
| Backend | OpenGL | LLVM/CUDA/Vulkan/Metal |
| Setup | Minimal | Requires Python |

---

## Contributing

### How do I report bugs?

Use GitHub Issues with the bug report template. Include:
- OS and GPU information
- Compiler version
- Minimal reproduction code
- Generated GLSL (from `GetCode()`)

### How do I request features?

Open a GitHub Issue with the feature request template. Describe:
- The use case
- Proposed API
- Alternative solutions you've considered
