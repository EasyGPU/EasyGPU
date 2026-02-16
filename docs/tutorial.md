# Tutorial: From Zero to GPU Programming

A comprehensive guide to GPU programming with EasyGPU. By the end, you'll understand how to write efficient parallel algorithms.

## Table of Contents

1. [Understanding GPU Architecture](#understanding-gpu-architecture)
2. [Your First Real Kernel](#your-first-real-kernel)
3. [Memory and Data Movement](#memory-and-data-movement)
4. [Parallel Thinking](#parallel-thinking)
5. [Control Flow on GPU](#control-flow-on-gpu)
6. [Working with 2D Data](#working-with-2d-data)
7. [Structs and Complex Data](#structs-and-complex-data)
8. [Reusable Functions](#reusable-functions)
9. [Debugging and Profiling](#debugging-and-profiling)
10. [Complete Example: Particle System](#complete-example-particle-system)

---

## Understanding GPU Architecture

Before writing GPU code, you need to understand how GPUs work.

### CPU vs GPU

**CPU (Central Processing Unit):**
- Few powerful cores (4-64)
- Optimized for sequential tasks
- Complex control logic
- Large cache hierarchy

**GPU (Graphics Processing Unit):**
- Thousands of simple cores
- Optimized for parallel tasks
- Simple control logic per core
- Designed for throughput, not latency

### The Thread Hierarchy

GPUs organize work hierarchically:

```
Grid (entire computation)
└── Work Groups (local execution units)
    └── Threads (individual workers)
```

**Example: Processing 1024 elements**

```cpp
// CPU approach: sequential loop
for (int i = 0; i < 1024; i++) {
    data[i] = Process(data[i]);
}

// GPU approach: 1024 threads run simultaneously
Kernel1D kernel([](Int i) {
    data[i] = Process(data[i]);
});
kernel.Dispatch(4, true);  // 4 groups * 256 threads = 1024 threads
```

### Memory Model

```
CPU Memory (RAM)
    │  Upload/Download
    ▼
GPU Global Memory (large, slow)
    │  Accessible by all threads
    ▼
GPU Shared Memory (small, fast, per work group)
    │  Shared within a work group
    ▼
GPU Registers (fastest, per thread)
```

In EasyGPU, you primarily work with Global Memory through `Buffer<T>`.

---

## Your First Real Kernel

Let's write a kernel that actually does something useful: vector addition.

### The Problem

Given two arrays `A` and `B`, compute `C[i] = A[i] + B[i]` for all elements.

### The Solution

```cpp
#include <GPU.h>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // Configuration
    constexpr size_t N = 1000000;  // 1 million elements
    
    // Initialize CPU data
    std::vector<float> A(N), B(N), C(N);
    for (size_t i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }
    
    // Create GPU buffers
    Buffer<float> gpu_A(A);
    Buffer<float> gpu_B(B);
    Buffer<float> gpu_C(N);  // Allocate only, no upload
    
    // Define kernel
    Kernel1D vector_add([](Int i) {
        // Bind buffers to access them
        auto a = gpu_A.Bind();
        auto b = gpu_B.Bind();
        auto c = gpu_C.Bind();
        
        // Each thread computes one element
        c[i] = a[i] + b[i];
    });
    
    // Calculate dispatch size
    // Default work size is 256, so we need ceil(N/256) groups
    constexpr int work_size = 256;
    int num_groups = (N + work_size - 1) / work_size;
    
    // Execute
    std::cout << "Dispatching " << num_groups << " work groups...\n";
    vector_add.Dispatch(num_groups, true);
    
    // Get results
    gpu_C.Download(C);
    
    // Verify
    bool correct = true;
    for (size_t i = 0; i < N && correct; i++) {
        float expected = A[i] + B[i];
        if (std::abs(C[i] - expected) > 0.0001f) {
            correct = false;
            std::cout << "Mismatch at " << i << ": got " << C[i] 
                      << ", expected " << expected << "\n";
        }
    }
    
    if (correct) {
        std::cout << "Success! All " << N << " elements computed correctly.\n";
    }
    
    return 0;
}
```

### Key Concepts

1. **Thread Indexing**: Each thread receives a unique `i` value
2. **Buffer Binding**: Must bind buffers inside the kernel lambda
3. **Dispatch Calculation**: `(N + work_size - 1) / work_size` computes ceiling division
4. **Synchronization**: `true` means wait for GPU to finish

---

## Memory and Data Movement

Understanding memory is crucial for GPU performance.

### Buffer Creation Patterns

```cpp
// Pattern 1: Allocate and upload separately
Buffer<float> buffer(1024);
buffer.Upload(cpu_data);

// Pattern 2: Upload at construction
Buffer<float> buffer(cpu_data);

// Pattern 3: Write-only buffer (GPU computes, CPU reads result)
Buffer<float> buffer(1024, BufferMode::Write);
kernel.Dispatch(groups, true);
buffer.Download(cpu_data);

// Pattern 4: Read-only buffer (GPU only reads)
Buffer<float> buffer(cpu_data, BufferMode::Read);
```

### Upload/Download Performance

```cpp
// Minimize data transfer between CPU and GPU

// BAD: Transfer data every frame
for (int frame = 0; frame < 1000; frame++) {
    buffer.Upload(data);        // Upload
    kernel.Dispatch(groups, true);
    buffer.Download(data);      // Download
}

// GOOD: Keep data on GPU
Buffer<float> buffer(data);
for (int frame = 0; frame < 1000; frame++) {
    kernel.Dispatch(groups, true);  // Data stays on GPU
}
buffer.Download(data);          // Download once at end
```

### Multiple Buffers

```cpp
// Processing pipeline with multiple stages
Buffer<float> stage1_input(data);
Buffer<float> stage1_output(N);
Buffer<float> stage2_output(N);

Kernel1D stage1([](Int i) {
    auto in = stage1_input.Bind();
    auto out = stage1_output.Bind();
    out[i] = ProcessStage1(in[i]);
});

Kernel1D stage2([](Int i) {
    auto in = stage1_output.Bind();
    auto out = stage2_output.Bind();
    out[i] = ProcessStage2(in[i]);
});

// Execute pipeline
stage1.Dispatch(groups, true);
stage2.Dispatch(groups, true);
```

### Uniforms (Dynamic Constants)

When you need to pass values that change between kernel dispatches, use `Uniform<T>` instead of capturing constants. Uniforms are uploaded to the GPU at dispatch time, allowing you to modify values without recompiling the kernel.

```cpp
// Create uniforms
Uniform<float> time;
Uniform<int> iteration;

Buffer<float> data(1024);

Kernel1D kernel([&](Int i) {
    auto buf = data.Bind();
    auto t = time.Load();        // Load uniform value in kernel
    auto iter = iteration.Load();
    
    // Use uniforms in computation
    buf[i] = buf[i] * Sin(t + ToFloat(iter));
});

// Simulation loop
for (int i = 0; i < 100; i++) {
    time = static_cast<float>(i) * 0.016f;  // Update time
    iteration = i;                           // Update iteration
    kernel.Dispatch(groups, true);           // Uses new uniform values
}
```

**When to use Uniforms vs Capture:**

| Approach | Use When | Example |
|:---------|:---------|:--------|
| Capture by value | Value is constant for kernel lifetime | Fixed array size, configuration flags |
| `Uniform<T>` | Value changes between dispatches | Time step, iteration count, dynamic parameters |

**Uniform Types:**
```cpp
Uniform<float>  f;     // float uniform
Uniform<int>    i;     // int uniform
Uniform<bool>   b;     // bool uniform
Uniform<Vec3>   v;     // vec3 uniform
Uniform<Mat4>   m;     // mat4 uniform
```

---

## Parallel Thinking

To write good GPU code, you must think in parallel.

### Pattern 1: Element-wise Operations

Each thread processes one element independently:

```cpp
Kernel1D element_wise([](Int i) {
    auto in = input.Bind();
    auto out = output.Bind();
    
    // No dependency on other elements
    out[i] = sqrt(in[i] * in[i] + 1.0f);
});
```

### Pattern 2: Stencil Operations

Each thread reads neighboring elements:

```cpp
// 1D blur: each output is average of 3 neighboring inputs
Kernel1D blur_1d([](Int i) {
    auto in = input.Bind();
    auto out = output.Bind();
    
    Float sum = MakeFloat(0.0f);
    int count = 0;
    
    For(-1, 2, [&](Int& offset) {
        Int idx = i + offset;
        If(idx >= 0 && idx < N, [&]() {
            sum = sum + in[idx];
            count = count + 1;
        });
    });
    
    out[i] = sum / MakeFloat(count);
});
```

### Pattern 3: Reduction

Combining many values into one:

```cpp
// Parallel reduction: sum all elements
// Strategy: each thread sums a chunk, then CPU sums partial results

constexpr int CHUNK_SIZE = 1024;
constexpr int N = 1024 * 1024;  // 1M elements
constexpr int NUM_CHUNKS = N / CHUNK_SIZE;  // 1024 chunks

Buffer<float> partial_sums(NUM_CHUNKS);

Kernel1D reduce([](Int chunk_id) {
    auto in = input.Bind();
    auto out = partial_sums.Bind();
    
    int start = chunk_id * CHUNK_SIZE;
    Float sum = MakeFloat(0.0f);
    
    For(start, start + CHUNK_SIZE, [&](Int& i) {
        sum = sum + in[i];
    });
    
    out[chunk_id] = sum;
});

reduce.Dispatch(NUM_CHUNKS, true);

// CPU completes the reduction
std::vector<float> partials;
partial_sums.Download(partials);
float total = std::accumulate(partials.begin(), partials.end(), 0.0f);
```

### Pattern 4: Gather vs Scatter

```cpp
// Gather: Threads read from multiple locations
Kernel1D gather([](Int i) {
    auto in = input.Bind();
    auto out = output.Bind();
    
    // Each thread reads from calculated indices
    Int idx1 = i * 2;
    Int idx2 = i * 2 + 1;
    out[i] = (in[idx1] + in[idx2]) * 0.5f;
});

// Scatter: Threads write to multiple locations
// WARNING: Can cause race conditions!
Kernel1D scatter([](Int i) {
    auto out = output.Bind();
    
    // Multiple threads might write to same location
    // Use carefully with atomics or ensure no overlap
    out[i % 10] = some_value;  // Dangerous!
});
```

---

## Control Flow on GPU

GPU control flow requires special handling.

### If-Else Statements

```cpp
If(condition, [&]() {
    // Then branch
}).Elif(another_condition, [&]() {
    // Else-if branch
}).Else([&]() {
    // Else branch
});
```

**Performance Note:** Divergent branches (threads taking different paths) can hurt performance.

```cpp
// BAD: Divergent within work group
Kernel1D bad([](Int i) {
    If(i % 2 == 0, [&]() {  // Half threads take different path
        // Path A
    }).Else([&]() {
        // Path B
    });
});

// BETTER: Group threads by path
Kernel1D better([](Int i) {
    // Process all even indices first
    If(i < N / 2, [&]() {
        // All threads in work group take same path
    });
});
```

### Loops

```cpp
// For loop with explicit range
For(0, 100, [&](Int& i) {
    // i goes from 0 to 99
});

// For loop with step
For(0, 100, 2, [&](Int& i) {
    // i goes 0, 2, 4, ..., 98
});

// While loop
While(x > threshold, [&]() {
    x = x * 0.5f;
});

// Do-While loop
DoWhile([&]() {
    x = x + dx;
}, x < limit);

// Break and Continue
For(0, 100, [&](Int& i) {
    If(ShouldSkip(i), [&]() {
        Continue();  // Skip to next iteration
    });
    
    If(ShouldStop(i), [&]() {
        Break();  // Exit loop immediately
    });
});
```

### Variable Declaration

```cpp
// Declare variables inside kernels
Kernel1D example([](Int i) {
    // Creating GPU variables from literals
    Float x = MakeFloat(0.0f);
    Int count = MakeInt(0);
    Bool flag = MakeBool(false);
    
    // Creating from expressions
    Float y = x + 1.0f;  // x is Var<float>, 1.0f is float literal
    
    // Type conversion (explicit)
    Int i_val = MakeInt(42);
    Float f_val = ToFloat(i_val);  // Convert int to float
});
```

> ⚠️ **CRITICAL WARNING: `Var` Initialization May Accidentally Create a Reference**
> 
> Due to move constructor optimizations, `Int val = buf[i]` may cause `val` to become an alias (reference) to `buf[i]` instead of creating a new independent variable with a copy of the value.
> 
> ```cpp
> auto buf = buffer.Bind();
> 
> // ✅ CORRECT: Explicitly create a new variable with the value
> Int val = MakeInt(buf[i]);
> val = 5;  // Only modifies val, NOT buf[i]
> 
> // ❌ DANGEROUS: val may become a reference to buf[i]
> Int val = buf[i];
> val = 5;  // May unexpectedly modify buf[i]!
> ```
> 
> **Why this happens:**
> - `buf[i]` returns a temporary `Var<T>` (rvalue)
> - `Int val = buf[i]` selects the **move constructor** `VarBase(VarBase&&)`, which transfers ownership of the underlying variable name
> - Result: `val` directly references `buffer[i]` in the generated GLSL
> 
> **Always use `Make*()`** when initializing from buffer elements to ensure value semantics:
> ```cpp
> Int    val = MakeInt(buf[i]);      // Value copy
> Float  f   = MakeFloat(buf[i]);    // Value copy  
> Float3 v   = MakeFloat3(buf[i]);   // Value copy
> ```

**Critical: Make vs Cast**

```cpp
// Make*: Wrap literals - NO type conversion
Float f1 = MakeFloat(3.14f);     // OK: float literal
Float f2 = MakeFloat(42);        // ERROR: int literal
Float f3 = ToFloat(MakeInt(42)); // OK: Convert after wrapping

// To*: Convert between Var types - WITH type conversion
Int i = MakeInt(42);
Float f = ToFloat(i);            // Convert Var<int> to Var<float>

Float pi = MakeFloat(3.14f);
Int approx = ToInt(pi);          // Convert Var<float> to Var<int> (truncates)
```

---

## Working with 2D Data

Image processing and grid computations use 2D kernels.

### Basic Image Kernel

```cpp
constexpr int WIDTH = 1024;
constexpr int HEIGHT = 1024;

Buffer<Vec4> image(WIDTH * HEIGHT);

Kernel2D render([](Int x, Int y) {
    auto img = image.Bind();
    
    // Compute 1D index from 2D coordinates
    Int idx = y * WIDTH + x;
    
    // Create gradient based on position
    Float r = Expr<float>(x) / WIDTH;   // Red increases left to right
    Float g = Expr<float>(y) / HEIGHT;  // Green increases top to bottom
    Float b = MakeFloat(0.5f);          // Constant blue
    
    img[idx] = MakeFloat4(r, g, b, 1.0f);
});

// Dispatch: 1024/16 = 64 groups per dimension
render.Dispatch(64, 64, true);
```

### Image Processing: Gaussian Blur

```cpp
Callable<Vec4(BufferRef<Vec4>&, Int, Int)> Sample = 
[](BufferRef<Vec4>& img, Int& x, Int& y) {
    // Clamp to image bounds
    Int cx = Clamp(x, 0, WIDTH - 1);
    Int cy = Clamp(y, 0, HEIGHT - 1);
    return img[cy * WIDTH + cx];
};

Kernel2D gaussian_blur([](Int x, Int y) {
    auto in = input_image.Bind();
    auto out = output_image.Bind();
    
    Int idx = y * WIDTH + x;
    
    // 3x3 Gaussian kernel
    // [1 2 1]
    // [2 4 2] / 16
    // [1 2 1]
    
    Float4 sum = MakeFloat4(0.0f);
    
    sum = sum + Sample(in, x - 1, y - 1) * 1.0f;
    sum = sum + Sample(in, x + 0, y - 1) * 2.0f;
    sum = sum + Sample(in, x + 1, y - 1) * 1.0f;
    sum = sum + Sample(in, x - 1, y + 0) * 2.0f;
    sum = sum + Sample(in, x + 0, y + 0) * 4.0f;
    sum = sum + Sample(in, x + 1, y + 0) * 2.0f;
    sum = sum + Sample(in, x - 1, y + 1) * 1.0f;
    sum = sum + Sample(in, x + 0, y + 1) * 2.0f;
    sum = sum + Sample(in, x + 1, y + 1) * 1.0f;
    
    out[idx] = sum / 16.0f;
});
```

---

## Local Arrays (VarArray)

For small, thread-private arrays that don't need to persist between kernels, use `VarArray` instead of `Buffer`.

### Basic Usage

```cpp
Kernel1D local_array_example([](Int i) {
    // Create a local array of 10 floats
    VarArray<float, 10> localData;
    
    // Initialize
    For(0, 10, [&](Int& j) {
        localData[j] = MakeFloat(j) * 2.0f;
    });
    
    // Use the local array
    Float sum = MakeFloat(0.0f);
    For(0, 10, [&](Int& j) {
        sum = sum + localData[j];
    });
    
    // Write result to global buffer
    auto out = output.Bind();
    out[i] = sum;
});
```

### Initialized from CPU Data

```cpp
// CPU side
std::array<int, 5> lookupTable = {10, 20, 30, 40, 50};

Kernel1D with_lookup([&, lookupTable](Int i) {
    // Copy lookup table to local array
    VarArray<int, 5> table(lookupTable);
    
    auto data = input.Bind();
    auto out = output.Bind();
    
    // Use lookup table
    Int idx = data[i];
    idx = Clamp(idx, 0, 4);
    out[i] = table[idx];  // Lookup value
});
```

### Stencil Computation Example

```cpp
// 1D blur using local array for caching
Kernel1D blur_with_local([](Int i) {
    auto in = input.Bind();
    auto out = output.Bind();
    
    // Load neighborhood into local array
    VarArray<float, 7> window;
    For(0, 7, [&](Int& j) {
        Int srcIdx = i + j - 3;  // -3 to +3
        If(srcIdx >= 0 && srcIdx < N, [&]() {
            window[j] = in[srcIdx];
        }).Else([&]() {
            window[j] = MakeFloat(0.0f);
        });
    });
    
    // Compute weighted average from local array
    Float sum = MakeFloat(0.0f);
    For(0, 7, [&](Int& j) {
        Float weight = MakeFloat(1.0f / 7.0f);
        sum = sum + window[j] * weight;
    });
    
    out[i] = sum;
});
```

### When to Use VarArray vs Buffer

**Use VarArray when:**
- Data is only needed within a single kernel
- Array size is small and known at compile time
- Each thread needs its own private copy
- You want to avoid global memory allocation overhead

**Use Buffer when:**
- Data needs to persist between kernel launches
- Data is large (thousands+ of elements)
- Data needs to be accessed by multiple threads
- Data needs to be uploaded from / downloaded to CPU

## Structs and Complex Data

Define custom data structures for GPU use.

### Defining Structs

```cpp
EASYGPU_STRUCT(Particle,
    (Float3, position),
    (Float3, velocity),
    (Float3, acceleration),
    (float, mass),
    (float, life)
);

// Use in buffer
Buffer<Particle> particles(10000);

// Access in kernel
Kernel1D update_particles([](Int i) {
    auto p = particles.Bind();
    
    // Read
    Float3 pos = p[i].position();
    Float3 vel = p[i].velocity();
    Float mass = p[i].mass();
    
    // Update physics
    Float dt = MakeFloat(0.016f);
    vel = vel + p[i].acceleration() * dt;
    pos = pos + vel * dt;
    
    // Write back
    p[i].velocity() = vel;
    p[i].position() = pos;
});
```

### Nested Structs

```cpp
EASYGPU_STRUCT(Material,
    (Float3, albedo),
    (Float, roughness),
    (Float, metallic)
);

EASYGPU_STRUCT(HitRecord,
    (Float3, position),
    (Float3, normal),
    (Float, t),
    (Material, material)
);
```

---

## Reusable Functions

Use `Callable` to define functions that can be called from kernels.

### Basic Callable

```cpp
// Define a function: float -> float
Callable<float(float)> Square = [](Float& x) {
    Return(x * x);
};

// Use in kernel
Kernel1D compute([](Int i) {
    auto data = buffer.Bind();
    data[i] = Square(data[i]);
});
```

### Multiple Parameters

```cpp
Callable<float(float, float)> Lerp = [](Float& a, Float& b, Float& t) {
    Return(a + (b - a) * t);
};

// Usage
Float result = Lerp(v0, v1, 0.5f);
```

### Reference Parameters (Output)

```cpp
// Return multiple values via reference
Callable<void(float, float, float&)> PolarToCartesian = 
[](Float& r, Float& theta, Float& x, Float& y) {
    x = r * Cos(theta);
    y = r * Sin(theta);
};

// Usage in kernel
Float x, y;
PolarToCartesian(radius, angle, x, y);
```

### Random Number Generation

```cpp
// Linear Congruential Generator
Callable<float(int&)> Random = [](Int& state) {
    state = (state * 747796405 + 2891336453) & 0x7FFFFFFF;
    Int result = Abs(state);
    Return(ToFloat(result) / 2147483647.0f);
};

// Random in unit sphere
Callable<Vec3(int&)> RandomInUnitSphere = [](Int& state) {
    Float3 p;
    For(0, 100, [&](Int&) {
        p = MakeFloat3(Random(state), Random(state), Random(state)) * 2.0f 
            - MakeFloat3(1.0f, 1.0f, 1.0f);
        If(Length2(p) < 1.0f, [&]() {
            Break();
        });
    });
    Return(p);
};

// Usage: Initialize seeds on CPU, use in kernel
std::vector<int> seeds(N);
for (int i = 0; i < N; i++) seeds[i] = i + 1;
Buffer<int> rng_states(seeds);

Kernel1D kernel([](Int i) {
    auto rng = rng_states.Bind();
    Int state = rng[i];
    
    Float3 random_dir = RandomInUnitSphere(state);
    
    rng[i] = state;  // Save state for next time
});
```

---

## Debugging and Profiling

### Viewing Generated GLSL

```cpp
Kernel1D kernel([](Int i) {
    auto data = buffer.Bind();
    data[i] = data[i] * 2.0f;
});

// Print generated shader code
std::cout << "Generated GLSL:\n";
std::cout << kernel.GetCode() << std::endl;
```

### Inspector Kernels

Inspector kernels compile but don't execute, useful for debugging:

```cpp
InspectorKernel1D inspector([](Int i) {
    auto data = buffer.Bind();
    
    // Complex computation
    Float x = data[i];
    If(x > 0, [&]() {
        x = Sqrt(x);
    }).Else([&]() {
        x = -Sqrt(-x);
    });
    data[i] = x;
});

// Check compilation
if (inspector.Compile()) {
    std::cout << "Compilation OK\n";
} else {
    std::cout << "Compilation failed\n";
}

// View generated code
inspector.PrintCode();
```

### Profiling Kernel Execution

```cpp
#include <Kernel/KernelProfiler.h>

Kernel1D kernel([](Int i) {
    // ... computation ...
});
kernel.SetName("MyKernel");

// Run multiple times
for (int i = 0; i < 10; i++) {
    kernel.Dispatch(groups, true);
}

// Print timing report
KernelProfiler::PrintReport(kernel);
// Output: MyKernel: avg 0.5ms, min 0.4ms, max 0.7ms
```

---

## Complete Example: Particle System

A complete particle system with physics and collision.

```cpp
#include <GPU.h>
#include <iostream>
#include <vector>
#include <cmath>

// Particle data structure
EASYGPU_STRUCT(Particle,
    (Float3, position),    // Current position
    (Float3, velocity),    // Current velocity
    (Float3, acceleration), // Applied forces
    (float, mass),         // Mass for physics
    (float, life),         // Remaining life in seconds
    (float, size)          // Particle size
);

// Emitter configuration
struct EmitterConfig {
    GPU::Math::Vec3 position = {0, 10, 0};
    GPU::Math::Vec3 direction = {0, 1, 0};
    float spread = 0.5f;
    float min_speed = 5.0f;
    float max_speed = 10.0f;
    float min_life = 2.0f;
    float max_life = 5.0f;
};

int main() {
    // Configuration
    constexpr int MAX_PARTICLES = 100000;
    constexpr int NEW_PER_FRAME = 100;
    constexpr int FRAMES = 300;  // Simulate 5 seconds at 60 FPS
    
    // Initialize particles
    std::vector<Particle> particle_data(MAX_PARTICLES);
    for (auto& p : particle_data) {
        p.life = 0;  // All particles start dead
    }
    
    // Create GPU buffers
    Buffer<Particle> particles(particle_data);
    Buffer<int> rng_states(MAX_PARTICLES);
    
    // Initialize RNG seeds
    std::vector<int> seeds(MAX_PARTICLES);
    for (int i = 0; i < MAX_PARTICLES; i++) {
        seeds[i] = i + 1;
    }
    rng_states.Upload(seeds);
    
    // Random number generator
    Callable<float(int&)> Random = [](Int& state) {
        state = (state * 747796405 + 2891336453) & 0x7FFFFFFF;
        Int result = Abs(state);
        Return(ToFloat(result) / 2147483647.0f);
    };
    
    Callable<float(int&, float, float)> RandomRange = 
    [](Int& state, Float& min, Float& max) {
        Return(min + Random(state) * (max - min));
    };
    
    // Emitter position (captured by value)
    Float3 emitter_pos = MakeFloat3(0.0f, 10.0f, 0.0f);
    
    // Physics kernel
    Kernel1D update_particles([&, emitter_pos](Int i) {
        auto p = particles.Bind();
        auto rng = rng_states.Bind();
        
        Float dt = MakeFloat(1.0f / 60.0f);  // 60 FPS
        Float3 gravity = MakeFloat3(0.0f, -9.8f, 0.0f);
        
        // Read particle state
        Float3 pos = p[i].position();
        Float3 vel = p[i].velocity();
        Float life = p[i].life();
        Float mass = p[i].mass();
        
        // Update or respawn
        If(life > 0.0f, [&]() {
            // Alive: apply physics
            
            // Acceleration from forces
            Float3 acc = gravity + p[i].acceleration();
            
            // Update velocity and position
            vel = vel + acc * dt;
            pos = pos + vel * dt;
            
            // Decrease life
            life = life - dt;
            
            // Floor collision
            If(pos.y() < 0.0f, [&]() {
                pos.y() = 0.0f;
                vel.y() = -vel.y() * 0.8f;  // Bounce with damping
                vel.x() = vel.x() * 0.9f;   // Friction
                vel.z() = vel.z() * 0.9f;
            });
            
        }).Else([&]() {
            // Dead: check if we should respawn
            // Use RNG to stagger respawns
            Int state = rng[i];
            Float spawn_chance = Random(state);
            rng[i] = state;
            
            If(spawn_chance < 0.01f, [&]() {
                // Respawn this particle
                Int rng_state = rng[i];
                
                // Random position in sphere around emitter
                Float angle = RandomRange(rng_state, 
                    MakeFloat(0.0f), MakeFloat(6.28318f));
                Float speed = RandomRange(rng_state, 
                    MakeFloat(5.0f), MakeFloat(10.0f));
                Float life_span = RandomRange(rng_state, 
                    MakeFloat(2.0f), MakeFloat(5.0f));
                
                pos = emitter_pos;
                vel = MakeFloat3(
                    Cos(angle) * speed,
                    speed,
                    Sin(angle) * speed
                );
                life = life_span;
                mass = MakeFloat(1.0f);
                
                rng[i] = rng_state;
            });
        });
        
        // Write back
        p[i].position() = pos;
        p[i].velocity() = vel;
        p[i].life() = life;
    });
    
    // Simulate
    std::cout << "Simulating " << FRAMES << " frames with " 
              << MAX_PARTICLES << " particles...\n";
    
    for (int frame = 0; frame < FRAMES; frame++) {
        // Dispatch enough groups for all particles
        update_particles.Dispatch((MAX_PARTICLES + 255) / 256, true);
        
        if (frame % 60 == 0) {
            std::cout << "Frame " << frame << "/" << FRAMES << "\n";
        }
    }
    
    // Get final state
    particles.Download(particle_data);
    
    // Report
    int alive = 0;
    for (const auto& p : particle_data) {
        if (p.life > 0) alive++;
    }
    std::cout << "Simulation complete. " << alive 
              << " particles alive.\n";
    
    return 0;
}
```

---

## Summary

You now understand:

1. **GPU Architecture** - How GPUs differ from CPUs
2. **Thread Hierarchy** - Grids, work groups, and threads
3. **Memory Model** - Buffer management and data movement
4. **Parallel Patterns** - Element-wise, stencil, reduction
5. **Control Flow** - If, For, While on GPU
6. **2D Processing** - Image kernels and indexing
7. **Complex Data** - Structs in GPU code
8. **Reusable Functions** - Callable for code reuse
9. **Debugging** - Viewing GLSL and profiling

Next steps:
- Review [API Reference](api-reference.md) for complete function list
- Explore [Patterns](patterns.md) for common algorithms
- Check [FAQ](faq.md) for troubleshooting
