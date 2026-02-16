# Getting Started

This guide will get you running your first GPU program in under 10 minutes.

## What is EasyGPU?

EasyGPU is an embedded domain-specific language (eDSL) that lets you write GPU compute kernels using standard C++ syntax. Instead of learning GLSL or dealing with complex graphics APIs, you write GPU code that looks like regular C++.

**Traditional GPU programming:**
```cpp
// Write C++ code on CPU
// Learn GLSL for GPU shaders
// Write 100+ lines of boilerplate to bridge them
```

**With EasyGPU:**
```cpp
// Write C++ code that runs on GPU
Buffer<float> data(1024);
Kernel1D kernel([](Int i) {
    data[i] = data[i] * 2.0f;
});
kernel.Dispatch(4, true);
```

## Prerequisites

- **Operating System:** Windows (Linux support coming)
- **Compiler:** GCC 11+, Clang 14+, or MSVC 2022+
- **C++ Standard:** C++20
- **GPU:** Any GPU supporting OpenGL 4.3+
- **Build System:** CMake 3.21+ (optional but recommended)

## Installation

### Method 1: CMake FetchContent (Recommended)

Add to your `CMakeLists.txt`:

```cmake
include(FetchContent)
FetchContent_Declare(
    easygpu
    GIT_REPOSITORY https://github.com/easygpu/EasyGPU.git
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(easygpu)

target_link_libraries(your_target EasyGPU)
```

### Method 2: Copy Headers

1. Copy the `include/` directory to your project
2. Add `#include <GPU.h>` to your source files
3. Link with OpenGL (`-lGL` on Linux, `-lopengl32` on Windows)

## Your First Program

Create a file named `first_kernel.cpp`:

```cpp
#include <GPU.h>
#include <iostream>
#include <vector>

int main() {
    // Step 1: Prepare data on the CPU
    std::vector<float> numbers = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    // Step 2: Create GPU buffers
    // This uploads the data to GPU memory
    Buffer<float> input(numbers);
    Buffer<float> output(numbers.size());
    
    // Step 3: Define a GPU kernel
    // This lambda will execute on the GPU, not the CPU
    Kernel1D double_values([](Int i) {
        // Bind buffers to access them in the kernel
        auto in = input.Bind();
        auto out = output.Bind();
        
        // Each thread processes one element
        out[i] = in[i] * 2.0f;
    });
    
    // Step 4: Execute the kernel
    // 1 group of 256 threads, wait for completion
    double_values.Dispatch(1, true);
    
    // Step 5: Retrieve results
    output.Download(numbers);
    
    // Verify
    std::cout << "Results: ";
    for (float n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    // Output: Results: 2 4 6 8 10
    
    return 0;
}
```

### Build and Run

With CMake:
```bash
mkdir build && cd build
cmake ..
cmake --build .
./first_kernel
```

Direct compilation (Linux):
```bash
g++ -std=c++20 first_kernel.cpp -lGL -o first_kernel
./first_kernel
```

Direct compilation (Windows with MSVC):
```bash
cl /std:c++20 first_kernel.cpp opengl32.lib
first_kernel.exe
```

## Understanding the Basics

### The GPU Programming Model

GPU programming is different from CPU programming:

| CPU | GPU |
|:----|:----|
| Sequential execution | Massively parallel |
| One instruction at a time | Thousands of threads simultaneously |
| Optimized for latency | Optimized for throughput |
| Complex control flow | Simple, uniform control flow |

In EasyGPU, you write kernels that run on thousands of threads in parallel.

### Buffers

`Buffer<T>` manages GPU memory and data transfer:

```cpp
// Allocate GPU memory for 1024 floats
Buffer<float> buffer1(1024);

// Allocate with specific access mode
Buffer<float> buffer2(1024, BufferMode::Read);      // GPU reads only
Buffer<float> buffer3(1024, BufferMode::Write);     // GPU writes only
Buffer<float> buffer4(1024, BufferMode::ReadWrite); // Both (default)

// Upload data from CPU vector
std::vector<float> data = {1.0f, 2.0f, 3.0f};
Buffer<float> buffer5(data);

// Upload and download data manually
buffer1.Upload(data);
buffer1.Download(data);
```

### Local Arrays (VarArray)

For small, thread-private arrays within kernels:

```cpp
Kernel1D example([](Int i) {
    // Create local array of 10 floats
    VarArray<float, 10> localData;
    
    // Initialize and use
    For(0, 10, [&](Int& j) {
        localData[j] = MakeFloat(j);
    });
    
    // Access elements
    Float sum = MakeFloat(0.0f);
    For(0, 10, [&](Int& j) {
        sum = sum + localData[j];
    });
});
```

**Buffer vs VarArray:**
- `Buffer<T>` - Global GPU memory, large, persists between kernels
- `VarArray<Type, N>` - Local thread memory, small (compile-time size), per-kernel only

### Kernels

Kernels are C++ lambdas that execute on the GPU:

```cpp
// 1D kernel: processes arrays
// The parameter 'i' is the thread index
Kernel1D kernel1([](Int i) {
    data[i] = data[i] * 2;
});

// 2D kernel: processes images/grids
// Parameters are (x, y) coordinates
Kernel2D kernel2([](Int x, Int y) {
    int pixel_index = y * width + x;
    image[pixel_index] = color;
});

// 3D kernel: processes volumes
Kernel3D kernel3([](Int x, Int y, Int z) {
    int voxel_index = z * width * height + y * width + x;
    volume[voxel_index] = density;
});
```

### Dispatch

Dispatch launches the kernel on the GPU:

```cpp
// 1D kernel: dispatch 100 work groups
// With default work size of 256, this creates 100 * 256 = 25,600 threads
kernel.Dispatch(100, true);  // true = wait for completion

// 2D kernel: dispatch 64x64 groups for a 1024x1024 image
// With default 16x16 work size, each dimension gets (1024/16) = 64 groups
kernel.Dispatch(64, 64, true);

// 3D kernel: dispatch groups in 3 dimensions
kernel.Dispatch(32, 32, 32, true);
```

### Thread IDs and Indexing

Each thread gets a unique ID:

```cpp
Kernel1D kernel([](Int i) {
    // 'i' ranges from 0 to (num_groups * work_group_size - 1)
    // Thread 0 processes element 0
    // Thread 1 processes element 1
    // etc.
    data[i] = Process(data[i]);
});
```

For 2D:
```cpp
Kernel2D kernel([](Int x, Int y) {
    // x ranges: 0 to (group_count_x * work_group_size_x - 1)
    // y ranges: 0 to (group_count_y * work_group_size_y - 1)
    int index = y * width + x;
    pixels[index] = ComputeColor(x, y);
});
```

### Control Flow

GPU control flow uses capitalized function names:

```cpp
// If-Else
If(x > 0, [&]() {
    // Then branch
    result = Sqrt(x);
}).Elif(x < 0, [&]() {
    // Else-if branch
    result = -Sqrt(-x);
}).Else([&]() {
    // Else branch
    result = 0;
});

// For loop
For(0, 100, [&](Int& i) {
    // i ranges from 0 to 99
    sum = sum + data[i];
});

// While loop
While(x < threshold, [&]() {
    x = x * 2;
});

// Break and Continue
For(0, 100, [&](Int& i) {
    If(i % 2 == 0, [&]() {
        Continue();  // Skip even numbers
    });
    If(data[i] > limit, [&]() {
        Break();  // Exit loop early
    });
});
```

### Make vs Cast: Critical Distinction

**`Make*` functions** wrap C++ literals into GPU variables. They do **NOT** perform type conversion.

```cpp
// Correct: literal types match
Float f = MakeFloat(3.14f);    // float literal -> Var<float>
Int i = MakeInt(42);           // int literal -> Var<int>
Float3 v = MakeFloat3(1, 2, 3); // Three floats -> Var<Vec3>

// WRONG: type mismatch - will not compile
Float f = MakeFloat(42);       // ERROR: 42 is int, not float

// Correct fixes:
Float f = MakeFloat(42.0f);           // Use float literal
Float f = ToFloat(MakeInt(42));       // Convert int Var to float Var
```

**`To*` functions** convert between GPU variable types. They **DO** perform type conversion.

```cpp
Int i = MakeInt(42);
Float f = ToFloat(i);          // Var<int> -> Var<float> (conversion)

Float pi = MakeFloat(3.14f);
Int approx = ToInt(pi);        // Var<float> -> Var<int> (truncates to 3)
```

| Operation | Function | Semantics |
|:----------|:---------|:----------|
| Create Var<float> from float literal | `MakeFloat(3.14f)` | Wrap, no conversion |
| Create Var<int> from int literal | `MakeInt(5)` | Wrap, no conversion |
| Convert Var<int> to Var<float> | `ToFloat(int_var)` | Convert with widening |
| Convert Var<float> to Var<int> | `ToInt(float_var)` | Convert with truncation |

## Common Mistakes

### Mistake 1: Binding Outside Kernel

```cpp
// WRONG
auto ref = buffer.Bind();  // ERROR: Bind must be inside kernel
Kernel1D kernel([](Int i) {
    data[i] = ref[i];
});

// CORRECT
Kernel1D kernel([](Int i) {
    auto ref = buffer.Bind();  // Bind inside kernel
    data[i] = ref[i];
});
```

### Mistake 2: Wrong Dispatch Size

```cpp
// If you have 1000 elements and default work size is 256:
// You need ceil(1000/256) = 4 work groups

// WRONG: Only processes first 256 elements
kernel.Dispatch(1, true);

// CORRECT: Processes all 1000 elements
kernel.Dispatch(4, true);  // 4 * 256 = 1024 threads (covers 1000)

// Alternative: Calculate properly
kernel.Dispatch((N + 255) / 256, true);
```

### Mistake 3: Forgetting Synchronization

```cpp
// WRONG: Download happens before kernel finishes
kernel.Dispatch(4, false);  // Don't wait
data.Download(results);      // Data may be incomplete

// CORRECT
kernel.Dispatch(4, true);   // Wait for completion
data.Download(results);      // Data is ready
```

### Mistake 4: Using C++ Types in Kernels

```cpp
// WRONG: Cannot use C++ types in GPU code
Kernel1D kernel([](Int i) {
    std::vector<float> temp;  // ERROR: std::vector is CPU-only
    temp.push_back(data[i]);
});

// CORRECT: Use GPU types only
Kernel1D kernel([](Int i) {
    Float temp = data[i];     // OK: Float is a GPU type
    result[i] = temp * 2;
});
```

## Next Steps

Now that you have the basics:

1. **[Tutorial](tutorial.md)** - Complete walkthrough with more examples
2. **[API Reference](api-reference.md)** - Full API documentation
3. **[Patterns](patterns.md)** - Common solutions and techniques
4. **[FAQ](faq.md)** - Troubleshooting and common questions

## Quick Reference Card

```cpp
// Include
#include <GPU.h>

// Buffers
Buffer<float> buf1(1024);                          // Allocate
Buffer<float> buf2(data);                          // Upload
Buffer<float> buf3(1024, BufferMode::Write);       // Write-only
buf1.Upload(data);
buf1.Download(data);

// Kernels
Kernel1D k1([](Int i) { ... });                    // 1D
Kernel2D k2([](Int x, Int y) { ... });             // 2D
Kernel3D k3([](Int x, Int y, Int z) { ... });      // 3D
k1.Dispatch(groups, true);                         // Execute

// Inside kernels
auto ref = buffer.Bind();                          // Access buffer

// Control flow
If(cond, [&]() { ... }).Else([&]() { ... });
For(start, end, [&](Int& i) { ... });
While(cond, [&]() { ... });
Break(); Continue();

// Values
Float f = MakeFloat(3.14f);                        // From literal
Int i = MakeInt(42);
Float f2 = ToFloat(i);                             // Convert
Int i2 = ToInt(f);                                 // Convert (truncates)

// Uniforms (dynamic constants)
Uniform<float> param;                              // Create uniform
param = 2.5f;                                      // Set value
kernel.Dispatch(groups, true);                     // Dispatch with current value
param = 5.0f;                                      // Change value
kernel.Dispatch(groups, true);                     // Dispatch with new value
```
