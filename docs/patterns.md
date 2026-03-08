# Common Patterns

Solutions to frequently encountered tasks in EasyGPU.

> **Note:** Throughout these patterns, remember:
> - `Unref(var)` - Create **independent copy** from buffer elements (avoids aliasing)
> - `MakeFloat(value)` / `MakeInt(value)` - Wrap **literals** into GPU variables (no type conversion)
> - `ToFloat(var)` / `ToInt(var)` - **Convert** between `Var` types (with type conversion)
> 
> Example:
> ```cpp
> // From buffer - use Unref to avoid aliasing
> Float val = Unref(buf[i]);               // Independent copy
> 
> // From literal - use Make
> Float f = MakeFloat(3.14f);              // Wrap float literal
> Int i = MakeInt(42);                     // Wrap int literal  
> 
> // Type conversion - use To
> Float f2 = ToFloat(MakeInt(42));         // Convert Var<int> to Var<float>
> Int i2 = ToInt(MakeFloat(3.9f));         // Convert and truncate to 3
> ```

## Table of Contents

- [Unref - Independent Variable Copies](#unref---independent-variable-copies)
- [Select Patterns (Ternary Operator)](#select-patterns-ternary-operator)
- [Resource Slots - Dynamic Resource Switching](#resource-slots---dynamic-resource-switching)
- [Local Array Patterns](#local-array-patterns)
- [Parallel Reduction (Sum/Max)](#parallel-reduction-summax)
- [Uniforms for Dynamic Parameters](#uniforms-for-dynamic-parameters)
- [Image Processing](#image-processing)
- [Particle Systems](#particle-systems)
- [Matrix Operations](#matrix-operations)
- [Random Number Generation](#random-number-generation)
- [Working with Indices](#working-with-indices)
- [Multi-Pass Rendering](#multi-pass-rendering)
- [Debugging Output](#debugging-output)
- [Async Data Transfer](#async-data-transfer)
- [Generic Callables with Templates](#generic-callables-with-templates)

---

## Unref - Independent Variable Copies

When reading from buffers, always use `Unref()` to create independent copies. Without it, you may accidentally create aliases.

### Basic Usage

```cpp
Kernel1D process([](Int i) {
    auto buf = buffer.Bind();
    
    // ❌ DANGEROUS: val aliases buf[i]
    Int val = buf[i];
    val = 5;  // May modify buf[i]!
    
    // ✅ CORRECT: Independent copy
    Int val = Unref(buf[i]);
    val = 5;  // Only modifies val
});
```

### In Arithmetic Expressions

For expressions, use `Unref` when storing to named variables:

```cpp
// Store intermediate results
Float a = Unref(buf[i]);
Float b = Unref(buf[i + 1]);
Float result = (a + b) * 0.5f;
buf[i] = result;
```

### With Callables

Use `Unref` when passing buffer elements to Callables that modify arguments:

```cpp
Callable<void(Float&)> ClampToZero = [](Float& x) {
    If(x < 0.0f, [&]() { x = 0.0f; });
};

Kernel1D process([](Int i) {
    auto buf = buffer.Bind();
    
    // Create independent copy before modifying
    Float val = Unref(buf[i]);
    ClampToZero(val);
    
    // Write back if needed
    buf[i] = val;
});
```

### Key Principle

| Scenario | Use Unref? |
|:---------|:-----------|
| `buf[i] = value` | No - direct assignment |
| `Float x = buf[i]` | **Yes** - avoid alias |
| `SomeFunc(buf[i])` | Depends - use if func modifies argument |
| `buf[i] * 2.0f` | No - temporary in expression |

See [Unref Documentation](unref.md) for complete details.

---

## Select Patterns (Ternary Operator)

The `Select` function provides expression-level conditionals, useful for concise value selection without the verbosity of `If` statements.

```cpp
// Basic syntax
ResultType result = Select(condition, trueValue, falseValue);
```

> ⚠️ **Performance Warning**: `Select` evaluates **both branches** before selecting the result. For simple operations this is fine, but for expensive computations this wastes work. Additionally, both `Select` and `If` can cause **warp divergence** when threads in a warp take different paths. See [When to Use Select vs If](#when-to-use-select-vs-if) for details.

### Pattern 1: Absolute Value

Compute the absolute value of a number:

```cpp
Float x = buf[i];
Float absX = Select(x < 0.0f, -x, x);

// For vectors (component-wise)
Vec3 v = positions[i];
Vec3 absV = Select(v < 0.0f, -v, v);
```

### Pattern 2: Min / Max

Find minimum or maximum of two values:

```cpp
Float a = buf[i];
Float b = buf[i + 1];

Float maxVal = Select(a > b, a, b);
Float minVal = Select(a < b, a, b);

// Clamp a value to range [min, max]
Float clamped = Select(x < minVal, minVal,
                      Select(x > maxVal, maxVal, x));

// Simplified clamp using helper
Callable<Float(Float, Float, Float)> Clamp = [](Float& val, Float& min, Float& max) {
    Return(Select(val < min, min, Select(val > max, max, val)));
};

Float clamped = Clamp(x, 0.0f, 1.0f);
```

### Pattern 3: Step and Sign Functions

```cpp
// Step function (Heaviside)
Float step = Select(x >= threshold, 1.0f, 0.0f);

// Sign function (-1, 0, or 1)
Float sign = Select(x > 0.0f, 1.0f,
                   Select(x < 0.0f, -1.0f, 0.0f));

// CopySign - Transfer sign from one value to another
Float magnitude = MakeFloat(5.0f);
Float signSource = MakeFloat(-3.0f);
Float result = CopySign(magnitude, signSource);  // Returns -5.0f
```

### Pattern 3b: CopySign Patterns

CopySign is useful when you need to preserve the magnitude of one value but use the sign from another:

```cpp
// Mirror reflection - flip direction based on surface normal
Vec3 incident = rayDirection;
Vec3 normal = surfaceNormal;
Vec3 reflected = CopySign(incident, -normal);  // Reflect across axis

// Broadcast scalar sign to all vector components
Vec3 v = MakeFloat3(1.0f, 2.0f, 3.0f);
Vec3 negative = CopySign(v, -1.0f);  // All components become negative

// Ensure vector points in same direction as reference
Vec3 velocity = ...;
Vec3 targetDir = ...;
Vec3 alignedVel = CopySign(velocity, Sign(Dot(velocity, targetDir)));

// Create symmetric values
Float offset = MakeFloat(2.5f);
Float left = CopySign(offset, -1.0f);   // -2.5f
Float right = CopySign(offset, 1.0f);   // +2.5f
```

### Pattern 4: Conditional Blending

Blend between values based on a condition:

```cpp
// Simple selection
Vec3 color = Select(isLit, litColor, shadowColor);

// Blend factor
Vec3 blended = Select(factor > 0.5f, colorA, colorB);

// Smooth blend (mix based on factor)
Float t = smoothness;  // 0.0 to 1.0
Vec3 mixed = colorA * t + colorB * (1.0f - t);
```

### Pattern 5: Grade/Level Classification

Classify values into discrete levels:

```cpp
// Grade calculation
Int grade = Select(score >= 90, 4,      // A
                  Select(score >= 80, 3,  // B
                        Select(score >= 70, 2,  // C
                              Select(score >= 60, 1, 0))));  // D : F

// Level of detail selection
Int lod = Select(distance < 10.0f, 3,   // High detail
                Select(distance < 50.0f, 2,  // Medium
                      Select(distance < 100.0f, 1, 0)));  // Low : None
```

### Pattern 6: Saturate (Clamp to [0, 1])

Common in graphics programming:

```cpp
// Saturate: clamp to [0, 1] range
Float saturate(Float x) {
    Return(Select(x < 0.0f, 0.0f, Select(x > 1.0f, 1.0f, x)));
}

// Usage
Float intensity = saturate(rawIntensity);
Vec3 color = MakeFloat3(saturate(r), saturate(g), saturate(b));
```

### Pattern 7: Safe Division

Avoid division by zero:

```cpp
// Safe division with epsilon check
Float safeDiv(Float a, Float b) {
    Return(Select(Abs(b) < 0.0001f, 0.0f, a / b));
}

// Or return a default value
Float ratio = Select(divisor != 0.0f, dividend / divisor, 0.0f);
```

### Pattern 8: Select with Vector Types

All vector types work with Select:

```cpp
// Vec2
Vec2 a = MakeFloat2(1.0f, 2.0f);
Vec2 b = MakeFloat2(3.0f, 4.0f);
Vec2 selected = Select(condition, a, b);

// Vec3 - common in graphics
Vec3 normal = Select(flipNormal, -originalNormal, originalNormal);

// Vec4
Vec4 rgba = Select(hasAlpha, original, MakeFloat4(rgb, 1.0f));

// Integer vectors
IVec2 gridPos = Select(isEven, posA, posB);
```

### Pattern 9: Combining with Other Operations

```cpp
// In arithmetic expressions
Float result = Select(a > b, a, b) * 2.0f + offset;

// Chained operations
Float final = Sqrt(Select(x > 0.0f, x, 0.0f));  // Sqrt only of positive values

// Multiple selects
Vec3 color = Select(isDay, dayColor, nightColor) * 
             Select(isWet, wetnessFactor, 1.0f);
```

### When to Use Select vs If

| Scenario | Use | Reason |
|:---------|:----|:-------|
| Simple value selection | `Select` | Cleaner, expression-level |
| Min/Max/Clamp/Abs | `Select` | One-liner patterns |
| Multiple statements | `If` | Side effects, multiple operations |
| Expensive computation | `If` | Avoid evaluating unused branch |
| Complex branching | `If` | Readability with `.Elif().Else()` |

```cpp
// GOOD: Select for simple selection
Float maxVal = Select(a > b, a, b);

// GOOD: If for side effects
If(shouldClear, [&]() {
    buf[i] = 0.0f;
    count[i] = 0;
    flags[i] = FLAG_NONE;
});

// BAD: Select when computation is expensive
// (Both branches are evaluated!)
Float result = Select(c, ExpensiveFunc(a), ExpensiveFunc(b));  // Don't do this!

// GOOD: If for expensive computations
Float result;
If(c, [&]() {
    result = ExpensiveFunc(a);
}).Else([&]() {
    result = ExpensiveFunc(b);
});
```

### Performance Considerations

**⚠️ Warp Divergence and Branch Execution:**

Understanding how GPUs execute branches is critical for performance:

```
GPU Execution Model:
- Threads are executed in warps (groups of 32 or 64)
- All threads in a warp execute the same instruction
- When threads diverge, the warp executes each path with masking
```

**Select vs If - Execution Model:**

```cpp
// Select: Both branches execute, then select
Float result = Select(condition, branchA, branchB);
// GLSL: result = condition ? branchA : branchB
// Both branchA and branchB are computed first!

// If: Only one branch executes, but may cause divergence
If(condition, [&]() {
    result = branchA;  // Only executed if condition is true
}).Else([&]() {
    result = branchB;  // Only executed if condition is false
});
```

**Warp Divergence Example:**

```cpp
// Problem: High divergence with data-dependent branching
Kernel1D process([](Int i) {
    // Threads 0,2,4... take PathA; threads 1,3,5... take PathB
    Float result = Select(i % 2 == 0, PathA(i), PathB(i));
    
    // What happens in a warp (threads 0-31):
    // 1. Execute PathA for all 32 threads (16 masked)
    // 2. Execute PathB for all 32 threads (16 masked)
    // Result: 2x work, but no serialization stalls
});

// Alternative with If - same divergence issue
Kernel1D process_if([](Int i) {
    Float result;
    If(i % 2 == 0, [&]() {
        result = PathA(i);
    }).Else([&]() {
        result = PathB(i);
    });
    
    // Warp execution:
    // 1. Execute PathA for threads where condition=true (others wait)
    // 2. Execute PathB for threads where condition=false (others wait)
    // Result: 2x work + serialization overhead
});

// Solution: Group threads to minimize divergence
Kernel1D process_grouped([](Int i) {
    // Process all evens first, then all odds
    If(i < N/2, [&]() {
        // Threads 0 to N/2-1: all execute PathA together
        Float result = PathA(i * 2);
    }).Else([&]() {
        // Threads N/2 to N-1: all execute PathB together
        Float result = PathB((i - N/2) * 2 + 1);
    });
    // Warps are homogeneous: each warp executes only one path
});
```

**Guidelines:**

1. **Simple operations**: Use `Select` (min, max, abs, clamp)
   - Both execute, but cheap enough that it doesn't matter
   - No divergence penalty since both paths run regardless

2. **Expensive operations**: Use `If` with thread grouping
   - Avoid evaluating both expensive branches
   - Structure data to minimize divergence

3. **Data layout matters**:
   ```cpp
   // BAD: Data causes natural divergence
   struct Particle { bool isActive; float x, y, z; };
   // Active and inactive particles intermixed in memory
   
   // GOOD: Separate arrays for different processing paths
   Buffer<float> activeParticles, inactiveParticles;
   // Process each buffer separately - no divergence within warps
   ```

4. **Count the cost**:
   ```cpp
   // Cheap: arithmetic, comparisons, Select
   Float a = Select(x > 0, x, -x);  // Fine, use Select
   
   // Expensive: transcendental functions, texture reads, loops
   Float b = Select(x > 0, Sin(x), Cos(x));  // Both computed! Use If
   ```

---

## Resource Slots - Dynamic Resource Switching

**Slots** are one of EasyGPU's most powerful features, enabling you to switch between different GPU resources (buffers, textures) at runtime **without recompiling kernels**.

### Why Slots?

In traditional GPU programming, when you want to process different data with the same kernel, you have two options:

1. **Recompile the kernel** for each different resource - extremely slow
2. **Use raw OpenGL handles** - breaks EasyGPU's type-safe abstraction

**Slots solve both problems:**

```cpp
// Define kernel once with slots
BufferSlot<float> dataSlot;

Kernel1D process([](Int i) {
    auto data = dataSlot.Bind();
    data[i] = data[i] * 2.0f;
});

// Switch between different buffers at runtime
Buffer<float> bufferA(1024), bufferB(1024);

dataSlot.Attach(bufferA);
process.Dispatch(4, true);  // Process buffer A

dataSlot.Attach(bufferB);   
process.Dispatch(4, true);  // Process buffer B - same kernel, no recompilation!
```

### How Slots Work

Slots act as **indirection layers** between your kernel and actual GPU resources:

```
Kernel Definition          Dispatch Time
     |                          |
     v                          v
+-----------+             +-------------+
| Slot.Bind | ----------> | Slot.Attach |
+-----------+             +-------------+
     |                          |
     | (GLSL)                   | (OpenGL)
     v                          v
+-----------+             +-------------+
|  slot_0   | <---------- |  Buffer A   |
|  slot_1   | <---------- |  Buffer B   |
+-----------+             +-------------+
```

**Key Design Points:**

1. **Late Binding**: Resources are bound at `Dispatch()` time, not kernel definition time
2. **Type Safety**: Slot type (`BufferSlot<float>`) ensures compile-time type checking
3. **Zero Overhead**: Slots compile to direct OpenGL bindings - no runtime indirection
4. **Resource Lifetime**: Slots don't own resources - you manage buffer/texture lifetime

### Slot Types

| Slot Type | For | Description |
|:----------|:----|:------------|
| `BufferSlot<T>` | Buffers | Dynamic switching of `Buffer<T>` |
| `TextureSlot<Format>` | 2D Textures | Dynamic switching of `Texture2D<Format>` |

### Basic Usage Pattern

```cpp
// 1. Declare slots (global or class member)
BufferSlot<Vec4> particleSlot;
TextureSlot<RGBA8> imageSlot;

// 2. Define kernel using slots
Kernel2D effect([](Int x, Int y) {
    auto particles = particleSlot.Bind();
    auto image = imageSlot.Bind();
    
    // Use slots like regular buffers/textures
    Vec4 p = particles[y * WIDTH + x];
    image.Write(x, y, p);
});

// 3. Attach resources and dispatch
Buffer<Vec4> frame1(N), frame2(N);
TextureRGBA8 tex1(W, H), tex2(W, H);

particleSlot.Attach(frame1);
imageSlot.Attach(tex1);
effect.Dispatch(groupsX, groupsY, true);  // Process frame1 -> tex1

particleSlot.Attach(frame2);               // Switch buffers
imageSlot.Attach(tex2);                    // Switch textures
effect.Dispatch(groupsX, groupsY, true);  // Process frame2 -> tex2 (same kernel!)
```

### Common Patterns with Slots

#### Pattern 1: Ping-Pong with Slots

The classic multi-pass technique made clean and type-safe:

```cpp
BufferSlot<float> readSlot;
BufferSlot<float> writeSlot;

Kernel1D jacobi([](Int i) {
    auto src = readSlot.Bind();
    auto dst = writeSlot.Bind();
    
    // Smoothing: new[i] = (left + center + right) / 3
    Float left  = src[Max(i - 1, 0)];
    Float center = src[i];
    Float right = src[Min(i + 1, 63)];
    dst[i] = (left + center + right) / 3.0f;
});

Buffer<float> bufA(64), bufB(64);

// Ping-pong between buffers
for (int iter = 0; iter < 100; iter++) {
    readSlot.Attach(bufA);
    writeSlot.Attach(bufB);
    jacobi.Dispatch(1, true);  // A -> B
    
    readSlot.Attach(bufB);
    writeSlot.Attach(bufA);
    jacobi.Dispatch(1, true);  // B -> A
}
```

#### Pattern 2: Multi-Texture Processing

Process multiple textures with the same kernel:

```cpp
TextureSlot<RGBA8> sourceSlot;
TextureSlot<RGBA8> outputSlot;

Kernel2D blur([](Int x, Int y) {
    auto src = sourceSlot.Bind();
    auto dst = outputSlot.Bind();
    
    // 3x3 box blur
    Float4 sum = MakeFloat4(0.0f);
    For(-1, 2, [&](Int& dy) {
        For(-1, 2, [&](Int& dx) {
            sum = sum + src.Read(Clamp(x + dx, 0, W-1), 
                                  Clamp(y + dy, 0, H-1));
        });
    });
    dst.Write(x, y, sum / 9.0f);
});

// Process multiple images
std::vector<TextureRGBA8> images = LoadImages(10);
TextureRGBA8 output(W, H);

outputSlot.Attach(output);
for (auto& img : images) {
    sourceSlot.Attach(img);
    blur.Dispatch(groupsX, groupsY, true);
    SaveTexture(output);
}
```

#### Pattern 3: Resolution-Independent Kernels

Write kernels that work with any size texture:

```cpp
TextureSlot<R32F> dataSlot;

// This kernel works with 512x512, 1024x1024, or any size!
Kernel2D normalize([](Int x, Int y) {
    auto data = dataSlot.Bind();
    
    uint32_t width, height;
    dataSlot.GetDimensions(width, height);  // Query attached size
    
    Float value = data.Read(x, y).x();
    Float normalized = value / 255.0f;
    data.Write(x, y, MakeFloat4(normalized, 0, 0, 1));
}, 16, 16);  // Workgroup size, not image size!

TextureR32F small(512, 512);
TextureR32F large(2048, 2048);

dataSlot.Attach(small);
normalize.Dispatch(32, 32, true);   // 512/16 = 32 groups

dataSlot.Attach(large);
normalize.Dispatch(128, 128, true); // 2048/16 = 128 groups
```

#### Pattern 4: Conditional Resource Switching

Dynamic resource selection based on runtime conditions:

```cpp
BufferSlot<Vec4> primarySlot;
BufferSlot<Vec4> secondarySlot;

Kernel1D blend([](Int i) {
    auto primary = primarySlot.Bind();
    auto secondary = secondarySlot.Bind();
    
    // Mix based on some condition
    Vec4 a = primary[i];
    Vec4 b = secondary[i];
    primary[i] = a * 0.7f + b * 0.3f;
});

Buffer<Vec4> highRes(N), lowRes(N);

void RenderFrame(bool useLOD) {
    if (useLOD) {
        primarySlot.Attach(lowRes);
        secondarySlot.Attach(highRes);
    } else {
        primarySlot.Attach(highRes);
        secondarySlot.Attach(lowRes);
    }
    blend.Dispatch(groups, true);
}
```

### Slot vs Direct Binding

| Feature | Direct Binding (`Buffer::Bind()`) | Slot (`BufferSlot::Bind()`) |
|:--------|:----------------------------------|:----------------------------|
| **Flexibility** | Fixed at kernel definition | Switchable at runtime |
| **Performance** | Zero overhead | Zero overhead |
| **Recompilation** | Required for each resource | Once for all resources |
| **Use Case** | Static pipelines | Dynamic/multi-pass |
| **Type Safety** | ✓ | ✓ |

### Best Practices

1. **Use Slots for multi-pass algorithms**
   ```cpp
   // Good: Clean ping-pong
   BufferSlot<float> read, write;
   ```

2. **Use direct binding for static pipelines**
   ```cpp
   // Good: Simple and clear
   Buffer<float> data(N);
   kernel([&](Int i) { auto d = data.Bind(); ... });
   ```

3. **Always check IsAttached() before dispatch**
   ```cpp
   if (!mySlot.IsAttached()) {
       throw std::runtime_error("Slot not attached!");
   }
   kernel.Dispatch(...);
   ```

4. **Manage resource lifetimes carefully**
   ```cpp
   // DANGER: Buffer destroyed while Slot holds reference
   {
       Buffer<float> temp(100);
       slot.Attach(temp);
   }  // temp destroyed here!
   kernel.Dispatch(...);  // Undefined behavior!
   ```

### Advanced: Slots with Uniforms

Combine slots with uniforms for maximum flexibility:

```cpp
BufferSlot<Vec4> particleSlot;
Uniform<float> timeStep;
Uniform<int> mode;

Kernel1D update([&](Int i) {
    auto p = particleSlot.Bind();
    auto dt = timeStep.Load();
    auto m = mode.Load();
    
    // Different physics based on runtime mode
    If(m == 0, [&]() {
        // Euler integration
        p[i].velocity() = p[i].velocity() + gravity * dt;
    }).ElseIf(m == 1, [&]() {
        // Verlet integration
        // ...
    });
});

// Different update modes without recompilation
mode = 0;  // Euler
particleSlot.Attach(physicsSetA);
update.Dispatch(groups, true);

mode = 1;  // Verlet
particleSlot.Attach(physicsSetB);
update.Dispatch(groups, true);
```

---

## Local Array Patterns

`VarArray` provides thread-local storage for small arrays within kernels.

### Pattern 1: Lookup Tables

```cpp
// Precompute expensive function on CPU, use lookup on GPU
std::array<float, 256> sinTable;
for (int i = 0; i < 256; i++) {
    sinTable[i] = std::sin(i * 2.0 * M_PI / 256.0);
}

Kernel1D fast_sin([&, sinTable](Int i) {
    auto data = input.Bind();
    auto out = output.Bind();
    
    // Copy table to local array
    VarArray<float, 256> lut(sinTable);
    
    // Fast lookup instead of computing sin
    Int idx = ToInt(data[i] * 256.0f) & 0xFF;
    out[i] = lut[idx];
});
```

### Pattern 2: Histogram in Local Array

```cpp
Kernel1D local_histogram([](Int i) {
    auto data = input.Bind();
    
    // Each thread computes histogram of its chunk
    VarArray<int, 16> hist;  // 16 bins
    
    // Initialize
    For(0, 16, [&](Int& j) {
        hist[j] = MakeInt(0);
    });
    
    // Process chunk
    int chunkStart = i * CHUNK_SIZE;
    For(chunkStart, chunkStart + CHUNK_SIZE, [&](Int& j) {
        Int val = data[j];
        Int bin = Clamp(val / 16, 0, 15);
        hist[bin] = hist[bin] + 1;
    });
    
    // Write histogram to global memory
    auto out = histograms.Bind();
    For(0, 16, [&](Int& j) {
        out[i * 16 + j] = hist[j];
    });
});
```

### Pattern 3: Sliding Window Buffer

```cpp
// Efficient 1D convolution with sliding window
Kernel1D sliding_conv([](Int i) {
    auto in = input.Bind();
    auto out = output.Bind();
    
    // Circular buffer for sliding window
    VarArray<float, 5> window;
    int windowIdx = 0;
    
    // Load initial window
    For(0, 5, [&](Int& j) {
        Int srcIdx = i + j - 2;
        If(srcIdx >= 0 && srcIdx < N, [&]() {
            window[j] = in[srcIdx];
        }).Else([&]() {
            window[j] = MakeFloat(0.0f);
        });
    });
    
    // Compute convolution from window
    Float sum = MakeFloat(0.0f);
    For(0, 5, [&](Int& j) {
        sum = sum + window[j] * kernel_weights[j];
    });
    
    out[i] = sum;
});
```

### Pattern 4: Small Array Sort

```cpp
// Bubble sort for small local arrays
Callable<void(VarArray<float, 8>&)> BubbleSort8 = [](VarArray<float, 8>& arr) {
    For(0, 7, [&](Int& i) {
        For(0, 7 - i, [&](Int& j) {
            If(arr[j] > arr[j + 1], [&]() {
                Float temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            });
        });
    });
};

Kernel1D sort_neighbors([](Int i) {
    auto data = input.Bind();
    auto out = output.Bind();
    
    // Collect 8 neighbors
    VarArray<float, 8> neighbors;
    For(0, 8, [&](Int& j) {
        Int idx = Clamp(i + j - 4, 0, N - 1);
        neighbors[j] = data[idx];
    });
    
    // Sort locally
    BubbleSort8(neighbors);
    
    // Output median
    out[i] = neighbors[4];
});
```

### Pattern 5: Stack/Queue Simulation

```cpp
// Stack for DFS on small graphs
Kernel1D dfs_local([](Int i) {
    VarArray<int, 32> stack;  // Fixed-size stack
    Var<int> stackPtr = MakeInt(0);
    
    // Push start node
    stack[stackPtr] = MakeInt(i);
    stackPtr = stackPtr + 1;
    
    // DFS
    While(stackPtr > 0, [&]() {
        // Pop
        stackPtr = stackPtr - 1;
        Int node = stack[stackPtr];
        
        // Process node
        // ...
        
        // Push neighbors (if space available)
        If(stackPtr < 32, [&]() {
            // Push left neighbor
            // stack[stackPtr] = left;
            // stackPtr = stackPtr + 1;
        });
    });
});
```

- [Parallel Reduction (Sum/Max)](#parallel-reduction-summax)
- [Image Processing](#image-processing)
- [Particle Systems](#particle-systems)
- [Matrix Operations](#matrix-operations)
- [Random Number Generation](#random-number-generation)
- [Working with Indices](#working-with-indices)
- [Multi-Pass Rendering](#multi-pass-rendering)
- [Debugging Output](#debugging-output)
- [Generic Callables with Templates](#generic-callables-with-templates)

---

## Parallel Reduction (Sum/Max)

Computing aggregates across large arrays.

### Simple Per-Thread Reduction

```cpp
// Each thread processes a chunk of data
constexpr int CHUNK_SIZE = 1024;

Kernel1D sum_chunks([](Int i) {
    auto in = input.Bind();
    auto out = partial_sums.Bind();
    
    int start = i * CHUNK_SIZE;
    Float sum = MakeFloat(0.0f);
    
    For(start, start + CHUNK_SIZE, [&](Int& j) {
        If(j < total_size, [&]() {
            sum = sum + in[j];
        });
    });
    
    out[i] = sum;
});

// Dispatch: ceil(total_size / CHUNK_SIZE) threads
sum_chunks.Dispatch((total_size + CHUNK_SIZE - 1) / CHUNK_SIZE, true);

// Final reduction on CPU
std::vector<float> partials;
partial_sums.Download(partials);
float total = std::accumulate(partials.begin(), partials.end(), 0.0f);
```

### Finding Maximum

```cpp
Kernel1D find_max([](Int i) {
    auto in = input.Bind();
    auto out = max_values.Bind();
    
    int start = i * CHUNK_SIZE;
    Float max_val = in[start];
    
    For(start + 1, start + CHUNK_SIZE, [&](Int& j) {
        If(j < total_size && in[j] > max_val, [&]() {
            max_val = in[j];
        });
    });
    
    out[i] = max_val;
});
```

---

## Uniforms for Dynamic Parameters

Use `Uniform<T>` when you need to change values between kernel dispatches without recompiling.

### Simulation Parameters

```cpp
Uniform<float> timeStep;
Uniform<int> maxIterations;
Uniform<float> damping;

timeStep = 0.016f;
maxIterations = 100;
damping = 0.99f;

Kernel1D physics_step([&](Int i) {
    auto p = particles.Bind();
    auto dt = timeStep.Load();
    auto maxIter = maxIterations.Load();
    auto damp = damping.Load();
    
    // Physics update using uniforms
    Float3 vel = p[i].velocity();
    vel = vel * damp;
    p[i].velocity() = vel;
});

// Run simulation with different parameters
for (int frame = 0; frame < 1000; frame++) {
    timeStep = 0.016f + 0.001f * sin(frame * 0.01f);  // Varying timestep
    physics_step.Dispatch(groups, true);
}
```

### Feature Toggles

```cpp
Uniform<bool> enableGravity;
Uniform<bool> enableCollision;

enableGravity = true;
enableCollision = false;

Kernel1D update([&](Int i) {
    auto p = particles.Bind();
    auto gravEnabled = enableGravity.Load();
    auto collEnabled = enableCollision.Load();
    
    If(gravEnabled, [&]() {
        // Apply gravity
    });
    
    If(collEnabled, [&]() {
        // Apply collision
    });
});

// Toggle features at runtime
enableCollision = true;
update.Dispatch(groups, true);
```

### Transform Matrices

```cpp
Uniform<Mat4> viewMatrix;
Uniform<Mat4> projectionMatrix;

// Update camera position each frame
void UpdateCamera(float time) {
    Mat4 view = Mat4::Rotate(time, Vec3(0, 1, 0)) * Mat4::Translate(Vec3(0, 0, -5));
    viewMatrix = view;
}

Kernel1D transform([&](Int i) {
    auto verts = vertices.Bind();
    auto view = viewMatrix.Load();
    auto proj = projectionMatrix.Load();
    
    Float4 pos = MakeFloat4(verts[i].x(), verts[i].y(), verts[i].z(), 1.0f);
    pos = proj * view * pos;
    verts[i] = pos.xyz();
});
```

---

## Image Processing

### Gaussian Blur

```cpp
Callable<Float4(BufferRef<Vec4>&, Int, Int, Int, Int)> GaussianBlur = 
[](BufferRef<Vec4>& img, Int& x, Int& y, Int& width, Int& height) {
    Float4 sum = MakeFloat4(0.0f);
    Float weight_sum = MakeFloat(0.0f);
    
    // 3x3 Gaussian kernel
    const float kernel[3][3] = {
        {1.0f/16, 2.0f/16, 1.0f/16},
        {2.0f/16, 4.0f/16, 2.0f/16},
        {1.0f/16, 2.0f/16, 1.0f/16}
    };
    
    For(-1, 2, [&](Int& ky) {
        For(-1, 2, [&](Int& kx) {
            Int px = Clamp(x + kx, 0, width - 1);
            Int py = Clamp(y + ky, 0, height - 1);
            Int idx = py * width + px;
            
            Float weight = MakeFloat(kernel[ky + 1][kx + 1]);
            sum = sum + img[idx] * weight;
            weight_sum = weight_sum + weight;
        });
    });
    
    Return(sum / weight_sum);
};

// Usage
Kernel2D blur([](Int x, Int y) {
    auto in = input_image.Bind();
    auto out = output_image.Bind();
    
    Int idx = y * WIDTH + x;
    out[idx] = GaussianBlur(in, x, y, WIDTH, HEIGHT);
});
```

### Sobel Edge Detection

```cpp
Callable<Float(BufferRef<Float>&, Int, Int, Int, Int)> Sobel = 
[](BufferRef<Float>& img, Int& x, Int& y, Int& width, Int& height) {
    Float gx = MakeFloat(0.0f);
    Float gy = MakeFloat(0.0f);
    
    // Sobel kernels
    // Gx: [-1 0 1]  Gy: [-1 -2 -1]
    //     [-2 0 2]       [ 0  0  0]
    //     [-1 0 1]       [ 1  2  1]
    
    auto sample = [&](Int dx, Int dy) -> Float {
        Int px = Clamp(x + dx, 0, width - 1);
        Int py = Clamp(y + dy, 0, height - 1);
        return img[py * width + px];
    };
    
    gx = (sample(1, -1) + 2.0f * sample(1, 0) + sample(1, 1)) -
         (sample(-1, -1) + 2.0f * sample(-1, 0) + sample(-1, 1));
    
    gy = (sample(-1, 1) + 2.0f * sample(0, 1) + sample(1, 1)) -
         (sample(-1, -1) + 2.0f * sample(0, -1) + sample(1, -1));
    
    Return(Sqrt(gx * gx + gy * gy));
};
```

---

## Particle Systems

### Basic Particle Update

```cpp
EASYGPU_STRUCT(Particle,
    (Float3, position),
    (Float3, velocity),
    (Float3, acceleration),
    (float, life)
);

Kernel1D update_particles([](Int i) {
    auto p = particles.Bind();
    Float dt = MakeFloat(0.016f);
    
    // Read
    Float3 pos = p[i].position();
    Float3 vel = p[i].velocity();
    Float3 acc = p[i].acceleration();
    Float life = p[i].life();
    
    // Update physics
    vel = vel + acc * dt;
    pos = pos + vel * dt;
    life = life - dt;
    
    // Reset dead particles
    If(life <= 0.0f, [&]() {
        pos = emitter_position;
        vel = RandomVelocity();
        life = MakeFloat(5.0f);
    });
    
    // Write back
    p[i].position() = pos;
    p[i].velocity() = vel;
    p[i].life() = life;
});
```

### Spatial Hashing (Nearest Neighbor)

```cpp
// Grid-based neighbor search
Callable<void(Int, Int3&)> GetGridCell = [](Int& particle_id, Int3& cell) {
    // Hash position to grid cell
    Float3 pos = GetParticlePosition(particle_id);
    Float grid_size = MakeFloat(1.0f);
    
    cell.x() = Floor(pos.x() / grid_size);
    cell.y() = Floor(pos.y() / grid_size);
    cell.z() = Floor(pos.z() / grid_size);
};

Kernel1D process_neighbors([](Int i) {
    auto p = particles.Bind();
    
    Int3 my_cell;
    GetGridCell(i, my_cell);
    
    // Check 27 neighboring cells
    For(-1, 2, [&](Int& cz) {
        For(-1, 2, [&](Int& cy) {
            For(-1, 2, [&](Int& cx) {
                // Process cell (my_cell + (cx, cy, cz))
                // ...
            });
        });
    });
});
```

---

## Matrix Operations

### Transform Matrices on GPU

```cpp
// Build transformation matrix
Callable<Mat4(Float3, Float3, Float3)> BuildTransform =
[](Float3& position, Float3& rotation, Float3& scale) {
    // Translation
    Matrix4 T = MakeMat4(
        MakeFloat4(1, 0, 0, 0),
        MakeFloat4(0, 1, 0, 0),
        MakeFloat4(0, 0, 1, 0),
        MakeFloat4(position.x(), position.y(), position.z(), 1)
    );

    // Scale
    Matrix4 S = MakeMat4(
        MakeFloat4(scale.x(), 0, 0, 0),
        MakeFloat4(0, scale.y(), 0, 0),
        MakeFloat4(0, 0, scale.z(), 0),
        MakeFloat4(0, 0, 0, 1)
    );

    // Rotation (simplified - around Y axis only)
    Float c = Cos(rotation.y());
    Float s = Sin(rotation.y());
    Matrix4 R = MakeMat4(
        MakeFloat4(c, 0, s, 0),
        MakeFloat4(0, 1, 0, 0),
        MakeFloat4(-s, 0, c, 0),
        MakeFloat4(0, 0, 0, 1)
    );

    Return(T * R * S);
};
```

---

## Random Number Generation

### LCG Random Number Generator

```cpp
Callable<Float(Int&)> Random = [](Int& state) {
    // Linear Congruential Generator
    // Constants from Numerical Recipes
    state = (state * 1664525 + 1013904223);
    Int result = Abs(state);
    Return(ToFloat(result) / 2147483647.0f);
};

Callable<Float(Int&, Float, Float)> RandomRange = 
[](Int& state, Float& min, Float& max) {
    Return(min + Random(state) * (max - min));
};

Callable<Float3(Int&)> RandomInUnitSphere = [](Int& state) {
    Float3 p;
    For(0, 50, [&](Int&) {
        p = MakeFloat3(
            Random(state) * 2.0f - 1.0f,
            Random(state) * 2.0f - 1.0f,
            Random(state) * 2.0f - 1.0f
        );
        If(Length2(p) < 1.0f, [&]() {
            Break();
        });
    });
    Return(p);
};

// Usage
Kernel1D kernel([](Int i) {
    auto rng = random_states.Bind();
    
    Int state = rng[i];
    Float value = Random(state);
    Float3 dir = RandomInUnitSphere(state);
    
    rng[i] = state;  // Save state for next frame
});
```

### Seeded Random

```cpp
// Initialize random states
std::vector<int> seeds(N);
for (int i = 0; i < N; i++) {
    seeds[i] = i + 1;  // Non-zero seed
}
Buffer<int> random_states(seeds);
```

---

## Working with Indices

### 1D Index to 2D Coordinates

```cpp
Kernel1D to_2d([](Int i) {
    Int x = i % WIDTH;
    Int y = i / WIDTH;
    
    // Process at (x, y)
});
```

### 2D Coordinates to 1D Index

```cpp
Kernel2D to_1d([](Int x, Int y) {
    Int idx = y * WIDTH + x;
    
    // Access flat array
    data[idx] = value;
});
```

### Bounds Checking

```cpp
Callable<Bool(Int, Int, Int, Int)> InBounds = 
[](Int& x, Int& y, Int& width, Int& height) {
    Return(x >= 0 && x < width && y >= 0 && y < height);
};

Kernel2D safe_access([](Int x, Int y) {
    If(InBounds(x, y, WIDTH, HEIGHT), [&]() {
        // Safe to access
        data[y * WIDTH + x] = value;
    });
});
```

---

## Multi-Pass Rendering

### Ping-Pong Buffers

```cpp
// Two buffers for alternating read/write
Buffer<float> buffer_a(1024);
Buffer<float> buffer_b(1024);

// Pass 1: Read from A, write to B
Kernel1D pass1([](Int i) {
    auto in = buffer_a.Bind();
    auto out = buffer_b.Bind();
    out[i] = Process(in[i]);
});

// Pass 2: Read from B, write to A
Kernel1D pass2([](Int i) {
    auto in = buffer_b.Bind();
    auto out = buffer_a.Bind();
    out[i] = Process(in[i]);
});

// Iterate
for (int iter = 0; iter < 100; iter++) {
    pass1.Dispatch(groups, true);
    pass2.Dispatch(groups, true);
}
```

---

## Debugging Output

### Writing Debug Values

```cpp
Buffer<int> debug_buffer(1024);

Kernel1D with_debug([](Int i) {
    auto data = input.Bind();
    auto debug = debug_buffer.Bind();
    
    Float value = Process(data[i]);
    
    // Check for invalid values
    If(IsNan(value) || IsInf(value), [&]() {
        debug[i] = 1;  // Mark as invalid
    }).Else([&]() {
        debug[i] = 0;
    });
    
    output[i] = value;
});

// Check after dispatch
std::vector<int> debug;
debug_buffer.Download(debug);
int invalid_count = std::count(debug.begin(), debug.end(), 1);
```

### Conditional Breakpoint Pattern

```cpp
// Set a specific condition to inspect
Int break_id = MakeInt(12345);

Kernel1D debug_kernel([](Int i) {
    If(i == break_id, [&]() {
        // Inspect values here
        Float value = data[i];
        // Value will be visible in generated GLSL when using InspectorKernel
    });
});
```

---

## Async Data Transfer

Use Pixel Buffer Objects (PBOs) for non-blocking CPU/GPU data transfers. Essential for real-time applications.

### Pattern 1: Double-Buffered Streaming

CPU prepares next frame while GPU uploads previous frame:

```cpp
Texture2D<PixelFormat::RGBA8> videoFrame(1920, 1080);
videoFrame.InitUploadPBOPool(2);  // Double buffering

Kernel2D processFrame([&](Int x, Int y) {
    auto frame = videoFrame.Bind();
    Float4 color = frame.Read(x, y);
    // Apply filter...
    frame.Write(x, y, filtered);
}, 16, 16);

// Generate frames
std::vector<std::vector<uint8_t>> frames = GenerateFrames(100);

for (size_t i = 0; i < frames.size(); ++i) {
    // Try async upload - returns false if both PBOs busy
    while (!videoFrame.UploadAsync(frames[i].data())) {
        // Both buffers in use, wait a bit
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    // Process frame (runs in parallel with next upload)
    processFrame.Dispatch(120, 68, true);
}

videoFrame.Sync();  // Wait for final upload
```

### Pattern 2: Triple Buffering (Maximum Throughput)

Tolerates more timing variance between CPU and GPU:

```cpp
Texture2D<PixelFormat::RGBA8> renderTarget(2048, 2048);
renderTarget.InitUploadPBOPool(3);  // Triple buffering

// Producer thread generates frames faster than GPU consumes
for (const auto& frameData : frameQueue) {
    // UploadStream blocks if all 3 PBOs busy, ensuring smooth flow
    renderTarget.UploadAsyncStream(frameData.data(), 1000);
    
    kernel.Dispatch(128, 128, false);  // Don't wait, keep pipeline full
}

renderTarget.Sync();
```

### Pattern 3: Async Readback

Download results without stalling the GPU pipeline:

```cpp
Texture2D<PixelFormat::RGBA8> output(1024, 1024);
output.InitDownloadPBOPool(2);

// Rendering passes
renderKernel.Dispatch(64, 64, true);

// Start async download (returns immediately)
output.DownloadAsync();

// Do more computation while GPU prepares download...
postProcess.Dispatch(32, 32, true);

// Check if download ready
std::vector<uint8_t> pixels(1024 * 1024 * 4);
if (output.GetDownloadData(pixels.data())) {
    SaveImage(pixels);
} else {
    // Not ready yet, either wait or process next frame
    output.Sync();
    output.GetDownloadData(pixels.data());
}
```

### Pattern 4: Ping-Pong with Async Transfer

Combine ping-pong rendering with async uploads:

```cpp
Texture2D<PixelFormat::RGBA8> texA(1024, 1024);
Texture2D<PixelFormat::RGBA8> texB(1024, 1024);

texA.InitUploadPBOPool(2);
texB.InitUploadPBOPool(2);

bool useA = true;
for (const auto& frameData : frames) {
    auto& currentTex = useA ? texA : texB;
    auto& otherTex = useA ? texB : texA;
    
    // Upload new frame while processing previous
    currentTex.UploadAsyncStream(frameData.data(), 1000);
    
    // Process (reads from otherTex, writes to currentTex)
    Kernel2D process([&](Int x, Int y) {
        auto in = otherTex.Bind();
        auto out = currentTex.Bind();
        // Read from in, write to out...
    }, 16, 16);
    
    process.Dispatch(64, 64, true);
    useA = !useA;
}
```

### Pattern 5: Producer-Consumer with PBO

CPU produces data while GPU consumes:

```cpp
Texture2D<PixelFormat::RGBA32F> simulationData(512, 512);
simulationData.InitUploadPBOPool(2);

// CPU thread: continuously generate simulation data
void ProducerThread() {
    for (int step = 0; step < 1000; ++step) {
        std::vector<float> data = ComputeSimulationStep(step);
        
        // Block if GPU hasn't finished with previous frame
        simulationData.UploadAsyncStream(data.data(), 5000);
    }
}

// GPU thread: process as fast as possible
void ConsumerThread() {
    Kernel2D simulate([&](Int x, Int y) {
        auto data = simulationData.Bind();
        // Process simulation data...
    }, 16, 16);
    
    for (int i = 0; i < 1000; ++i) {
        simulate.Dispatch(32, 32, true);
    }
}
```

### When to Use PBO vs Synchronous Transfer

| Scenario | Use | Reason |
|:---------|:----|:-------|
| Real-time video | PBO | Maximize CPU/GPU parallelism |
| Single image processing | Sync | Simple, no pipeline needed |
| Large dataset streaming | PBO | Overlap transfer and compute |
| Interactive applications | PBO | Maintain responsive framerate |
| Small textures (<1MB) | Either | Overhead negligible |

### Buffer Count Selection

| Count | Latency | Throughput | Use Case |
|:------|:--------|:-----------|:---------|
| 1 | Low | Low | Simple async |
| 2 | Medium | High | Standard streaming |
| 3+ | Higher | Max | High-latency tolerance |


---

## Generic Callables with Templates

Use C++ templates to write reusable, type-generic GPU functions that work across multiple data types.

### Basic Generic Callable

Create a callable that accepts any GPU type convertible to `Float`:

```cpp
// Generic weighted sum - works with any two GPU types
template <class T1, class T2>
Callable<Float(T1, T2)> WeightedSum = [&](T1 X1, T2 X2) {
    // Convert both inputs to Float for computation
    Return(ToFloat(X1) * 0.7f + ToFloat(X2) * 0.3f);
};

Kernel1D kernel([](Int i) {
    auto buf = data.Bind();
    
    // Mix Int and Float
    Int count = MakeInt(100);
    Float intensity = MakeFloat(0.5f);
    Float mixed = WeightedSum<Int, Float>(count, intensity);
    
    // Mix two Floats
    Float a = MakeFloat(1.0f);
    Float b = MakeFloat(2.0f);
    Float blended = WeightedSum<Float, Float>(a, b);
    
    buf[i] = mixed + blended;
});
```

> **Important:** Template parameters must be **GPU types** (`Int`, `Float`, `Float2`, `Float3`, etc.), not C++ literal types (`int`, `float`, `Vec2`). The DSL operates on `Var<T>` types, not native C++ types.

### Generic Type Conversion Callable

Create utilities that normalize any numeric type to 0-1 range:

```cpp
// Normalize any GPU integer type to [0, 1] Float
template <class T>
Callable<Float(T, T, T)> NormalizeUint = [&](T value, T minVal, T maxVal) {
    // All operations promoted to Float
    Float v = ToFloat(value);
    Float min = ToFloat(minVal);
    Float max = ToFloat(maxVal);
    Return((v - min) / (max - min));
};

Kernel2D processImage([](Int x, Int y) {
    auto img = image.Bind();
    
    // Works with 8-bit values (stored as Int)
    Int pixelValue = img[y * WIDTH + x];
    Float normalized = NormalizeUint<Int>(pixelValue, MakeInt(0), MakeInt(255));
    
    // Apply curve and convert back
    Float adjusted = Pow(normalized, 1.0f / 2.2f);  // Gamma correction
    img[y * WIDTH + x] = ToInt(adjusted * 255.0f);
});
```

### Generic Vector Operations

Define operations that work on any vector dimension:

```cpp
// Component-wise minimum for any Float vector type
template <class VecType>
Callable<VecType(VecType, VecType)> ComponentMin = [&](VecType a, VecType b) {
    // Works with Float2, Float3, Float4
    Return(Min(a, b));
};

// Generic lerp that works with Float, Float2, Float3, Float4
template <class T>
Callable<T(T, T, Float)> GenericLerp = [&](T a, T b, Float t) {
    Return(a + (b - a) * t);
};

Kernel1D interpolate([](Int i) {
    auto data = buffer.Bind();
    
    Float2 start = MakeFloat2(0.0f, 0.0f);
    Float2 end = MakeFloat2(1.0f, 1.0f);
    Float t = MakeFloat(0.5f);
    
    // Lerp on Float2
    Float2 mid = GenericLerp<Float2>(start, end, t);
    
    // Lerp on Float
    Float alpha = GenericLerp<Float>(MakeFloat(0.0f), MakeFloat(1.0f), t);
    
    data[i] = mid + MakeFloat2(alpha, alpha);
});
```

### Generic Clamp Function

```cpp
// Clamp any comparable GPU type
template <class T>
Callable<T(T, T, T)> GenericClamp = [&](T value, T minVal, T maxVal) {
    If(value < minVal, [&]() { Return(minVal); });
    If(value > maxVal, [&]() { Return(maxVal); });
    Return(value);
};

// Usage examples
Kernel1D clampDemo([](Int i) {
    auto buf = buffer.Bind();
    
    // Clamp Float to [0, 1]
    Float f = buf[i];
    Float clampedF = GenericClamp<Float>(f, MakeFloat(0.0f), MakeFloat(1.0f));
    
    // Clamp Int to valid index range
    Int idx = MakeInt(i * 2);
    Int safeIdx = GenericClamp<Int>(idx, MakeInt(0), MakeInt(1023));
    
    // Clamp Float3 to color range
    Float3 color = MakeFloat3(1.5f, -0.2f, 0.8f);
    Float3 clampedColor = GenericClamp<Float3>(
        color, 
        MakeFloat3(0.0f, 0.0f, 0.0f), 
        MakeFloat3(1.0f, 1.0f, 1.0f)
    );
});
```

### Generic Distance Metric

```cpp
// Manhattan distance for any vector type (L1 norm)
template <class VecType>
Callable<Float(VecType, VecType)> ManhattanDistance = [&](VecType a, VecType b) {
    Return(Sum(Abs(a - b)));
};

// Chebyshev distance for any vector type (L∞ norm)  
template <class VecType>
Callable<Float(VecType, VecType)> ChebyshevDistance = [&](VecType a, VecType b) {
    Return(Max(Abs(a - b)));
};

Kernel2D distanceField([](Int x, Int y) {
    auto output = out.Bind();
    
    Float2 point = MakeFloat2(ToFloat(x), ToFloat(y));
    Float2 center = MakeFloat2(512.0f, 512.0f);
    
    // Generic distance works with Float2
    Float dist = ManhattanDistance<Float2>(point, center);
    
    // Convert to intensity
    Float intensity = MakeFloat(1.0f) - GenericClamp<Float>(dist / 512.0f, 
                                                             MakeFloat(0.0f), 
                                                             MakeFloat(1.0f));
    output[y * WIDTH + x] = intensity;
});
```

### When to Use Generic Callables

| Scenario | Use Generic? | Example |
|:---------|:-------------|:--------|
| Same logic, different types | **Yes** | Math utilities, clamp, lerp |
| Type-specific optimizations | No | Specialized SIMD operations |
| Simple one-off functions | No | Inline in kernel |
| Library of reusable utilities | **Yes** | Build your own GPU math library |

### Template Parameter Guidelines

```cpp
// ✅ CORRECT: Use GPU types (Var<T> aliases)
template <class T> Callable<Float(T)> Convert = [&](T value) {
    Return(ToFloat(value));
};
// T can be: Int, Float, Float2, Float3, Float4, etc.

// ❌ WRONG: C++ literal types don't work in GPU context
template <class T> Callable<float(T)> Wrong = [&](T value) {
    Return(value * 2.0f);  // ERROR: T is C++ type, not GPU type
};
```

**Available GPU Types for Templates:**

| Category | Types |
|:---------|:------|
| Scalar | `Int`, `Float`, `Bool` |
| Vector | `Float2`, `Float3`, `Float4`, `Int2`, `Int3`, `Int4` |
| Matrix | `Mat2`, `Mat3`, `Mat4`, etc. |

**Type Conversion in Generic Callables:**

```cpp
template <class T>
Callable<Float(T)> Process = [&](T input) {
    // Always convert to a known type for computation
    Float f = ToFloat(input);    // Safe: T -> Float
    Int i = ToInt(input);        // Safe: T -> Int (truncation)
    
    // Perform operations in known type space
    Float result = Sqrt(Abs(f));
    Return(result);
};
```
