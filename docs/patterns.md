# Common Patterns

Solutions to frequently encountered tasks in EasyGPU.

> **Note:** Throughout these patterns, remember:
> - `MakeFloat(value)` / `MakeInt(value)` - Wrap **literals** into GPU variables (no type conversion)
> - `ToFloat(var)` / `ToInt(var)` - **Convert** between `Var` types (with type conversion)
> 
> Example:
> ```cpp
> Float f = MakeFloat(3.14f);              // Wrap float literal
> Int i = MakeInt(42);                     // Wrap int literal  
> Float f2 = ToFloat(MakeInt(42));         // Convert Var<int> to Var<float>
> Int i2 = ToInt(MakeFloat(3.9f));         // Convert and truncate to 3
> ```

## Table of Contents

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
Callable<Vec4(BufferRef<Vec4>&, Int, Int, Int, Int)> GaussianBlur = 
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
Callable<float(int&)> Random = [](Int& state) {
    // Linear Congruential Generator
    // Constants from Numerical Recipes
    state = (state * 1664525 + 1013904223);
    Int result = Abs(state);
    Return(ToFloat(result) / 2147483647.0f);
};

Callable<float(int&, float, float)> RandomRange = 
[](Int& state, Float& min, Float& max) {
    Return(min + Random(state) * (max - min));
};

Callable<Vec3(int&)> RandomInUnitSphere = [](Int& state) {
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
Callable<bool(Int, Int, Int, Int)> InBounds = 
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
