# Shared Memory, Atomics, and Parallel Primitives

High-performance GPU computing with workgroup-level cooperation.

> **Note:** These features require understanding of GPU execution models. See [Understanding Thread Hierarchy](#understanding-thread-hierarchy) for details.

## Table of Contents

- [Shared Memory](#shared-memory)
  - [Basic Usage](#basic-usage)
  - [When to Use Shared Memory](#when-to-use-shared-memory)
- [Atomic Operations](#atomic-operations)
  - [Supported Atomic Operations](#supported-atomic-operations)
  - [Atomic Usage Patterns](#atomic-usage-patterns)
- [Parallel Primitives](#parallel-primitives)
  - [WorkgroupReduce](#workgroupreduce)
  - [WorkgroupScanInclusive](#workgroupscaninclusive)
  - [WorkgroupScanExclusive](#workgroupscanexclusive)
- [Understanding Thread Hierarchy](#understanding-thread-hierarchy)
- [Performance Guidelines](#performance-guidelines)
- [Common Patterns](#common-patterns)

---

## Shared Memory

Shared memory provides **fast, workgroup-local storage** that is accessible to all threads within a workgroup. It is significantly faster than global memory (buffers) for data that needs to be shared among threads.

### Basic Usage

```cpp
// Declare shared memory with type and size
SharedMemory<float, 256> shared;

// Each thread writes to shared memory
Int localId = LocalThreadId();  // Get local thread ID (0-255)
shared[localId] = inputValue;

// Synchronize before reading other threads' data
Kernel1D::WorkgroupBarrier();

// Now safe to read values written by other threads
Float neighborValue = shared[(localId + 1) % 256];
```

**Generated GLSL:**
```glsl
shared float v1[256];

void main() {
    int v2 = (int(gl_LocalInvocationID.x));
    v1[v2] = inputValue;
    barrier();
    float v3 = v1[(v2 + 1) % 256];
}
```

### Comparison: Buffer vs Shared Memory

| Feature | Buffer&lt;T&gt; | SharedMemory&lt;T, N&gt; |
|:--------|:----------------|:------------------------|
| **Scope** | Global (all workgroups) | Workgroup-local |
| **Speed** | Slower (~100s of cycles) | Fast (~1-10 cycles) |
| **Size** | Millions of elements | Limited (typically < 32KB) |
| **Lifetime** | Persistent across kernels | Kernel-only |
| **Initialization** | From CPU | Uninitialized or kernel-written |

### When to Use Shared Memory

**Use Shared Memory for:**
- Data that is read/written multiple times within a kernel
- Thread cooperation within a workgroup (reduction, scan)
- Caching frequently accessed global memory
- Stencil computations (each thread loads neighbors)

**Don't use for:**
- Data that persists between kernel launches
- Large datasets that exceed workgroup shared memory limits
- Data only accessed once per thread

---

## Atomic Operations

Atomic operations perform **read-modify-write** operations that are guaranteed to complete without interference from other threads. Essential for:
- Counters and histograms
- Work queues and task distribution
- Lock-free data structures

### Supported Atomic Operations

```cpp
// Integer atomics (int32)
AtomicAdd(target, value);      // Add value, return old value
AtomicSub(target, value);      // Subtract value, return old value
AtomicMin(target, value);      // Min of current and value
AtomicMax(target, value);      // Max of current and value
AtomicAnd(target, value);      // Bitwise AND
AtomicOr(target, value);       // Bitwise OR
AtomicXor(target, value);      // Bitwise XOR
AtomicExchange(target, value); // Swap values
AtomicCompSwap(target, compare, value); // Compare and swap

// Floating-point atomics (float32) - Add/Min/Max/Exchange only
AtomicAdd(floatTarget, floatValue);
AtomicMin(floatTarget, floatValue);
AtomicMax(floatTarget, floatValue);
AtomicExchange(floatTarget, floatValue);
```

### Atomic Usage Patterns

#### Pattern 1: Global Counter

```cpp
Buffer<int> counter(1);  // Single element buffer

Kernel1D countActive([&](Int i) {
    auto data = input.Bind();
    auto cnt = counter.Bind();
    
    If(data[i] > threshold, [&]() {
        // Atomically increment counter
        ExprBase::NotUse(AtomicAdd(cnt[0], MakeInt(1)));
    });
});
```

#### Pattern 2: Histogram

```cpp
Buffer<int> histogram(256);  // 256 bins

Kernel1D computeHistogram([&](Int i) {
    auto in = input.Bind();
    auto hist = histogram.Bind();
    
    // Compute bin from input value
    Int bin = Clamp(ToInt(in[i] * 256.0f), 0, 255);
    
    // Atomically increment bin count
    ExprBase::NotUse(AtomicAdd(hist[bin], MakeInt(1)));
});
```

#### Pattern 3: Find Global Maximum

```cpp
Buffer<float> globalMax(1);

Kernel1D findMax([&](Int i) {
    auto in = input.Bind();
    auto maxVal = globalMax.Bind();
    
    Float localMax = in[i];
    
    // Atomically update global maximum
    Float oldMax = AtomicMax(maxVal[0], localMax);
});
```

> **Note:** Use `ExprBase::NotUse()` to explicitly discard atomic return values when not needed. This avoids compiler warnings about unused return values.

---

## Parallel Primitives

Built-in parallel algorithms using shared memory for efficient workgroup-level cooperation.

### WorkgroupReduce

Compute a single aggregate value (sum, min, max, etc.) from all threads in a workgroup.

```cpp
SharedMemory<float, 256> shared;

Kernel1D reduceKernel([&](Int i) {
    // Each thread provides one value
    Expr<float> myValue = ...;
    
    // Reduce across all threads in workgroup
    Expr<float> workgroupSum = WorkgroupReduce(shared, myValue);
    // workgroupSum is valid in ALL threads (not just thread 0)
    
    // Typically only thread 0 writes the result
    Int localId = LocalThreadId();
    If(localId == 0, [&]() {
        result[WorkgroupId()] = workgroupSum;
    });
}, 256);  // Workgroup size must match SharedMemory size
```

**Available Operations:**

```cpp
// Built-in operations
WorkgroupReduce(shared, value, Parallel::AddOp());  // Sum
WorkgroupReduce(shared, value, Parallel::MulOp());  // Product
WorkgroupReduce(shared, value, Parallel::MinOp());  // Minimum
WorkgroupReduce(shared, value, Parallel::MaxOp());  // Maximum

// Default is Add
WorkgroupReduce(shared, value);  // Same as Add
```

### WorkgroupScanInclusive

Compute prefix sums where each thread gets the sum of all previous elements **including itself**.

```cpp
SharedMemory<float, 256> shared;

Kernel1D scanKernel([&](Int i) {
    auto in = input.Bind();
    auto out = output.Bind();
    
    Int lid = LocalThreadId();
    
    // Inclusive scan: result[i] = sum(in[0]..in[i])
    Var<float> scanned = WorkgroupScanInclusive(shared, in[lid]);
    
    // Each thread gets its prefix sum
    out[i] = scanned;
}, 256);
```

**Example:**
```
Input:  [1, 2, 3, 4, 5, 6, 7, 8]
Output: [1, 3, 6, 10, 15, 21, 28, 36]
         ↑  ↑  ↑   ↑   ↑   ↑   ↑   ↑
         1 1+2 1+2+3 ...
```

### WorkgroupScanExclusive

Compute prefix sums where each thread gets the sum of all **previous elements only** (exclusive).

```cpp
SharedMemory<float, 256> shared;

Kernel1D exclusiveScanKernel([&](Int i) {
    auto in = input.Bind();
    auto out = output.Bind();
    
    Int lid = LocalThreadId();
    
    // Exclusive scan: result[i] = sum(in[0]..in[i-1]), result[0] = identity
    Var<float> scanned = WorkgroupScanExclusive(shared, in[lid], 0.0f);
    
    out[i] = scanned;
}, 256);
```

**Example:**
```
Input:  [1, 2, 3, 4, 5, 6, 7, 8]
Output: [0, 1, 3, 6, 10, 15, 21, 28]
         ↑  ↑  ↑  ↑   ↑   ↑   ↑   ↑
         0  1 1+2 1+2+3 ...
```

**Use Case - Load Balancing:**
```cpp
// Convert per-thread work counts to start indices
Buffer<int> workCounts(256);    // How much work each thread has
Buffer<int> startIndices(256);   // Where each thread should start

Kernel1D computeOffsets([&](Int i) {
    auto counts = workCounts.Bind();
    auto offsets = startIndices.Bind();
    
    SharedMemory<int, 256> shared;
    
    // Exclusive scan gives starting position for each thread
    Int offset = WorkgroupScanExclusive(shared, counts[i], 0);
    offsets[i] = offset;
});
```

---

## Understanding Thread Hierarchy

Understanding how GPUs execute threads is crucial for using these features effectively.

```
Grid (entire kernel dispatch)
├── Workgroup 0
│   ├── Thread 0
│   ├── Thread 1
│   └── ... (up to workgroup size)
├── Workgroup 1
│   ├── Thread 0
│   └── ...
└── ... (many workgroups)
```

### Key Concepts

| Concept | GLSL | EasyGPU Access | Scope |
|:--------|:-----|:---------------|:------|
| Global ID | `gl_GlobalInvocationID` | Kernel parameter `i` | Unique across all threads |
| Local ID | `gl_LocalInvocationID` | `LocalThreadId()` | Within workgroup (0 to size-1) |
| Workgroup ID | `gl_WorkGroupID` | `WorkgroupId()` | Which workgroup |
| Barrier | `barrier()` | `Kernel1D::WorkgroupBarrier()` | Synchronizes workgroup |

### Memory Model

```
CPU Memory (RAM)
    │
    ▼ Upload/Download
Global GPU Memory (Buffers, Textures)
    │ Accessible by all threads
    ▼
Shared Memory (per workgroup)
    │ Accessible by threads in workgroup
    ▼
Registers (per thread)
    Fastest, thread-private
```

---

## Performance Guidelines

### Shared Memory Performance

**Optimal access pattern:**
```cpp
// GOOD: Sequential access by thread ID
shared[localId] = value;
barrier();
Float x = shared[(localId + 1) % 256];  // Neighbor access

// BAD: Random/strided access causes bank conflicts
Float x = shared[(localId * 17) % 256];
```

### Atomic Performance

**Minimize contention:**
```cpp
// BAD: All threads hit same counter
AtomicAdd(globalCounter[0], 1);  // Serializes all threads

// GOOD: Use shared memory for local aggregation, then global atomic
SharedMemory<int, 256> localCounts;
localCounts[localId] = localCount;
barrier();
// ...reduce local counts...
If(localId == 0, [&]() {
    AtomicAdd(globalCounter[0], workgroupTotal);  // One atomic per workgroup
});
```

### Workgroup Size Selection

```cpp
// Must be power of 2 for primitives
Kernel1D kernel([](Int i) { ... }, 256);  // Good: power of 2

// SharedMemory size must match workgroup size
SharedMemory<float, 256> shared;  // Matches workgroup size 256
```

---

## Common Patterns

### Pattern 1: Matrix Transpose with Shared Memory

```cpp
// TILE_SIZE x TILE_SIZE block transpose
constexpr int TILE_SIZE = 16;

Kernel2D transpose([](Int x, Int y) {
    SharedMemory<float, TILE_SIZE * TILE_SIZE> tile;
    
    auto in = input.Bind();
    auto out = output.Bind();
    
    auto localId = LocalThreadId2D();
    Int localX = localId.x();
    Int localY = localId.y();
    
    // Coalesced read from global memory
    tile[localY * TILE_SIZE + localX] = in[y * width + x];
    
    Kernel2D::WorkgroupBarrier();
    
    // Write transposed
    auto wgId = WorkgroupId2D();
    Int globalX = wgId.x() * TILE_SIZE + localY;
    Int globalY = wgId.y() * TILE_SIZE + localX;
    out[globalY * height + globalX] = tile[localX * TILE_SIZE + localY];
}, TILE_SIZE, TILE_SIZE);
```

### Pattern 2: Workgroup-Level Histogram

```cpp
// Each workgroup computes local histogram, then atomically adds to global
Kernel1D fastHistogram([](Int i) {
    auto in = input.Bind();
    auto globalHist = histogram.Bind();
    
    Int lid = LocalThreadId();
    Int wgId = WorkgroupId();
    
    // Shared memory for local histogram
    SharedMemory<int, 256> localHist;
    
    // Initialize local histogram
    localHist[lid] = MakeInt(0);
    Kernel1D::WorkgroupBarrier();
    
    // Each thread processes multiple elements
    For(i, totalSize, 256, [&](Int& idx) {
        Int bin = ToInt(in[idx] * 256.0f);
        ExprBase::NotUse(AtomicAdd(localHist[Clamp(bin, 0, 255)], MakeInt(1)));
    });
    
    Kernel1D::WorkgroupBarrier();
    
    // Add local histogram to global (one thread per bin)
    If(lid < 256, [&]() {
        Int bin = lid;
        ExprBase::NotUse(AtomicAdd(globalHist[bin], localHist[bin]));
    });
}, 256);
```

### Pattern 3: Parallel Prefix Sum (Multi-Workgroup)

```cpp
// Phase 1: Each workgroup computes local scan
Kernel1D localScan([](Int i) {
    SharedMemory<float, 256> shared;
    auto in = input.Bind();
    auto out = partialSums.Bind();
    auto blockSums = blockTotals.Bind();
    
    Int lid = LocalThreadId();
    Int wgId = WorkgroupId();
    
    // Load and scan within workgroup
    Var<float> scanned = WorkgroupScanInclusive(shared, in[i]);
    out[i] = scanned;
    
    // Last thread writes block sum
    If(lid == 255, [&]() {
        blockSums[wgId] = scanned;
    });
}, 256);

// Phase 2: Scan block sums (another kernel)
// Phase 3: Add block offset to each element
```

---

## API Reference

### SharedMemory

```cpp
template <ScalarType Type, int N>
class SharedMemory;

// Construction
SharedMemory<Type, N> shared;  // Declare at kernel scope

// Element access
shared[index]           // index can be Var<int>, Expr<int>, or int
shared.GetName()        // Get the GLSL variable name
shared.GetSize()        // Get N (compile-time constant)
```

### Atomic Functions

```cpp
// Integer atomics
[[nodiscard]] Expr<int> AtomicAdd(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicAdd(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicSub(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicSub(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicMin(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicMin(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicMax(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicMax(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicAnd(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicAnd(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicOr(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicOr(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicXor(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicXor(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicExchange(const Expr<int>& target, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicExchange(const Expr<int>& target, int value);
[[nodiscard]] Expr<int> AtomicCompSwap(const Expr<int>& target, const Expr<int>& compare, const Expr<int>& value);
[[nodiscard]] Expr<int> AtomicCompSwap(const Expr<int>& target, int compare, int value);

// Floating-point atomics
[[nodiscard]] Expr<float> AtomicAdd(const Expr<float>& target, const Expr<float>& value);
[[nodiscard]] Expr<float> AtomicAdd(const Expr<float>& target, float value);
[[nodiscard]] Expr<float> AtomicMin(const Expr<float>& target, const Expr<float>& value);
[[nodiscard]] Expr<float> AtomicMin(const Expr<float>& target, float value);
[[nodiscard]] Expr<float> AtomicMax(const Expr<float>& target, const Expr<float>& value);
[[nodiscard]] Expr<float> AtomicMax(const Expr<float>& target, float value);
[[nodiscard]] Expr<float> AtomicExchange(const Expr<float>& target, const Expr<float>& value);
[[nodiscard]] Expr<float> AtomicExchange(const Expr<float>& target, float value);
```

### Parallel Primitives

```cpp
// Reduction
template <typename T, int N, typename Op>
[[nodiscard]] Expr<T> WorkgroupReduce(SharedMemory<T, N>& shared, const Expr<T>& value, Op op);

template <typename T, int N>
[[nodiscard]] Expr<T> WorkgroupReduce(SharedMemory<T, N>& shared, const Expr<T>& value);  // Default: Add

// Inclusive scan
template <typename T, int N, typename Op>
[[nodiscard]] Var<T> WorkgroupScanInclusive(SharedMemory<T, N>& shared, const Expr<T>& value, Op op);

template <typename T, int N>
[[nodiscard]] Var<T> WorkgroupScanInclusive(SharedMemory<T, N>& shared, const Expr<T>& value);  // Default: Add

// Exclusive scan
template <typename T, int N, typename Op>
[[nodiscard]] Var<T> WorkgroupScanExclusive(SharedMemory<T, N>& shared, const Expr<T>& value, T identity, Op op);

template <typename T, int N>
[[nodiscard]] Var<T> WorkgroupScanExclusive(SharedMemory<T, N>& shared, const Expr<T>& value, T identity = T{});  // Default: Add

// Operations
namespace Parallel {
    struct Add;   // a + b
    struct Mul;   // a * b
    struct Min;   // min(a, b)
    struct Max;   // max(a, b)
}
```

### Barrier Functions

```cpp
class KernelBase {
public:
    // Execution barrier - wait for all threads in workgroup to reach this point
    static void WorkgroupBarrier();
    
    // Memory barrier - ensure memory writes are visible
    static void MemoryBarrier();
    
    // Combined barrier
    static void FullBarrier();  // Memory + Execution barrier
};
```

---

## See Also

- [API Reference](api-reference.md) - Complete API documentation
- [Tutorial](tutorial.md) - Learn GPU programming basics
- [Common Patterns](patterns.md) - Solutions to common tasks
- [FAQ](faq.md) - Frequently asked questions
