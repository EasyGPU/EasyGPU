# Automatic Differentiation

Compute gradients of GPU kernels automatically — no hand-written derivative code.

> **Note:** Automatic differentiation (AD) records every differentiable operation during the forward pass, then walks the recording in reverse to generate gradient code. This works for both inspection (view generated GLSL without GPU) and execution (train parameters directly on GPU).

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [API Overview](#api-overview)
  - [AdjointInspector — Offline Inspection](#adjointinspector--offline-inspection)
  - [ADKernel1D — GPU Training](#adkernel1d--gpu-training)
  - [AdjointKernel1D — Forward+Backward Combined](#adjointkernel1d--forwardbackward-combined)
- [Low-Level API](#low-level-api)
- [Supported Operations](#supported-operations)
  - [Arithmetic Operations](#arithmetic-operations)
  - [Intrinsic Functions](#intrinsic-functions)
  - [Vector Operations](#vector-operations)
  - [Zero-Gradient Operations](#zero-gradient-operations)
- [Callable AD](#callable-ad)
- [Control Flow AD](#control-flow-ad)
- [Gradient Buffers](#gradient-buffers)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## Quick Start

The simplest way to use AD is with `AdjointInspector1D` — no GPU required:

```cpp
#include <GPU.h>
#include <iostream>

int main() {
    // Define a computation: y = w * x, loss = y * y
    AD::AdjointInspector1D inspector([](Var<int>& i, auto& ctx) {
        Var<float> w; w = 2.0f;
        Var<float> x; x = 3.0f;
        Var<float> y = w * x;
        Var<float> loss = y * y;

        ctx.RegisterParameter(w);
        ctx.MarkLoss(loss);
    });

    std::cout << "=== Forward ===\n" << inspector.GetForwardCode() << "\n";
    std::cout << "=== Backward ===\n" << inspector.GetBackwardCode() << "\n";
    return 0;
}
```

**Generated forward GLSL:**
```glsl
#version 430 core
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    float v1 = float(2.0);
    float v2 = float(3.0);
    float v3 = (v1) * (v2);
    float v4 = (v3) * (v3);
}
```

**Generated backward GLSL:**
```glsl
#version 430 core
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    float d_v4 = float(0);      // adjoint of loss
    float d_v1 = float(0);      // adjoint of w (parameter)

    d_v4 = float(1.0);          // seed loss gradient

    float d_v3 = float(0);
    d_v3 += (d_v4) * (v3);      // d(loss)/d(y): d(y²)/dy = 2y = y + y
    d_v3 += (v3) * (d_v4);
    d_v1 += (d_v3) * (v2);      // d(loss)/d(w): chain rule through multiplication
}
```

For GPU execution with real buffers, use `ADKernel1D` (see [ADKernel1D — GPU Training](#adkernel1d--gpu-training)).

---

## How It Works

EasyGPU's AD system is a **source-to-source** reverse-mode automatic differentiation engine. It works in three phases:

```
Forward Pass (C++ DSL)         Tape Recording          Backward Pass (GLSL)
─────────────────────     ─────────────────────     ──────────────────────
Var<float> w; w = 2.0f;   [0] Store v1 = 2.0       float d_v4 = 0;
Var<float> x; x = 3.0f;   [1] Store v2 = 3.0       float d_v1 = 0;
Var<float> y = w * x;     [2] BinaryOp(v3=v1*v2)   d_v4 = 1.0;  // seed
Var<float> loss = y * y;  [3] BinaryOp(v4=v3*v3)   d_v3 += d_v4 * v3;
                                                     d_v3 += v3 * d_v4;
                                                     d_v1 += d_v3 * v2;
```

1. **Forward pass** — Your C++ DSL code is translated to GLSL as usual, but the `Builder` also sends each operation to the active `GradientTape`.
2. **Tape recording** — The tape stores metadata for each differentiable operation: what kind of operation, which variables are inputs/outputs, and their GLSL types.
3. **Backward generation** — The `AdjointGenerator` walks the tape in reverse, applying the chain rule to generate adjoint accumulation statements.

### Key Concepts

| Term | Meaning |
|:-----|:--------|
| **Tape** (Wengert list) | Ordered list of all differentiable operations from the forward pass |
| **Adjoint** | Gradient accumulator variable, prefixed with `d_` (e.g., `d_v5`) |
| **Seed** | The adjoint of the loss variable, initialized to `1.0` |
| **Parameter** | A variable whose gradient is needed for optimization |
| **Active variable** | Any variable on a path from a parameter to the loss |

> ⚠️ **The tape only records top-level stores.** Compound expressions like `Var<float> y = a*x + b*y + c*z` record a single `Store` of the final result, not each intermediate `Mul`/`Add`. This is by design — the IR already decomposes complex expressions before they reach the tape.

---

## API Overview

EasyGPU provides three API levels for AD, from simple inspection to full GPU training:

| API | GPU Required | Use Case |
|:----|:-------------|:---------|
| `AdjointInspector1D/2D/3D` | No | Inspect generated GLSL, debug gradients, write tests |
| `AdjointKernel1D/2D/3D` | Yes | Combined forward+backward shader, single dispatch |
| `ADKernel1D` | Yes | Separate Forward/Backward calls, gradient download |

### AdjointInspector — Offline Inspection

`AdjointInspector` builds on `InspectorKernel` and adds automatic differentiation. It generates both forward and backward GLSL without requiring a GPU — ideal for testing and debugging.

```cpp
AD::AdjointInspector1D inspector([](Var<int>& i, auto& ctx) {
    Var<float> a; a = 2.0f;
    Var<float> b; b = 3.0f;
    Var<float> c = a * b;

    ctx.RegisterParameter(a);
    ctx.RegisterParameter(b);
    ctx.MarkLoss(c);
});

// Access generated code
std::string forward  = inspector.GetForwardCode();
std::string backward = inspector.GetBackwardCode();

// Debug the tape
inspector.PrintTape();
// Output: [0] kind=2 out=v3 ins:v1,v2,
```

**Kernel function signatures:**

```cpp
// 1D: void(Var<int>& threadId, AdjointContext& ctx)
AD::AdjointInspector1D inspector1d(func, workSizeX);

// 2D: void(Var<int>& idX, Var<int>& idY, AdjointContext& ctx)
AD::AdjointInspector2D inspector2d(func, workSizeX, workSizeY);

// 3D: void(Var<int>& idX, Var<int>& idY, Var<int>& idZ, AdjointContext& ctx)
AD::AdjointInspector3D inspector3d(func, workSizeX, workSizeY, workSizeZ);
```

**AdjointContext methods:**

```cpp
// Register a parameter (for gradient computation)
ctx.RegisterParameter(variableName, "float");     // by name + type string
ctx.RegisterParameter(var);                       // pass Var<T> directly (recommended)
ctx.RegisterParameter(variableName, "float");    // by name + type string
ctx.RegisterParameter<float>(variableName);      // by name + template type (legacy)

// Mark the scalar loss variable
ctx.MarkLoss(variableName, "float");               // by name + type string
ctx.MarkLoss(var);                               // pass Var<T> directly (recommended)
ctx.MarkLoss(variableName, "float");             // by name + type string
ctx.MarkLoss<float>(variableName);               // by name + template type (legacy)
```

**Inspector methods:**

```cpp
inspector.GetForwardCode();       // GLSL source of forward pass
inspector.GetBackwardCode();      // GLSL source of backward pass
inspector.GetTapeSummary();       // Text summary of all tape entries
inspector.PrintTape();            // Print tape summary to stdout
inspector.HasBackwardCode();      // Whether backward code was generated
inspector.Tape();                 // Access the underlying GradientTape
```

### ADKernel1D — GPU Training

`ADKernel1D` is the primary API for training on GPU. It wraps a regular `Kernel1D`, records the tape, and generates a combined forward+backward shader. Use `Forward()` for the forward pass and `Backward()` to compute gradients.

```cpp
#include <GPU.h>

int main() {
    const int N = 1024;

    // Training data
    std::vector<float> xData(N), yData(N);
    for (int i = 0; i < N; i++) {
        xData[i] = i * 0.01f;
        yData[i] = 2.0f * xData[i] + 1.0f;
    }

    Buffer<float> buf_x(xData);
    Buffer<float> buf_y(yData);
    Buffer<float> buf_w(N);  // parameter: weight (one per thread)
    Buffer<float> buf_b(N);  // parameter: bias (one per thread)

    // Initialize parameters
    std::vector<float> wInit(N, 0.5f), bInit(N, 0.5f);
    buf_w.Upload(wInit);
    buf_b.Upload(bInit);

    // Define AD kernel
    AD::ADKernel1D kernel([](Var<int>& i) {
        auto x = buf_x[i];
        auto y_true = buf_y[i];
        auto w = buf_w[i];
        auto b = buf_b[i];

        // Forward computation
        auto y_pred = w * x + b;
        auto diff = y_pred - y_true;
        auto loss = diff * diff;  // MSE loss

        // Mark parameter indices (order matters for Gradient())
        int iw = AD::Param(w);
        int ib = AD::Param(b);
        AD::Loss(loss);
    }, N);

    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        kernel.Backward(4, true);  // Forward + backward, 4 groups of 256

        // Get gradients
        auto grad_w = kernel.Gradient(0);  // index 0 = w (first AD::Param call)
        auto grad_b = kernel.Gradient(1);  // index 1 = b

        // SGD update on CPU
        wInit = buf_w.Download();  // re-download current params
        bInit = buf_b.Download();
        float lr = 0.01f;
        for (int i = 0; i < N; i++) {
            wInit[i] -= lr * grad_w[i];
            bInit[i] -= lr * grad_b[i];
        }
        buf_w.Upload(wInit);
        buf_b.Upload(bInit);
    }

    return 0;
}
```

**Key points:**
- `AD::Param(var)` registers a variable as a trainable parameter and returns its index (0, 1, 2, ...)
- `AD::Loss(var)` marks the scalar loss variable
- `Backward()` runs the combined forward+backward shader, computing both loss and gradients
- `Gradient(index)` downloads the gradient from GPU to a `std::vector<float>`
- `Backward(groups, false)` dispatches asynchronously. Use this in training loops when the next operation is another GPU operation such as an optimizer step.

> **Performance note:** Gradient and adjoint buffers are cleared by a small GPU clear kernel before the combined backward dispatch. Training loops no longer need to upload zero-filled CPU arrays just to reset gradients.

### AdjointKernel1D — Forward+Backward Combined

`AdjointKernel1D` generates a single combined shader that runs both forward and backward in one dispatch. Useful when you want to inspect the merged shader or run everything in a single GPU pass.

```cpp
AD::AdjointKernel1D kernel([](Var<int>& i, auto& ctx) {
    Var<float> w; w = 2.0f;
    Var<float> x; x = 3.0f;
    Var<float> y = w * x;
    Var<float> loss = y * y;

    ctx.RegisterParameter(w);
    ctx.MarkLoss(loss);
});

std::string combined = kernel.GetCombinedCode();
// Contains: forward declarations + buffer decls + main() with forward body
//           + adjoint declarations + adjoint body + gradient writebacks
```

**2D and 3D variants:**

```cpp
AD::AdjointKernel2D kernel2d([](Var<int>& x, Var<int>& y, auto& ctx) {
    // 2D computation...
}, workSizeX, workSizeY);

AD::AdjointKernel3D kernel3d([](Var<int>& x, Var<int>& y, Var<int>& z, auto& ctx) {
    // 3D computation...
}, workSizeX, workSizeY, workSizeZ);
```

---

## Low-Level API

For advanced use cases, you can work directly with the tape and generator:

```cpp
#include <AD/GradientTape.h>
#include <AD/AdjointGenerator.h>

// Step 1: Create a tape and activate it
GPU::AD::GradientTape tape;
GPU::IR::Builder::Builder::Get().SetGradientTape(&tape);

// Step 2: Build your kernel (tape records automatically)
// ... use InspectorKernel1D or Kernel1D here ...

GPU::IR::Builder::Builder::Get().SetGradientTape(nullptr);

// Step 3: Generate backward code
GPU::AD::AdjointGenerator gen;
std::string backwardGLSL = gen.Generate(tape, true);

// Step 4: Inspect the tape
for (size_t i = 0; i < tape.Size(); i++) {
    const auto& entry = tape[i];
    // entry.kind, entry.output, entry.inputs ...
}
```

### GradientTape Methods

```cpp
// Parameter management
tape.RegisterParameter("v1", "float");
tape.IsParameter("v1");               // returns true
tape.ParameterCount();                // number of registered parameters

// Loss
tape.MarkLoss("v5", "float");
tape.LossVar();                       // returns std::optional<TapeVar>

// Variable queries
tape.IsActive("v3");                  // is variable on gradient path?
tape.GetVarType("v3");                // returns "float", "vec3", etc.

// Tape access
tape.Size();                          // number of entries
tape[i];                              // access entry by index
tape.Entries();                       // all entries

// Sub-tapes (for callable bodies)
tape.PushSubTape();
int idx = tape.PopSubTape();
tape.SubTapeCount();
tape.SubTape(index);

// Control flow
tape.BeginIfBranch("v2 > 0");
tape.BeginElifBranch("v2 < 0");
tape.BeginElseBranch();
tape.EndIfChain();
tape.BeginForLoop("i", "0", "10", "1");
tape.EndForLoop();
```

### AdjointGenerator Methods

```cpp
GPU::AD::AdjointGenerator gen;

// Generate complete backward shader (with #version, layout, main)
std::string glsl = gen.Generate(tape, writeBackParams);

// Generate body parts only (for merging into existing shader)
GPU::AD::AdjointBody body = gen.GenerateBody(tape, writeBackParams);
// body.declarations — (adjName, glslType) pairs
// body.lines        — adjoint accumulation statements
// body.writebacks   — (paramName, adjName) pairs for gradient writeback

gen.GetAdjointTable();  // query adjoint variable names
```

---

## Supported Operations

### Arithmetic Operations

All basic arithmetic operations are differentiable:

| Operation | C++ DSL | Gradient Rule (d_out = adjoint of result) |
|:----------|:--------|:------------------------------------------|
| Add | `c = a + b` | `d_a += d_c`, `d_b += d_c` |
| Subtract | `c = a - b` | `d_a += d_c`, `d_b -= d_c` |
| Multiply | `c = a * b` | `d_a += d_c * b`, `d_b += d_c * a` |
| Divide | `c = a / b` | `d_a += d_c / b`, `d_b -= d_c * a / (b * b)` |
| Negate | `c = -a` | `d_a -= d_c` |
| Compound Add | `a += b` | `d_a += d_c`, `d_b += d_c` |
| Compound Sub | `a -= b` | `d_a += d_c`, `d_b -= d_c` |

```cpp
Var<float> a; a = 2.0f;
Var<float> b; b = 3.0f;
Var<float> c = a + b * a - b / a;  // Chain of differentiable ops
```

### Intrinsic Functions

All supported intrinsics and their gradient rules:

| Function | GLSL | Gradient Rule |
|:---------|:-----|:--------------|
| `sin(x)` | `sin(x)` | `d_x += d_out * cos(x)` |
| `cos(x)` | `cos(x)` | `d_x += d_out * -sin(x)` |
| `tan(x)` | `tan(x)` | `d_x += d_out * (1 + tan(x)²)` |
| `asin(x)` | `asin(x)` | `d_x += d_out / sqrt(1 - x²)` |
| `acos(x)` | `acos(x)` | `d_x += d_out / -sqrt(1 - x²)` |
| `atan(x)` | `atan(x)` | `d_x += d_out / (1 + x²)` |
| `exp(x)` | `exp(x)` | `d_x += d_out * exp(x)` |
| `log(x)` | `log(x)` | `d_x += d_out / x` |
| `exp2(x)` | `exp2(x)` | `d_x += d_out * exp2(x) * log(2)` |
| `log2(x)` | `log2(x)` | `d_x += d_out / (x * log(2))` |
| `sqrt(x)` | `sqrt(x)` | `d_x += d_out / (2 * sqrt(x))` |
| `inversesqrt(x)` | `1/sqrt(x)` | `d_x += d_out * -0.5 / (x * sqrt(x))` |
| `abs(x)` | `abs(x)` | `d_x += d_out * sign(x)` |
| `pow(a, b)` | `pow(a,b)` | `d_a += d_out * b * pow(a, b-1)`, `d_b += d_out * pow(a,b) * log(a)` |
| `atan2(y, x)` | `atan(y,x)` | `d_y += d_out * x/(x²+y²)`, `d_x += d_out * -y/(x²+y²)` |
| `min(a, b)` | `min(a,b)` | `d_a += d_out * (a < b)`, `d_b += d_out * (b <= a)` |
| `max(a, b)` | `max(a,b)` | `d_a += d_out * (a > b)`, `d_b += d_out * (b >= a)` |
| `clamp(x, lo, hi)` | `clamp(x,lo,hi)` | `d_x += d_out * (lo < x && x < hi)` |
| `mix(a, b, t)` | `mix(a,b,t)` | `d_a += d_out * (1-t)`, `d_b += d_out * t`, `d_t += d_out * (b-a)` |
| `step(edge, x)` | `step(edge,x)` | Zero gradient (step is non-differentiable) |
| `smoothstep(e0,e1,x)` | `smoothstep(e0,e1,x)` | Cubic Hermite derivative |
| `sinh(x)` | `sinh(x)` | `d_x += d_out * cosh(x)` |
| `cosh(x)` | `cosh(x)` | `d_x += d_out * sinh(x)` |
| `tanh(x)` | `tanh(x)` | `d_x += d_out * (1 - tanh(x)²)` |

```cpp
// Example: sigmoid-like computation
Var<float> x; x = 0.5f;
Var<float> y = 1.0f / (1.0f + Exp(-x));  // All differentiable
Var<float> z = Log(y);                     // Chained through tape
```

### Vector Operations

Vector math operations are fully supported with correct gradient rules:

| Operation | C++ DSL | Gradient Rule |
|:----------|:--------|:--------------|
| `dot(a, b)` | `Dot(a, b)` | `d_a += d_out * b`, `d_b += d_out * a` |
| `cross(a, b)` | `Cross(a, b)` | `d_a += cross(d_out, b)`, `d_b += cross(a, d_out)` |
| `length(v)` | `Length(v)` | `d_v += d_out * v / length(v)` |
| `distance(a, b)` | `Distance(a, b)` | `d_a += d_out * (a-b) / dist`, `d_b -= same` |
| `normalize(v)` | `Normalize(v)` | `d_v += d_out * (I - n*nᵀ) / length(v)` |
| `reflect(I, N)` | `Reflect(I, N)` | Standard reflect derivative |
| `refract(I, N, eta)` | `Refract(I, N, eta)` | Standard refract derivative |

```cpp
Var<Vec3> normal; normal = MakeFloat3(0, 1, 0);
Var<Vec3> lightDir; lightDir = Normalize(MakeFloat3(1, 1, 1));
Var<float> ndotl = Dot(normal, lightDir);
Var<float> diffuse = Max(ndotl, 0.0f);  // Differentiable through max
```

### Zero-Gradient Operations

These operations do not propagate gradients. The tape either skips them or generates zero adjoint contributions:

- `floor(x)`, `ceil(x)`, `trunc(x)`, `round(x)` — step functions
- `sign(x)` — discontinuous
- `step(edge, x)`, `faceforward(N, I, Nref)` — non-differentiable by design
- Bit-casting: `floatBitsToInt`, `intBitsToFloat`, `floatBitsToUint`, `uintBitsToFloat`

---

## Callable AD

Callables (user-defined GPU functions) work with automatic differentiation. When a Callable is invoked during the forward pass, its body operations are recorded in a **sub-tape**. During backward generation, the sub-tape is walked in reverse and the adjoint code is inlined at the call site.

### Basic Callable with Gradients

```cpp
// Define a differentiable callable
Callable<Float(Float, Float)> MultiplyAdd = [](Float& a, Float& b) {
    Return(a * 2.0f + b);
};

// Use in an AD kernel
AD::AdjointInspector1D inspector([](Var<int>& i, auto& ctx) {
    Var<float> x; x = 3.0f;
    Var<float> y; y = 4.0f;
    Var<float> z = MultiplyAdd(x, y);
    Var<float> loss = z * z;

    ctx.RegisterParameter(x);
    ctx.MarkLoss(loss);
});
```

The backward pass inlines the adjoint of the Callable body automatically — no manual derivative needed.

### Callable with Multiple Parameters and Control Flow

```cpp
Callable<Float(Float, Float)> ClampedMix = [](Float& a, Float& b) {
    Float t = MakeFloat(0.5f);
    Float m = Mix(a, b, t);
    If(m > 1.0f, [&]() { m = 1.0f; });
    Return(m);
};
```

### Void Callables (No Return Value)

```cpp
Callable<void(Float&)> ScaleByTwo = [](Float& x) {
    x = x * 2.0f;
};

// Usage in kernel
Var<float> v; v = 1.0f;
ScaleByTwo(v);  // v is now 2.0f, gradients propagated correctly
```

> ⚠️ **Callable AD limitations:**
> - Each Callable invocation creates a sub-tape. Deeply nested Callables (5+ levels) may increase shader complexity.
> - The same Callable called multiple times creates separate sub-tapes — gradients are correct but code size grows linearly.

---

## Control Flow AD

The AD system records control flow boundaries on the tape and generates matching backward control flow.

### If/Else

```cpp
Var<float> x; x = 1.0f;
Var<float> y; y = 0.0f;

If(x > 0, [&]() {
    y = x * 2.0f;
}).Else([&]() {
    y = x * -1.0f;
});

Var<float> loss = y * y;
```

**Generated backward (control flow is preserved):**
```glsl
float d_v4 = 0;  // adjoint of loss
d_v4 = 1.0;

// Reverse tape: loss = y * y
float d_v2 = 0;
d_v2 += d_v4 * v2;  // d_y
d_v2 += v2 * d_v4;

// Reverse tape: if/else
if(v1 > 0) {
    // y = x * 2.0 → d_x += d_y * 2.0
    float d_v0 = 0;
    d_v0 += d_v2 * 2.0;
} else {
    // y = x * -1.0 → d_x += d_y * -1.0
    float d_v0 = 0;
    d_v0 -= d_v2;
}
```

### For Loops

For loops are reversed in the backward pass — the tape records the loop bounds, and the adjoint generator produces a descending loop:

```cpp
Var<float> sum; sum = 0.0f;
For(0, 10, [&](Int& j) {
    sum = sum + MakeFloat(j) * x;
});
```

**Generated backward (descending loop):**
```glsl
for(int j = 10 - 1; j >= 0; j--) {
    d_v0 += d_v1 * float(j);
}
```

### Nested Control Flow

If/else inside for loops, for inside if/else, and arbitrary nesting are all supported:

```cpp
For(0, N, [&](Int& j) {
    If(data[j] > threshold, [&]() {
        sum = sum + data[j] * weight;
    }).Else([&]() {
        sum = sum - data[j] * bias;
    });
});
```

---

## Gradient Buffers

### Buffer Grouping (Interleaved Layout)

Multiple parameters from the same source buffer share a single gradient SSBO to stay within shader storage block limits:

```cpp
Buffer<float> buf_W(N);  // Contains 3 parameters per element (w1, w2, w3)
Buffer<float> buf_b(N);  // Contains 1 parameter per element (bias)

AD::ADKernel1D kernel([](Var<int>& i) {
    auto W = buf_W.Bind();
    auto b = buf_b.Bind();

    // These three parameters share one gradient buffer
    auto w1 = W[i * 3 + 0];
    auto w2 = W[i * 3 + 1];
    auto w3 = W[i * 3 + 2];

    AD::Param(w1);  // index 0, group 'buf_W', offset 0
    AD::Param(w2);  // index 1, group 'buf_W', offset 1
    AD::Param(w3);  // index 2, group 'buf_W', offset 2
    AD::Param(b[i]); // index 3, group 'buf_b', offset 0

    AD::Loss(/* ... */);
}, N);

// Single gradient buffer for buf_W stores all 3 params interleaved:
// [thread0_w1, thread0_w2, thread0_w3, thread1_w1, thread1_w2, thread1_w3, ...]
```

**Generated gradient buffer layout:**
```glsl
layout(std430, binding = 10) buffer _ad_gradbuf_buf_W {
    float _ad_grad_buf_W_data[];  // stride=3 interleaved
};
layout(std430, binding = 11) buffer _ad_gradbuf_buf_b {
    float _ad_grad_buf_b_data[];  // stride=1 (single param)
};
```

### Downloading Gradients

```cpp
// By parameter index (matching AD::Param() call order)
auto grad_w1 = kernel.Gradient(0);  // first AD::Param call
auto grad_w2 = kernel.Gradient(1);  // second AD::Param call
auto grad_b  = kernel.Gradient(3);

// By variable name
auto grad_b = kernel.Gradient("v42");  // GLSL variable name
```

### Batch Gradient Download

For inspection or custom CPU-side optimizers, calling `Gradient(i)` in a loop re-downloads shared gradient buffers multiple times. Use `DownloadAllGradients()` for efficient batch download:

```cpp
// ❌ Slow — downloads each shared buffer once per parameter in the group
for (size_t i = 0; i < kernel.ParameterCount(); i++) {
    auto grad_i = kernel.Gradient(i);  // redundant downloads
}

// ✅ Fast — each shared buffer downloaded exactly once
auto allGrads = kernel.DownloadAllGradients();
for (size_t i = 0; i < allGrads.size(); i++) {
    const auto &grad_i = allGrads[i];  // already in CPU memory
}
```

`DownloadAllGradients()` uses an internal cache map: each unique gradient buffer handle is downloaded once, then per-parameter slices are extracted from the cached data.

> **Note:** The built-in NN optimizers (`Adam`, `SGD`, `RMSprop`) do not use this CPU download path during normal training. They consume the AD gradient buffers directly on the GPU.

---

## Training with NN Components

The AD engine integrates with the NN module (`include/NN/`) to eliminate boilerplate. Tensor, Optimizer, and Layers are designed to work with `ADKernel1D`:

### Tensor + AD::Param

`Tensor<T, Dims...>::ForEachParam()` registers every scalar element as a trainable parameter:

```cpp
Tensor<float, 128, 64> W(xavierData);

// Inside kernel lambda — 8192 AD::Param calls, one line
auto W_ref = W.Bind();
W_ref.ForEachParam([](auto &w) { AD::Param(w); });
```

`ForEachParam` uses `std::index_sequence` + fold expressions to unroll at compile time. Each `AD::Param(w)` call records the scalar's GLSL variable name on the gradient tape and returns its index. The AD kernel groups parameters from the same source buffer into shared interleaved gradient SSBOs automatically.

### Optimizer + ADKernel1D

Optimizers consume gradients directly from the AD kernel:

```cpp
Adam adam(0.001f);
adam.AddTensor(W);           // register weights
adam.AddTensor(b);           // register biases

for (int step = 0; step < 1000; step++) {
    kernel.Backward(groups, false); // forward + backward, write gradients to GPU
    adam.Step(kernel);              // GPU aggregate + update
}
```

`Adam::Step(kernel)` internally:
1. Reads AD gradient buffers directly on the GPU
2. Averages per-thread gradients for each scalar parameter
3. Updates flat optimizer state (`m`/`v`) in GPU buffers
4. Applies Adam update to all registered tensor buffers in one combined dispatch when binding limits allow

The optimizer tracks per-parameter `m` and `v` state vectors, matching the AD kernel's scalar parameter count exactly. This avoids the common pitfall of averaging gradients across entire tensors.

`SGD` and `RMSprop` use the same GPU optimizer path. If a model exceeds the backend buffer binding limit, the optimizer automatically falls back to one GPU dispatch per tensor instead of downloading gradients to the CPU.

### Layers

Layers provide `Setup()` (register parameters) and `Forward()` (emit DSL code):

```cpp
// Outside kernel — construct with Xavier-initialized weights
Linear<float, 784, 128> fc1(42);
ReLU<float> relu(128);
Linear<float, 128, 10>  fc2(123);

// Inside kernel lambda
fc1.Setup(); fc2.Setup(); relu.Setup();  // register all parameters
// ... use Forward() in the computation:
fc1.Forward(input, threadId, hidden);
relu.Forward(hidden, threadId, activated);
fc2.Forward(activated, threadId, output);
```

See [API Reference](api-reference.md#neural-network) for the full NN API.

---

## Limitations

### What the AD system CAN do

- Differentiate through arithmetic (+, -, *, /, -x)
- Differentiate through ~30 intrinsic functions (sin, cos, exp, log, sqrt, pow, etc.)
- Differentiate through vector operations (dot, cross, length, normalize, distance, reflect, refract)
- Differentiate through user-defined Callables (including nested and control-flow-containing Callables)
- Differentiate through if/else branches, for loops, and nested combinations
- Handle compound assignments (+=, -=) with correct gradient accumulation
- Support float, vec2, vec3, vec4, ivec*, and matrix types
- Share gradient buffers for parameters from the same source buffer (interleaved layout)

### What the AD system does NOT support

- **While loops** — The tape cannot determine the number of iterations at record time, which is needed for reversing the loop. Use `For` with a fixed bound instead.
- **Break/Continue** inside loops — The tape records linear sequences; early exits break the reversal invariant.
- **Swizzle writes** (`v.xyz = something`) — Swizzle reads are supported (e.g., `v.xyz * 2.0f`), but swizzle writes require special handling not yet implemented.
- **ArrayAccess and MemberAccess** — Direct struct member access on the RHS of an assignment is not yet differentiated.
- **Integer-only operations** — Operations on `Int` types (beyond type conversion) are not differentiated. Only float-typed computations contribute gradients.
- **Shared memory atomics** — `AtomicAdd`/`AtomicMin` etc. are not recorded on the tape.
- **Buffer element indexing with variables** — `buf[varIndex]` where `varIndex` is not a compile-time constant prevents the system from tracking which parameter is being updated.

> ⚠️ **Compound expressions only record the final Store.** `Var<float> y = a*x + b*y + c*z` records one tape entry, not five. This is usually what you want — the IR already decomposes the expression — but it means you can't inspect intermediate gradients from sub-expressions.

---

## Troubleshooting

### No backward code generated

Check that:
1. At least one parameter is registered via `ctx.RegisterParameter()` or `AD::Param()`
2. A loss variable is marked via `ctx.MarkLoss()` or `AD::Loss()`
3. The loss is connected to the parameters through differentiable operations
4. The loss variable is of type `float` (not `int` or `bool`)

### "WARNING: _code is empty" in forward code

The kernel body is empty — your computation may be entirely inside Callables or control flow that didn't emit top-level code. Try adding a simple statement outside any Callable/If/For.

### Gradient values are all zero

Common causes:
- The loss variable is not on a differentiable path from the parameters
- Intermediate variables use non-differentiable operations (floor, ceil, round, sign)
- The variable registered as parameter is a constant literal, not a buffer element
- `AD::Param()` was called but the variable name doesn't match the actual GLSL variable name

### Incorrect gradient values

- **Swizzle-heavy code**: Swizzle writes are not supported. The gradient through a swizzle chain may be incorrect.
- **Mixing float and int**: Integer operations don't produce gradients. Ensure all computation is in float domain.
- **Aliasing**: Using `buf[i]` directly without `Unref()` can cause aliasing where modifying it also modifies the original. Use `Var<float> val = Unref(buf[i])` to create independent copies.

### Debugging the tape

```cpp
// Print tape entries to understand what was recorded
inspector.PrintTape();
// Or get as string
std::string summary = inspector.GetTapeSummary();
// [0] kind=2 out=v3 ins:v1,v2,
// [1] kind=2 out=v4 ins:v3,v3,
```

Each entry shows:
- `kind`: 0=BinaryOp, 1=UnaryOp, 2=Intrinsic1, 3=Intrinsic2, 4=Intrinsic3, 5=Ternary, 6=CompoundAssign, 7=Call, 8=Return, 9=ControlFlowBegin, 10=ControlFlowEnd
- `out`: The variable being assigned to
- `ins`: The input variables (comma-separated)
- `fn`: Intrinsic function name (for intrinsic entries)

---

## API Reference

### AdjointContext

```cpp
class AdjointContext {
public:
    // Recommended: pass Var<T> directly — type is deduced
    template<typename T> void RegisterParameter(const Var<T>& var);
    template<typename T> void MarkLoss(const Var<T>& var);

    // String-based overloads (advanced / dynamic-name use)
    void RegisterParameter(const std::string& name, const std::string& glslType);
    template<typename T> void RegisterParameter(const std::string& name);
    void MarkLoss(const std::string& name, const std::string& glslType);
    template<typename T> void MarkLoss(const std::string& name);

    GradientTape& Tape();
};
```

### Free Functions (ADKernel1D only)

```cpp
// Mark a Var<T> as a parameter. Returns parameter index (0, 1, 2, ...)
template<typename T> int AD::Param(const Var<T>& var);

// Mark a Var<T> as the scalar loss
template<typename T> void AD::Loss(const Var<T>& var);
```

### AdjointInspector1D / 2D / 3D

```cpp
template<typename Func>
class AdjointInspector1D {
public:
    AdjointInspector1D(Func&& func, int workSizeX = 256);
    std::string GetForwardCode() const;
    std::string GetBackwardCode() const;
    std::string GetTapeSummary() const;
    void        PrintTape();
    bool        HasBackwardCode() const;
    const GradientTape& Tape() const;
};

// AdjointInspector2D, AdjointInspector3D — same interface
```

### ADKernel1D

```cpp
class ADKernel1D {
public:
    template<typename Func>
    ADKernel1D(Func&& func, size_t elementCount, int groupSize = 256);

    void Forward(int groupCount, bool sync = false);
    void Backward(int groupCount, bool sync = false);
    std::vector<float> Gradient(int paramIndex) const;
    std::vector<float> Gradient(const std::string& paramVarName) const;

    /** Batch-download all parameter gradients efficiently.
     *  Shared gradient buffers are downloaded once and cached,
     *  avoiding redundant transfers for interleaved groups. */
    std::vector<std::vector<float>> DownloadAllGradients() const;

    std::string ForwardCode() const;
    std::string CombinedCode() const;
    const GradientTape& Tape() const;
    size_t ParameterCount() const;
};
```

### AdjointKernel1D / 2D / 3D

```cpp
template<typename Func>
class AdjointKernel1D {
public:
    AdjointKernel1D(Func&& func, int workSizeX = 256);
    std::string GetForwardCode() const;
    std::string GetCombinedCode() const;
    std::string GetBackwardBodyCode() const;
    const GradientTape& Tape() const;
    bool HasCombinedCode() const;
};

// AdjointKernel2D(Func&&, int workSizeX = 16, int workSizeY = 16)
// AdjointKernel3D(Func&&, int workSizeX = 8, int workSizeY = 8, int workSizeZ = 4)
```

### GradientTape (Low-Level)

```cpp
class GradientTape {
public:
    void Record(const GPU::IR::Node::Node& node, bool isStatement);

    // Parameters & loss
    void RegisterParameter(const std::string& name, const std::string& glslType);
    bool IsParameter(const std::string& name) const;
    void MarkLoss(const std::string& name, const std::string& glslType);
    const std::optional<TapeVar>& LossVar() const;

    // Queries
    size_t Size() const;
    const TapeEntry& operator[](int32_t i) const;
    bool IsActive(const std::string& name) const;
    const std::string* GetVarType(const std::string& name) const;
    size_t ParameterCount() const;

    // Control flow
    void BeginIfBranch(const std::string& cond);
    void BeginElifBranch(const std::string& cond);
    void BeginElseBranch();
    void EndIfChain();
    void BeginForLoop(const std::string& var, const std::string& start,
                      const std::string& end, const std::string& step);
    void EndForLoop();

    // Sub-tapes
    void PushSubTape();
    int  PopSubTape();
    size_t SubTapeCount() const;
    const GradientTape& SubTape(int i) const;
};
```

### TapeEntry (Low-Level)

```cpp
enum class TapeOpKind : uint8_t {
    BinaryOp, UnaryOp, Intrinsic1, Intrinsic2, Intrinsic3,
    Ternary, CompoundAssign, Call, Return,
    ControlFlowBegin, ControlFlowEnd, Loss
};

struct TapeEntry {
    int32_t id;
    TapeOpKind kind;
    TapeVar output;
    std::vector<TapeVar> inputs;
    OperationCode binaryOp;           // for BinaryOp/UnaryOp
    CompoundAssignmentCode compoundOp; // for CompoundAssign
    std::string intrinsicName;         // for Intrinsic1/2/3
    std::string callableFuncName;      // for Call
    ControlFlowKind controlFlowKind;   // for ControlFlowBegin
    std::string conditionVarName;      // if/elif condition
    std::string forVarName;            // loop variable
    std::string forStart, forEnd, forStep; // loop bounds
};
```

### AdjointGenerator (Low-Level)

```cpp
class AdjointGenerator {
public:
    std::string Generate(const GradientTape& tape, bool writeBackParams = true);
    AdjointBody GenerateBody(const GradientTape& tape, bool writeBackParams = true);
    const AdjointTable& GetAdjointTable() const;
};

struct AdjointBody {
    std::vector<std::pair<std::string, std::string>> declarations;  // (name, type)
    std::vector<std::string> lines;                                  // body statements
    std::vector<std::pair<std::string, std::string>> writebacks;     // (param, adj)
    std::string callableAdjointFunctions;
};
```

---

## See Also

- [Getting Started](getting-started.md) — First GPU program with EasyGPU
- [Tutorial](tutorial.md) — Complete walkthrough
- [Common Patterns](patterns.md) — Unref, Select, Callables, and more
- [API Reference](api-reference.md) — Full API documentation
- [FAQ](faq.md) — Troubleshooting and common questions
