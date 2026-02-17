# API Reference

Complete reference for all EasyGPU classes and functions.

## Table of Contents

- [Core Types](#core-types)
- [Kernels](#kernels)
- [Buffers](#buffers)
- [Uniforms](#uniforms)
- [Variables and Expressions](#variables-and-expressions)
- [Control Flow](#control-flow)
- [Math Functions](#math-functions)
- [Vector Types](#vector-types)
- [Matrix Types](#matrix-types)
- [Callable](#callable)
- [Structs](#structs)
- [Textures](#textures)
- [PBO Async Transfer](#pbo-async-transfer)

---

## Core Types

### Type Aliases

```cpp
using Int   = Var<int>;      // 32-bit signed integer
using Float = Var<float>;    // 32-bit float
using Bool  = Var<bool>;     // Boolean

using Kernel1D = Kernel::Kernel1D;
using Kernel2D = Kernel::Kernel2D;
using Kernel3D = Kernel::Kernel3D;
```

---

## Kernels

### Kernel1D

1D compute kernel for parallel array processing.

```cpp
// Constructor
Kernel1D kernel(
    const std::function<void(Var<int>&)>& func,  // Kernel function
    int workSizeX = 256                           // Threads per work group
);

// With name
Kernel1D kernel(
    const std::string& name,
    const std::function<void(Var<int>&)>& func,
    int workSizeX = 256
);
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Dispatch(int groupX, bool sync = false)` | Execute kernel |
| `SetName(const std::string& name)` | Set kernel name |
| `GetName() const` | Get kernel name |
| `GetCode()` | Get generated GLSL code |

**Example:**

```cpp
Kernel1D kernel([](Int i) {
    data[i] = data[i] * 2;
}, 256);

kernel.Dispatch(100, true);  // 100 groups, wait for completion
```

### Kernel2D

2D compute kernel for image/grid processing.

```cpp
Kernel2D kernel(
    const std::function<void(Var<int>&, Var<int>&)>& func,  // (x, y)
    int workSizeX = 16,
    int workSizeY = 16
);
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Dispatch(int groupX, int groupY, bool sync = false)` | Execute kernel |

### Kernel3D

3D compute kernel for volume processing.

```cpp
Kernel3D kernel(
    const std::function<void(Var<int>&, Var<int>&, Var<int>&)>& func,  // (x, y, z)
    int workSizeX = 8,
    int workSizeY = 8,
    int workSizeZ = 4
);
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Dispatch(int groupX, int groupY, int groupZ, bool sync = false)` | Execute kernel |

### Inspector Kernels

For debugging - compiles but doesn't execute.

```cpp
InspectorKernel1D inspector([](Int i) { ... });
inspector.PrintCode();                    // Print GLSL
std::string code = inspector.GetCode();   // Get GLSL
bool ok = inspector.Compile();            // Test compilation
```

### Kernel Barriers

Synchronization within work groups:

```cpp
Kernel1D::WorkgroupBarrier();  // Synchronize threads in work group
Kernel1D::MemoryBarrier();     // Ensure memory writes are visible
Kernel1D::FullBarrier();       // Both barriers combined
```

---

## Buffers

### Buffer<T>

GPU buffer for data storage and transfer.

```cpp
template<typename T>
class Buffer;
```

**Constructors:**

| Constructor | Description |
|:------------|:------------|
| `Buffer(size_t count, BufferMode mode = BufferMode::ReadWrite)` | Allocate |
| `Buffer(const std::vector<T>& data, BufferMode mode = BufferMode::ReadWrite)` | Upload from vector |
| `Buffer(Buffer&& other)` | Move constructor |

**BufferMode:**

| Mode | Description |
|:-----|:------------|
| `BufferMode::Read` | Read-only on GPU |
| `BufferMode::Write` | Write-only on GPU |
| `BufferMode::ReadWrite` | Read-write on GPU (default) |

**Methods:**

| Method | Description |
|:-------|:------------|
| `Bind()` | Bind to current kernel (returns BufferRef) |
| `Upload(const T* data, size_t count)` | Upload data to GPU |
| `Upload(const std::vector<T>& data)` | Upload from vector |
| `Download(T* outData, size_t count)` | Download data from GPU |
| `Download(std::vector<T>& outData)` | Download to vector |
| `GetCount() const` | Get element count |
| `GetElementSize() const` | Get element size in bytes |
| `GetBufferSize() const` | Get total size in bytes |
| `GetHandle() const` | Get OpenGL buffer ID |

**Example:**

```cpp
Buffer<float> buf1(1024);                          // Allocate
Buffer<float> buf2(data);                          // Upload
Buffer<float> buf3(1024, BufferMode::Write);       // Write-only

// In kernel
Kernel1D kernel([](Int i) {
    auto b = buf1.Bind();
    b[i] = b[i] * 2;
});

// After kernel
buf1.Download(data);
```

---

## Uniforms

### Uniform<T>

Uniform variables for passing constants from CPU to GPU. Unlike captured values which are embedded directly into the generated GLSL code, uniforms are dynamically uploaded to the GPU at dispatch time, allowing you to change values between kernel executions without recompiling.

```cpp
template<typename T>
class Uniform;
```

**Supported Types:**

| C++ Type | GLSL Type | Description |
|:---------|:----------|:------------|
| `float` | `float` | 32-bit floating point |
| `int` | `int` | 32-bit signed integer |
| `bool` | `bool` | Boolean value |
| `Math::Vec2` | `vec2` | 2-component float vector |
| `Math::Vec3` | `vec3` | 3-component float vector |
| `Math::Vec4` | `vec4` | 4-component float vector |
| `Math::IVec2` | `ivec2` | 2-component int vector |
| `Math::IVec3` | `ivec3` | 3-component int vector |
| `Math::IVec4` | `ivec4` | 4-component int vector |
| `Math::Mat2` | `mat2` | 2x2 float matrix |
| `Math::Mat3` | `mat3` | 3x3 float matrix |
| `Math::Mat4` | `mat4` | 4x4 float matrix |
| `Math::Mat2x3` | `mat2x3` | 2 columns, 3 rows |
| `Math::Mat2x4` | `mat2x4` | 2 columns, 4 rows |
| `Math::Mat3x2` | `mat3x2` | 3 columns, 2 rows |
| `Math::Mat3x4` | `mat3x4` | 3 columns, 4 rows |
| `Math::Mat4x2` | `mat4x2` | 4 columns, 2 rows |
| `Math::Mat4x3` | `mat4x3` | 4 columns, 3 rows |

**Constructors:**

| Constructor | Description |
|:------------|:------------|
| `Uniform()` | Default constructor - creates uninitialized uniform |
| `Uniform(T value)` | Constructor with initial value |
| `Uniform(const Uniform& other)` | Copy constructor |

**Methods:**

| Method | Description |
|:-------|:------------|
| `Load()` | Load the uniform in kernel context, returns `Var<T>` |
| `GetValue() const` | Get the current CPU-side value |
| `SetValue(T value)` | Set the CPU-side value |
| `operator=(T value)` | Assign value from literal |
| `operator=(const Uniform& other)` | Assign from another uniform |
| `operator T() const` | Implicit conversion to value type |

**Example:**

```cpp
// Create uniforms
Uniform<int> offset;
Uniform<float> scale(2.5f);

// Set values on CPU
offset = 100;
scale.SetValue(3.0f);

// Use in kernel
Buffer<float> data(1024);
Kernel1D kernel([&](Int i) {
    auto buf = data.Bind();
    auto off = offset.Load();    // Load uniform as Var<int>
    auto s = scale.Load();       // Load uniform as Var<float>
    buf[i] = (buf[i] + ToFloat(off)) * s;
});

// Dispatch with current uniform values
kernel.Dispatch(4, true);

// Change uniform values and dispatch again
offset = 200;
scale = 1.5f;
kernel.Dispatch(4, true);  // Uses new values without recompilation
```

**Multiple Uniforms:**

```cpp
Uniform<int> threshold;
Uniform<float> factor1;
Uniform<float> factor2;

threshold = 50;
factor1 = 0.5f;
factor2 = 2.0f;

Kernel1D kernel([&](Int i) {
    auto buf = buffer.Bind();
    auto t = threshold.Load();
    auto f1 = factor1.Load();
    auto f2 = factor2.Load();
    
    If(buf[i] > ToFloat(t), [&]() {
        buf[i] = buf[i] * f1;
    }).Else([&]() {
        buf[i] = buf[i] * f2;
    });
});
```

**Uniform<bool> for Conditional Logic:**

```cpp
Uniform<bool> enableFeature;
enableFeature = true;

Kernel1D kernel([&](Int i) {
    auto buf = buffer.Bind();
    auto enabled = enableFeature.Load();
    
    If(enabled, [&]() {
        buf[i] = Process(buf[i]);
    });
});

// Toggle feature off
enableFeature = false;
kernel.Dispatch(4, true);
```

---

## Variables and Expressions

### Var<T>

Mutable GPU variable.

```cpp
template<typename T>
class Var;

// Type aliases
using Int   = Var<int>;
using Float = Var<float>;
using Bool  = Var<bool>;
```

**Construction:**

```cpp
Int i;                          // Uninitialized
Int i = MakeInt(5);            // From literal
Int i = otherVar;              // Copy
```

> ⚠️ **CRITICAL: `Var` Initialization May Accidentally Create a Reference**
> 
> When initializing a `Var` from a buffer element, **always** use `Make*()` to ensure value semantics:
> 
> ```cpp
> auto buf = buffer.Bind();
> 
> // ✅ CORRECT: Explicitly create a new variable with a copy of the value
> Int val = MakeInt(buf[i]);
> val = 5;  // Only modifies val, NOT buf[i]
> 
> // ❌ DANGEROUS: Direct initialization may create a reference
> Int val = buf[i];
> val = 5;  // May unexpectedly modify buf[i] in the generated GLSL!
> ```
> 
> **Why this happens:**
> - `buf[i]` returns a temporary `Var<T>` (rvalue)
> - `Int val = buf[i]` selects the **move constructor** `VarBase(VarBase&&)`
> - The move transfers ownership of the underlying variable name (e.g., `"buffer[i]"`)
> - Result: `val` becomes an alias to `buffer[i]` in the generated shader
> 
> **Always use `Make*()`** to force creation of a new independent variable:
> ```cpp
> Int    val = MakeInt(buf[i]);
> Float  f   = MakeFloat(buf[i]);
> Float3 v   = MakeFloat3(buf[i]);
> ```

**Assignment:**

```cpp
Var<int> a = MakeInt(5);
Var<int> b = MakeInt(10);
a = b;        // Copy value
a = b + 5;    // Arithmetic result
```

### VarArray<Type, N>

Fixed-size array for GPU-local storage. Unlike `Buffer<T>` which resides in global GPU memory, `VarArray` creates a local array within the kernel (similar to `float arr[N]` in GLSL).

```cpp
template<ScalarType Type, int N>
class VarArray;
```

**Construction:**

```cpp
// Empty array (uninitialized)
VarArray<float, 10> localFloats;

// Initialized from CPU array
std::array<int, 5> cpuData = {1, 2, 3, 4, 5};
VarArray<int, 5> localInts(cpuData);
```

**Element Access:**

```cpp
VarArray<float, 10> arr;

// Index with literal
arr[0] = 5.0f;
Float val = arr[3];

// Index with Var<int>
Int idx = MakeInt(5);
arr[idx] = arr[idx] + 1.0f;

// Index with Expr<int>
For(0, 10, [&](Int& i) {
    arr[i] = arr[i] * 2.0f;  // Dynamic indexing
});
```

**Use Cases:**
- Local scratch space within a thread
- Small lookup tables
- Stencil computation buffers
- Sorting small arrays locally

**Comparison: Buffer vs VarArray**

| Feature | Buffer<T> | VarArray<Type, N> |
|:--------|:----------|:------------------|
| Memory | Global GPU memory | Local/thread-private memory |
| Lifetime | Survives kernel exit | Created/destroyed per thread |
| Size | Large (millions of elements) | Small (typically < 1000) |
| Access | All threads can access | Only owning thread can access |
| Persistence | Data persists between kernels | Data lost after kernel |
| Binding | Requires `.Bind()` | Created directly in kernel |

### Expr<T>

Immutable GPU expression (read-only).

```cpp
template<typename T>
class Expr;
```

Used for values that cannot be assigned to:

```cpp
Expr<float> e = a + b;  // Expression result
e = 5.0f;               // Error: Expr is read-only
```

### Constructors (Make)

**Important:** `Make` APIs wrap C++ literals into GPU `Var` types **without type conversion**. They are NOT the same as `Cast` APIs.

```cpp
// Helper functions to create GPU values from C++ literals
MakeInt(int value)           -> Var<int>      // Wrap int literal
MakeFloat(float value)       -> Var<float>    // Wrap float literal
MakeBool(bool value)         -> Var<bool>     // Wrap bool literal

MakeFloat3(float, float, float)  -> Var<Vec3>   // Wrap 3 floats
MakeFloat4(float, float, float, float) -> Var<Vec4>  // Wrap 4 floats
MakeInt2(int, int)           -> Var<IVec2>    // Wrap 2 ints
MakeInt3(int, int, int)      -> Var<IVec3>    // Wrap 3 ints
```

**Key difference:**
```cpp
// Make: No conversion, just wrapping
Float f = MakeFloat(3.14f);   // OK: float literal -> Var<float>
Float f = MakeFloat(42);      // ERROR: int literal, use MakeFloat(42.0f) or ToFloat(MakeInt(42))

// Cast: Type conversion
Int i = MakeInt(3);
Float f = ToFloat(i);         // OK: Var<int> -> Var<float> with conversion

**Implicit Conversions:**

```cpp
Var<int> accepts: Var<int>, int, Expr<int>
Expr<int> accepts: Expr<int>, Var<int>, int
```

---

## Control Flow

### If

Conditional execution.

```cpp
IfChain If(Expr<bool> condition, const std::function<void()>& body);
```

**Chaining:**

```cpp
If(condition1, [&]() {
    // if body
}).Elif(condition2, [&]() {
    // else if body
}).Else([&]() {
    // else body
});
```

### For

For loop with integer index.

```cpp
// Default step = 1
void For(Expr<int> start, Expr<int> end, 
         const std::function<void(Var<int>&)>& body);

// Explicit step
void For(Expr<int> start, Expr<int> end, Expr<int> step,
         const std::function<void(Var<int>&)>& body);
```

**Example:**

```cpp
For(0, 100, [&](Int& i) {
    // i ranges from 0 to 99
    data[i] = data[i] * 2;
});

// With step
For(0, 100, 2, [&](Int& i) {
    // i = 0, 2, 4, ..., 98
});
```

### While

While loop.

```cpp
void While(Expr<bool> condition, const std::function<void()>& body);
```

**Example:**

```cpp
Float x = MakeFloat(1.0f);
While(x < 100.0f, [&]() {
    x = x * 1.1f;
});
```

### DoWhile

Do-while loop.

```cpp
void DoWhile(const std::function<void()>& body, Expr<bool> condition);
```

### Break and Continue

```cpp
void Break();     // Exit current loop
void Continue();  // Skip to next iteration
```

**Example:**

```cpp
For(0, 100, [&](Int& i) {
    If(i % 2 == 0, [&]() {
        Continue();  // Skip even numbers
    });
    
    If(data[i] > threshold, [&]() {
        Break();  // Exit loop early
    });
});
```

### Return

Return from Callable.

```cpp
template<typename T>
void Return(Expr<T> value);
```

**Example:**

```cpp
Callable<float(float)> Square = [](Float& x) {
    Return(x * x);
};
```

---

## Math Functions

### Arithmetic

```cpp
// Built-in operators: +, -, *, /, %, - (negation)
// Built-in comparisons: ==, !=, <, >, <=, >=
// Built-in logical: &&, ||, !

Expr<T> Abs(Expr<T> x);           // Absolute value
Expr<T> Min(Expr<T> a, Expr<T> b); // Minimum
Expr<T> Max(Expr<T> a, Expr<T> b); // Maximum
Expr<T> Clamp(Expr<T> x, Expr<T> min, Expr<T> max); // Clamp to range
```

### Power and Roots

```cpp
Expr<float> Sqrt(Expr<float> x);      // Square root
Expr<float> Pow(Expr<float> x, Expr<float> y);  // x^y
Expr<float> Exp(Expr<float> x);       // e^x
Expr<float> Log(Expr<float> x);       // Natural log
Expr<float> Log2(Expr<float> x);      // Base-2 log
```

### Trigonometry

```cpp
Expr<float> Sin(Expr<float> x);   // Sine (radians)
Expr<float> Cos(Expr<float> x);   // Cosine (radians)
Expr<float> Tan(Expr<float> x);   // Tangent (radians)
Expr<float> Asin(Expr<float> x);  // Arcsine
Expr<float> Acos(Expr<float> x);  // Arccosine
Expr<float> Atan(Expr<float> x);  // Arctangent
Expr<float> Atan2(Expr<float> y, Expr<float> x);  // Arctangent(y/x)
```

### Type Conversion (Cast)

**Important:** Do not confuse `Cast` APIs with `Make` APIs. Cast APIs perform type **conversion** between `Var` types.

```cpp
Expr<float> ToFloat(Expr<int> x);   // Convert int to float (widening conversion)
Expr<int> ToInt(Expr<float> x);     // Convert float to int (truncate toward zero)
Expr<int> Round(Expr<float> x);     // Round to nearest int
Expr<int> Floor(Expr<float> x);     // Floor (round down)
Expr<int> Ceil(Expr<float> x);      // Ceiling (round up)
```

**Cast vs Make:**

| API | Purpose | Has Conversion Semantics |
|:----|:--------|:-------------------------|
| `ToFloat(Var<int>)` | Convert `Var<int>` to `Var<float>` | Yes (int → float conversion) |
| `MakeFloat(1.0f)` | Create `Var<float>` from literal | No (just wraps the value) |
| `ToInt(Var<float>)` | Convert `Var<float>` to `Var<int>` | Yes (truncation) |
| `MakeInt(5)` | Create `Var<int>` from literal | No (just wraps the value) |

### Vector Math

```cpp
Expr<float> Dot(Expr<Vec3> a, Expr<Vec3> b);       // Dot product
Expr<Vec3> Cross(Expr<Vec3> a, Expr<Vec3> b);      // Cross product
Expr<float> Length(Expr<Vec3> v);                  // Vector length
Expr<float> Length2(Expr<Vec3> v);                 // Squared length
Expr<Vec3> Normalize(Expr<Vec3> v);                // Normalize vector
Expr<Vec3> Reflect(Expr<Vec3> v, Expr<Vec3> n);    // Reflect vector
Expr<Vec3> Refract(Expr<Vec3> v, Expr<Vec3> n, Expr<float> eta);  // Refract
```

---

## Vector Types

### CPU Types (Host)

```cpp
struct Vec2 { float x, y; };
struct Vec3 { float x, y, z; };
struct Vec4 { float x, y, z, w; };

struct IVec2 { int x, y; };
struct IVec3 { int x, y, z; };
struct IVec4 { int x, y, z, w; };
```

**CPU Operations:**

```cpp
Vec3 a(1, 2, 3);
Vec3 b = a + Vec3(4, 5, 6);  // (5, 7, 9)
float d = a.Dot(b);
Vec3 n = a.Normalized();
```

### GPU Types (Device)

In kernels, use `Var<Vec3>`, `Var<Vec2>`, etc.

```cpp
// Construction
Float3 v = MakeFloat3(1.0f, 2.0f, 3.0f);
Float3 v = MakeFloat3(1.0f);  // (1, 1, 1)

// Component access
Float x = v.x();
Float y = v.y();
Float z = v.z();

// Swizzling
Float2 xy = v.xy();
Float2 yz = v.yz();

// Assignment
v.x() = 5.0f;
```

**GPU Operations:**

```cpp
Float3 a = MakeFloat3(1, 2, 3);
Float3 b = MakeFloat3(4, 5, 6);

Float3 c = a + b;      // Addition
Float3 c = a - b;      // Subtraction
Float3 c = a * 2.0f;   // Scalar multiplication
Float3 c = a / 2.0f;   // Scalar division
Float3 c = a * b;      // Component-wise multiplication

Float d = Dot(a, b);   // Dot product
Float3 c = Cross(a, b); // Cross product
Float len = Length(a);  // Length
Float3 n = Normalize(a); // Normalization
```

---

## Matrix Types

### CPU Types (Host)

```cpp
struct Mat2;   // 2x2 matrix
struct Mat3;   // 3x3 matrix
struct Mat4;   // 4x4 matrix
struct Mat2x3; // 2 columns, 3 rows
struct Mat2x4; // 2 columns, 4 rows
struct Mat3x2; // 3 columns, 2 rows
struct Mat3x4; // 3 columns, 4 rows
struct Mat4x2; // 4 columns, 2 rows
struct Mat4x3; // 4 columns, 3 rows
```

**CPU Construction:**

```cpp
// From columns
Mat4 m(
    Vec4(1, 0, 0, 0),  // Column 0
    Vec4(0, 1, 0, 0),  // Column 1
    Vec4(0, 0, 1, 0),  // Column 2
    Vec4(0, 0, 0, 1)   // Column 3
);

// Transform matrices
Mat4 translation = Mat4::Translate(Vec3(1, 2, 3));
Mat4 rotation = Mat4::Rotate(45.0f * 3.14159f / 180.0f, Vec3(0, 1, 0));
Mat4 scale = Mat4::Scale(Vec3(2, 2, 2));
Mat4 perspective = Mat4::Perspective(60.0f, 16.0f/9.0f, 0.1f, 100.0f);
Mat4 ortho = Mat4::Ortho(-1, 1, -1, 1, 0.1f, 100.0f);
```

**CPU Operations:**

```cpp
Mat4 a, b;
Mat4 c = a * b;        // Matrix multiplication
Vec4 v = a * Vec4(1, 2, 3, 1);  // Matrix-vector multiplication
Mat4 inv = a.Inverse();
Mat4 trans = a.Transpose();
```

### GPU Types (Device)

```cpp
// In kernels
Var<Mat4> m;
Float4 v = m * MakeFloat4(1, 2, 3, 1);
```

---

## Callable

Define reusable functions.

```cpp
template<typename Signature>
class Callable;

// Example: float(float, float)
Callable<float(float, float)> Add = [](Float& a, Float& b) {
    Return(a + b);
};
```

**Features:**
- Can be called from any kernel
- Supports reference parameters for output
- Can capture host values (constants)
- Supports recursion (limited)

**Reference Parameters:**

```cpp
Callable<void(float, float&)> GetComponents = [](Float& v, Float& x, Float& y) {
    x = v;
    y = v * 2;
};

// Usage in kernel
Float x, y;
GetComponents(value, x, y);
```

---

## Structs

### EASYGPU_STRUCT Macro

Define GPU-compatible structs.

```cpp
EASYGPU_STRUCT(Name,
    (Type1, field1),
    (Type2, field2),
    ...
);
```

**Supported Types:**
- `float`, `int`, `bool`
- `Vec2`, `Vec3`, `Vec4`
- `IVec2`, `IVec3`, `IVec4`
- `Mat2`, `Mat3`, `Mat4`, etc.
- Other registered structs

**Example:**

```cpp
EASYGPU_STRUCT(Particle,
    (Float3, position),
    (Float3, velocity),
    (float, mass)
);

// Use in buffer
Buffer<Particle> particles(1000);

// Access in kernel
Kernel1D update([](Int i) {
    auto p = particles.Bind();
    
    // Read
    Float3 pos = p[i].position();
    
    // Write
    p[i].position() = pos + velocity * dt;
});
```

### Nested Structs

```cpp
EASYGPU_STRUCT(Material,
    (Float3, albedo),
    (Float, roughness)
);

EASYGPU_STRUCT(Triangle,
    (Vec3, v0),
    (Vec3, v1),
    (Vec3, v2),
    (Material, mat)
);
```

---

## Textures

### Texture2D

2D texture for image data.

```cpp
Texture2D<PixelFormat::RGBA8> texture(width, height);
```

**PixelFormat:**

| Format | Description |
|:-------|:------------|
| `PixelFormat::R8` | Single channel, 8-bit |
| `PixelFormat::RG8` | Two channels, 8-bit each |
| `PixelFormat::RGBA8` | Four channels, 8-bit each |
| `PixelFormat::R32F` | Single channel, 32-bit float |
| `PixelFormat::RG32F` | Two channels, 32-bit float |
| `PixelFormat::RGBA32F` | Four channels, 32-bit float |
| `PixelFormat::R16F` | Single channel, 16-bit float |
| `PixelFormat::RG16F` | Two channels, 16-bit float |
| `PixelFormat::RGBA16F` | Four channels, 16-bit float |
| `PixelFormat::R32I` | Single channel, 32-bit signed int |
| `PixelFormat::RG32I` | Two channels, 32-bit signed int |
| `PixelFormat::RGBA32I` | Four channels, 32-bit signed int |
| `PixelFormat::R32UI` | Single channel, 32-bit unsigned int |
| `PixelFormat::RG32UI` | Two channels, 32-bit unsigned int |
| `PixelFormat::RGBA32UI` | Four channels, 32-bit unsigned int |

**Constructors:**

```cpp
Texture2D<PixelFormat::RGBA8> tex(width, height);              // Empty texture
Texture2D<PixelFormat::RGBA8> tex(width, height, data);        // With initial data
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Upload(const void* data)` | Upload pixel data to GPU (synchronous) |
| `UploadSubRegion(x, y, w, h, data)` | Upload partial data |
| `Download(void* outData)` | Download pixel data from GPU (synchronous) |
| `Download(std::vector<T>& outData)` | Download to vector |
| `Bind()` | Bind to current kernel (returns TextureRef) |
| `GetWidth()` | Get texture width |
| `GetHeight()` | Get texture height |
| `GetHandle()` | Get OpenGL texture ID |
| `GetSizeInBytes()` | Get total size in bytes |

**PBO Async Methods:**

| Method | Description |
|:-------|:------------|
| `InitUploadPBOPool(bufferCount)` | Initialize PBO pool for async upload (typically 2-3) |
| `InitDownloadPBOPool(bufferCount)` | Initialize PBO pool for async download |
| `UploadAsync(data)` | Asynchronous upload using PBO (non-blocking) |
| `UploadAsyncStream(data, timeoutMs)` | Async upload with blocking wait for idle PBO |
| `DownloadAsync()` | Start asynchronous download |
| `GetDownloadData(outData)` | Get data from completed async download |
| `Sync()` | Wait for all async operations to complete |
| `IsIdle()` | Check if all async operations are complete |

**Usage in Kernel:**

```cpp
Texture2D<PixelFormat::RGBA8> texture(1024, 1024);

Kernel2D kernel([&](Int x, Int y) {
    auto img = texture.Bind();
    
    // Read pixel
    Float4 color = img.Read(x, y);
    
    // Write pixel
    img.Write(x, y, color * 0.5f);
});

kernel.Dispatch(64, 64, true);
```

**Type Aliases:**

```cpp
using TextureRGBA8   = Texture2D<PixelFormat::RGBA8>;
using TextureRGBA32F = Texture2D<PixelFormat::RGBA32F>;
using TextureR32F    = Texture2D<PixelFormat::R32F>;
using TextureRG32F   = Texture2D<PixelFormat::RG32F>;
using TextureR8      = Texture2D<PixelFormat::R8>;
using image2d<Format> = IR::Value::TextureRef<Format>;  // Inside kernel
```

---

### Texture3D

3D texture for volume data.

```cpp
Texture3D<PixelFormat::RGBA8> volume(width, height, depth);
```

**Constructors:**

```cpp
Texture3D<PixelFormat::RGBA8> vol(width, height, depth);              // Empty volume
Texture3D<PixelFormat::RGBA8> vol(width, height, depth, data);        // With initial data
```

**Methods:**

| Method | Description |
|:-------|:------------|
| `Upload(const void* data)` | Upload voxel data to GPU (synchronous) |
| `UploadSubRegion(x, y, z, w, h, d, data)` | Upload partial data |
| `Download(void* outData)` | Download voxel data from GPU (synchronous) |

**PBO Async Methods:**

| Method | Description |
|:-------|:------------|
| `InitUploadPBOPool(bufferCount)` | Initialize PBO pool for async upload |
| `InitDownloadPBOPool(bufferCount)` | Initialize PBO pool for async download |
| `UploadAsync(data)` | Asynchronous upload using PBO |
| `UploadAsyncStream(data, timeoutMs)` | Async upload with blocking wait |
| `DownloadAsync()` | Start asynchronous download |
| `GetDownloadData(outData)` | Get data from completed download |
| `Sync()` | Wait for all async operations to complete |
| `IsIdle()` | Check if all async operations are complete |
| `Download(std::vector<T>& outData)` | Download to vector |
| `Bind()` | Bind to current kernel (returns Texture3DRef) |
| `GetWidth()` | Get volume width |
| `GetHeight()` | Get volume height |
| `GetDepth()` | Get volume depth |
| `GetHandle()` | Get OpenGL texture ID |
| `GetSizeInBytes()` | Get total size in bytes |

**Usage in Kernel:**

```cpp
Texture3D<PixelFormat::R32F> volume(256, 256, 256);

Kernel3D kernel([&](Int x, Int y, Int z) {
    auto vol = volume.Bind();
    
    // Read voxel
    Float4 value = vol.Read(x, y, z);
    
    // Write voxel
    vol.Write(x, y, z, value * 2.0f);
});

kernel.Dispatch(32, 32, 32, true);
```

**Type Aliases:**

```cpp
using Texture3DRGBA8   = Texture3D<PixelFormat::RGBA8>;
using Texture3DRGBA32F = Texture3D<PixelFormat::RGBA32F>;
using Texture3DR32F    = Texture3D<PixelFormat::R32F>;
using Texture3DRG32F   = Texture3D<PixelFormat::RG32F>;
using Texture3DR8      = Texture3D<PixelFormat::R8>;
using image3d<Format> = IR::Value::Texture3DRef<Format>;  // Inside kernel
```

---

### TextureRef (2D)

Reference to a 2D texture inside a kernel, returned by `Texture2D::Bind()`.

**Read Methods:**

```cpp
// All combinations of Var<int>, Expr<int>, and literal int
Float4 color = img.Read(x, y);
```

**Write Methods:**

```cpp
// All combinations of Var<int>, Expr<int>, literal int for coordinates
// and Var<Vec4>, Expr<Vec4> for color
img.Write(x, y, color);
```

---

### Texture3DRef (3D)

Reference to a 3D texture inside a kernel, returned by `Texture3D::Bind()`.

**Read Methods:**

```cpp
// All combinations of Var<int>, Expr<int>, and literal int
Float4 value = vol.Read(x, y, z);
```

**Write Methods:**

```cpp
// All combinations of Var<int>, Expr<int>, literal int for coordinates
// and Var<Vec4>, Expr<Vec4> for value
vol.Write(x, y, z, value);
```

---

## PBO Async Transfer

Pixel Buffer Objects (PBOs) enable asynchronous CPU/GPU data transfers, allowing CPU and GPU to work in parallel. This is essential for real-time applications like video streaming and interactive simulations.

### Overview

```
CPU Memory              GPU Memory
     │                       │
     │  Synchronous Upload   │
     │ ─────────────────────>│  CPU waits for GPU
     │                       │
     │  Async with PBO       │
     │ ─────────────────────>│  CPU continues immediately
     │ (non-blocking)        │  GPU copies in background
```

### Basic Async Upload

```cpp
Texture2D<PixelFormat::RGBA8> texture(1920, 1080);

// Initialize PBO pool with 2 buffers (double buffering)
texture.InitUploadPBOPool(2);

// Upload without blocking
std::vector<uint8_t> frame(1920 * 1080 * 4);
// ... fill frame data ...

texture.UploadAsync(frame.data());  // Returns immediately
// CPU can continue processing while GPU uploads

kernel.Dispatch(120, 68, true);
```

### Streaming Pattern

For continuous streaming (e.g., video playback), use `UploadAsyncStream` which blocks if no PBO is available:

```cpp
Texture2D<PixelFormat::RGBA8> videoFrame(1920, 1080);
videoFrame.InitUploadPBOPool(3);  // Triple buffering

Kernel2D processFrame([&](Int x, Int y) {
    auto frame = videoFrame.Bind();
    Float4 color = frame.Read(x, y);
    // Apply filter...
    frame.Write(x, y, filtered);
}, 16, 16);

// Stream loop
for (const auto& frameData : videoFrames) {
    // Upload blocks if all PBOs busy (waits up to 1000ms)
    videoFrame.UploadAsyncStream(frameData.data(), 1000);
    
    // Process while next frame uploads
    processFrame.Dispatch(120, 68, true);
}

// Wait for final upload
videoFrame.Sync();
```

### Async Download

Download GPU-computed results without blocking:

```cpp
Texture2D<PixelFormat::RGBA8> result(1024, 1024);
result.InitDownloadPBOPool(2);

// Render to texture
renderKernel.Dispatch(64, 64, true);

// Start async download (returns immediately)
result.DownloadAsync();

// Do other work while GPU prepares data...
otherKernel.Dispatch(32, 32, true);

// Try to get data
std::vector<uint8_t> pixels(1024 * 1024 * 4);
if (result.GetDownloadData(pixels.data())) {
    // Data ready
    SaveToFile(pixels);
} else {
    // Still pending, sync and retry
    result.Sync();
    result.GetDownloadData(pixels.data());
}
```

### Synchronization

| Method | Use When |
|:-------|:---------|
| `Sync()` | Need all operations complete before next step |
| `IsIdle()` | Polling to check if operations finished |
| Non-blocking | `UploadAsync` returns `false` if no PBO available |

```cpp
// Poll for completion
while (!texture.IsIdle()) {
    // Do other CPU work
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// Or block until complete
texture.Sync();
```

### Buffer Count Guidelines

| Count | Pattern | Use Case |
|:------|:--------|:---------|
| 1 | Single buffering | Simple async, may stall |
| 2 | Double buffering | Most common, good balance |
| 3+ | Triple buffering | High-latency tolerance, max throughput |

```cpp
// Double buffering: CPU fills one PBO while GPU uploads from another
texture.InitUploadPBOPool(2);

// Triple buffering: More tolerance for timing variations
texture.InitUploadPBOPool(3);
```

### Error Handling

```cpp
// UploadAsync returns false if no PBO available
if (!texture.UploadAsync(data)) {
    // Option 1: Sync and retry
    texture.Sync();
    texture.UploadAsync(data);  // Now succeeds
    
    // Option 2: Use streaming version (blocks until ready)
    // texture.UploadAsyncStream(data, timeoutMs);
}
```

---

## Error Handling

### ShaderException

Thrown on GPU errors.

```cpp
try {
    kernel.Dispatch(100, true);
} catch (const ShaderException& e) {
    std::cerr << "Shader error: " << e.what() << std::endl;
}
```

### Common Errors

| Error | Solution |
|:------|:---------|
| `Buffer::Bind() called outside of Kernel` | Move Bind() inside kernel lambda |
| `No active OpenGL context` | Check OpenGL support and drivers |
| `GLSL compilation failed` | Use `GetCode()` to debug generated code |
