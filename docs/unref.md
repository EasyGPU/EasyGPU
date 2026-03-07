# Unref: Creating Independent Variable Copies

When working with GPU variables, particularly when reading from buffers, it's important to understand how variable initialization works in EasyGPU.

## The Problem

Due to C++ move semantics optimizations, direct initialization of a `Var` from a buffer element may create an alias rather than an independent copy:

```cpp
Kernel1D kernel([](Int i) {
    auto buf = buffer.Bind();
    
    // DANGEROUS: val may become a reference to buf[i]
    Int val = buf[i];
    val = 5;  // May unexpectedly modify buf[i] in the generated GLSL!
});
```

**Why this happens:**
- `buf[i]` returns a temporary `Var<T>` (an rvalue)
- `Int val = buf[i]` selects the **move constructor** `Var(Var&&)`, which transfers ownership of the underlying variable name
- Result: `val` directly references `buffer[i]` in the generated shader

## The Solution: Unref

Use `Unref()` to create an independent copy with value semantics:

```cpp
Kernel1D kernel([](Int i) {
    auto buf = buffer.Bind();
    
    // CORRECT: Creates a new independent variable
    Int val = Unref(buf[i]);
    val = 5;  // Only modifies val, NOT buf[i]
});
```

## How It Works

`Unref()` forces the use of the copy constructor instead of the move constructor. The copy constructor:
1. Allocates a new variable name in the GLSL shader
2. Generates an IR load/store pair to copy the value
3. Creates a truly independent variable

## Usage Examples

### Basic Buffer Access

```cpp
Kernel1D process([](Int i) {
    auto data = buffer.Bind();
    
    // Create independent copies
    Float current = Unref(data[i]);
    Float previous = Unref(data[i - 1]);
    
    // Modify without affecting the buffer
    current = current * 2.0f;
    previous = previous + 1.0f;
    
    // Write back explicitly when needed
    data[i] = current + previous;
});
```

### Working with Structs

```cpp
EASYGPU_STRUCT(Particle,
    (Float3, position),
    (Float3, velocity),
    (float, mass)
);

Kernel1D update([](Int i) {
    auto particles = buffer.Bind();
    
    // Create independent copy
    Particle p = Unref(particles[i]);
    
    // Modify local copy
    p.position() = p.position() + p.velocity() * dt;
    
    // Write back
    particles[i] = p;
});
```

### In Callables

```cpp
Callable<Float(Float, Float)> Lerp = [](Float& a, Float& b, Float& t) {
    Return(a + (b - a) * t);
};

Kernel1D interpolate([](Int i) {
    auto data = buffer.Bind();
    
    // Ensure independent copies for callable arguments
    Float v0 = Unref(data[i]);
    Float v1 = Unref(data[i + 1]);
    
    Float result = Lerp(v0, v1, 0.5f);
    data[i] = result;
});
```

## When to Use Unref

| Scenario | Use Unref? | Reason |
|:---------|:-----------|:-------|
| Reading from buffer into temp | Yes | Avoid aliasing the buffer element |
| Copying to another variable | Yes | Ensure independent copy |
| Passing to Callable by reference | Yes | Prevent unintended modifications |
| Creating local working copy | Yes | Safe local modifications |
| Direct buffer operations | No | `buf[i] = value` is fine |
| Reading once without storing | No | Direct use is safe |

## Comparison with Make Functions

Do not confuse `Unref()` with `Make*()` functions:

```cpp
// Unref: Creates independent copy from existing Var
Int a = Unref(buf[i]);        // Copy value from buffer element

// Make: Wraps C++ literal into new Var
Int b = MakeInt(42);          // Create from literal
Float c = MakeFloat(3.14f);   // Create from literal
```

| Function | Purpose | Input |
|:---------|:--------|:------|
| `Unref(var)` | Copy existing GPU variable | `Var<T>` or `Expr<T>` |
| `MakeInt(5)` | Create from C++ int literal | C++ literal |
| `MakeFloat(3.14f)` | Create from C++ float literal | C++ literal |

## Technical Details

### Template Signature

```cpp
template <typename T>
[[nodiscard]] Var<T> Unref(const Var<T>& var);

template <typename T>
[[nodiscard]] Var<T> Unref(Var<T>&& var);
```

### Return Value

Returns a new `Var<T>` that is guaranteed to be an independent variable with its own storage in the generated GLSL.

### Performance Notes

- `Unref()` generates an additional load/store pair in the IR
- This adds a variable declaration and assignment in the final GLSL
- The overhead is minimal for scalar types
- For large structs, consider whether a copy is truly necessary

## Best Practices

1. **Always use Unref when storing buffer elements to named variables:**
   ```cpp
   // Good
   Float val = Unref(buf[i]);
   
   // Risky - val may alias buf[i]
   Float val = buf[i];
   ```

2. **Use in arithmetic expressions is safe without Unref:**
   ```cpp
   // Safe - temporary is used directly
   Float result = buf[i] * 2.0f + 1.0f;
   ```

3. **Chain Unref for multiple reads:**
   ```cpp
   Float a = Unref(buf[i]);
   Float b = Unref(buf[i + 1]);
   Float c = Unref(buf[i + 2]);
   ```

4. **Document intent when not using Unref:**
   ```cpp
   // Intentional alias for in-place modification
   Float ref = buf[i];
   ref = ref * 2.0f;  // Modifies buf[i] by design
   ```
