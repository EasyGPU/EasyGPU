# Support Status

EasyGPU keeps its C++ kernel DSL backend-independent, but backend and platform capabilities are not identical. Production users should qualify the exact GPU, driver, and workload used by their application.

## Capability Levels

| Area | Status | Notes |
|:--|:--|:--|
| Compute DSL, buffers, textures, control flow | Stable | Covered by GPU-backed tests |
| Vulkan compute backend | Stable | Default backend; requires Vulkan SDK, glslang, and SPIRV-Tools |
| OpenGL compute backend | Stable | OpenGL 4.3+ on Windows and Linux |
| Vulkan graphics pipeline | Preview | Vulkan-only; API may still evolve |
| Automatic differentiation | Preview | Broad code-generation coverage; qualify numerical behavior per model |
| Neural-network helpers | Preview | Intended for small models and research workflows |
| UniformBuffer | Stable | Implemented as a read-only std430 storage buffer for cross-backend consistency |
| Installed CMake package | Stable | Exports `EasyGPU::EasyGPU` and optional `EasyGPU::Window` targets |

## Platform Notes

| Platform | Supported backend paths | Notes |
|:--|:--|:--|
| Windows | OpenGL, Vulkan | OpenGL is continuously built in CI |
| Linux | OpenGL, Vulkan | OpenGL GCC and Clang builds are continuously checked |
| macOS | Vulkan via MoltenVK | GPU timestamp profiling uses a synchronized CPU fallback |

## Verification Policy

- Every `tests/Test*.cpp` file, except the optional Windows EasyX fragment tester, is automatically registered with CTest.
- Release tests explicitly keep assertions enabled.
- The Vulkan Release suite currently contains 48 independently registered tests.
- The package-consumer fixture verifies installation, `find_package(EasyGPU)`, compilation, linking, and execution.
- Core-only packages are verified with `EASYGPU_BUILD_WINDOW=OFF`.
- Sanitizer builds are recommended when changing resource ownership or backend memory code.
- A successful build on one backend does not establish support on another backend.

## Runtime Guarantees

- Shader handles are released through exception-safe lifetime management during pipeline creation.
- Buffer and texture slots reject binding after the attached resource has been destroyed.
- `UniformBuffer<T>::GetValue()` returns a synchronized value copy rather than exposing a reference after releasing its lock.
- Vulkan pipeline-cache entries are used only as pipeline creation acceleration data; a valid shader module remains live throughout pipeline creation.
- Native GPU timestamp queries are used only when the backend reports reliable support. MoltenVK uses synchronized CPU timing as a stability fallback.

These guarantees reduce common lifetime and backend failure modes, but do not replace application-level validation on the target driver and workload.
