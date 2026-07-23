# Support Status

EasyGPU keeps its C++ kernel DSL backend-independent, but backend and platform capabilities are not identical. Production users should qualify the exact GPU, driver, and workload used by their application.

## Capability Levels

| Area | Status | Notes |
|:--|:--|:--|
| Compute DSL, buffers, textures, control flow | Stable | Covered by GPU-backed tests |
| Vulkan compute backend | Stable | Default backend; requires Vulkan SDK, glslang, and SPIRV-Tools |
| OpenGL compute backend | Stable | OpenGL 4.3+ on Windows and Linux |
| Vulkan SPIR-V optimization inspection | Stable | Vulkan-only; OpenGL accepts related APIs silently and returns empty inspection strings |
| Vulkan graphics pipeline | Preview | Vulkan-only; API may still evolve |
| GLFW window component | Stable | Optional `EasyGPU::Window` target; X11 backend on Linux |
| Dear ImGui overlay | Stable | Available through `UIContext` with Vulkan and Windows/Linux OpenGL window builds |
| Automatic differentiation | Preview | Broad code-generation coverage; qualify numerical behavior per model |
| Neural-network helpers | Preview | Intended for small models and research workflows |
| UniformBuffer | Stable | Implemented as a read-only std430 storage buffer for cross-backend consistency |
| Installed CMake package | Stable | Exports `EasyGPU::EasyGPU` and optional `EasyGPU::Window` targets |

## Platform Notes

| Platform | Supported backend paths | Notes |
|:--|:--|:--|
| Windows | OpenGL, Vulkan | Window + ImGui supported |
| Linux | OpenGL, Vulkan | Window + ImGui uses GLFW X11 backend |
| macOS | Vulkan via MoltenVK | Window + ImGui supported through Vulkan; OpenGL backend is intentionally disabled |

## Verification Policy

- Every `tests/Test*.cpp` file, except the optional Windows EasyX fragment tester, is automatically registered with CTest.
- Release tests explicitly keep assertions enabled.
- CTest enumerates all configured `tests/Test*.cpp` targets so the release suite grows with the repository.
- The package-consumer fixture verifies installation, `find_package(EasyGPU)`, compilation, linking, and execution.
- Core-only packages are verified with `EASYGPU_BUILD_WINDOW=OFF`.
- Sanitizer builds are recommended when changing resource ownership or backend memory code.
- A successful build on one backend does not establish support on another backend.

## Runtime Guarantees

- Shader handles are released through exception-safe lifetime management during pipeline creation.
- Buffer and texture slots reject binding after the attached resource has been destroyed.
- `UniformBuffer<T>::GetValue()` returns a synchronized value copy rather than exposing a reference after releasing its lock.
- Persistent Vulkan SPIR-V entries are version-keyed and validated before use; invalid entries fall back to source compilation.
- Vulkan pipeline-cache entries are used only as pipeline creation acceleration data; a valid shader module remains live throughout pipeline creation.
- Native GPU timestamp queries are used only when the backend reports reliable support. MoltenVK uses synchronized CPU timing as a stability fallback.

These guarantees reduce common lifetime and backend failure modes, but do not replace application-level validation on the target driver and workload.
