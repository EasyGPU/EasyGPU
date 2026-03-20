# EasyGPU Window Component Proposal

## Goal

Integrate a lightweight window module into EasyGPU so examples, tools, and interactive compute applications can open a native window without forcing users to pull in a separate framework.

The window layer should feel like a natural extension of EasyGPU:

- small and easy to embed
- optional at build time
- usable from both OpenGL and Vulkan projects
- simple enough for beginners
- structured enough for long-term industrial maintenance

This proposal assumes `minifb` is used as a vendor base or reference implementation, but the final module should expose an EasyGPU-owned API and lifecycle instead of leaking `minifb` details into user code.

## Why This Is Worth Doing

EasyGPU already solves kernel authoring, resource management, and backend abstraction. What is still missing is the last mile:

- opening a window
- presenting images or compute output
- handling resize and input
- running a basic app loop

Without that layer, the first interactive program still feels incomplete. Users must glue EasyGPU to another library just to show a texture or a pixel buffer. That creates friction in examples, tutorials, experiments, and small tools.

Adding a window module improves:

- onboarding
- demo quality
- example readability
- portability of sample applications
- long-term positioning of EasyGPU as a practical compute toolkit

## Product Positioning

This should not become a GUI framework.

The target is a lightweight application shell for GPU compute visualization:

- native window creation
- event polling
- keyboard and mouse input
- framebuffer presentation
- optional backend-aware display path

This means:

- yes to simple viewers, toy apps, debug tools, and compute demos
- no to widgets, docking, retained UI trees, or a full scene framework

## Recommended Scope

### Phase A: Minimal Window Layer

Deliver a small, reliable window abstraction:

- create/destroy window
- set title
- query size
- resize callback
- close request
- keyboard and mouse state
- frame loop helpers

### Phase B: CPU Framebuffer Presentation

Provide a portable "just show pixels" path:

- upload `RGBA8` CPU image to window
- blit to screen
- resize-safe present path

This is the easiest path to adoption and the best first milestone.

### Phase C: EasyGPU Texture Presentation

Add a backend-aware path that presents EasyGPU textures:

- copy GPU texture to staging/display buffer
- present to window
- optional fast path when backend/platform permits

This keeps the API simple while allowing better performance later.

### Phase D: Power Features

Only after the core is stable:

- vsync control
- frame pacing
- clipboard
- drag and drop
- raw mouse input
- IMGUI hosting hooks

## Non-Goals

- full GUI toolkit
- swapchain abstraction for arbitrary rendering engines
- deep OS-specific window customization
- game engine style scene graph
- mandatory dependency for headless EasyGPU builds

## Feasibility of Using minifb

## Short Answer

Yes, with constraints.

`minifb` is a good fit if used as a low-level platform shim rather than as the public API.

## Why It Fits

- tiny footprint
- straightforward window/event model
- cross-platform support
- CPU framebuffer centric, which aligns with an easy first milestone
- easier to vendor than GLFW or SDL

## Why It Should Not Leak Into the Public API

- it is not designed as a backend-aware GPU presentation framework
- its public surface is narrower than what EasyGPU may want later
- direct exposure would freeze EasyGPU to `minifb`'s design choices
- deeper local modifications would make upstream sync painful

## Recommendation

Use one of these two strategies:

### Preferred: Vendor as a Private Backend

- place `minifb` under `third/minifb`
- wrap it with EasyGPU-owned classes
- avoid exposing `minifb` headers in public includes
- keep local patches minimal and documented

### Acceptable: Fork and Rename Internal Layer

If more invasive platform work is needed:

- fork a pinned version
- treat it as `EasyGPUWindowPlatform`
- isolate platform-facing code from EasyGPU-facing API

This is more maintainable than gradually mutating upstream files without boundaries.

## Proposed Module Layout

```text
include/
  Window/
    Window.h
    WindowConfig.h
    WindowEvents.h
    PixelBuffer.h
    TexturePresenter.h
    Input.h

source/
  Window/
    Window.cpp
    PixelBuffer.cpp
    TexturePresenter.cpp
    Platform/
      MiniFBWindowPlatform.cpp
      MiniFBWindowPlatform.h

third/
  minifb/
    ...
```

Optional future split:

```text
include/
  Window/
    VulkanPresent.h
    OpenGLPresent.h
```

Only add backend-specific headers if they provide clear value. The default public API should stay backend-neutral.

## Build System Design

Add new CMake options:

```cmake
option(EASYGPU_BUILD_WINDOW "Build EasyGPU window component" ON)
option(EASYGPU_WINDOW_VENDOR_MINIFB "Use bundled minifb backend" ON)
option(EASYGPU_WINDOW_BUILD_EXAMPLES "Build window examples" ON)
```

Recommended target layout:

- keep `EasyGPU` as the core library
- add `EasyGPUWindow` as a separate static library
- link `EasyGPUWindow` to `EasyGPU`
- make window support optional

Example:

```cmake
add_library(EasyGPUWindow STATIC
    include/Window/Window.h
    include/Window/WindowConfig.h
    include/Window/WindowEvents.h
    include/Window/PixelBuffer.h
    include/Window/TexturePresenter.h
    include/Window/Input.h
    source/Window/Window.cpp
    source/Window/PixelBuffer.cpp
    source/Window/TexturePresenter.cpp
    source/Window/Platform/MiniFBWindowPlatform.cpp
)

target_link_libraries(EasyGPUWindow PUBLIC EasyGPU)
target_include_directories(EasyGPUWindow PUBLIC include)
```

Platform linkage should stay private to `EasyGPUWindow`.

On Linux, the window target may need:

- `X11`
- `Xcursor`
- `Xrandr`
- `Xinerama`
- `Xi`
- `pthread`

depending on how the vendored layer is structured.

## Public API Design

The API should feel modern but not overengineered.

## Core Types

### `WindowConfig`

```cpp
namespace GPU::Window {

struct WindowConfig {
    uint32_t width = 1280;
    uint32_t height = 720;
    std::string title = "EasyGPU";
    bool resizable = true;
    bool visible = true;
    bool vsync = true;
    bool highDPI = true;
    bool centerOnCreate = true;
};

}
```

### `Key`, `MouseButton`, `ModifierFlags`

```cpp
namespace GPU::Window {

enum class Key {
    Unknown,
    Escape,
    Enter,
    Space,
    Left,
    Right,
    Up,
    Down,
    A, B, C, D,
    F1, F2, F3,
    // ...
};

enum class MouseButton {
    Left,
    Right,
    Middle,
    Button4,
    Button5
};

enum class ModifierFlags : uint32_t {
    None  = 0,
    Shift = 1 << 0,
    Ctrl  = 1 << 1,
    Alt   = 1 << 2,
    Super = 1 << 3
};

}
```

### `WindowEvent`

```cpp
namespace GPU::Window {

struct WindowResizeEvent {
    uint32_t width;
    uint32_t height;
};

struct WindowCloseEvent {};

struct KeyEvent {
    Key key;
    bool pressed;
    ModifierFlags modifiers;
};

struct MouseButtonEvent {
    MouseButton button;
    bool pressed;
    int32_t x;
    int32_t y;
    ModifierFlags modifiers;
};

struct MouseMoveEvent {
    int32_t x;
    int32_t y;
    int32_t dx;
    int32_t dy;
};

struct ScrollEvent {
    float dx;
    float dy;
};

using WindowEvent = std::variant<
    WindowResizeEvent,
    WindowCloseEvent,
    KeyEvent,
    MouseButtonEvent,
    MouseMoveEvent,
    ScrollEvent>;

}
```

### `PixelBuffer`

`PixelBuffer` is the easiest bridge between compute output and on-screen display.

```cpp
namespace GPU::Window {

class PixelBuffer {
public:
    PixelBuffer(uint32_t width, uint32_t height);

    uint32_t Width() const noexcept;
    uint32_t Height() const noexcept;

    uint32_t* Data() noexcept;
    const uint32_t* Data() const noexcept;

    void Resize(uint32_t width, uint32_t height);
    void Clear(uint32_t rgba);
};

}
```

Use packed `RGBA8` or `BGRA8` internally, but define it explicitly in the docs and implementation. Do not leave channel order ambiguous.

### `Window`

```cpp
namespace GPU::Window {

class Window {
public:
    explicit Window(const WindowConfig& config = {});
    ~Window();

    Window(Window&&) noexcept;
    Window& operator=(Window&&) noexcept;

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool IsOpen() const noexcept;
    void Close();

    uint32_t Width() const noexcept;
    uint32_t Height() const noexcept;
    float Aspect() const noexcept;

    void SetTitle(std::string title);
    void SetVSync(bool enabled);
    void SetResizable(bool enabled);

    void PollEvents();
    bool PollEvent(WindowEvent& event);
    void WaitEvents();

    bool IsKeyDown(Key key) const;
    bool IsMouseDown(MouseButton button) const;
    std::pair<int32_t, int32_t> MousePosition() const noexcept;

    void Present(const PixelBuffer& buffer);

    void SetResizeCallback(std::function<void(uint32_t, uint32_t)> callback);
    void SetCloseCallback(std::function<void()> callback);
};

}
```

This should be the first public API milestone.

## EasyGPU-Aware Presentation API

Once CPU blit is stable, add a high-level presenter for GPU resources.

### `TexturePresenter`

```cpp
namespace GPU::Window {

enum class PresentMode {
    CopyToCPU,
    Auto
};

class TexturePresenter {
public:
    explicit TexturePresenter(Window& window);
    ~TexturePresenter();

    void Present(Texture2D& texture);
    void Present(Texture2D& texture, PresentMode mode);

    void Present(Buffer<uint32_t>& rgbaBuffer, uint32_t width, uint32_t height);
};

}
```

Behavior:

- `Texture2D` present should work even if the backend is OpenGL or Vulkan
- first implementation may copy through CPU memory
- later implementation can add backend-specific fast paths behind the same API

This keeps the API stable while the internals evolve.

## Optional Convenience API

These are helpful, but should not block the first release.

### `Run` Helper

```cpp
namespace GPU::Window {

template <typename FrameFunc>
void Run(Window& window, FrameFunc&& frame);

}
```

Usage:

```cpp
Window win({.width = 800, .height = 600, .title = "Julia"});

GPU::Window::Run(win, [&] {
    kernel.Dispatch(...);
    presenter.Present(texture);
});
```

### `DisplaySurface`

If you want a richer abstraction later:

```cpp
class DisplaySurface {
public:
    virtual ~DisplaySurface() = default;
    virtual void Resize(uint32_t width, uint32_t height) = 0;
    virtual void Present() = 0;
};
```

This is useful only if multiple presentation backends appear. Do not introduce it too early unless it reduces code complexity immediately.

## Backend Strategy

The window system should be backend-neutral above the platform layer.

### Always Supported

- native window creation
- input
- CPU framebuffer present

### Supported Through EasyGPU Integration

- present compute results from `Texture2D`
- present CPU `Buffer<uint32_t>`

### Backend-Specific Internals

#### OpenGL

- optional textured fullscreen blit
- potential direct upload using existing GL context

#### Vulkan

- copy image to staging buffer
- convert to CPU-visible pixel buffer
- present through platform window path

Important: do not make the first implementation depend on Vulkan swapchain integration. That sounds attractive, but it is much more expensive than a generic display path and is not required for the first industrially useful release.

## Internal Architecture

## Layer 1: Public API

Files:

- `Window.h`
- `WindowEvents.h`
- `PixelBuffer.h`
- `TexturePresenter.h`

Responsibilities:

- stable user-facing types
- no platform headers
- no direct dependency leaks from `minifb`

## Layer 2: Platform Abstraction

Example internal interface:

```cpp
class IWindowPlatform {
public:
    virtual ~IWindowPlatform() = default;

    virtual bool IsOpen() const = 0;
    virtual void Close() = 0;
    virtual uint32_t Width() const = 0;
    virtual uint32_t Height() const = 0;
    virtual void SetTitle(const std::string& title) = 0;
    virtual void PollEvents() = 0;
    virtual bool PollEvent(GPU::Window::WindowEvent& event) = 0;
    virtual void PresentRGBA8(const uint32_t* pixels, uint32_t width, uint32_t height) = 0;
};
```

Responsibilities:

- translate native input/events
- own OS handles
- own vendored `minifb` interaction

## Layer 3: EasyGPU Integration

Files:

- `TexturePresenter.cpp`

Responsibilities:

- download texture/buffer content
- convert formats if required
- manage staging/readback buffers
- keep compute-side and display-side logic separate

This split matters. It prevents the platform layer from becoming polluted with GPU backend logic.

## API Semantics and Rules

These rules should be explicit in the docs and tests.

### Ownership

- `Window` owns the native window
- `TexturePresenter` does not own `Window`
- `TexturePresenter` may allocate reusable staging resources

### Threading

Initial version should be single-thread affinity:

- all `Window` calls must happen on the creating thread
- event polling and present must happen on that thread

This matches typical platform rules and keeps the model simple.

### Resize

- resize events should be delivered after internal size state updates
- `Present()` must validate input dimensions against current window state
- scaling behavior should be defined explicitly

Recommended default:

- preserve input resolution
- center content
- nearest or bilinear scale selected by policy later

### Pixel Format

Pick one canonical display format for version 1:

- `RGBA8_UNORM`

Any unsupported source format should be converted before present.

### Failure Model

Use the same general error style as the rest of EasyGPU:

- programmer misuse -> throw descriptive `std::runtime_error`
- unsupported platform/backend path -> throw explicit unsupported error
- silent fallback only when behavior stays correct and obvious

## Concrete Implementation Plan

## Milestone 1: Infrastructure

- add `EASYGPU_BUILD_WINDOW`
- vendor `minifb`
- create `EasyGPUWindow` target
- add platform compile branches for Windows and Linux
- keep all platform specifics private

Deliverable:

- empty window can open and close

## Milestone 2: Event Model

- define `WindowEvent`
- implement polling
- map key/mouse enums
- add resize and close callbacks

Deliverable:

- input test app

## Milestone 3: CPU Presentation

- add `PixelBuffer`
- implement `Window::Present(const PixelBuffer&)`
- handle resize safely
- document channel order

Deliverable:

- Mandelbrot or plasma example with CPU framebuffer

## Milestone 4: EasyGPU Texture Presentation

- add `TexturePresenter`
- support `Texture2D` readback
- support `Buffer<uint32_t>` display
- cache staging/readback allocations

Deliverable:

- compute-to-window example using EasyGPU textures

## Milestone 5: Industrial Hardening

- add tests for resize, event pump, and presentation edge cases
- validate teardown order
- document thread model and unsupported cases
- test on Windows and Linux

Deliverable:

- shipping-quality window module for examples and small tools

## Required Changes to EasyGPU

## CMake

Add:

- `EASYGPU_BUILD_WINDOW`
- `EasyGPUWindow` target
- vendored `minifb` source integration
- platform-specific linkage

## Public Headers

Add:

- `include/Window/*`
- umbrella include path updates if EasyGPU uses a central `GPU.h`

## Core Runtime

Prefer not to couple window creation into `Context` or backend init.

The window module should depend on EasyGPU, not the reverse.

This is important because:

- headless compute must remain clean
- test environments may not have display access
- backend initialization and platform windowing are separate concerns

## Examples

Add:

- `examples/window_hello.cpp`
- `examples/window_pixels.cpp`
- `examples/window_texture.cpp`

These examples should demonstrate:

- no EasyGPU usage
- CPU framebuffer display
- compute texture display

## Tests

Add at least:

- event translation tests
- pixel format conversion tests
- resize handling tests
- texture presenter validation tests

On CI, headless platforms may require these tests to be optional or smoke-only.

## API Examples

## Example 1: Minimal Window

```cpp
#include <Window/Window.h>

int main() {
    GPU::Window::Window window({
        .width = 800,
        .height = 600,
        .title = "EasyGPU Window"
    });

    while (window.IsOpen()) {
        window.PollEvents();
    }
}
```

## Example 2: CPU Framebuffer

```cpp
#include <Window/Window.h>
#include <Window/PixelBuffer.h>

int main() {
    using namespace GPU::Window;

    Window window({.width = 800, .height = 600, .title = "Pixels"});
    PixelBuffer pixels(800, 600);

    while (window.IsOpen()) {
        window.PollEvents();

        for (uint32_t y = 0; y < pixels.Height(); ++y) {
            for (uint32_t x = 0; x < pixels.Width(); ++x) {
                bool checker = ((x / 16) ^ (y / 16)) & 1;
                pixels.Data()[y * pixels.Width() + x] = checker ? 0xFFFFCC44u : 0xFF202030u;
            }
        }

        window.Present(pixels);
    }
}
```

## Example 3: EasyGPU Compute Texture

```cpp
#include <GPU.h>
#include <Window/Window.h>
#include <Window/TexturePresenter.h>

int main() {
    using namespace GPU;
    using namespace GPU::Window;

    Texture2D image(1024, 1024, PixelFormat::RGBA32F);

    Kernel2D fill([&](Int x, Int y) {
        auto tex = image.Bind();
        tex[Int2(x, y)] = Vec4(Float(x) / 1024.0f, Float(y) / 1024.0f, 0.25f, 1.0f);
    });

    Window window({.width = 1024, .height = 1024, .title = "EasyGPU Viewer"});
    TexturePresenter presenter(window);

    while (window.IsOpen()) {
        window.PollEvents();
        fill.Dispatch(1024, 1024, true);
        presenter.Present(image);
    }
}
```

## Industrial Design Principles

To keep this maintainable, follow these rules.

### 1. Public API First, Vendor Details Hidden

Users should depend on EasyGPU abstractions, not `minifb`.

### 2. Stable Semantics Before Fast Paths

Correctness and predictable lifecycle matter more than immediate zero-copy presentation.

### 3. Headless Remains First-Class

The window module must remain optional.

### 4. Separate Platform and GPU Responsibilities

Do not bury Vulkan/OpenGL logic inside platform event code.

### 5. Optimize After Observability

Before adding backend-specific fast present paths, add:

- clear logging hooks if EasyGPU has a debug facility later
- explicit error messages
- tests for resize, teardown, and format conversion

## Risks and Mitigations

## Risk: `minifb` Is Too CPU-Centric

Impact:

- limits direct GPU present strategies

Mitigation:

- keep public API generic
- start with CPU present
- add backend-specific fast path under `TexturePresenter`

## Risk: Linux Platform Surface Complexity

Impact:

- extra X11 dependency and maintenance burden

Mitigation:

- keep Linux window code isolated
- document build dependencies clearly
- test Windows and Linux from the first milestone

## Risk: API Overreach

Impact:

- window layer becomes a second framework

Mitigation:

- define non-goals early
- require strong justification before adding more than presentation and input

## Risk: Tight Coupling to Current EasyGPU Texture APIs

Impact:

- future backend evolution becomes harder

Mitigation:

- isolate texture readback/present logic in `TexturePresenter`
- avoid teaching `Texture2D` about windows directly

## Recommended Documentation Additions

When implementation starts, add:

- `docs/window.md` for user-facing usage
- README feature bullet for native window support
- getting started section for interactive examples
- CMake documentation for `EASYGPU_BUILD_WINDOW`

This proposal document can later be split into:

- architecture decision record
- user guide
- implementation checklist

## Industrial References

These projects are worth studying for design patterns, not for wholesale copying.

### GLFW

Useful for:

- minimal but stable public API
- strong cross-platform boundaries
- practical event and input model

Takeaway:

Public window APIs should stay boring and predictable.

### SDL2 / SDL3

Useful for:

- platform abstraction discipline
- event system design
- long-term maintenance lessons

Takeaway:

Separate platform layer from app-facing API very aggressively.

### Dear ImGui Backends

Useful for:

- backend-platform split
- incremental integration model
- keeping platform and renderer responsibilities separate

Takeaway:

Windowing and rendering should remain composable, not fused.

### bgfx Examples

Useful for:

- cross-backend sample application structure
- practical frame loop organization

Takeaway:

Examples matter almost as much as the API.

### wgpu + winit

Useful for:

- clean separation between GPU abstraction and window/event management
- strong ownership boundaries

Takeaway:

Do not force the core GPU runtime to own window creation.

## Final Recommendation

Yes, this is worth building.

The best engineering path is:

1. vendor `minifb` as a private platform layer
2. expose an EasyGPU-owned `Window` API
3. ship CPU framebuffer presentation first
4. add `TexturePresenter` for EasyGPU resource display
5. keep the module optional and decoupled from the core runtime

If you follow that path, EasyGPU gains a highly practical feature without turning into a bloated engine.

## Proposed First Delivery Checklist

- add `EasyGPUWindow` target
- vendor `minifb`
- implement `WindowConfig`, `WindowEvent`, `Window`
- implement `PixelBuffer`
- implement `Window::Present(PixelBuffer)`
- add one CPU example
- add one EasyGPU texture viewer example
- document build options
- test on Windows and Linux

That would already be a strong, industrially credible first release of an EasyGPU window component.
