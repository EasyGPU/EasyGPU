# EasyGPU Window Component

The EasyGPU Window component provides a lightweight, cross-platform windowing system for interactive GPU compute visualization. It is designed to be simple, modern, and seamlessly integrated with EasyGPU's compute capabilities.

## Overview

```cpp
#include <GPU.h>

int main() {
    using namespace GPU;
    
    // Create a window
    AppWindow window({
        .width = 1024,
        .height = 768,
        .title = "My App",
        .resizable = true,
        .vsync = true
    });
    
    // Main loop
    while (window.IsOpen()) {
        window.PollEvents();
        
        // Handle events
        WindowEvent event;
        while (window.PollEvent(event)) {
            if (std::holds_alternative<KeyEvent>(event)) {
                auto& key = std::get<KeyEvent>(event);
                if (key.key == Key::Escape && key.pressed) {
                    window.Close();
                }
            }
        }
        
        // Present pixel data
        PixelBuffer pixels(1024, 768);
        // ... fill pixels ...
        window.Present(pixels);
    }
}
```

## Features

- **Cross-platform**: Windows (Win32), Linux (X11), and macOS (Cocoa/CoreGraphics) support
- **Simple API**: Modern C++20 design with minimal boilerplate
- **Event-driven**: Keyboard, mouse, resize, and focus events
- **CPU Rendering**: `PixelBuffer` for software-rendered graphics
- **GPU Integration**: `TexturePresenter` for displaying GPU textures
- **Optional**: Can be disabled at build time if not needed

## API Reference

### AppWindow

The main window class for creating and managing application windows.

```cpp
class AppWindow {
public:
    // Construction
    explicit AppWindow(const WindowConfig& config = {});
    ~AppWindow();
    
    // Window state
    [[nodiscard]] bool IsOpen() const noexcept;
    void Close();
    
    // Properties
    [[nodiscard]] uint32_t Width() const noexcept;
    [[nodiscard]] uint32_t Height() const noexcept;
    [[nodiscard]] float Aspect() const noexcept;
    void SetTitle(const std::string& title);
    void SetVSync(bool enabled);
    
    // Events
    void PollEvents();
    bool PollEvent(WindowEvent& event);
    void WaitEvents();
    
    // Input state
    [[nodiscard]] bool IsKeyDown(Key key) const;
    [[nodiscard]] bool IsMouseDown(MouseButton button) const;
    [[nodiscard]] std::pair<int32_t, int32_t> MousePosition() const noexcept;
    [[nodiscard]] std::pair<float, float> MouseScroll() const noexcept;
    
    // Presentation
    void Present(const PixelBuffer& buffer);
    void Present(const uint32_t* pixels, uint32_t width, uint32_t height);
    
    // Callbacks
    void SetResizeCallback(std::function<void(uint32_t, uint32_t)> callback);
    void SetCloseCallback(std::function<bool()> callback);
    void SetFocusCallback(std::function<void(bool)> callback);
};
```

### WindowConfig

Configuration options for window creation.

```cpp
struct WindowConfig {
    uint32_t    width           = 1280;     // Window width in pixels
    uint32_t    height          = 720;      // Window height in pixels
    std::string title           = "EasyGPU"; // Window title
    bool        resizable       = true;     // Allow window resizing
    bool        visible         = true;     // Start visible
    bool        vsync           = true;     // Enable vertical sync
    bool        highDPI         = true;     // Enable high DPI support
    bool        centerOnCreate  = true;     // Center window on screen
};
```

### PixelBuffer

CPU-side RGBA8 pixel buffer for software rendering.

```cpp
class PixelBuffer {
public:
    // Construction
    PixelBuffer(uint32_t width, uint32_t height);
    PixelBuffer(uint32_t width, uint32_t height, const uint32_t* data);
    
    // Properties
    [[nodiscard]] uint32_t Width() const noexcept;
    [[nodiscard]] uint32_t Height() const noexcept;
    [[nodiscard]] uint32_t* Data() noexcept;
    [[nodiscard]] const uint32_t* Data() const noexcept;
    
    // Operations
    void Resize(uint32_t width, uint32_t height);
    void Clear(uint32_t rgba);
    void SetPixel(uint32_t x, uint32_t y, uint32_t rgba);
    [[nodiscard]] uint32_t GetPixel(uint32_t x, uint32_t y) const;
    
    // Unchecked operations (for performance)
    void SetPixelUnchecked(uint32_t x, uint32_t y, uint32_t rgba);
    [[nodiscard]] uint32_t GetPixelUnchecked(uint32_t x, uint32_t y) const;
};
```

### TexturePresenter

Helper class for displaying EasyGPU textures in a window.

```cpp
class TexturePresenter {
public:
    explicit TexturePresenter(AppWindow& window);
    
    // Present GPU texture
    template <Runtime::PixelFormat Format>
    void Present(Runtime::Texture2D<Format>& texture, PresentMode mode = PresentMode::Auto);
    
    // Present GPU buffer
    void Present(Runtime::Buffer<uint32_t>& buffer, uint32_t width, uint32_t height, 
                 PresentMode mode = PresentMode::Auto);
    
    // Present raw pixels
    void Present(const uint32_t* pixels, uint32_t width, uint32_t height);
    
    // Staging buffer access
    [[nodiscard]] PixelBuffer& StagingBuffer();
    void Present();  // Present the staging buffer
};
```

### Input Enums

```cpp
// Keyboard keys
enum class Key : int32_t {
    Unknown, Space, Num0, Num1, /* ... */ A, B, C, /* ... */ 
    Escape, Enter, Tab, Left, Right, Up, Down,
    F1, F2, /* ... */ F12,
    // ... and more
};

// Mouse buttons
enum class MouseButton : uint8_t {
    Left, Right, Middle, Button4, Button5, Button6, Button7
};

// Modifier flags
enum class ModifierFlags : uint32_t {
    None, Shift, Ctrl, Alt, Super, CapsLock, NumLock
};
```

### Events

Events are delivered via `std::variant`:

```cpp
using WindowEvent = std::variant<
    WindowResizeEvent,    // { uint32_t width, height; }
    WindowCloseEvent,     // {}
    KeyEvent,             // { Key key; bool pressed; ModifierFlags modifiers; }
    CharInputEvent,       // { uint32_t codepoint; }
    MouseButtonEvent,     // { MouseButton button; bool pressed; int32_t x, y; ModifierFlags modifiers; }
    MouseMoveEvent,       // { int32_t x, y, dx, dy; }
    MouseScrollEvent,     // { float dx, dy; }
    WindowFocusEvent      // { bool focused; }
>;
```

## Examples

### Basic Window

```cpp
#include <GPU.h>
#include <iostream>

int main() {
    using namespace GPU;
    
    AppWindow window({.width = 800, .height = 600, .title = "Basic Window"});
    
    while (window.IsOpen()) {
        window.PollEvents();
        
        WindowEvent event;
        while (window.PollEvent(event)) {
            if (std::holds_alternative<KeyEvent>(event)) {
                auto& key = std::get<KeyEvent>(event);
                if (key.key == Key::Escape && key.pressed) {
                    window.Close();
                }
            }
        }
    }
    
    return 0;
}
```

### CPU Pixel Rendering

```cpp
#include <GPU.h>

int main() {
    using namespace GPU;
    
    AppWindow window({.width = 800, .height = 600, .title = "Pixel Buffer"});
    PixelBuffer pixels(800, 600);
    
    while (window.IsOpen()) {
        window.PollEvents();
        
        // Draw checkerboard pattern
        for (uint32_t y = 0; y < pixels.Height(); ++y) {
            for (uint32_t x = 0; x < pixels.Width(); ++x) {
                bool checker = ((x / 16) ^ (y / 16)) & 1;
                pixels.SetPixelUnchecked(x, y, checker ? 0xFFFFFFFF : 0xFF000000);
            }
        }
        
        window.Present(pixels);
    }
    
    return 0;
}
```

### Real-time GPU Compute

```cpp
#include <GPU.h>

int main() {
    using namespace GPU;
    using namespace GPU::Runtime;
    
    AppWindow window({.width = 1024, .height = 768, .title = "GPU Compute"});
    Texture2D<PixelFormat::RGBA8> texture(1024, 768);
    TexturePresenter presenter(window);
    
    float time = 0.0f;
    Uniform<float> timeUniform(0.0f);
    
    Kernel2D plasma([&](Int x, Int y) {
        auto tex = texture.Bind();
        float t = timeUniform.Get();
        
        Float u = Float(x) / 1024.0f;
        Float v = Float(y) / 768.0f;
        
        Float r = 0.5f + 0.5f * Sin(u * 10.0f + t);
        Float g = 0.5f + 0.5f * Sin(v * 10.0f + t * 0.5f);
        Float b = 0.5f + 0.5f * Sin((u + v) * 5.0f + t * 0.3f);
        
        tex.Write(x, y, MakeFloat4(r, g, b, 1.0f));
    });
    
    while (window.IsOpen()) {
        window.PollEvents();
        
        // Update and render
        time += 0.016f;
        timeUniform.Set(time);
        plasma.Dispatch(64, 48);
        
        // Display result
        presenter.Present(texture);
    }
    
    return 0;
}
```

## Building

The Window component is enabled by default. To disable it:

```bash
cmake -B build -DEASYGPU_BUILD_WINDOW=OFF
```

Platform-specific requirements:

- **Windows**: No additional dependencies (uses Win32 API)
- **Linux**: Requires X11 development libraries
  ```bash
  sudo apt-get install libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev
  ```
- **macOS**: No additional dependencies. Uses Cocoa/CoreGraphics (built into macOS), no external frameworks required.

## Design Philosophy

The Window component follows EasyGPU's core principles:

1. **Simplicity**: Minimal API surface, easy to learn
2. **Modern C++**: RAII, move semantics, no raw pointers
3. **Cross-platform**: Works on Windows, Linux, and macOS without changes
4. **Non-intrusive**: Completely optional, doesn't affect core compute functionality
5. **Performance**: Efficient event handling and presentation paths

## Limitations

- Not a full GUI framework (no widgets, buttons, etc.)
- Designed for compute visualization, not complex UIs
- Single-window applications (multi-window support is limited)
- No hardware-accelerated 3D rendering (use with FragmentKernel for that)

## See Also

- [Getting Started](getting-started.md)
- [API Reference](api-reference.md)
- [Examples](../examples/)
