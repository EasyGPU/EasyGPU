#pragma once

/**
 * @file MiniFBWindowPlatform.h
 * @brief Internal platform abstraction using minifb
 */

#ifndef EASYGPU_MINIFB_WINDOW_PLATFORM_H
#define EASYGPU_MINIFB_WINDOW_PLATFORM_H

#include <Window/WindowConfig.h>
#include <Window/WindowEvents.h>

#include <cstdint>
#include <functional>
#include <queue>
#include <string>

// Forward declare minifb types
struct mfb_window;
struct mfb_timer;

namespace GPU::Window {

/**
 * @brief Internal platform abstraction interface
 * 
 * This is the bridge between the public Window API and minifb.
 * It is intentionally kept internal to allow future backend changes.
 */
class IWindowPlatform {
public:
    virtual ~IWindowPlatform() = default;

    // Window state
    [[nodiscard]] virtual bool IsOpen() const = 0;
    virtual void Close() = 0;

    // Window properties
    [[nodiscard]] virtual uint32_t Width() const = 0;
    [[nodiscard]] virtual uint32_t Height() const = 0;
    virtual void SetTitle(const std::string& title) = 0;

    // Presentation
    virtual void Present(const uint32_t* pixels, uint32_t width, uint32_t height) = 0;

    // Events
    virtual void PollEvents() = 0;
    virtual void WaitEvents() = 0;
    virtual bool PollEvent(WindowEvent& event) = 0;

    // Input state
    [[nodiscard]] virtual bool IsKeyDown(int keyCode) const = 0;
    [[nodiscard]] virtual bool IsMouseDown(int button) const = 0;
    [[nodiscard]] virtual std::pair<int32_t, int32_t> MousePosition() const = 0;
    [[nodiscard]] virtual std::pair<float, float> MouseScroll() const = 0;

    // Callbacks (used by Window class to inject events)
    std::function<void(uint32_t, uint32_t)> resizeCallback;
    std::function<bool()> closeCallback;
    std::function<void(bool)> focusCallback;
};

/**
 * @brief minifb-based platform implementation
 */
class MiniFBWindowPlatform : public IWindowPlatform {
public:
    MiniFBWindowPlatform(const WindowConfig& config, std::queue<WindowEvent>& eventQueue);
    ~MiniFBWindowPlatform() override;

    // Disable copy/move (window handles are not transferable)
    MiniFBWindowPlatform(const MiniFBWindowPlatform&) = delete;
    MiniFBWindowPlatform& operator=(const MiniFBWindowPlatform&) = delete;
    MiniFBWindowPlatform(MiniFBWindowPlatform&&) = delete;
    MiniFBWindowPlatform& operator=(MiniFBWindowPlatform&&) = delete;

    // IWindowPlatform implementation
    [[nodiscard]] bool IsOpen() const override;
    void Close() override;

    [[nodiscard]] uint32_t Width() const override;
    [[nodiscard]] uint32_t Height() const override;
    void SetTitle(const std::string& title) override;

    void Present(const uint32_t* pixels, uint32_t width, uint32_t height) override;

    void PollEvents() override;
    void WaitEvents() override;
    bool PollEvent(WindowEvent& event) override;

    [[nodiscard]] bool IsKeyDown(int keyCode) const override;
    [[nodiscard]] bool IsMouseDown(int button) const override;
    [[nodiscard]] std::pair<int32_t, int32_t> MousePosition() const override;
    [[nodiscard]] std::pair<float, float> MouseScroll() const override;

    // Instance handlers for callbacks (called by C callback trampolines)
    // These are public so the extern "C" callbacks can access them
    void HandleActive(bool isActive);
    void HandleResize(int width, int height);
    bool HandleClose();
    void HandleKeyboard(int key, int mod, bool isPressed);
    void HandleCharInput(unsigned int code);
    void HandleMouseButton(int button, int mod, bool isPressed);
    void HandleMouseMove(int x, int y);
    void HandleMouseScroll(int mod, float deltaX, float deltaY);

private:

    // Helper: convert minifb key/mod to our enums
    static Key ConvertKey(int mfbKey);
    static ModifierFlags ConvertMods(int mfbMods);
    static MouseButton ConvertMouseButton(int mfbButton);

private:
    mfb_window* _window = nullptr;
    mfb_timer* _timer = nullptr;
    std::queue<WindowEvent>& _eventQueue;
    
    // Current state
    uint32_t _width = 0;
    uint32_t _height = 0;
    int32_t _mouseX = 0;
    int32_t _mouseY = 0;
    float _scrollX = 0.0f;
    float _scrollY = 0.0f;
    bool _isOpen = false;
};

} // namespace GPU::Window

#endif // EASYGPU_MINIFB_WINDOW_PLATFORM_H
