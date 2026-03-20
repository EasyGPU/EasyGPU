#pragma once

/**
 * @file PixelBuffer.h
 * @brief CPU-side pixel buffer for window presentation
 */

#ifndef EASYGPU_PIXEL_BUFFER_H
#define EASYGPU_PIXEL_BUFFER_H

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace GPU::Window {

/**
 * @brief A CPU-side RGBA8 pixel buffer
 * 
 * This is the bridge between compute output and on-screen display.
 * Pixels are stored in RGBA8 format (4 bytes per pixel).
 * 
 * Usage:
 *   PixelBuffer pixels(800, 600);
 *   pixels.Clear(0xFF202030);  // Dark blue background
 *   pixels.SetPixel(10, 20, 0xFFFF0000);  // Red pixel at (10, 20)
 *   window.Present(pixels);
 */
class PixelBuffer {
public:
    /**
     * @brief Create a pixel buffer with the specified dimensions
     * @param width Width in pixels
     * @param height Height in pixels
     */
    PixelBuffer(uint32_t width, uint32_t height);

    /**
     * @brief Create from existing data
     * @param width Width in pixels
     * @param height Height in pixels
     * @param data Pointer to RGBA8 pixel data (will be copied)
     */
    PixelBuffer(uint32_t width, uint32_t height, const uint32_t* data);

    ~PixelBuffer() = default;

    // Move operations
    PixelBuffer(PixelBuffer&& other) noexcept = default;
    PixelBuffer& operator=(PixelBuffer&& other) noexcept = default;

    // Disable copy (expensive, use explicit if needed)
    PixelBuffer(const PixelBuffer&) = delete;
    PixelBuffer& operator=(const PixelBuffer&) = delete;

public:
    /**
     * @brief Get buffer width
     */
    [[nodiscard]] uint32_t Width() const noexcept { return _width; }

    /**
     * @brief Get buffer height
     */
    [[nodiscard]] uint32_t Height() const noexcept { return _height; }

    /**
     * @brief Get total pixel count
     */
    [[nodiscard]] size_t PixelCount() const noexcept { return _data.size(); }

    /**
     * @brief Get raw pixel data pointer
     */
    [[nodiscard]] uint32_t* Data() noexcept { return _data.data(); }
    [[nodiscard]] const uint32_t* Data() const noexcept { return _data.data(); }

    /**
     * @brief Get data size in bytes
     */
    [[nodiscard]] size_t SizeInBytes() const noexcept { return _data.size() * sizeof(uint32_t); }

public:
    /**
     * @brief Resize the buffer (contents become undefined)
     * @param width New width
     * @param height New height
     */
    void Resize(uint32_t width, uint32_t height);

    /**
     * @brief Clear entire buffer with a color
     * @param rgba Color in RGBA8 format (e.g., 0xFF0000FF for red)
     */
    void Clear(uint32_t rgba);

    /**
     * @brief Set a single pixel
     * @param x X coordinate
     * @param y Y coordinate
     * @param rgba Color in RGBA8 format
     * @throws std::out_of_range if coordinates are invalid
     */
    void SetPixel(uint32_t x, uint32_t y, uint32_t rgba);

    /**
     * @brief Get a single pixel value
     * @param x X coordinate
     * @param y Y coordinate
     * @return Color in RGBA8 format
     * @throws std::out_of_range if coordinates are invalid
     */
    [[nodiscard]] uint32_t GetPixel(uint32_t x, uint32_t y) const;

    /**
     * @brief Set pixel (unchecked, for performance)
     */
    void SetPixelUnchecked(uint32_t x, uint32_t y, uint32_t rgba) {
        _data[y * _width + x] = rgba;
    }

    /**
     * @brief Get pixel (unchecked, for performance)
     */
    [[nodiscard]] uint32_t GetPixelUnchecked(uint32_t x, uint32_t y) const {
        return _data[y * _width + x];
    }

private:
    uint32_t _width;
    uint32_t _height;
    std::vector<uint32_t> _data;
};

/**
 * @brief Helper to pack RGBA components into a single uint32_t
 */
[[nodiscard]] inline uint32_t PackRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return (static_cast<uint32_t>(a) << 24) |
           (static_cast<uint32_t>(r) << 16) |
           (static_cast<uint32_t>(g) << 8)  |
           (static_cast<uint32_t>(b));
}

/**
 * @brief Helper to unpack RGBA components from a uint32_t
 */
inline void UnpackRGBA(uint32_t rgba, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) {
    a = static_cast<uint8_t>((rgba >> 24) & 0xFF);
    r = static_cast<uint8_t>((rgba >> 16) & 0xFF);
    g = static_cast<uint8_t>((rgba >> 8) & 0xFF);
    b = static_cast<uint8_t>(rgba & 0xFF);
}

} // namespace GPU::Window

#endif // EASYGPU_PIXEL_BUFFER_H
