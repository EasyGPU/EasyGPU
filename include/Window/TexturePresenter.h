#pragma once

/**
 * @file TexturePresenter.h
 * @brief Present EasyGPU textures and buffers to a window
 */

#ifndef EASYGPU_TEXTURE_PRESENTER_H
#define EASYGPU_TEXTURE_PRESENTER_H

#include <Window/AppWindow.h>
#include <Window/PixelBuffer.h>

// Forward declarations for Runtime types - only include what we need
namespace GPU::Runtime {

template <typename T> class Buffer;

// Forward declare PixelFormat enum
enum class PixelFormat : uint8_t;

// Forward declare Texture2D with template parameter
template <PixelFormat Format> class Texture2D;

} // namespace GPU::Runtime

#include <memory>

namespace GPU::Window {

// Forward declaration
class TexturePresenterImpl;

/**
 * @brief Presentation mode for texture display
 */
enum class PresentMode {
	/**
	 * Always copy through CPU memory (most compatible)
	 */
	CopyToCPU,

	/**
	 * Automatically select best method (may use GPU direct path when available)
	 */
	Auto
};

/**
 * @brief Helper class to present EasyGPU textures and buffers to a window
 *
 * This class handles the transfer from GPU resources to the window display,
 * managing staging buffers and format conversion as needed.
 *
 * Usage:
 *   Window window({.width = 1024, .height = 1024});
 *   TexturePresenter presenter(window);
 *
 *   Texture2D<PixelFormat::RGBA8> texture(1024, 1024);
 *   // ... render to texture ...
 *   presenter.Present(texture);
 */
class TexturePresenter {
public:
	/**
	 * @brief Create a texture presenter for the given window
	 * @param window The window to present to (must outlive the presenter)
	 */
	explicit TexturePresenter(AppWindow &window);

	/**
	 * @brief Destructor
	 */
	~TexturePresenter();

	// Disable copy/move
	TexturePresenter(const TexturePresenter &)			  = delete;
	TexturePresenter &operator=(const TexturePresenter &) = delete;
	TexturePresenter(TexturePresenter &&)				  = delete;
	TexturePresenter &operator=(TexturePresenter &&)	  = delete;

public:
	/**
	 * @brief Present a texture to the window
	 * @tparam Format Pixel format of the texture
	 * @param texture The texture to display
	 * @param mode Presentation mode (default: Auto)
	 *
	 * The texture is downloaded from GPU and displayed in the window.
	 * This is a synchronous operation.
	 */
	template <Runtime::PixelFormat Format>
	void					   Present(Runtime::Texture2D<Format> &texture, PresentMode mode = PresentMode::Auto);

	/**
	 * @brief Present a buffer containing RGBA8 pixel data
	 * @param buffer The buffer containing pixel data
	 * @param width Width of the image in pixels
	 * @param height Height of the image in pixels
	 * @param mode Presentation mode (default: Auto)
	 *
	 * The buffer must contain at least width * height uint32_t RGBA8 pixels.
	 */
	void					   Present(Runtime::Buffer<uint32_t> &buffer, uint32_t width, uint32_t height,
									   PresentMode mode = PresentMode::Auto);

	/**
	 * @brief Present raw pixel data directly
	 * @param pixels Pointer to RGBA8 pixel data
	 * @param width Width of the image
	 * @param height Height of the image
	 */
	void					   Present(const uint32_t *pixels, uint32_t width, uint32_t height);

	/**
	 * @brief Get the staging pixel buffer used internally
	 *
	 * This can be used to directly write pixel data for display.
	 * Call Present() with no arguments to display the staging buffer.
	 */
	[[nodiscard]] PixelBuffer &StagingBuffer();

	/**
	 * @brief Present the internal staging buffer
	 */
	void					   Present();

private:
	// Implementation details
	std::unique_ptr<TexturePresenterImpl> _impl;
};

} // namespace GPU::Window

// Template implementation
#include <Window/TexturePresenter.inl>

#endif // EASYGPU_TEXTURE_PRESENTER_H
