#pragma once

/**
 * @file TexturePresenter.h
 * @brief Present EasyGPU textures and buffers to a window
 *
 * NOTE: This header does NOT include EasyGPU core headers.
 * You must include <GPU.h> BEFORE including this header to use
 * the template methods Present(Texture2D) and Present(Buffer).
 */

#ifndef EASYGPU_TEXTURE_PRESENTER_H
#define EASYGPU_TEXTURE_PRESENTER_H

#include <Window/AppWindow.h>
#include <Window/PixelBuffer.h>

#include <cstdint>
#include <memory>

namespace GPU::Window {

// Forward declaration
class TexturePresenterImpl;

/**
 * @brief Presentation mode for texture display
 */
enum class PresentMode {
	CopyToCPU,
	Auto
};

/**
 * @brief Helper class to present EasyGPU textures and buffers to a window
 *
 * Usage:
 *   #include <GPU.h>  // Must include first!
 *   #include <Window/TexturePresenter.h>
 *
 *   Window window({.width = 1024, .height = 1024});
 *   TexturePresenter presenter(window);
 *   Texture2D<PixelFormat::RGBA8> texture(1024, 1024);
 *   presenter.Present(texture);
 */
class TexturePresenter {
public:
	explicit TexturePresenter(AppWindow &window);
	~TexturePresenter();

	TexturePresenter(const TexturePresenter &)						 = delete;
	TexturePresenter &operator=(const TexturePresenter &)			 = delete;
	TexturePresenter(TexturePresenter &&)							 = delete;
	TexturePresenter				 &operator=(TexturePresenter &&) = delete;

	/**
	 * @brief Present a texture to the window
	 * @tparam Format Pixel format of the texture
	 *
	 * NOTE: Implementation is in TexturePresenter.inl which must be included
	 * AFTER including <GPU.h>
	 */
	template <typename TextureT> void Present(TextureT &texture, PresentMode mode = PresentMode::Auto);

	/**
	 * @brief Present a buffer containing RGBA8 pixel data
	 *
	 * NOTE: Implementation is in TexturePresenter.inl which must be included
	 * AFTER including <GPU.h>
	 */
	template <typename BufferT>
	void Present(BufferT &buffer, uint32_t width, uint32_t height, PresentMode mode = PresentMode::Auto);

	/**
	 * @brief Present raw pixel data directly (always available)
	 */
	void Present(const uint32_t *pixels, uint32_t width, uint32_t height);

	[[nodiscard]] PixelBuffer &StagingBuffer();
	void					   Present(); // Present staging buffer

private:
	std::unique_ptr<TexturePresenterImpl> _impl;
};

} // namespace GPU::Window

// Implementation - only include if GPU.h was already included
#ifdef GPU_H_INCLUDED
#include <Window/TexturePresenter.inl>
#endif

#endif // EASYGPU_TEXTURE_PRESENTER_H
