#pragma once

/**
 * @file TexturePresenter.inl
 * @brief Template implementation for TexturePresenter
 *
 * This file is automatically included by TexturePresenter.h when
 * GPU_H_INCLUDED is defined (i.e., when <GPU.h> was included first).
 */

#ifndef EASYGPU_TEXTURE_PRESENTER_INL
#define EASYGPU_TEXTURE_PRESENTER_INL

#include <Runtime/Buffer.h>
#include <Runtime/PixelFormat.h>
#include <Runtime/Texture.h>

#include <cstring>
#include <vector>

namespace GPU::Window {

// TexturePresenterImpl needs to be fully defined here for inline methods
// It's defined in TexturePresenter.cpp but we need access to _stagingBuffer
// So we make the template methods call a helper that does the actual work

template <typename TextureT> void TexturePresenter::Present(TextureT &texture, PresentMode mode) {
	if constexpr (TextureT::GetFormat() == Runtime::PixelFormat::RGBA8) {
		if (mode == PresentMode::Auto) {
#ifdef EASYGPU_BACKEND_VULKAN
			PresentTextureHandle(texture.GetHandle());
			return;
#endif
		}
	}

	const uint32_t width   = texture.GetWidth();
	const uint32_t height  = texture.GetHeight();

	PixelBuffer	  &staging = StagingBuffer();
	if (staging.Width() != width || staging.Height() != height) {
		staging.Resize(width, height);
	}

	// Use if constexpr to check format at compile time
	if constexpr (TextureT::GetFormat() == Runtime::PixelFormat::RGBA8) {
		texture.Download(staging.Data());
	} else {
		// For other formats, download to temporary buffer
		std::vector<uint8_t> tempData(width * height * Runtime::GetBytesPerPixel(TextureT::GetFormat()));
		texture.Download(tempData.data());
		if (Runtime::GetBytesPerPixel(TextureT::GetFormat()) == 4) {
			std::memcpy(staging.Data(), tempData.data(), tempData.size());
		}
	}

	Present(staging.Data(), width, height);
}

template <typename BufferT>
void TexturePresenter::Present(BufferT &buffer, uint32_t width, uint32_t height, PresentMode mode) {
	(void)mode;

	PixelBuffer &staging = StagingBuffer();
	if (staging.Width() != width || staging.Height() != height) {
		staging.Resize(width, height);
	}

	buffer.Download(staging.Data(), width * height);
	Present(staging.Data(), width, height);
}

} // namespace GPU::Window

#endif // EASYGPU_TEXTURE_PRESENTER_INL
