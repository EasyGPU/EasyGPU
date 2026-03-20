#pragma once

/**
 * @file TexturePresenter.inl
 * @brief Template implementation for TexturePresenter
 */

#ifndef EASYGPU_TEXTURE_PRESENTER_INL
#define EASYGPU_TEXTURE_PRESENTER_INL

#include <Runtime/Texture.h>
#include <Runtime/PixelFormat.h>

#include <cstring>
#include <vector>

namespace GPU::Window {

template <Runtime::PixelFormat Format>
void TexturePresenter::Present(Runtime::Texture2D<Format>& texture, PresentMode mode) {
    const uint32_t width = texture.GetWidth();
    const uint32_t height = texture.GetHeight();
    
    // Ensure staging buffer is large enough
    PixelBuffer& staging = StagingBuffer();
    if (staging.Width() != width || staging.Height() != height) {
        staging.Resize(width, height);
    }
    
    // Download texture data
    // For now, we always download as RGBA8
    // TODO: Handle other formats with conversion
    if constexpr (Format == Runtime::PixelFormat::RGBA8) {
        texture.Download(staging.Data());
    } else {
        // For other formats, download to temporary buffer and convert
        // This is a simplified implementation - full implementation would
        // handle format conversion properly
        std::vector<uint8_t> tempData(width * height * Runtime::GetBytesPerPixel(Format));
        texture.Download(tempData.data());
        
        // Convert to RGBA8 (simplified - assumes float formats)
        // Full implementation would use proper color space conversion
        // For now, just copy as-is which may look wrong for non-RGBA8 formats
        if (Runtime::GetBytesPerPixel(Format) == 4) {
            std::memcpy(staging.Data(), tempData.data(), tempData.size());
        }
    }
    
    // Present to window
    Present(staging.Data(), width, height);
}

} // namespace GPU::Window

#endif // EASYGPU_TEXTURE_PRESENTER_INL
