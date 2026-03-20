// TexturePresenter.cpp - Implementation
// Note: This file intentionally does NOT include GPU.h or Runtime headers
// to avoid template instantiation order issues with EasyGPU core.
// The Buffer::Download() call is handled in the inline template file.

#include <Window/TexturePresenter.h>
#include <Window/AppWindow.h>

namespace GPU::Window {

// Implementation class to hide details
class TexturePresenterImpl {
public:
    explicit TexturePresenterImpl(AppWindow &window) : _window(window), _stagingBuffer(1, 1) {
    }

    AppWindow  &_window;
    PixelBuffer _stagingBuffer;
};

TexturePresenter::TexturePresenter(AppWindow &window) : _impl(std::make_unique<TexturePresenterImpl>(window)) {
}

TexturePresenter::~TexturePresenter() = default;

// Buffer presentation is implemented in the header as inline template
// to avoid linking issues

void TexturePresenter::Present(const uint32_t *pixels, uint32_t width, uint32_t height) {
    _impl->_window.Present(pixels, width, height);
}

PixelBuffer &TexturePresenter::StagingBuffer() {
    return _impl->_stagingBuffer;
}

void TexturePresenter::Present() {
    Present(_impl->_stagingBuffer.Data(), _impl->_stagingBuffer.Width(), _impl->_stagingBuffer.Height());
}

} // namespace GPU::Window
