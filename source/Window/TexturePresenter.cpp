// TexturePresenter.cpp
// Implementation of non-template methods only.
// Template methods are in TexturePresenter.inl

#include <Window/TexturePresenter.h>

namespace GPU::Window {

class TexturePresenterImpl {
public:
    explicit TexturePresenterImpl(AppWindow &window) 
        : _window(window), _stagingBuffer(1, 1) {}

    AppWindow  &_window;
    PixelBuffer _stagingBuffer;
};

TexturePresenter::TexturePresenter(AppWindow &window) 
    : _impl(std::make_unique<TexturePresenterImpl>(window)) {
}

TexturePresenter::~TexturePresenter() = default;

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
