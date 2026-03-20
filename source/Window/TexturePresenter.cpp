#include <Window/AppWindow.h>
#include <Window/TexturePresenter.h>

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

void TexturePresenter::Present(Runtime::Buffer<uint32_t> &buffer, uint32_t width, uint32_t height, PresentMode mode) {
	(void)mode; // Currently only CopyToCPU is implemented

	// Ensure staging buffer is large enough
	if (_impl->_stagingBuffer.Width() != width || _impl->_stagingBuffer.Height() != height) {
		_impl->_stagingBuffer.Resize(width, height);
	}

	// Download from GPU buffer
	buffer.Download(_impl->_stagingBuffer.Data(), width * height);

	// Present to window
	Present(_impl->_stagingBuffer.Data(), width, height);
}

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
