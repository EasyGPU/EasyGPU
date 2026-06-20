/**
 * @file TexturePresenter.cpp
 * @brief Implementation of texture presenter for displaying GPU output in a window.
 */

// TexturePresenter.cpp
// Implementation of non-template methods only.
// Template methods are in TexturePresenter.inl

#include <Window/TexturePresenter.h>

#ifdef EASYGPU_BACKEND_VULKAN
#include "Platform/GLFWWindowPlatform.h"
#include "Swapchain.h"
#include <Backend/VulkanBackend.h>
#include <Runtime/Context.h>
#endif

namespace GPU::Window {

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

void TexturePresenter::Present(const uint32_t *pixels, uint32_t width, uint32_t height) {
	_impl->_window.Present(pixels, width, height);
}

void TexturePresenter::PresentTextureHandle(Backend::TextureHandle texture) {
#ifdef EASYGPU_BACKEND_VULKAN
	auto *platform = dynamic_cast<GLFWWindowPlatform *>(_impl->_window.Platform());
	if (!platform || !platform->GetSwapchain()) {
		return;
	}
	auto &backend = GPU::Runtime::Context::GetBackend<GPU::Backend::VulkanBackend>();
	backend.Finish();
	auto  overlay = _impl->_window.TakeNextVulkanOverlay();
	if (overlay) {
		platform->GetSwapchain()->PresentTexture(
			backend, texture, [overlay = std::move(overlay)](VkCommandBuffer cmd, uint32_t imageIndex) mutable {
				if (overlay) {
					overlay(cmd, imageIndex);
				}
			});
	} else {
		platform->GetSwapchain()->PresentTexture(backend, texture);
	}
#else
	(void)texture;
#endif
}

PixelBuffer &TexturePresenter::StagingBuffer() {
	return _impl->_stagingBuffer;
}

void TexturePresenter::Present() {
	Present(_impl->_stagingBuffer.Data(), _impl->_stagingBuffer.Width(), _impl->_stagingBuffer.Height());
}

} // namespace GPU::Window
