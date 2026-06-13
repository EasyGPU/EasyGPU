#pragma once

/**
 * @file DepthBuffer.h
 * @brief RAII depth buffer for graphics pipeline rendering.
 */

#ifndef EASYGPU_DEPTH_BUFFER_H
#define EASYGPU_DEPTH_BUFFER_H

#include <Backend/Backend.h>
#include <Runtime/Context.h>

#include <cstdint>

namespace GPU::Runtime {

/**
 * @brief RAII depth buffer for depth testing in graphics pipelines.
 *
 * Usage:
 *   DepthBuffer depth(1920, 1080);
 *   pipeline.DrawIndexed(rt, depth, indexCount, true);
 */
class DepthBuffer {
public:
	/**
	 * @brief Create a depth buffer.
	 * @param width Buffer width in pixels.
	 * @param height Buffer height in pixels.
	 */
	DepthBuffer(uint32_t width, uint32_t height) : _width(width), _height(height) {
		auto *backend = Context::GetBackend();
		if (backend) {
			_handle = backend->CreateDepthBuffer(width, height);
		}
	}

	~DepthBuffer() {
		if (_handle != Backend::INVALID_TEXTURE_HANDLE) {
			auto *backend = Context::GetBackend();
			if (backend) {
				backend->DestroyDepthBuffer(_handle);
			}
		}
	}

	DepthBuffer(const DepthBuffer &)			= delete;
	DepthBuffer &operator=(const DepthBuffer &) = delete;
	DepthBuffer(DepthBuffer &&other) noexcept : _handle(other._handle), _width(other._width), _height(other._height) {
		other._handle = Backend::INVALID_TEXTURE_HANDLE;
	}
	DepthBuffer &operator=(DepthBuffer &&other) noexcept {
		if (this != &other) {
			if (_handle != Backend::INVALID_TEXTURE_HANDLE) {
				auto *backend = Context::GetBackend();
				if (backend)
					backend->DestroyDepthBuffer(_handle);
			}
			_handle		  = other._handle;
			_width		  = other._width;
			_height		  = other._height;
			other._handle = Backend::INVALID_TEXTURE_HANDLE;
		}
		return *this;
	}

	[[nodiscard]] uint32_t Width() const {
		return _width;
	}
	[[nodiscard]] uint32_t Height() const {
		return _height;
	}
	[[nodiscard]] Backend::TextureHandle GetHandle() const {
		return _handle;
	}

private:
	Backend::TextureHandle _handle = Backend::INVALID_TEXTURE_HANDLE;
	uint32_t			   _width  = 0;
	uint32_t			   _height = 0;
};

} // namespace GPU::Runtime

#endif // EASYGPU_DEPTH_BUFFER_H
