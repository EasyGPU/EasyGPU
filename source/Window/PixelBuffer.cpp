#include <Window/PixelBuffer.h>

namespace GPU::Window {

PixelBuffer::PixelBuffer(uint32_t width, uint32_t height)
	: _width(width), _height(height), _data(static_cast<size_t>(width) * height) {
	if (width == 0 || height == 0) {
		throw std::invalid_argument("PixelBuffer dimensions must be non-zero");
	}
}

PixelBuffer::PixelBuffer(uint32_t width, uint32_t height, const uint32_t *data)
	: _width(width), _height(height), _data(data, data + static_cast<size_t>(width) * height) {
	if (width == 0 || height == 0) {
		throw std::invalid_argument("PixelBuffer dimensions must be non-zero");
	}
	if (data == nullptr) {
		throw std::invalid_argument("PixelBuffer data cannot be null");
	}
}

void PixelBuffer::Resize(uint32_t width, uint32_t height) {
	if (width == 0 || height == 0) {
		throw std::invalid_argument("PixelBuffer dimensions must be non-zero");
	}
	_width	= width;
	_height = height;
	_data.resize(static_cast<size_t>(width) * height);
}

void PixelBuffer::Clear(uint32_t rgba) {
	std::fill(_data.begin(), _data.end(), rgba);
}

void PixelBuffer::SetPixel(uint32_t x, uint32_t y, uint32_t rgba) {
	if (x >= _width || y >= _height) {
		throw std::out_of_range("Pixel coordinates out of bounds");
	}
	_data[y * _width + x] = rgba;
}

uint32_t PixelBuffer::GetPixel(uint32_t x, uint32_t y) const {
	if (x >= _width || y >= _height) {
		throw std::out_of_range("Pixel coordinates out of bounds");
	}
	return _data[y * _width + x];
}

} // namespace GPU::Window
