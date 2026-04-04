#pragma once

/**
 * @file Buffer.h
 * @brief GPU buffer for compute shader with backend support
 */
#ifndef EASYGPU_BUFFER_H
#define EASYGPU_BUFFER_H

#include <Backend/Backend.h>
#include <IR/Builder/Builder.h>
#include <IR/Value/BufferRef.h>
#include <Runtime/Context.h>
#include <Utility/Matrix.h>
#include <Utility/Meta/Std430Layout.h>
#include <Utility/Meta/StructMeta.h>
#include <Utility/Vec.h>

#include <cstdint>
#include <cstring>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace GPU::Runtime {

/**
 * The access mode for the buffer
 * This mirrors Backend::BufferMode for user convenience
 */
enum class BufferMode {
	Read,	  // Readonly access
	Write,	  // Writeonly access
	ReadWrite // Read-write access
};

/**
 * Convert Runtime BufferMode to Backend BufferMode
 */
inline Backend::BufferMode ToBackendBufferMode(BufferMode mode) {
	switch (mode) {
	case BufferMode::Read:
		return Backend::BufferMode::Read;
	case BufferMode::Write:
		return Backend::BufferMode::Write;
	case BufferMode::ReadWrite:
		return Backend::BufferMode::ReadWrite;
	default:
		return Backend::BufferMode::ReadWrite;
	}
}

/**
 * Helper function to get GLSL type name for buffer elements
 */
template <typename T> struct IsStructRegistered {
	static constexpr bool value = GPU::Meta::StructMeta<T>::isRegistered;
};

template <typename T> std::string GetGLSLTypeNameForBuffer(T *) {
	auto sv = std::string(GPU::Meta::StructMeta<T>::glslTypeName);
	return std::string(sv.data(), sv.size());
}

inline std::string GetGLSLTypeNameForBuffer(float *) {
	return "float";
}
inline std::string GetGLSLTypeNameForBuffer(int *) {
	return "int";
}
inline std::string GetGLSLTypeNameForBuffer(bool *) {
	return "bool";
}
inline std::string GetGLSLTypeNameForBuffer(Math::Vec2 *) {
	return "vec2";
}
inline std::string GetGLSLTypeNameForBuffer(Math::Vec3 *) {
	return "vec3";
}
inline std::string GetGLSLTypeNameForBuffer(Math::Vec4 *) {
	return "vec4";
}
inline std::string GetGLSLTypeNameForBuffer(Math::IVec2 *) {
	return "ivec2";
}
inline std::string GetGLSLTypeNameForBuffer(Math::IVec3 *) {
	return "ivec3";
}
inline std::string GetGLSLTypeNameForBuffer(Math::IVec4 *) {
	return "ivec4";
}
inline std::string GetGLSLTypeNameForBuffer(Math::Mat2 *) {
	return "mat2";
}
inline std::string GetGLSLTypeNameForBuffer(Math::Mat3 *) {
	return "mat3";
}
inline std::string GetGLSLTypeNameForBuffer(Math::Mat4 *) {
	return "mat4";
}

template <typename T> std::string GetGLSLTypeNameForBuffer() {
	return GetGLSLTypeNameForBuffer(static_cast<T *>(nullptr));
}
template <> inline std::string GetGLSLTypeNameForBuffer<Math::Mat3>() {
	return "mat3";
}
template <> inline std::string GetGLSLTypeNameForBuffer<Math::Mat4>() {
	return "mat4";
}

/**
 * The GPU buffer for compute shader
 * @tparam T The element type of the buffer
 */
template <typename T> class Buffer {
public:
	Buffer(size_t Count, BufferMode Mode = BufferMode::ReadWrite) : _count(Count), _mode(Mode) {
		InitLayout();
		CreateBuffer();
	}

	Buffer(const std::vector<T> &Data, BufferMode Mode = BufferMode::ReadWrite) : _count(Data.size()), _mode(Mode) {
		InitLayout();
		CreateBuffer();
		if (!Data.empty()) {
			Upload(Data.data(), Data.size());
		}
	}

	Buffer(Buffer &&other) noexcept
		: _bufferHandle(other._bufferHandle), _count(other._count), _elementSize(other._elementSize),
		  _mode(other._mode), _boundBinding(other._boundBinding), _layoutConverter(std::move(other._layoutConverter)),
		  _moved(other._moved) {
		other._bufferHandle = Backend::INVALID_BUFFER_HANDLE;
		other._count		= 0;
		other._elementSize	= 0;
		other._boundBinding = -1;
		other._moved		= true;
	}

	Buffer &operator=(Buffer &&other) noexcept {
		if (this != &other) {
			DestroyBuffer();
			_bufferHandle		= other._bufferHandle;
			_count				= other._count;
			_elementSize		= other._elementSize;
			_mode				= other._mode;
			_boundBinding		= other._boundBinding;
			_layoutConverter	= std::move(other._layoutConverter);
			_moved				= other._moved;
			other._bufferHandle = Backend::INVALID_BUFFER_HANDLE;
			other._count		= 0;
			other._elementSize	= 0;
			other._boundBinding = -1;
			other._moved		= true;
		}
		return *this;
	}

	~Buffer() {
		DestroyBuffer();
	}

	Buffer(const Buffer &)			  = delete;
	Buffer &operator=(const Buffer &) = delete;

private:
	void InitLayout() {
		_layoutConverter = std::make_unique<Meta::Std430Converter<T>>();
		_elementSize	 = _layoutConverter->GetGPULayoutSize();
		if (_elementSize < sizeof(T)) {
			_elementSize = sizeof(T);
		}
	}

public:
	[[nodiscard]] IR::Value::BufferRef<T> Bind() {
		if (_moved) {
			throw std::runtime_error("Buffer::Bind() called on a moved-from buffer");
		}
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("Buffer::Bind() called outside of Kernel definition");
		}

		uint32_t	binding	   = context->AllocateBindingSlot();
		std::string bufferName = std::format("buf{}", binding);
		std::string typeName   = GetGLSLTypeNameForBuffer<T>();

		context->RegisterBuffer(binding, typeName, bufferName, static_cast<int>(_mode));
		context->BindRuntimeBuffer(binding, _bufferHandle);
		_boundBinding = binding;

		return IR::Value::BufferRef<T>(bufferName, binding);
	}

	void Upload(const T *data, size_t count) {
		if (_bufferHandle == Backend::INVALID_BUFFER_HANDLE || data == nullptr || count == 0) {
			return;
		}
		if (count > _count) {
			count = _count;
		}

		Runtime::Context::GetInstance().MakeCurrent();

		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}

		if (_layoutConverter && _layoutConverter->NeedsConversion()) {
			std::vector<char> gpuBuffer(count * _elementSize);
			_layoutConverter->ConvertToGPU(data, gpuBuffer.data(), count);
			backend->UploadBuffer(_bufferHandle, 0, count * _elementSize, gpuBuffer.data());
		} else {
			backend->UploadBuffer(_bufferHandle, 0, count * _elementSize, data);
		}
	}

	void Upload(const std::vector<T> &data) {
		if (!data.empty()) {
			Upload(data.data(), data.size());
		}
	}

	void Download(T *outData, size_t count) {
		if (_bufferHandle == Backend::INVALID_BUFFER_HANDLE || outData == nullptr || count == 0) {
			return;
		}
		if (count > _count) {
			count = _count;
		}

		Runtime::Context::GetInstance().MakeCurrent();

		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}

		if (_layoutConverter && _layoutConverter->NeedsConversion()) {
			std::vector<char> gpuBuffer(count * _elementSize);
			backend->DownloadBuffer(_bufferHandle, 0, count * _elementSize, gpuBuffer.data());
			_layoutConverter->ConvertFromGPU(gpuBuffer.data(), outData, count);
		} else {
			backend->DownloadBuffer(_bufferHandle, 0, count * _elementSize, outData);
		}
	}

	void Download(std::vector<T> &outData) {
		if (outData.size() < _count) {
			outData.resize(_count);
		}
		if (!outData.empty()) {
			Download(outData.data(), outData.size());
		}
	}

public:
	[[nodiscard]] Backend::BufferHandle GetHandle() const {
		return _bufferHandle;
	}

	[[nodiscard]] size_t GetCount() const {
		return _count;
	}

	[[nodiscard]] BufferMode GetMode() const {
		return _mode;
	}

	[[nodiscard]] size_t GetElementSize() const {
		return _elementSize;
	}

	[[nodiscard]] size_t GetBufferSize() const {
		return _count * _elementSize;
	}

	[[nodiscard]] int GetBinding() const {
		return _boundBinding;
	}

private:
	void CreateBuffer() {
		Runtime::AutoInitContext();
		Runtime::Context::GetInstance().MakeCurrent();

		if (_count == 0) {
			return;
		}

		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}

		Backend::BufferDesc desc;
		desc.sizeInBytes = _count * _elementSize;
		desc.mode		 = ToBackendBufferMode(_mode);
		desc.initialData = nullptr;

		_bufferHandle	 = backend->CreateBuffer(desc);
		if (_bufferHandle == Backend::INVALID_BUFFER_HANDLE) {
			throw std::runtime_error("Failed to create GPU buffer");
		}
	}

	void DestroyBuffer() {
		if (_bufferHandle != Backend::INVALID_BUFFER_HANDLE) {
			auto *backend = Context::GetBackend();
			if (backend) {
				backend->DestroyBuffer(_bufferHandle);
			}
			_bufferHandle = Backend::INVALID_BUFFER_HANDLE;
		}
	}

private:
	Backend::BufferHandle				   _bufferHandle	= Backend::INVALID_BUFFER_HANDLE;
	size_t								   _count			= 0;
	size_t								   _elementSize		= sizeof(T);
	BufferMode							   _mode			= BufferMode::ReadWrite;
	int									   _boundBinding	= -1;
	std::unique_ptr<Meta::LayoutConverter> _layoutConverter = nullptr;
	bool								   _moved			= false; // Track if buffer has been moved from
};

} // namespace GPU::Runtime

#endif // EASYGPU_BUFFER_H
