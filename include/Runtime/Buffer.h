#pragma once

/**
 * @file Buffer.h
 * @brief GPU buffer for compute shader with backend support.
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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace GPU::Runtime {

/**
 * @brief The access mode for the buffer.
 *
 * Mirrors Backend::BufferMode for user convenience.
 */
enum class BufferMode {
	Read,	  ///< Readonly access
	Write,	  ///< Writeonly access
	ReadWrite ///< Read-write access
};

/**
 * @brief Convert Runtime BufferMode to Backend BufferMode.
 * @param mode The runtime buffer access mode.
 * @return The equivalent Backend::BufferMode enum value.
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
 * @brief Convert Runtime BufferMode to Backend buffer mode integer constant.
 * @param mode The runtime buffer access mode.
 * @return The equivalent Backend integer constant (e.g., BUFFER_MODE_READ_WRITE).
 */
inline int ToBackendBufferModeInt(BufferMode mode) {
	switch (mode) {
	case BufferMode::Read:
		return Backend::BUFFER_MODE_READ_ONLY;
	case BufferMode::Write:
		return Backend::BUFFER_MODE_WRITE_ONLY;
	case BufferMode::ReadWrite:
		return Backend::BUFFER_MODE_READ_WRITE;
	default:
		return Backend::BUFFER_MODE_READ_WRITE;
	}
}

/**
 * @brief Trait to check whether a struct type is registered with the metadata system.
 * @tparam T The type to check.
 */
template <typename T> struct IsStructRegistered {
	static constexpr bool value = GPU::Meta::StructMeta<T>::isRegistered;
};

/**
 * @brief Get the GLSL type name for registered struct types.
 * @tparam T The registered struct type.
 * @return The GLSL struct type name as defined by StructMeta.
 */
template <typename T> std::string GetGLSLTypeNameForBuffer(T *) {
	auto sv = std::string(GPU::Meta::StructMeta<T>::glslTypeName);
	return std::string(sv.data(), sv.size());
}

/** @brief Get GLSL type name for float buffers. */
inline std::string GetGLSLTypeNameForBuffer(float *) {
	return "float";
}
/** @brief Get GLSL type name for int buffers. */
inline std::string GetGLSLTypeNameForBuffer(int *) {
	return "int";
}
/** @brief Get GLSL type name for bool buffers. */
inline std::string GetGLSLTypeNameForBuffer(bool *) {
	return "bool";
}
/** @brief Get GLSL type name for vec2 buffers. */
inline std::string GetGLSLTypeNameForBuffer(Math::Vec2 *) {
	return "vec2";
}
/** @brief Get GLSL type name for vec3 buffers. */
inline std::string GetGLSLTypeNameForBuffer(Math::Vec3 *) {
	return "vec3";
}
/** @brief Get GLSL type name for vec4 buffers. */
inline std::string GetGLSLTypeNameForBuffer(Math::Vec4 *) {
	return "vec4";
}
/** @brief Get GLSL type name for ivec2 buffers. */
inline std::string GetGLSLTypeNameForBuffer(Math::IVec2 *) {
	return "ivec2";
}
/** @brief Get GLSL type name for ivec3 buffers. */
inline std::string GetGLSLTypeNameForBuffer(Math::IVec3 *) {
	return "ivec3";
}
/** @brief Get GLSL type name for ivec4 buffers. */
inline std::string GetGLSLTypeNameForBuffer(Math::IVec4 *) {
	return "ivec4";
}
/** @brief Get GLSL type name for mat2 buffers. */
inline std::string GetGLSLTypeNameForBuffer(Math::Mat2 *) {
	return "mat2";
}
/** @brief Get GLSL type name for mat3 buffers. */
inline std::string GetGLSLTypeNameForBuffer(Math::Mat3 *) {
	return "mat3";
}
/** @brief Get GLSL type name for mat4 buffers. */
inline std::string GetGLSLTypeNameForBuffer(Math::Mat4 *) {
	return "mat4";
}

/**
 * @brief Deduce GLSL type name for buffer from template parameter.
 * @tparam T The element type to deduce GLSL type name for.
 * @return The GLSL type name string.
 */
template <typename T> std::string GetGLSLTypeNameForBuffer() {
	return GetGLSLTypeNameForBuffer(static_cast<T *>(nullptr));
}
/** @brief Specialization: GLSL type name for mat3. */
template <> inline std::string GetGLSLTypeNameForBuffer<Math::Mat3>() {
	return "mat3";
}
/** @brief Specialization: GLSL type name for mat4. */
template <> inline std::string GetGLSLTypeNameForBuffer<Math::Mat4>() {
	return "mat4";
}

/**
 * The GPU buffer for compute shader
 * @tparam T The element type of the buffer
 */
template <typename T> class Buffer {
public:
	/**
	 * @brief Construct a GPU buffer with a given element count.
	 * @param Count Number of elements to allocate.
	 * @param Mode Access mode (default: ReadWrite).
	 */
	Buffer(size_t Count, BufferMode Mode = BufferMode::ReadWrite) : _count(Count), _mode(Mode) {
		InitLayout();
		CreateBuffer();
	}

	/**
	 * @brief Construct a GPU buffer and upload initial data from a vector.
	 * @param Data Initial data to upload to the GPU.
	 * @param Mode Access mode (default: ReadWrite).
	 */
	Buffer(const std::vector<T> &Data, BufferMode Mode = BufferMode::ReadWrite) : _count(Data.size()), _mode(Mode) {
		InitLayout();
		CreateBuffer();
		if (!Data.empty()) {
			Upload(Data.data(), Data.size());
		}
	}

	/** @brief Move constructor. Transfers ownership of the GPU resource. */
	Buffer(Buffer &&other) noexcept
		: _bufferHandle(other._bufferHandle), _count(other._count), _elementSize(other._elementSize),
		  _mode(other._mode), _boundBinding(other._boundBinding), _layoutConverter(std::move(other._layoutConverter)),
		  _moved(other._moved), _lifetimeToken(std::move(other._lifetimeToken)) {
		other._bufferHandle = Backend::INVALID_BUFFER_HANDLE;
		other._count		= 0;
		other._elementSize	= 0;
		other._boundBinding = -1;
		other._moved		= true;
	}

	/** @brief Move assignment. Destroys the existing resource and transfers ownership. */
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
			_lifetimeToken		= std::move(other._lifetimeToken);
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
	/**
	 * @brief Bind this buffer to the current kernel being defined.
	 * @return BufferRef<T> for DSL access inside a kernel body.
	 * @throw std::runtime_error if called outside a kernel definition or on a moved-from buffer.
	 */
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

		context->RegisterBuffer(binding, typeName, bufferName, ToBackendBufferModeInt(_mode));
		context->BindRuntimeBuffer(binding, _bufferHandle);
		_boundBinding = binding;

		return IR::Value::BufferRef<T>(bufferName, binding);
	}

	/**
	 * @brief Upload data from host memory to the GPU buffer.
	 * @param data Pointer to host memory containing the element data.
	 * @param count Number of elements to upload (must not exceed buffer capacity).
	 * @throw std::out_of_range if count exceeds buffer element count.
	 */
	void Upload(const T *data, size_t count) {
		if (_bufferHandle == Backend::INVALID_BUFFER_HANDLE || data == nullptr || count == 0) {
			return;
		}
		if (count > _count) {
			throw std::out_of_range(std::format("Upload count ({}) exceeds buffer element count ({})", count, _count));
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

	/**
	 * @brief Upload data from a vector to the GPU buffer.
	 * @param data Vector of elements to upload.
	 */
	void Upload(const std::vector<T> &data) {
		if (!data.empty()) {
			Upload(data.data(), data.size());
		}
	}

	/**
	 * @brief Download data from the GPU buffer to host memory.
	 * @param[out] outData Pointer to host memory to receive the element data.
	 * @param count Number of elements to download (must not exceed buffer capacity).
	 * @throw std::out_of_range if count exceeds buffer element count.
	 */
	void Download(T *outData, size_t count) {
		if (_bufferHandle == Backend::INVALID_BUFFER_HANDLE || outData == nullptr || count == 0) {
			return;
		}
		if (count > _count) {
			throw std::out_of_range(
				std::format("Download count ({}) exceeds buffer element count ({})", count, _count));
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
	/**
	 * @brief Get the backend buffer handle.
	 * @return The backend buffer handle.
	 * @return The backend buffer handle.
	 */
	[[nodiscard]] Backend::BufferHandle GetHandle() const {
		return _bufferHandle;
	}
	[[nodiscard]] std::weak_ptr<void> GetLifetimeToken() const {
		return _lifetimeToken;
	}

	/**
	 * @brief Get the number of elements in the buffer.
	 * @return Element count.
	 * @return Element count.
	 */
	[[nodiscard]] size_t GetCount() const {
		return _count;
	}

	/**
	 * @brief Get the buffer access mode.
	 * @return The BufferMode of this buffer.
	 */
	[[nodiscard]] BufferMode GetMode() const {
		return _mode;
	}

	/**
	 * @brief Get the GPU-side size of a single element in bytes.
	 * @return Element size in bytes (may differ from sizeof(T) due to std430 padding).
	 */
	[[nodiscard]] size_t GetElementSize() const {
		return _elementSize;
	}

	/**
	 * @brief Get the total GPU buffer size in bytes.
	 * @return Total size in bytes (element count * element size).
	 */
	[[nodiscard]] size_t GetBufferSize() const {
		return _count * _elementSize;
	}

	/**
	 * @brief Get the shader binding slot index.
	 * @return The binding slot index, or -1 if not yet bound.
	 */
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
	std::shared_ptr<void>				   _lifetimeToken	= std::make_shared<int>(0);
};

} // namespace GPU::Runtime

#endif // EASYGPU_BUFFER_H
