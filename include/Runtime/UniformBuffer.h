#pragma once

/**
 * @file UniformBuffer.h
 * @brief Uniform Buffer Object (UBO) for passing large structs to GPU shaders.
 */

#ifndef EASYGPU_UNIFORMBUFFER_H
#define EASYGPU_UNIFORMBUFFER_H

#include <Backend/Backend.h>
#include <IR/Builder/Builder.h>
#include <IR/Value/Var.h>
#include <Runtime/Context.h>
#include <Utility/Meta/Std430Layout.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace GPU::Runtime {

/**
 * @brief Non-template base class for UniformBuffer, used for type erasure.
 */
class UniformBufferBase {
public:
	virtual ~UniformBufferBase()					 = default;

	/**
	 * @brief Get the backend buffer handle of this UBO.
	 * @return The backend buffer handle.
	 */
	virtual Backend::BufferHandle GetHandle() const	 = 0;

	/**
	 * @brief Get the GPU-side layout size in bytes.
	 * @return The std430 layout size.
	 */
	virtual size_t				  GetGPUSize() const = 0;

	/** @brief Upload the current host-side value to the GPU. */
	virtual void				  UploadToGPU()		 = 0;
};

/**
 * @brief Uniform Buffer Object (UBO) for passing structured data to GPU shaders.
 *
 * Uses std430 layout for automatic GPU memory layout conversion.
 * @tparam T The struct type stored in this UBO (must be registered with GPU_META_STRUCT).
 */
template <typename T> class UniformBuffer : public UniformBufferBase {
public:
	/** @brief Default constructor. Creates an empty UBO with default-initialized value. */
	UniformBuffer() : _uboHandle(Backend::INVALID_BUFFER_HANDLE) {
		CreateUBO();
	}

	/**
	 * @brief Construct a UBO with an initial value and upload it to the GPU.
	 * @param value The initial value to store.
	 */
	explicit UniformBuffer(const T &value) : _value(value), _uboHandle(Backend::INVALID_BUFFER_HANDLE) {
		CreateUBO();
		Upload();
	}

	~UniformBuffer() {
		auto *backend = Context::GetBackend();
		if (backend && _uboHandle != Backend::INVALID_BUFFER_HANDLE) {
			backend->DestroyBuffer(_uboHandle);
		}
	}

	UniformBuffer(const UniformBuffer &)			= delete;
	UniformBuffer &operator=(const UniformBuffer &) = delete;
	UniformBuffer(UniformBuffer &&other) noexcept : _value(std::move(other._value)), _uboHandle(other._uboHandle) {
		other._uboHandle = Backend::INVALID_BUFFER_HANDLE;
	}
	UniformBuffer &operator=(UniformBuffer &&other) noexcept {
		if (this != &other) {
			_value			 = std::move(other._value);
			_uboHandle		 = other._uboHandle;
			other._uboHandle = Backend::INVALID_BUFFER_HANDLE;
		}
		return *this;
	}

public:
	/**
	 * @brief Assign a new value and upload it to the GPU.
	 * @param value The new value to store.
	 * @return Reference to this UBO.
	 */
	UniformBuffer &operator=(const T &value) {
		{
			std::lock_guard<std::mutex> lock(_mutex);
			_value = value;
		}
		Upload();
		return *this;
	}

	/**
	 * @brief Set the value and upload it to the GPU (thread-safe).
	 * @param value The new value.
	 */
	void SetValue(const T &value) {
		{
			std::lock_guard<std::mutex> lock(_mutex);
			_value = value;
		}
		Upload();
	}

	/**
	 * @brief Get the current host-side value (thread-safe).
	 * @return A copy of the stored value.
	 */
	[[nodiscard]] T GetValue() const {
		std::lock_guard<std::mutex> lock(_mutex);
		return _value;
	}

	/**
	 * @brief Get the backend buffer handle of this UBO.
	 * @return The backend buffer handle.
	 */
	[[nodiscard]] Backend::BufferHandle GetHandle() const override {
		return _uboHandle;
	}

	/**
	 * @brief Get the GPU-side layout size of this UBO in bytes.
	 * @return The std430 layout size.
	 */
	[[nodiscard]] size_t GetGPUSize() const override {
		GPU::Meta::Std430Converter<T> converter;
		return converter.GetGPULayoutSize();
	}

	void UploadToGPU() override {
		Upload();
	}

public:
	/**
	 * @brief Load this UBO in a kernel definition context.
	 *
	 * Registers the UBO with the kernel and returns an external Var<T> referencing
	 * the UBO members. The returned Var can be read but not written.
	 *
	 * @return Var<T> referencing the UBO data for DSL access.
	 * @throw std::runtime_error if called outside a Kernel definition.
	 */
	[[nodiscard]] IR::Value::Var<T> Load() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("UniformBuffer::Load() called outside of Kernel definition");
		}

		// Register UBO with context
		std::string varName =
			context->RegisterUniformBuffer(GPU::Meta::StructMeta<T>::glslTypeName, this, GPU::Meta::GetStd430Size<T>());

		// Return external Var referencing the UBO member
		return IR::Value::Var<T>(varName, true);
	}

private:
	void CreateUBO() {
		auto *backend = Context::GetBackend();
		if (!backend)
			return;

		GPU::Meta::Std430Converter<T> converter;
		size_t						  gpuSize = converter.GetGPULayoutSize();

		std::vector<unsigned char>	  gpuData(gpuSize, 0);
		converter.ConvertToGPU(&_value, gpuData.data(), 1);

		Backend::BufferDesc desc;
		desc.sizeInBytes = gpuSize;
		desc.mode = Backend::BufferMode::Read;
		desc.initialData = gpuData.data();
		_uboHandle = backend->CreateBuffer(desc);
	}

	void Upload() {
		auto *backend = Context::GetBackend();
		if (!backend || _uboHandle == Backend::INVALID_BUFFER_HANDLE)
			return;

		GPU::Meta::Std430Converter<T> converter;
		size_t						  gpuSize = converter.GetGPULayoutSize();

		std::vector<unsigned char>	  gpuData(gpuSize, 0);
		{
			std::lock_guard<std::mutex> lock(_mutex);
			converter.ConvertToGPU(&_value, gpuData.data(), 1);
		}

		backend->UploadBuffer(_uboHandle, 0, gpuSize, gpuData.data());
	}

	T					  _value{};
	Backend::BufferHandle _uboHandle;
	mutable std::mutex	  _mutex;
};

} // namespace GPU::Runtime

#endif // EASYGPU_UNIFORMBUFFER_H
