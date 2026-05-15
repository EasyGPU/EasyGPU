#pragma once

/**
 * @file Texture.h
 * @brief 2D Texture for GPU compute shader with backend support.
 */

#ifndef EASYGPU_TEXTURE_H
#define EASYGPU_TEXTURE_H

#include <Backend/Backend.h>
#include <Runtime/Context.h>
#include <Runtime/PixelFormat.h>

#include <IR/Builder/Builder.h>
#include <IR/Value/TextureRef.h>
#include <IR/Value/TextureSampler.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(EASYGPU_BACKEND_VULKAN)
#include <GLAD/glad.h>
#endif

namespace GPU::Runtime {

// Forward declaration
class PBOBuffer;

/**
 * @brief Convert Runtime PixelFormat to Backend PixelFormat.
 * @param format The runtime pixel format.
 * @return The equivalent Backend::PixelFormat enum value.
 */
inline Backend::PixelFormat ToBackendPixelFormat(PixelFormat format) {
	switch (format) {
	case PixelFormat::R8:
		return Backend::PixelFormat::R8;
	case PixelFormat::RG8:
		return Backend::PixelFormat::RG8;
	case PixelFormat::RGBA8:
		return Backend::PixelFormat::RGBA8;
	case PixelFormat::R32F:
		return Backend::PixelFormat::R32F;
	case PixelFormat::RG32F:
		return Backend::PixelFormat::RG32F;
	case PixelFormat::RGBA32F:
		return Backend::PixelFormat::RGBA32F;
	case PixelFormat::R16F:
		return Backend::PixelFormat::R16F;
	case PixelFormat::RG16F:
		return Backend::PixelFormat::RG16F;
	case PixelFormat::RGBA16F:
		return Backend::PixelFormat::RGBA16F;
	case PixelFormat::R32I:
		return Backend::PixelFormat::R32I;
	case PixelFormat::RG32I:
		return Backend::PixelFormat::RG32I;
	case PixelFormat::RGBA32I:
		return Backend::PixelFormat::RGBA32I;
	case PixelFormat::R32UI:
		return Backend::PixelFormat::R32UI;
	case PixelFormat::RG32UI:
		return Backend::PixelFormat::RG32UI;
	case PixelFormat::RGBA32UI:
		return Backend::PixelFormat::RGBA32UI;
	default:
		return Backend::PixelFormat::RGBA8;
	}
}

/**
 * @brief Pixel Buffer Object wrapper for asynchronous texture uploads/downloads.
 */
class PBOBuffer {
public:
	/** @brief PBO lifecycle state. */
	enum class State {
		Idle,
		Uploading,
		Downloading,
		Ready
	};

public:
	/**
	 * @brief Create a PBO buffer.
	 * @param size Size in bytes to allocate on the GPU.
	 * @param isDownload If true, this PBO is used for download (readback) operations.
	 */
	PBOBuffer(size_t size, bool isDownload = false) : _size(size), _isDownload(isDownload), _state(State::Idle) {
		Runtime::AutoInitContext();
		Runtime::Context::GetInstance().MakeCurrent();

		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}

		Backend::BufferDesc desc;
		desc.sizeInBytes = size;
		desc.mode		 = Backend::BufferMode::ReadWrite;
		desc.initialData = nullptr;

		_pboHandle		 = backend->CreateBuffer(desc);
		if (_pboHandle == Backend::INVALID_BUFFER_HANDLE) {
			throw std::runtime_error("Failed to create PBO");
		}
	}

	~PBOBuffer() {
		if (_pboHandle != Backend::INVALID_BUFFER_HANDLE) {
#if !defined(EASYGPU_BACKEND_VULKAN)
			if (_fence) {
				Wait(0);
			}
#endif
			auto *backend = Context::GetBackend();
			if (backend) {
				backend->DestroyBuffer(_pboHandle);
			}
		}
	}

	PBOBuffer(const PBOBuffer &)			= delete;
	PBOBuffer &operator=(const PBOBuffer &) = delete;

	PBOBuffer(PBOBuffer &&other) noexcept
		: _pboHandle(other._pboHandle), _size(other._size), _isDownload(other._isDownload), _state(other._state)
#if !defined(EASYGPU_BACKEND_VULKAN)
		  ,
		  _fence(other._fence)
#endif
	{
		other._pboHandle = Backend::INVALID_BUFFER_HANDLE;
		other._size		 = 0;
#if !defined(EASYGPU_BACKEND_VULKAN)
		other._fence = nullptr;
#endif
	}

	PBOBuffer &operator=(PBOBuffer &&other) noexcept {
		if (this != &other) {
			if (_pboHandle != Backend::INVALID_BUFFER_HANDLE) {
				auto *backend = Context::GetBackend();
				if (backend) {
					backend->DestroyBuffer(_pboHandle);
				}
			}
			_pboHandle		 = other._pboHandle;
			_size			 = other._size;
			_isDownload		 = other._isDownload;
			_state			 = other._state;
			other._pboHandle = Backend::INVALID_BUFFER_HANDLE;
			other._size		 = 0;
		}
		return *this;
	}

public:
	/**
	 * @brief Map the PBO for CPU write access (upload PBOs only).
	 * @return Pointer to the mapped GPU memory, or nullptr on failure.
	 */
	void *MapWrite() {
		if (_isDownload) {
			throw std::runtime_error("Cannot map write on download PBO");
		}
		auto *backend = Context::GetBackend();
		if (!backend) {
			return nullptr;
		}
		return backend->MapBuffer(_pboHandle, false, true);
	}

	/**
	 * @brief Map the PBO for CPU read access (download PBOs only).
	 * @return Pointer to the mapped GPU memory, or nullptr on failure.
	 */
	const void *MapRead() {
		if (!_isDownload) {
			throw std::runtime_error("Cannot map read on upload PBO");
		}
		auto *backend = Context::GetBackend();
		if (!backend) {
			return nullptr;
		}
		return backend->MapBuffer(_pboHandle, true, false);
	}

	/** @brief Unmap the PBO buffer, making GPU writes visible. */
	void Unmap() {
		auto *backend = Context::GetBackend();
		if (!backend) {
			return;
		}
		backend->UnmapBuffer(_pboHandle);
	}

	/**
	 * @brief Copy host data into the PBO for upload.
	 * @param data Pointer to the source data on the host.
	 * @param size Number of bytes to copy (must not exceed PBO size).
	 */
	void CopyData(const void *data, size_t size) {
		if (_isDownload) {
			throw std::runtime_error("Cannot copy data to download PBO");
		}
		if (size > _size) {
			throw std::runtime_error("Data size exceeds PBO size");
		}
		auto *backend = Context::GetBackend();
		if (!backend) {
			return;
		}
		backend->UploadBuffer(_pboHandle, 0, size, data);
	}

	Backend::BufferHandle GetHandle() const {
		return _pboHandle;
	}

	size_t GetSize() const {
		return _size;
	}

	bool IsDownload() const {
		return _isDownload;
	}

	State GetState() const {
		return _state;
	}

	void SetState(State state) {
		_state = state;
	}

	void InsertFence() {
#if !defined(EASYGPU_BACKEND_VULKAN)
		if (_fence) {
			glDeleteSync(_fence);
		}
		_fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
#endif
	}

	bool IsComplete() {
#if !defined(EASYGPU_BACKEND_VULKAN)
		if (!_fence) {
			return true;
		}
		GLenum result = glClientWaitSync(_fence, 0, 0);
		return (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED);
#else
		return true;
#endif
	}

	void Wait(uint64_t timeout = 0) {
#if !defined(EASYGPU_BACKEND_VULKAN)
		if (_fence) {
			GLuint64 timeoutNs = timeout ? (timeout * 1000000ULL) : ~0ULL;
			glClientWaitSync(_fence, GL_SYNC_FLUSH_COMMANDS_BIT, timeoutNs);
			glDeleteSync(_fence);
			_fence = nullptr;
		}
#else
		(void)timeout;
		auto *backend = Context::GetBackend();
		if (backend) {
			backend->Finish();
		}
#endif
	}

	void DeleteFence() {
	}

private:
	Backend::BufferHandle _pboHandle  = Backend::INVALID_BUFFER_HANDLE;
	size_t				  _size		  = 0;
	bool				  _isDownload = false;
	State				  _state	  = State::Idle;
#if !defined(EASYGPU_BACKEND_VULKAN)
	GLsync _fence = nullptr;
#endif
};

/**
 * @brief Pool of PBO buffers for asynchronous multi-buffered texture transfers.
 */
class PBOPool {
public:
	/**
	 * @brief Create a PBO pool.
	 * @param bufferSize Size in bytes for each PBO in the pool.
	 * @param bufferCount Number of PBOs to create.
	 * @param isDownload If true, PBOs are for download (readback) operations.
	 */
	PBOPool(size_t bufferSize, uint32_t bufferCount, bool isDownload = false)
		: _bufferSize(bufferSize), _isDownload(isDownload) {
		for (uint32_t i = 0; i < bufferCount; ++i) {
			_buffers.push_back(std::make_unique<PBOBuffer>(bufferSize, isDownload));
		}
	}

	/**
	 * @brief Get an idle PBO from the pool (non-blocking).
	 * @return Pointer to an idle PBO, or nullptr if all are busy.
	 */
	PBOBuffer *AcquireIdle() {
		for (auto &pbo : _buffers) {
			if (pbo->GetState() == PBOBuffer::State::Idle) {
				return pbo.get();
			}
		}
		return nullptr;
	}

	PBOBuffer *AcquireIdleBlocking(uint32_t timeoutMs = 1000) {
		auto start = std::chrono::steady_clock::now();
		while (true) {
			PBOBuffer *pbo = AcquireIdle();
			if (pbo)
				return pbo;

			UpdateStates();

			auto elapsed = std::chrono::steady_clock::now() - start;
			if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeoutMs) {
				return nullptr;
			}

			std::this_thread::sleep_for(std::chrono::microseconds(100));
		}
	}

	void UpdateStates() {
		for (auto &pbo : _buffers) {
			if (pbo->GetState() == PBOBuffer::State::Uploading || pbo->GetState() == PBOBuffer::State::Downloading) {
				if (pbo->IsComplete()) {
					pbo->DeleteFence();
					if (pbo->IsDownload()) {
						pbo->SetState(PBOBuffer::State::Ready);
					} else {
						pbo->SetState(PBOBuffer::State::Idle);
					}
				}
			}
		}
	}

	void SyncAll() {
		for (auto &pbo : _buffers) {
			if (pbo->GetState() != PBOBuffer::State::Idle) {
				pbo->Wait();
				pbo->DeleteFence();
				pbo->SetState(PBOBuffer::State::Idle);
			}
		}
	}

	const std::vector<std::unique_ptr<PBOBuffer>> &GetBuffers() const {
		return _buffers;
	}

private:
	size_t									_bufferSize;
	bool									_isDownload;
	std::vector<std::unique_ptr<PBOBuffer>> _buffers;
};

/**
 * @brief 2D GPU texture for compute shader image access.
 * @tparam Format The pixel format of the texture.
 */
template <PixelFormat Format> class Texture2D {
public:
	/**
	 * @brief Create a 2D texture with uninitialized data.
	 * @param width Texture width in pixels.
	 * @param height Texture height in pixels.
	 */
	Texture2D(uint32_t width, uint32_t height) : _width(width), _height(height), _format(Format) {
		CreateTexture(nullptr);
	}

	/**
	 * @brief Create a 2D texture and upload initial data.
	 * @param width Texture width in pixels.
	 * @param height Texture height in pixels.
	 * @param data Pointer to the initial pixel data to upload.
	 */
	Texture2D(uint32_t width, uint32_t height, const void *data) : _width(width), _height(height), _format(Format) {
		CreateTexture(data);
	}

	Texture2D(Texture2D &&other) noexcept
		: _textureHandle(other._textureHandle), _width(other._width), _height(other._height), _format(other._format),
		  _boundBinding(other._boundBinding), _uploadPool(std::move(other._uploadPool)),
		  _downloadPool(std::move(other._downloadPool)), _currentUploadPBO(other._currentUploadPBO),
		  _currentDownloadPBO(other._currentDownloadPBO) {
		other._textureHandle	  = Backend::INVALID_TEXTURE_HANDLE;
		other._width			  = 0;
		other._height			  = 0;
		other._boundBinding		  = -1;
		other._currentUploadPBO	  = nullptr;
		other._currentDownloadPBO = nullptr;
	}

	Texture2D &operator=(Texture2D &&other) noexcept {
		if (this != &other) {
			DestroyTexture();
			_textureHandle			  = other._textureHandle;
			_width					  = other._width;
			_height					  = other._height;
			_format					  = other._format;
			_boundBinding			  = other._boundBinding;
			_uploadPool				  = std::move(other._uploadPool);
			_downloadPool			  = std::move(other._downloadPool);
			_currentUploadPBO		  = other._currentUploadPBO;
			_currentDownloadPBO		  = other._currentDownloadPBO;
			other._textureHandle	  = Backend::INVALID_TEXTURE_HANDLE;
			other._width			  = 0;
			other._height			  = 0;
			other._boundBinding		  = -1;
			other._currentUploadPBO	  = nullptr;
			other._currentDownloadPBO = nullptr;
		}
		return *this;
	}

	~Texture2D() {
		DestroyTexture();
	}

	Texture2D(const Texture2D &)			= delete;
	Texture2D &operator=(const Texture2D &) = delete;

public:
	/**
	 * @brief Upload pixel data to the full texture.
	 * @param data Pointer to pixel data in the texture's pixel format.
	 */
	void Upload(const void *data) {
		if (_textureHandle == Backend::INVALID_TEXTURE_HANDLE || data == nullptr) {
			return;
		}

		Runtime::Context::GetInstance().MakeCurrent();

		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}

		backend->UploadTexture(_textureHandle, 0, 0, _width, _height, data);
	}

	void UploadSubRegion(uint32_t x, uint32_t y, uint32_t w, uint32_t h, const void *data) {
		if (_textureHandle == Backend::INVALID_TEXTURE_HANDLE || data == nullptr) {
			return;
		}
		if (x + w > _width || y + h > _height) {
			throw std::out_of_range("Upload region exceeds texture bounds");
		}

		Runtime::Context::GetInstance().MakeCurrent();

		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}

		backend->UploadTexture(_textureHandle, x, y, w, h, data);
	}

	/**
	 * @brief Download pixel data from the full texture to host memory.
	 * @param[out] outData Pointer to host memory to receive pixel data.
	 */
	void Download(void *outData) const {
		if (_textureHandle == Backend::INVALID_TEXTURE_HANDLE || outData == nullptr) {
			return;
		}

		Runtime::Context::GetInstance().MakeCurrent();

		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}

		backend->DownloadTexture(_textureHandle, 0, 0, _width, _height, outData);
	}

	/**
	 * @brief Download pixel data into a typed vector, resizing if needed.
	 * @param[out] outData Vector to receive pixel data.
	 */
	template <typename T> void Download(std::vector<T> &outData) const {
		size_t requiredSize = (_width * _height * GetBytesPerPixel() + sizeof(T) - 1) / sizeof(T);
		if (outData.size() < requiredSize) {
			outData.resize(requiredSize);
		}
		Download(outData.data());
	}

public:
	void InitUploadPBOPool(uint32_t bufferCount = 2) {
		if (!_uploadPool) {
			size_t size = _width * _height * GetBytesPerPixel();
			_uploadPool = std::make_unique<PBOPool>(size, bufferCount, false);
		}
	}

	void InitDownloadPBOPool(uint32_t bufferCount = 2) {
		if (!_downloadPool) {
			size_t size	  = _width * _height * GetBytesPerPixel();
			_downloadPool = std::make_unique<PBOPool>(size, bufferCount, true);
		}
	}

	bool UploadAsync(const void *data) {
		if (!_uploadPool) {
			InitUploadPBOPool(2);
		}
		_uploadPool->UpdateStates();
		PBOBuffer *pbo = _uploadPool->AcquireIdle();
		if (!pbo) {
			return false;
		}

		Runtime::Context::GetInstance().MakeCurrent();

		size_t dataSize = _width * _height * GetBytesPerPixel();
		pbo->CopyData(data, dataSize);
		Upload(data);
		pbo->SetState(PBOBuffer::State::Uploading);
		pbo->InsertFence();
		return true;
	}

	bool UploadAsyncStream(const void *data, uint32_t timeoutMs = 1000) {
		if (!_uploadPool) {
			InitUploadPBOPool(2);
		}

		PBOBuffer *pbo = _uploadPool->AcquireIdleBlocking(timeoutMs);
		if (!pbo) {
			throw std::runtime_error("UploadAsyncStream timeout - no idle PBO available");
		}

		Runtime::Context::GetInstance().MakeCurrent();

		size_t dataSize = _width * _height * GetBytesPerPixel();
		pbo->CopyData(data, dataSize);
		Upload(data);

		pbo->SetState(PBOBuffer::State::Uploading);
		pbo->InsertFence();

		return true;
	}

	/**
	 * Experimental / Not fully implemented
	 */
	bool DownloadAsync() {
		throw std::runtime_error("Async texture transfer is not yet fully implemented");
	}

	bool GetDownloadData(void *outData) {
		if (!_downloadPool)
			return false;

		_downloadPool->UpdateStates();

		for (auto &pbo : _downloadPool->GetBuffers()) {
			if (pbo->GetState() == PBOBuffer::State::Ready) {
				const void *mapped = pbo->MapRead();
				if (mapped) {
					std::memcpy(outData, mapped, pbo->GetSize());
					pbo->Unmap();
					pbo->SetState(PBOBuffer::State::Idle);
					return true;
				}
			}
		}
		return false;
	}

	void Sync() {
		if (_uploadPool) {
			_uploadPool->SyncAll();
		}
		if (_downloadPool) {
			_downloadPool->SyncAll();
		}
	}

	bool IsIdle() {
		if (_uploadPool) {
			_uploadPool->UpdateStates();
			for (auto &pbo : _uploadPool->GetBuffers()) {
				if (pbo->GetState() != PBOBuffer::State::Idle) {
					return false;
				}
			}
		}
		if (_downloadPool) {
			_downloadPool->UpdateStates();
			for (auto &pbo : _downloadPool->GetBuffers()) {
				if (pbo->GetState() != PBOBuffer::State::Idle) {
					return false;
				}
			}
		}
		return true;
	}

public:
	[[nodiscard]] IR::Value::TextureRef<Format> Bind() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("Texture2D::Bind() called outside of Kernel definition");
		}

		uint32_t	binding		= context->AllocateTextureBinding();
		std::string textureName = std::format("tex{}", binding);
		context->RegisterTexture(binding, _format, textureName, _width, _height);
		context->BindRuntimeTexture(binding, static_cast<uint32_t>(_textureHandle));
		_boundBinding = static_cast<int>(binding);

		return IR::Value::TextureRef<Format>(textureName, binding, _width, _height);
	}

	[[nodiscard]] IR::Value::TextureSampler2D<Format> BindSampler() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("Texture2D::BindSampler() called outside of Kernel definition");
		}

		uint32_t	binding		= context->AllocateTextureBinding();
		std::string textureName = std::format("tex{}", binding);
		context->RegisterTexture(binding, _format, textureName, _width, _height, true);
		context->BindRuntimeTexture(binding, static_cast<uint32_t>(_textureHandle));
		_boundBinding = static_cast<int>(binding);

		return IR::Value::TextureSampler2D<Format>(textureName, binding, _width, _height);
	}

public:
	[[nodiscard]] Backend::TextureHandle GetHandle() const {
		return _textureHandle;
	}
	[[nodiscard]] uint32_t GetWidth() const {
		return _width;
	}
	[[nodiscard]] uint32_t GetHeight() const {
		return _height;
	}
	static constexpr PixelFormat GetFormat() {
		return Format;
	}
	[[nodiscard]] size_t GetBytesPerPixel() const {
		return Runtime::GetBytesPerPixel(_format);
	}
	[[nodiscard]] size_t GetSizeInBytes() const {
		return _width * _height * GetBytesPerPixel();
	}
	[[nodiscard]] int GetBinding() const {
		return _boundBinding;
	}

private:
	void CreateTexture(const void *initialData) {
		Runtime::AutoInitContext();
		Runtime::Context::GetInstance().MakeCurrent();

		if (_width == 0 || _height == 0) {
			throw std::invalid_argument("Texture dimensions must be non-zero");
		}

		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}

		Backend::TextureDesc desc;
		desc.width		 = _width;
		desc.height		 = _height;
		desc.depth		 = 1;
		desc.format		 = ToBackendPixelFormat(_format);
		desc.initialData = initialData;

		_textureHandle	 = backend->CreateTexture(desc);
		if (_textureHandle == Backend::INVALID_TEXTURE_HANDLE) {
			throw std::runtime_error("Failed to create GPU texture");
		}
	}

	void DestroyTexture() {
		Sync();
		if (_textureHandle != Backend::INVALID_TEXTURE_HANDLE) {
			auto *backend = Context::GetBackend();
			if (backend) {
				backend->DestroyTexture(_textureHandle);
			}
			_textureHandle = Backend::INVALID_TEXTURE_HANDLE;
		}
	}

private:
	Backend::TextureHandle	 _textureHandle = Backend::INVALID_TEXTURE_HANDLE;
	uint32_t				 _width			= 0;
	uint32_t				 _height		= 0;
	PixelFormat				 _format		= Format;
	int						 _boundBinding	= -1;

	std::unique_ptr<PBOPool> _uploadPool;
	std::unique_ptr<PBOPool> _downloadPool;
	PBOBuffer				*_currentUploadPBO	 = nullptr;
	PBOBuffer				*_currentDownloadPBO = nullptr;
};

/** @brief Convenience typedef for RGBA8 2D texture. */
using TextureRGBA8	 = Texture2D<PixelFormat::RGBA8>;
/** @brief Convenience typedef for RGBA32F 2D texture. */
using TextureRGBA32F = Texture2D<PixelFormat::RGBA32F>;
/** @brief Convenience typedef for R32F 2D texture. */
using TextureR32F	 = Texture2D<PixelFormat::R32F>;
/** @brief Convenience typedef for RG32F 2D texture. */
using TextureRG32F	 = Texture2D<PixelFormat::RG32F>;
/** @brief Convenience typedef for R8 2D texture. */
using TextureR8		 = Texture2D<PixelFormat::R8>;

/**
 * @brief 3D GPU texture for compute shader image access.
 * @tparam Format The pixel format of the texture.
 */
template <PixelFormat Format> class Texture3D {
public:
	/**
	 * @brief Create a 3D texture with uninitialized data.
	 * @param width Texture width in pixels.
	 * @param height Texture height in pixels.
	 * @param depth Texture depth in pixels.
	 */
	Texture3D(uint32_t width, uint32_t height, uint32_t depth)
		: _width(width), _height(height), _depth(depth), _format(Format) {
		CreateTexture(nullptr);
	}

	Texture3D(uint32_t width, uint32_t height, uint32_t depth, const void *data)
		: _width(width), _height(height), _depth(depth), _format(Format) {
		CreateTexture(data);
	}

	Texture3D(Texture3D &&other) noexcept
		: _textureHandle(other._textureHandle), _width(other._width), _height(other._height), _depth(other._depth),
		  _format(other._format), _boundBinding(other._boundBinding), _uploadPool(std::move(other._uploadPool)),
		  _downloadPool(std::move(other._downloadPool)), _currentUploadPBO(other._currentUploadPBO),
		  _currentDownloadPBO(other._currentDownloadPBO) {
		other._textureHandle	  = Backend::INVALID_TEXTURE_HANDLE;
		other._width			  = 0;
		other._height			  = 0;
		other._depth			  = 0;
		other._boundBinding		  = -1;
		other._currentUploadPBO	  = nullptr;
		other._currentDownloadPBO = nullptr;
	}

	Texture3D &operator=(Texture3D &&other) noexcept {
		if (this != &other) {
			DestroyTexture();
			_textureHandle			  = other._textureHandle;
			_width					  = other._width;
			_height					  = other._height;
			_depth					  = other._depth;
			_format					  = other._format;
			_boundBinding			  = other._boundBinding;
			_uploadPool				  = std::move(other._uploadPool);
			_downloadPool			  = std::move(other._downloadPool);
			_currentUploadPBO		  = other._currentUploadPBO;
			_currentDownloadPBO		  = other._currentDownloadPBO;
			other._textureHandle	  = Backend::INVALID_TEXTURE_HANDLE;
			other._width			  = 0;
			other._height			  = 0;
			other._depth			  = 0;
			other._boundBinding		  = -1;
			other._currentUploadPBO	  = nullptr;
			other._currentDownloadPBO = nullptr;
		}
		return *this;
	}

	~Texture3D() {
		DestroyTexture();
	}

	Texture3D(const Texture3D &)			= delete;
	Texture3D &operator=(const Texture3D &) = delete;

public:
	/**
	 * @brief Upload pixel data to the full 3D texture.
	 * @param data Pointer to pixel data in the texture's pixel format.
	 */
	void Upload(const void *data) {
		if (_textureHandle == Backend::INVALID_TEXTURE_HANDLE || data == nullptr) {
			return;
		}
		Runtime::Context::GetInstance().MakeCurrent();
		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}
		backend->UploadTexture3D(_textureHandle, 0, 0, 0, _width, _height, _depth, data);
	}

	void UploadSubRegion(uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t h, uint32_t d, const void *data) {
		if (_textureHandle == Backend::INVALID_TEXTURE_HANDLE || data == nullptr) {
			return;
		}
		if (x + w > _width || y + h > _height || z + d > _depth) {
			throw std::out_of_range("Upload region exceeds texture bounds");
		}
		Runtime::Context::GetInstance().MakeCurrent();
		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}
		backend->UploadTexture3D(_textureHandle, x, y, z, w, h, d, data);
	}

	void Download(void *outData) const {
		if (_textureHandle == Backend::INVALID_TEXTURE_HANDLE || outData == nullptr) {
			return;
		}
		Runtime::Context::GetInstance().MakeCurrent();
		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}
		backend->DownloadTexture3D(_textureHandle, 0, 0, 0, _width, _height, _depth, outData);
	}

	template <typename T> void Download(std::vector<T> &outData) const {
		size_t requiredSize = (_width * _height * _depth * GetBytesPerPixel() + sizeof(T) - 1) / sizeof(T);
		if (outData.size() < requiredSize) {
			outData.resize(requiredSize);
		}
		Download(outData.data());
	}

public:
	void InitUploadPBOPool(uint32_t bufferCount = 2) {
		if (!_uploadPool) {
			size_t size = _width * _height * _depth * GetBytesPerPixel();
			_uploadPool = std::make_unique<PBOPool>(size, bufferCount, false);
		}
	}

	void InitDownloadPBOPool(uint32_t bufferCount = 2) {
		if (!_downloadPool) {
			size_t size	  = _width * _height * _depth * GetBytesPerPixel();
			_downloadPool = std::make_unique<PBOPool>(size, bufferCount, true);
		}
	}

	bool UploadAsync(const void *data) {
		if (!_uploadPool) {
			InitUploadPBOPool(2);
		}
		_uploadPool->UpdateStates();
		PBOBuffer *pbo = _uploadPool->AcquireIdle();
		if (!pbo) {
			return false;
		}
		Runtime::Context::GetInstance().MakeCurrent();
		size_t dataSize = _width * _height * _depth * GetBytesPerPixel();
		pbo->CopyData(data, dataSize);
		Upload(data);
		pbo->SetState(PBOBuffer::State::Uploading);
		pbo->InsertFence();
		return true;
	}

	bool UploadAsyncStream(const void *data, uint32_t timeoutMs = 1000) {
		if (!_uploadPool) {
			InitUploadPBOPool(2);
		}
		PBOBuffer *pbo = _uploadPool->AcquireIdleBlocking(timeoutMs);
		if (!pbo) {
			throw std::runtime_error("UploadAsyncStream timeout - no idle PBO available");
		}
		Runtime::Context::GetInstance().MakeCurrent();
		size_t dataSize = _width * _height * _depth * GetBytesPerPixel();
		pbo->CopyData(data, dataSize);
		Upload(data);
		pbo->SetState(PBOBuffer::State::Uploading);
		pbo->InsertFence();
		return true;
	}

	bool DownloadAsync() {
		if (!_downloadPool) {
			InitDownloadPBOPool(2);
		}
		_downloadPool->UpdateStates();
		PBOBuffer *pbo = _downloadPool->AcquireIdle();
		if (!pbo) {
			return false;
		}
		Runtime::Context::GetInstance().MakeCurrent();
		pbo->SetState(PBOBuffer::State::Downloading);
		pbo->InsertFence();
		return true;
	}

	bool GetDownloadData(void *outData) {
		if (!_downloadPool)
			return false;
		_downloadPool->UpdateStates();
		for (auto &pbo : _downloadPool->GetBuffers()) {
			if (pbo->GetState() == PBOBuffer::State::Ready) {
				const void *mapped = pbo->MapRead();
				if (mapped) {
					std::memcpy(outData, mapped, pbo->GetSize());
					pbo->Unmap();
					pbo->SetState(PBOBuffer::State::Idle);
					return true;
				}
			}
		}
		return false;
	}

	void Sync() {
		if (_uploadPool) {
			_uploadPool->SyncAll();
		}
		if (_downloadPool) {
			_downloadPool->SyncAll();
		}
	}

	bool IsIdle() {
		if (_uploadPool) {
			_uploadPool->UpdateStates();
			for (auto &pbo : _uploadPool->GetBuffers()) {
				if (pbo->GetState() != PBOBuffer::State::Idle) {
					return false;
				}
			}
		}
		if (_downloadPool) {
			_downloadPool->UpdateStates();
			for (auto &pbo : _downloadPool->GetBuffers()) {
				if (pbo->GetState() != PBOBuffer::State::Idle) {
					return false;
				}
			}
		}
		return true;
	}

public:
	[[nodiscard]] IR::Value::TextureRef3D<Format> Bind() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("Texture3D::Bind() called outside of Kernel definition");
		}
		uint32_t	binding		= context->AllocateTextureBinding();
		std::string textureName = std::format("tex{}", binding);
		context->RegisterTexture3D(binding, _format, textureName, _width, _height, _depth);
		context->BindRuntimeTexture(binding, static_cast<uint32_t>(_textureHandle));
		_boundBinding = static_cast<int>(binding);
		return IR::Value::TextureRef3D<Format>(textureName, binding, _width, _height, _depth);
	}

	[[nodiscard]] IR::Value::TextureSampler3D<Format> BindSampler() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("Texture3D::BindSampler() called outside of Kernel definition");
		}
		uint32_t	binding		= context->AllocateTextureBinding();
		std::string textureName = std::format("tex{}", binding);
		context->RegisterTexture3D(binding, _format, textureName, _width, _height, _depth, true);
		context->BindRuntimeTexture(binding, static_cast<uint32_t>(_textureHandle));
		_boundBinding = static_cast<int>(binding);
		return IR::Value::TextureSampler3D<Format>(textureName, binding, _width, _height, _depth);
	}

public:
	[[nodiscard]] Backend::TextureHandle GetHandle() const {
		return _textureHandle;
	}
	[[nodiscard]] uint32_t GetWidth() const {
		return _width;
	}
	[[nodiscard]] uint32_t GetHeight() const {
		return _height;
	}
	[[nodiscard]] uint32_t GetDepth() const {
		return _depth;
	}
	static constexpr PixelFormat GetFormat() {
		return Format;
	}
	[[nodiscard]] size_t GetBytesPerPixel() const {
		return Runtime::GetBytesPerPixel(_format);
	}
	[[nodiscard]] size_t GetSizeInBytes() const {
		return _width * _height * _depth * GetBytesPerPixel();
	}
	[[nodiscard]] int GetBinding() const {
		return _boundBinding;
	}

private:
	void CreateTexture(const void *initialData) {
		Runtime::AutoInitContext();
		Runtime::Context::GetInstance().MakeCurrent();
		if (_width == 0 || _height == 0 || _depth == 0) {
			throw std::invalid_argument("Texture dimensions must be non-zero");
		}
		auto *backend = Context::GetBackend();
		if (!backend) {
			throw std::runtime_error("Backend not available");
		}
		Backend::TextureDesc desc;
		desc.width		 = _width;
		desc.height		 = _height;
		desc.depth		 = _depth;
		desc.format		 = ToBackendPixelFormat(_format);
		desc.initialData = initialData;
		_textureHandle	 = backend->CreateTexture(desc);
		if (_textureHandle == Backend::INVALID_TEXTURE_HANDLE) {
			throw std::runtime_error("Failed to create GPU texture");
		}
	}

	void DestroyTexture() {
		Sync();
		if (_textureHandle != Backend::INVALID_TEXTURE_HANDLE) {
			auto *backend = Context::GetBackend();
			if (backend) {
				backend->DestroyTexture(_textureHandle);
			}
			_textureHandle = Backend::INVALID_TEXTURE_HANDLE;
		}
	}

private:
	Backend::TextureHandle	 _textureHandle = Backend::INVALID_TEXTURE_HANDLE;
	uint32_t				 _width			= 0;
	uint32_t				 _height		= 0;
	uint32_t				 _depth			= 0;
	PixelFormat				 _format		= Format;
	int						 _boundBinding	= -1;
	std::unique_ptr<PBOPool> _uploadPool;
	std::unique_ptr<PBOPool> _downloadPool;
	PBOBuffer				*_currentUploadPBO	 = nullptr;
	PBOBuffer				*_currentDownloadPBO = nullptr;
};

/** @brief Convenience typedef for RGBA8 3D texture. */
using Texture3DRGBA8   = Texture3D<PixelFormat::RGBA8>;
/** @brief Convenience typedef for RGBA32F 3D texture. */
using Texture3DRGBA32F = Texture3D<PixelFormat::RGBA32F>;
/** @brief Convenience typedef for R32F 3D texture. */
using Texture3DR32F	   = Texture3D<PixelFormat::R32F>;
/** @brief Convenience typedef for RG32F 3D texture. */
using Texture3DRG32F   = Texture3D<PixelFormat::RG32F>;
/** @brief Convenience typedef for R8 3D texture. */
using Texture3DR8	   = Texture3D<PixelFormat::R8>;

} // namespace GPU::Runtime

#endif // EASYGPU_TEXTURE_H
