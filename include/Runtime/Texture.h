#pragma once

/**
 * Texture.h:
 *      @Descripiton    :   2D/3D Texture for GPU compute shader with PBO support
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 * 
 * Supports:
 *   - Creating empty texture of specified size
 *   - Creating from raw pixel buffer
 *   - Uploading raw pixel data (sync and async with PBO)
 *   - Downloading to raw pixel buffer (sync and async with PBO)
 *   - Read/Write access in compute shaders
 *   - Multi-PBO async streaming for efficient CPU/GPU parallelism
 */
#ifndef EASYGPU_TEXTURE_H
#define EASYGPU_TEXTURE_H

#include <Runtime/PixelFormat.h>
#include <Runtime/Context.h>

#include <IR/Value/TextureRef.h>
#include <IR/Builder/Builder.h>

#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <format>
#include <memory>
#include <queue>
#include <thread>
#include <chrono>

namespace GPU::Runtime {

    // Forward declaration
    class PBOBuffer;

    /**
     * PBO buffer for async texture transfer
     * Manages a single PBO with its state
     */
    class PBOBuffer {
    public:
        enum class State {
            Idle,       // Ready for use
            Uploading,  // Data being uploaded to texture
            Downloading,// Data being downloaded from texture
            Ready       // Download complete, data available
        };

    public:
        PBOBuffer(size_t size, bool isDownload = false)
            : _size(size), _isDownload(isDownload), _state(State::Idle) {
            Runtime::AutoInitContext();
            Runtime::Context::GetInstance().MakeCurrent();

            glGenBuffers(1, &_pboId);
            if (_pboId == 0) {
                throw std::runtime_error("Failed to create PBO");
            }

            glBindBuffer(isDownload ? GL_PIXEL_PACK_BUFFER : GL_PIXEL_UNPACK_BUFFER, _pboId);
            // For download PBO, we use GL_DYNAMIC_READ
            // For upload PBO, we use GL_STREAM_DRAW
            GLenum usage = isDownload ? GL_DYNAMIC_READ : GL_STREAM_DRAW;
            glBufferData(isDownload ? GL_PIXEL_PACK_BUFFER : GL_PIXEL_UNPACK_BUFFER, size, nullptr, usage);
            glBindBuffer(isDownload ? GL_PIXEL_PACK_BUFFER : GL_PIXEL_UNPACK_BUFFER, 0);
        }

        ~PBOBuffer() {
            if (_pboId != 0) {
                glDeleteBuffers(1, &_pboId);
                _pboId = 0;
            }
        }

        // Disable copy
        PBOBuffer(const PBOBuffer&) = delete;
        PBOBuffer& operator=(const PBOBuffer&) = delete;

        // Enable move
        PBOBuffer(PBOBuffer&& other) noexcept
            : _pboId(other._pboId), _size(other._size), 
              _isDownload(other._isDownload), _state(other._state) {
            other._pboId = 0;
            other._size = 0;
        }

        PBOBuffer& operator=(PBOBuffer&& other) noexcept {
            if (this != &other) {
                if (_pboId != 0) {
                    glDeleteBuffers(1, &_pboId);
                }
                _pboId = other._pboId;
                _size = other._size;
                _isDownload = other._isDownload;
                _state = other._state;
                other._pboId = 0;
                other._size = 0;
            }
            return *this;
        }

    public:
        /**
         * Map PBO for CPU write (upload)
         * @return Pointer to mapped memory
         */
        void* MapWrite() {
            if (_isDownload) {
                throw std::runtime_error("Cannot map write on download PBO");
            }
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pboId);
            void* ptr = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
            return ptr;
        }

        /**
         * Map PBO for CPU read (download)
         * @return Pointer to mapped memory
         */
        const void* MapRead() {
            if (!_isDownload) {
                throw std::runtime_error("Cannot map read on upload PBO");
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, _pboId);
            void* ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
            return ptr;
        }

        /**
         * Unmap the PBO
         */
        void Unmap() {
            if (_isDownload) {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, _pboId);
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            } else {
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pboId);
                glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            }
        }

        /**
         * Copy data to PBO (for upload)
         * @param data Source data pointer
         * @param size Size in bytes to copy
         */
        void CopyData(const void* data, size_t size) {
            if (_isDownload) {
                throw std::runtime_error("Cannot copy data to download PBO");
            }
            if (size > _size) {
                throw std::runtime_error("Data size exceeds PBO size");
            }
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pboId);
            glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, size, data);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        /**
         * Get PBO handle
         */
        uint32_t GetHandle() const { return _pboId; }

        /**
         * Get PBO size
         */
        size_t GetSize() const { return _size; }

        /**
         * Check if this is a download PBO
         */
        bool IsDownload() const { return _isDownload; }

        /**
         * Get current state
         */
        State GetState() const { return _state; }

        /**
         * Set state
         */
        void SetState(State state) { _state = state; }

        /**
         * Insert fence for async operation
         */
        void InsertFence() {
            if (_fence) {
                glDeleteSync(_fence);
            }
            _fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }

        /**
         * Check if operation is complete
         * @return true if complete, false if still pending
         */
        bool IsComplete() {
            if (!_fence) return true;
            
            GLenum result = glClientWaitSync(_fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
            return (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED);
        }

        /**
         * Wait for operation to complete
         * @param timeout Timeout in nanoseconds (0 = wait forever)
         */
        void Wait(uint64_t timeout = 0) {
            if (!_fence) return;
            
            GLenum result = glClientWaitSync(_fence, GL_SYNC_FLUSH_COMMANDS_BIT, timeout);
            if (result == GL_WAIT_FAILED) {
                throw std::runtime_error("Fence wait failed");
            }
        }

        /**
         * Delete fence if exists
         */
        void DeleteFence() {
            if (_fence) {
                glDeleteSync(_fence);
                _fence = nullptr;
            }
        }

    private:
        uint32_t _pboId = 0;
        size_t _size = 0;
        bool _isDownload = false;
        State _state = State::Idle;
        GLsync _fence = nullptr;
    };

    /**
     * PBO pool for managing multiple PBOs
     * Implements a simple ring buffer for PBOs
     */
    class PBOPool {
    public:
        PBOPool(size_t bufferSize, uint32_t bufferCount, bool isDownload = false)
            : _bufferSize(bufferSize), _isDownload(isDownload) {
            for (uint32_t i = 0; i < bufferCount; ++i) {
                _buffers.push_back(std::make_unique<PBOBuffer>(bufferSize, isDownload));
            }
        }

        /**
         * Acquire an idle PBO
         * @return Pointer to idle PBO or nullptr if none available
         */
        PBOBuffer* AcquireIdle() {
            for (auto& pbo : _buffers) {
                if (pbo->GetState() == PBOBuffer::State::Idle) {
                    return pbo.get();
                }
            }
            return nullptr;
        }

        /**
         * Acquire an idle PBO, waiting if necessary
         * @param timeoutMs Maximum time to wait in milliseconds
         * @return Pointer to idle PBO or nullptr if timeout
         */
        PBOBuffer* AcquireIdleBlocking(uint32_t timeoutMs = 1000) {
            auto start = std::chrono::steady_clock::now();
            while (true) {
                // First try to find idle buffer
                PBOBuffer* pbo = AcquireIdle();
                if (pbo) return pbo;

                // Check if any busy buffer has completed
                UpdateStates();

                // Check timeout
                auto elapsed = std::chrono::steady_clock::now() - start;
                if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeoutMs) {
                    return nullptr;
                }

                // Small yield
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        /**
         * Update states of all PBOs based on fence status
         */
        void UpdateStates() {
            for (auto& pbo : _buffers) {
                if (pbo->GetState() == PBOBuffer::State::Uploading ||
                    pbo->GetState() == PBOBuffer::State::Downloading) {
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

        /**
         * Wait for all PBOs to complete
         */
        void SyncAll() {
            for (auto& pbo : _buffers) {
                if (pbo->GetState() != PBOBuffer::State::Idle) {
                    pbo->Wait();
                    pbo->DeleteFence();
                    pbo->SetState(PBOBuffer::State::Idle);
                }
            }
        }

        /**
         * Get all buffers (for iteration)
         */
        const std::vector<std::unique_ptr<PBOBuffer>>& GetBuffers() const {
            return _buffers;
        }

    private:
        size_t _bufferSize;
        bool _isDownload;
        std::vector<std::unique_ptr<PBOBuffer>> _buffers;
    };

    /**
     * 2D Texture class for GPU compute operations
     * @tparam Format The pixel format of the texture
     *
     * Usage:
     *   // Create empty texture
     *   Texture2D<PixelFormat::RGBA8> tex1(1024, 1024);
     *
     *   // Create from raw buffer
     *   std::vector<uint8_t> pixels(1024 * 1024 * 4, 255);
     *   Texture2D<PixelFormat::RGBA8> tex2(1024, 1024, pixels.data());
     *
     *   // Upload new data (sync)
     *   tex.Upload(newPixels.data());
     *
     *   // Async upload with PBO
     *   tex.UploadAsync(newPixels.data());
     *   tex.Sync(); // Wait for completion
     *
     *   // Multi-PBO async streaming
     *   tex.InitUploadPBOPool(3); // 3 PBOs for triple buffering
     *   for (auto& frame : frames) {
     *       tex.UploadAsyncStream(frame.data());
     *       kernel.Dispatch(...); // GPU works in parallel
     *   }
     *   tex.Sync();
     *
     *   // Use in kernel
     *   Kernel1D kernel([&](Var<int>& id) {
     *       auto img = tex.Bind();
     *       Var<Vec4> color = img.Read(x, y);
     *       img.Write(x, y, color * 0.5f);
     *   });
     */
    template<PixelFormat Format>
    class Texture2D {
    public:
        /**
         * Create an empty texture with specified dimensions
         * @param width Texture width in pixels
         * @param height Texture height in pixels
         */
        Texture2D(uint32_t width, uint32_t height)
                : _width(width), _height(height), _format(Format) {
            CreateTexture(nullptr);
        }

        /**
         * Create a texture and upload pixel data
         * @param width Texture width in pixels
         * @param height Texture height in pixels
         * @param data Raw pixel data pointer (must be width * height * bytesPerPixel)
         */
        Texture2D(uint32_t width, uint32_t height, const void *data)
                : _width(width), _height(height), _format(Format) {
            CreateTexture(data);
        }

        /**
         * Move constructor
         */
        Texture2D(Texture2D &&other) noexcept
                : _textureId(other._textureId), _width(other._width), _height(other._height), _format(other._format),
                  _boundBinding(other._boundBinding),
                  _uploadPool(std::move(other._uploadPool)),
                  _downloadPool(std::move(other._downloadPool)),
                  _currentUploadPBO(other._currentUploadPBO),
                  _currentDownloadPBO(other._currentDownloadPBO) {
            other._textureId = 0;
            other._width = 0;
            other._height = 0;
            other._boundBinding = -1;
            other._currentUploadPBO = nullptr;
            other._currentDownloadPBO = nullptr;
        }

        /**
         * Move assignment
         */
        Texture2D &operator=(Texture2D &&other) noexcept {
            if (this != &other) {
                DestroyTexture();
                _textureId = other._textureId;
                _width = other._width;
                _height = other._height;
                _format = other._format;
                _boundBinding = other._boundBinding;
                _uploadPool = std::move(other._uploadPool);
                _downloadPool = std::move(other._downloadPool);
                _currentUploadPBO = other._currentUploadPBO;
                _currentDownloadPBO = other._currentDownloadPBO;
                other._textureId = 0;
                other._width = 0;
                other._height = 0;
                other._boundBinding = -1;
                other._currentUploadPBO = nullptr;
                other._currentDownloadPBO = nullptr;
            }
            return *this;
        }

        /**
         * Destructor
         */
        ~Texture2D() {
            DestroyTexture();
        }

        // Disable copy
        Texture2D(const Texture2D &) = delete;
        Texture2D &operator=(const Texture2D &) = delete;

    public:
        // ===================================================================
        // Synchronous Upload/Download
        // ===================================================================

        /**
         * Upload raw pixel data to GPU texture (synchronous)
         * @param data Raw pixel data pointer
         */
        void Upload(const void *data) {
            if (_textureId == 0 || data == nullptr) {
                return;
            }

            Runtime::Context::GetInstance().MakeCurrent();

            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);

            glBindTexture(GL_TEXTURE_2D, _textureId);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, format, type, data);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        /**
         * Upload a portion of the texture (synchronous)
         * @param x Offset X
         * @param y Offset Y
         * @param w Width to upload
         * @param h Height to upload
         * @param data Raw pixel data
         */
        void UploadSubRegion(uint32_t x, uint32_t y, uint32_t w, uint32_t h, const void *data) {
            if (_textureId == 0 || data == nullptr) {
                return;
            }
            if (x + w > _width || y + h > _height) {
                throw std::out_of_range("Upload region exceeds texture bounds");
            }

            Runtime::Context::GetInstance().MakeCurrent();

            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);

            glBindTexture(GL_TEXTURE_2D, _textureId);
            glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, w, h, format, type, data);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        /**
         * Download texture data to CPU buffer (synchronous)
         * @param outData Output buffer pointer (must be large enough)
         */
        void Download(void *outData) const {
            if (_textureId == 0 || outData == nullptr) {
                return;
            }

            Runtime::Context::GetInstance().MakeCurrent();

            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);

            glBindTexture(GL_TEXTURE_2D, _textureId);
            glGetTexImage(GL_TEXTURE_2D, 0, format, type, outData);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        /**
         * Download to a vector (resizes automatically)
         */
        template<typename T>
        void Download(std::vector<T> &outData) const {
            size_t requiredSize = (_width * _height * GetBytesPerPixel(_format) + sizeof(T) - 1) / sizeof(T);
            if (outData.size() < requiredSize) {
                outData.resize(requiredSize);
            }
            Download(outData.data());
        }

    public:
        // ===================================================================
        // PBO Asynchronous Upload/Download
        // ===================================================================

        /**
         * Initialize upload PBO pool for async streaming
         * @param bufferCount Number of PBOs (typically 2 or 3 for double/triple buffering)
         */
        void InitUploadPBOPool(uint32_t bufferCount = 2) {
            if (!_uploadPool) {
                size_t size = _width * _height * GetBytesPerPixel();
                _uploadPool = std::make_unique<PBOPool>(size, bufferCount, false);
            }
        }

        /**
         * Initialize download PBO pool for async streaming
         * @param bufferCount Number of PBOs (typically 2 or 3)
         */
        void InitDownloadPBOPool(uint32_t bufferCount = 2) {
            if (!_downloadPool) {
                size_t size = _width * _height * GetBytesPerPixel();
                _downloadPool = std::make_unique<PBOPool>(size, bufferCount, true);
            }
        }

        /**
         * Asynchronous upload using PBO
         * @param data Raw pixel data pointer
         * @return true if upload started, false if no PBO available (call Sync() or wait)
         */
        bool UploadAsync(const void* data) {
            if (!_uploadPool) {
                InitUploadPBOPool(2);
            }

            // Update states first
            _uploadPool->UpdateStates();

            // Try to acquire an idle PBO
            PBOBuffer* pbo = _uploadPool->AcquireIdle();
            if (!pbo) {
                return false; // No idle PBO available
            }

            Runtime::Context::GetInstance().MakeCurrent();
            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);

            // Copy data to PBO
            size_t dataSize = _width * _height * GetBytesPerPixel();
            pbo->CopyData(data, dataSize);

            // Initiate async transfer from PBO to texture
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->GetHandle());
            glBindTexture(GL_TEXTURE_2D, _textureId);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, format, type, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            // Set state and insert fence
            pbo->SetState(PBOBuffer::State::Uploading);
            pbo->InsertFence();

            return true;
        }

        /**
         * Streaming upload - waits for an idle PBO if necessary
         * @param data Raw pixel data pointer
         * @param timeoutMs Maximum time to wait for an idle PBO
         * @return true if upload started
         */
        bool UploadAsyncStream(const void* data, uint32_t timeoutMs = 1000) {
            if (!_uploadPool) {
                InitUploadPBOPool(2);
            }

            // This will block until an idle PBO is available
            PBOBuffer* pbo = _uploadPool->AcquireIdleBlocking(timeoutMs);
            if (!pbo) {
                throw std::runtime_error("UploadAsyncStream timeout - no idle PBO available");
            }

            Runtime::Context::GetInstance().MakeCurrent();
            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);

            size_t dataSize = _width * _height * GetBytesPerPixel();
            pbo->CopyData(data, dataSize);

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->GetHandle());
            glBindTexture(GL_TEXTURE_2D, _textureId);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, format, type, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            pbo->SetState(PBOBuffer::State::Uploading);
            pbo->InsertFence();

            return true;
        }

        /**
         * Asynchronous download using PBO
         * @return true if download started, false if no PBO available
         */
        bool DownloadAsync() {
            if (!_downloadPool) {
                InitDownloadPBOPool(2);
            }

            _downloadPool->UpdateStates();

            PBOBuffer* pbo = _downloadPool->AcquireIdle();
            if (!pbo) {
                return false;
            }

            Runtime::Context::GetInstance().MakeCurrent();
            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);

            // Initiate async transfer from texture to PBO
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo->GetHandle());
            glBindTexture(GL_TEXTURE_2D, _textureId);
            glGetTexImage(GL_TEXTURE_2D, 0, format, type, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

            pbo->SetState(PBOBuffer::State::Downloading);
            pbo->InsertFence();

            return true;
        }

        /**
         * Get downloaded data from a ready PBO
         * @param outData Output buffer to copy data into
         * @return true if data was available and copied
         */
        bool GetDownloadData(void* outData) {
            if (!_downloadPool) return false;

            _downloadPool->UpdateStates();

            // Find a ready PBO
            for (auto& pbo : _downloadPool->GetBuffers()) {
                if (pbo->GetState() == PBOBuffer::State::Ready) {
                    const void* mapped = pbo->MapRead();
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

        /**
         * Wait for all async operations to complete
         */
        void Sync() {
            if (_uploadPool) {
                _uploadPool->SyncAll();
            }
            if (_downloadPool) {
                _downloadPool->SyncAll();
            }
        }

        /**
         * Check if all async operations are complete
         * @return true if all operations completed
         */
        bool IsIdle() {
            if (_uploadPool) {
                _uploadPool->UpdateStates();
                for (auto& pbo : _uploadPool->GetBuffers()) {
                    if (pbo->GetState() != PBOBuffer::State::Idle) {
                        return false;
                    }
                }
            }
            if (_downloadPool) {
                _downloadPool->UpdateStates();
                for (auto& pbo : _downloadPool->GetBuffers()) {
                    if (pbo->GetState() != PBOBuffer::State::Idle) {
                        return false;
                    }
                }
            }
            return true;
        }

    public:
        /**
         * Bind this texture to the current kernel being defined
         * Automatically allocates a binding slot and registers the texture
         * @return TextureRef for DSL access
         */
        [[nodiscard]] IR::Value::TextureRef<Format> Bind() {
            auto *context = IR::Builder::Builder::Get().Context();
            if (!context) {
                throw std::runtime_error("Texture2D::Bind() called outside of Kernel definition");
            }

            uint32_t binding = context->AllocateTextureBinding();
            std::string textureName = std::format("tex{}", binding);
            context->RegisterTexture(binding, _format, textureName, _width, _height);
            context->BindRuntimeTexture(binding, _textureId);
            _boundBinding = static_cast<int>(binding);

            return IR::Value::TextureRef<Format>(textureName, binding, _width, _height);
        }

    public:
        [[nodiscard]] uint32_t GetHandle() const { return _textureId; }
        [[nodiscard]] uint32_t GetWidth() const { return _width; }
        [[nodiscard]] uint32_t GetHeight() const { return _height; }
        static constexpr PixelFormat GetFormat() { return Format; }
        [[nodiscard]] size_t GetBytesPerPixel() const { return Runtime::GetBytesPerPixel(_format); }
        [[nodiscard]] size_t GetSizeInBytes() const { return _width * _height * GetBytesPerPixel(); }
        [[nodiscard]] int GetBinding() const { return _boundBinding; }

    private:
        void CreateTexture(const void *initialData) {
            Runtime::AutoInitContext();
            Runtime::Context::GetInstance().MakeCurrent();

            if (_width == 0 || _height == 0) {
                throw std::invalid_argument("Texture dimensions must be non-zero");
            }

            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);

            glGenTextures(1, &_textureId);
            if (_textureId == 0) {
                throw std::runtime_error("Failed to create OpenGL texture");
            }

            glBindTexture(GL_TEXTURE_2D, _textureId);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, _width, _height, 0, format, type, initialData);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        void DestroyTexture() {
            Sync(); // Wait for all async operations
            if (_textureId != 0) {
                glDeleteTextures(1, &_textureId);
                _textureId = 0;
            }
        }

    private:
        uint32_t _textureId = 0;
        uint32_t _width = 0;
        uint32_t _height = 0;
        PixelFormat _format = Format;
        int _boundBinding = -1;

        // PBO pools for async transfer
        std::unique_ptr<PBOPool> _uploadPool;
        std::unique_ptr<PBOPool> _downloadPool;
        PBOBuffer* _currentUploadPBO = nullptr;
        PBOBuffer* _currentDownloadPBO = nullptr;
    };

    // Type aliases for common 2D texture formats
    using TextureRGBA8 = Texture2D<PixelFormat::RGBA8>;
    using TextureRGBA32F = Texture2D<PixelFormat::RGBA32F>;
    using TextureR32F = Texture2D<PixelFormat::R32F>;
    using TextureRG32F = Texture2D<PixelFormat::RG32F>;
    using TextureR8 = Texture2D<PixelFormat::R8>;

    /**
     * 3D Texture class with PBO support
     * @tparam Format The pixel format of the texture
     */
    template<PixelFormat Format>
    class Texture3D {
    public:
        Texture3D(uint32_t width, uint32_t height, uint32_t depth)
                : _width(width), _height(height), _depth(depth), _format(Format) {
            CreateTexture(nullptr);
        }

        Texture3D(uint32_t width, uint32_t height, uint32_t depth, const void *data)
                : _width(width), _height(height), _depth(depth), _format(Format) {
            CreateTexture(data);
        }

        Texture3D(Texture3D &&other) noexcept
                : _textureId(other._textureId), _width(other._width), _height(other._height), 
                  _depth(other._depth), _format(other._format), _boundBinding(other._boundBinding),
                  _uploadPool(std::move(other._uploadPool)),
                  _downloadPool(std::move(other._downloadPool)) {
            other._textureId = 0;
            other._width = 0;
            other._height = 0;
            other._depth = 0;
            other._boundBinding = -1;
        }

        Texture3D &operator=(Texture3D &&other) noexcept {
            if (this != &other) {
                DestroyTexture();
                _textureId = other._textureId;
                _width = other._width;
                _height = other._height;
                _depth = other._depth;
                _format = other._format;
                _boundBinding = other._boundBinding;
                _uploadPool = std::move(other._uploadPool);
                _downloadPool = std::move(other._downloadPool);
                other._textureId = 0;
                other._width = 0;
                other._height = 0;
                other._depth = 0;
                other._boundBinding = -1;
            }
            return *this;
        }

        ~Texture3D() {
            DestroyTexture();
        }

        Texture3D(const Texture3D &) = delete;
        Texture3D &operator=(const Texture3D &) = delete;

    public:
        // Synchronous operations
        void Upload(const void *data) {
            if (_textureId == 0 || data == nullptr) return;
            Runtime::Context::GetInstance().MakeCurrent();
            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);
            glBindTexture(GL_TEXTURE_3D, _textureId);
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, _width, _height, _depth, format, type, data);
            glBindTexture(GL_TEXTURE_3D, 0);
        }

        void UploadSubRegion(uint32_t x, uint32_t y, uint32_t z, 
                            uint32_t w, uint32_t h, uint32_t d, const void *data) {
            if (_textureId == 0 || data == nullptr) return;
            if (x + w > _width || y + h > _height || z + d > _depth) {
                throw std::out_of_range("Upload region exceeds texture bounds");
            }
            Runtime::Context::GetInstance().MakeCurrent();
            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);
            glBindTexture(GL_TEXTURE_3D, _textureId);
            glTexSubImage3D(GL_TEXTURE_3D, 0, x, y, z, w, h, d, format, type, data);
            glBindTexture(GL_TEXTURE_3D, 0);
        }

        void Download(void *outData) const {
            if (_textureId == 0 || outData == nullptr) return;
            Runtime::Context::GetInstance().MakeCurrent();
            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);
            glBindTexture(GL_TEXTURE_3D, _textureId);
            glGetTexImage(GL_TEXTURE_3D, 0, format, type, outData);
            glBindTexture(GL_TEXTURE_3D, 0);
        }

        template<typename T>
        void Download(std::vector<T> &outData) const {
            size_t requiredSize = (_width * _height * _depth * GetBytesPerPixel(_format) + sizeof(T) - 1) / sizeof(T);
            if (outData.size() < requiredSize) {
                outData.resize(requiredSize);
            }
            Download(outData.data());
        }

    public:
        // PBO Async operations
        void InitUploadPBOPool(uint32_t bufferCount = 2) {
            if (!_uploadPool) {
                size_t size = _width * _height * _depth * GetBytesPerPixel();
                _uploadPool = std::make_unique<PBOPool>(size, bufferCount, false);
            }
        }

        void InitDownloadPBOPool(uint32_t bufferCount = 2) {
            if (!_downloadPool) {
                size_t size = _width * _height * _depth * GetBytesPerPixel();
                _downloadPool = std::make_unique<PBOPool>(size, bufferCount, true);
            }
        }

        bool UploadAsync(const void* data) {
            if (!_uploadPool) InitUploadPBOPool(2);
            _uploadPool->UpdateStates();
            PBOBuffer* pbo = _uploadPool->AcquireIdle();
            if (!pbo) return false;

            Runtime::Context::GetInstance().MakeCurrent();
            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);
            size_t dataSize = _width * _height * _depth * GetBytesPerPixel();
            pbo->CopyData(data, dataSize);

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->GetHandle());
            glBindTexture(GL_TEXTURE_3D, _textureId);
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, _width, _height, _depth, format, type, nullptr);
            glBindTexture(GL_TEXTURE_3D, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            pbo->SetState(PBOBuffer::State::Uploading);
            pbo->InsertFence();
            return true;
        }

        bool UploadAsyncStream(const void* data, uint32_t timeoutMs = 1000) {
            if (!_uploadPool) InitUploadPBOPool(2);
            PBOBuffer* pbo = _uploadPool->AcquireIdleBlocking(timeoutMs);
            if (!pbo) throw std::runtime_error("UploadAsyncStream timeout");

            Runtime::Context::GetInstance().MakeCurrent();
            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);
            size_t dataSize = _width * _height * _depth * GetBytesPerPixel();
            pbo->CopyData(data, dataSize);

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->GetHandle());
            glBindTexture(GL_TEXTURE_3D, _textureId);
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, _width, _height, _depth, format, type, nullptr);
            glBindTexture(GL_TEXTURE_3D, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            pbo->SetState(PBOBuffer::State::Uploading);
            pbo->InsertFence();
            return true;
        }

        bool DownloadAsync() {
            if (!_downloadPool) InitDownloadPBOPool(2);
            _downloadPool->UpdateStates();
            PBOBuffer* pbo = _downloadPool->AcquireIdle();
            if (!pbo) return false;

            Runtime::Context::GetInstance().MakeCurrent();
            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo->GetHandle());
            glBindTexture(GL_TEXTURE_3D, _textureId);
            glGetTexImage(GL_TEXTURE_3D, 0, format, type, nullptr);
            glBindTexture(GL_TEXTURE_3D, 0);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

            pbo->SetState(PBOBuffer::State::Downloading);
            pbo->InsertFence();
            return true;
        }

        bool GetDownloadData(void* outData) {
            if (!_downloadPool) return false;
            _downloadPool->UpdateStates();
            for (auto& pbo : _downloadPool->GetBuffers()) {
                if (pbo->GetState() == PBOBuffer::State::Ready) {
                    const void* mapped = pbo->MapRead();
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
            if (_uploadPool) _uploadPool->SyncAll();
            if (_downloadPool) _downloadPool->SyncAll();
        }

        bool IsIdle() {
            if (_uploadPool) {
                _uploadPool->UpdateStates();
                for (auto& pbo : _uploadPool->GetBuffers()) {
                    if (pbo->GetState() != PBOBuffer::State::Idle) return false;
                }
            }
            if (_downloadPool) {
                _downloadPool->UpdateStates();
                for (auto& pbo : _downloadPool->GetBuffers()) {
                    if (pbo->GetState() != PBOBuffer::State::Idle) return false;
                }
            }
            return true;
        }

    public:
        [[nodiscard]] IR::Value::Texture3DRef<Format> Bind() {
            auto *context = IR::Builder::Builder::Get().Context();
            if (!context) {
                throw std::runtime_error("Texture3D::Bind() called outside of Kernel definition");
            }
            uint32_t binding = context->AllocateTextureBinding();
            std::string textureName = std::format("tex3d{}", binding);
            context->RegisterTexture3D(binding, _format, textureName, _width, _height, _depth);
            context->BindRuntimeTexture(binding, _textureId);
            _boundBinding = static_cast<int>(binding);
            return IR::Value::Texture3DRef<Format>(textureName, binding, _width, _height, _depth);
        }

        [[nodiscard]] uint32_t GetHandle() const { return _textureId; }
        [[nodiscard]] uint32_t GetWidth() const { return _width; }
        [[nodiscard]] uint32_t GetHeight() const { return _height; }
        [[nodiscard]] uint32_t GetDepth() const { return _depth; }
        static constexpr PixelFormat GetFormat() { return Format; }
        [[nodiscard]] size_t GetBytesPerPixel() const { return Runtime::GetBytesPerPixel(_format); }
        [[nodiscard]] size_t GetSizeInBytes() const { return _width * _height * _depth * GetBytesPerPixel(); }
        [[nodiscard]] int GetBinding() const { return _boundBinding; }

    private:
        void CreateTexture(const void *initialData) {
            Runtime::AutoInitContext();
            Runtime::Context::GetInstance().MakeCurrent();
            if (_width == 0 || _height == 0 || _depth == 0) {
                throw std::invalid_argument("Texture dimensions must be non-zero");
            }
            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);
            glGenTextures(1, &_textureId);
            if (_textureId == 0) {
                throw std::runtime_error("Failed to create OpenGL 3D texture");
            }
            glBindTexture(GL_TEXTURE_3D, _textureId);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, _width, _height, _depth, 0, format, type, initialData);
            glBindTexture(GL_TEXTURE_3D, 0);
        }

        void DestroyTexture() {
            Sync();
            if (_textureId != 0) {
                glDeleteTextures(1, &_textureId);
                _textureId = 0;
            }
        }

    private:
        uint32_t _textureId = 0;
        uint32_t _width = 0;
        uint32_t _height = 0;
        uint32_t _depth = 0;
        PixelFormat _format = Format;
        int _boundBinding = -1;
        std::unique_ptr<PBOPool> _uploadPool;
        std::unique_ptr<PBOPool> _downloadPool;
    };

    // Type aliases for common 3D texture formats
    using Texture3DRGBA8 = Texture3D<PixelFormat::RGBA8>;
    using Texture3DRGBA32F = Texture3D<PixelFormat::RGBA32F>;
    using Texture3DR32F = Texture3D<PixelFormat::R32F>;
    using Texture3DRG32F = Texture3D<PixelFormat::RG32F>;
    using Texture3DR8 = Texture3D<PixelFormat::R8>;

} // namespace GPU::Runtime

#endif // EASYGPU_TEXTURE_H
