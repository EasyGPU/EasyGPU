#pragma once

/**
 * TexturePBOManager.h:
 *      @Descripiton    :   Pixel Buffer Object manager for async texture download
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/18/2026
 * 
 * Provides double-buffered PBO support for asynchronous GPU->CPU texture data transfer.
 * Uses OpenGL Pixel Buffer Objects to enable pipelined download with GPU computation overlap.
 */
#ifndef EASYGPU_TEXTURE_PBO_MANAGER_H
#define EASYGPU_TEXTURE_PBO_MANAGER_H

#include <GLAD/glad.h>
#include <cstdint>
#include <memory>

namespace GPU::Runtime {

    /**
     * Download strategy for texture readback
     */
    enum class TextureDownloadStrategy {
        Auto,       ///< Automatically choose based on texture size
        Direct,     ///< Use glGetTexImage directly (synchronous)
        PBO         ///< Use PBO for asynchronous transfer
    };

    /**
     * PBO entry for double-buffered download
     */
    struct PBOEntry {
        uint32_t pboId = 0;         ///< OpenGL PBO buffer ID
        size_t size = 0;            ///< Buffer size in bytes
        GLsync fence = nullptr;     ///< GPU fence for async completion
        bool isActive = false;      ///< Whether this PBO has an active transfer
    };

    /**
     * Token for tracking asynchronous download operations
     */
    struct AsyncDownloadToken {
        uint32_t pboId = 0;         ///< PBO ID used for this download
        GLsync fence = nullptr;     ///< Fence to check completion
        size_t size = 0;            ///< Size of the download in bytes
        void* mappedPtr = nullptr;  ///< Mapped pointer (valid after WaitForDownload)
        int pboIndex = -1;          ///< Index in the PBO pool (-1 if invalid)
    };

    /**
     * Pixel Buffer Object manager for efficient texture download
     * 
     * Implements double-buffered PBO scheme:
     * - PBO[0]: Currently being read by CPU
     * - PBO[1]: Being filled by GPU from texture
     * 
     * This allows GPU->CPU transfer to overlap with CPU processing and GPU computation.
     */
    class TexturePBOManager {
    public:
        /**
         * Default constructor
         */
        TexturePBOManager();

        /**
         * Destructor - releases all PBO resources
         */
        ~TexturePBOManager();

        // Disable copy
        TexturePBOManager(const TexturePBOManager&) = delete;
        TexturePBOManager& operator=(const TexturePBOManager&) = delete;

        /**
         * Move constructor
         */
        TexturePBOManager(TexturePBOManager&& other) noexcept;

        /**
         * Move assignment
         */
        TexturePBOManager& operator=(TexturePBOManager&& other) noexcept;

    public:
        /**
         * Initialize PBO manager with specified buffer size
         * Creates double PBO buffers for ping-pong download
         * @param size Size of each PBO buffer in bytes
         */
        void Initialize(size_t size);

        /**
         * Destroy all PBO resources
         */
        void Destroy();

        /**
         * Check if PBO manager is initialized
         */
        [[nodiscard]] bool IsInitialized() const { return _initialized; }

        /**
         * Get the size of each PBO buffer
         */
        [[nodiscard]] size_t GetBufferSize() const { return _bufferSize; }

        /**
         * Get the current download strategy
         */
        [[nodiscard]] TextureDownloadStrategy GetStrategy() const { return _strategy; }

        /**
         * Set the download strategy
         */
        void SetStrategy(TextureDownloadStrategy strategy) { _strategy = strategy; }

    public:
        /**
         * Synchronous download using PBO
         * Initiates async transfer then waits for completion
         * 
         * @param textureId OpenGL texture ID
         * @param target Texture target (GL_TEXTURE_2D or GL_TEXTURE_3D)
         * @param format Pixel format (GL_RGBA, GL_RED, etc.)
         * @param type Pixel type (GL_UNSIGNED_BYTE, GL_FLOAT, etc.)
         * @param outData Output buffer to receive the data
         * @param directDownloadIfSmall If true, use direct glGetTexImage for small textures
         * @param smallTextureThreshold Threshold in bytes for "small" texture
         */
        void DownloadSync(uint32_t textureId, GLenum target,
                         GLenum format, GLenum type,
                         void* outData,
                         bool directDownloadIfSmall = true,
                         size_t smallTextureThreshold = 1024 * 1024);

        /**
         * Begin asynchronous download from texture to PBO
         * Returns immediately, GPU transfer happens in background
         * 
         * @param textureId OpenGL texture ID
         * @param target Texture target (GL_TEXTURE_2D or GL_TEXTURE_3D)
         * @param format Pixel format
         * @param type Pixel type
         * @param token Output token to track this download
         * @return true if download was initiated successfully
         */
        bool BeginDownloadAsync(uint32_t textureId, GLenum target,
                               GLenum format, GLenum type,
                               AsyncDownloadToken& token);

        /**
         * Check if an async download has completed (non-blocking)
         * 
         * @param token The download token
         * @return true if download is complete and data is ready
         */
        [[nodiscard]] bool IsDownloadComplete(const AsyncDownloadToken& token) const;

        /**
         * Wait for async download to complete (blocking)
         * 
         * @param token The download token
         * @param timeoutMs Timeout in milliseconds (0 = wait forever)
         * @return true if download completed, false if timeout
         */
        bool WaitForDownload(const AsyncDownloadToken& token, uint32_t timeoutMs = 0) const;

        /**
         * Complete the async download and copy data to CPU buffer
         * Must be called after WaitForDownload returns true
         * 
         * @param token The download token
         * @param outData Output buffer (must be large enough)
         */
        void CompleteDownload(AsyncDownloadToken& token, void* outData);

        /**
         * Map the PBO data for reading (advanced usage)
         * Allows zero-copy access to downloaded data
         * 
         * @param token The download token
         * @return Mapped pointer or nullptr on failure
         */
        void* MapDownloadBuffer(AsyncDownloadToken& token);

        /**
         * Unmap the PBO data
         * 
         * @param token The download token
         */
        void UnmapDownloadBuffer(AsyncDownloadToken& token);

        /**
         * Release the download token and free associated resources
         * Should be called after CompleteDownload or when discarding async operation
         * 
         * @param token The download token to release
         */
        void ReleaseToken(AsyncDownloadToken& token);

    private:
        /**
         * Get the next available PBO entry for writing (round-robin)
         */
        PBOEntry& GetNextPBOForWrite();

        /**
         * Wait for a PBO entry's fence and clean up
         */
        void WaitAndCleanupPBO(PBOEntry& entry);

        /**
         * Create PBO buffers
         */
        void CreatePBOs();

        /**
         * Delete PBO buffers and fences
         */
        void DeletePBOs();

    private:
        PBOEntry _pboEntries[2];            ///< Double-buffered PBOs
        size_t _bufferSize = 0;             ///< Size of each PBO buffer
        int _currentWriteIndex = 0;         ///< Current PBO for writing (round-robin)
        bool _initialized = false;          ///< Whether PBOs are created
        TextureDownloadStrategy _strategy = TextureDownloadStrategy::Auto;  ///< Download strategy
    };

} // namespace GPU::Runtime

#endif // EASYGPU_TEXTURE_PBO_MANAGER_H
