/**
 * TexturePBOManager.cpp:
 *      @Descripiton    :   Pixel Buffer Object manager implementation
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/18/2026
 */

#include <Runtime/TexturePBOManager.h>
#include <Runtime/Context.h>

#include <cstring>
#include <stdexcept>

namespace GPU::Runtime {

    // =================================================================================
    // Construction / Destruction
    // =================================================================================

    TexturePBOManager::TexturePBOManager() = default;

    TexturePBOManager::~TexturePBOManager() {
        Destroy();
    }

    TexturePBOManager::TexturePBOManager(TexturePBOManager&& other) noexcept {
        // Move all data
        _pboEntries[0] = other._pboEntries[0];
        _pboEntries[1] = other._pboEntries[1];
        _bufferSize = other._bufferSize;
        _currentWriteIndex = other._currentWriteIndex;
        _initialized = other._initialized;
        _strategy = other._strategy;

        // Clear other
        other._pboEntries[0] = PBOEntry{};
        other._pboEntries[1] = PBOEntry{};
        other._bufferSize = 0;
        other._currentWriteIndex = 0;
        other._initialized = false;
    }

    TexturePBOManager& TexturePBOManager::operator=(TexturePBOManager&& other) noexcept {
        if (this != &other) {
            // Clean up existing resources first
            Destroy();

            // Move all data
            _pboEntries[0] = other._pboEntries[0];
            _pboEntries[1] = other._pboEntries[1];
            _bufferSize = other._bufferSize;
            _currentWriteIndex = other._currentWriteIndex;
            _initialized = other._initialized;
            _strategy = other._strategy;

            // Clear other
            other._pboEntries[0] = PBOEntry{};
            other._pboEntries[1] = PBOEntry{};
            other._bufferSize = 0;
            other._currentWriteIndex = 0;
            other._initialized = false;
        }
        return *this;
    }

    // =================================================================================
    // Initialization
    // =================================================================================

    void TexturePBOManager::Initialize(size_t size) {
        if (_initialized) {
            // If already initialized with different size, recreate
            if (_bufferSize != size) {
                Destroy();
            } else {
                return; // Already initialized with correct size
            }
        }

        if (size == 0) {
            return;
        }

        _bufferSize = size;
        CreatePBOs();
        _initialized = true;
    }

    void TexturePBOManager::Destroy() {
        if (!_initialized) {
            return;
        }

        DeletePBOs();
        _bufferSize = 0;
        _currentWriteIndex = 0;
        _initialized = false;
    }

    // =================================================================================
    // PBO Management
    // =================================================================================

    void TexturePBOManager::CreatePBOs() {
        // Ensure context is current
        Runtime::Context::GetInstance().MakeCurrent();

        uint32_t pboIds[2];
        glGenBuffers(2, pboIds);

        for (int i = 0; i < 2; ++i) {
            _pboEntries[i].pboId = pboIds[i];
            _pboEntries[i].size = _bufferSize;
            _pboEntries[i].fence = nullptr;
            _pboEntries[i].isActive = false;

            // Initialize buffer storage
            glBindBuffer(GL_PIXEL_PACK_BUFFER, _pboEntries[i].pboId);
            glBufferData(GL_PIXEL_PACK_BUFFER, _bufferSize, nullptr, GL_STREAM_READ);
        }

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }

    void TexturePBOManager::DeletePBOs() {
        // Ensure context is current
        Runtime::Context::GetInstance().MakeCurrent();

        for (int i = 0; i < 2; ++i) {
            // Wait for any pending operations
            if (_pboEntries[i].fence) {
                glClientWaitSync(_pboEntries[i].fence, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
                glDeleteSync(_pboEntries[i].fence);
                _pboEntries[i].fence = nullptr;
            }

            if (_pboEntries[i].pboId != 0) {
                glDeleteBuffers(1, &_pboEntries[i].pboId);
                _pboEntries[i].pboId = 0;
            }

            _pboEntries[i].size = 0;
            _pboEntries[i].isActive = false;
        }
    }

    PBOEntry& TexturePBOManager::GetNextPBOForWrite() {
        PBOEntry& entry = _pboEntries[_currentWriteIndex];

        // Wait for this PBO if it has an active transfer
        if (entry.isActive && entry.fence) {
            WaitAndCleanupPBO(entry);
        }

        // Advance to next PBO for next call (round-robin)
        _currentWriteIndex = (_currentWriteIndex + 1) % 2;

        return entry;
    }

    void TexturePBOManager::WaitAndCleanupPBO(PBOEntry& entry) {
        if (!entry.fence) {
            return;
        }

        // Wait for GPU to finish
        glClientWaitSync(entry.fence, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);

        // Clean up fence
        glDeleteSync(entry.fence);
        entry.fence = nullptr;
        entry.isActive = false;
    }

    // =================================================================================
    // Synchronous Download
    // =================================================================================

    void TexturePBOManager::DownloadSync(uint32_t textureId, GLenum target,
                                        GLenum format, GLenum type,
                                        void* outData,
                                        bool directDownloadIfSmall,
                                        size_t smallTextureThreshold) {
        if (!outData) {
            return;
        }

        // Ensure context is current
        Runtime::Context::GetInstance().MakeCurrent();

        // Check strategy
        bool useDirect = (_strategy == TextureDownloadStrategy::Direct) ||
                        (directDownloadIfSmall && _bufferSize < smallTextureThreshold && 
                         _strategy == TextureDownloadStrategy::Auto);

        if (useDirect) {
            // Direct synchronous download
            glBindTexture(target, textureId);
            glGetTexImage(target, 0, format, type, outData);
            glBindTexture(target, 0);
            return;
        }

        // Ensure PBOs are initialized
        if (!_initialized) {
            Initialize(_bufferSize);
        }

        // Get next available PBO
        PBOEntry& pbo = GetNextPBOForWrite();

        // Step 1: Initiate async transfer from texture to PBO
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo.pboId);
        glBindTexture(target, textureId);
        glGetTexImage(target, 0, format, type, nullptr); // nullptr = write to bound PBO
        glBindTexture(target, 0);

        // Step 2: Create fence to track completion
        pbo.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        pbo.isActive = true;

        // Step 3: Wait for transfer to complete
        glClientWaitSync(pbo.fence, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);

        // Step 4: Map PBO and copy to output
        void* mapped = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, _bufferSize, GL_MAP_READ_BIT);
        if (mapped) {
            std::memcpy(outData, mapped, _bufferSize);
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        }

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        // Clean up fence
        glDeleteSync(pbo.fence);
        pbo.fence = nullptr;
        pbo.isActive = false;
    }

    // =================================================================================
    // Asynchronous Download
    // =================================================================================

    bool TexturePBOManager::BeginDownloadAsync(uint32_t textureId, GLenum target,
                                              GLenum format, GLenum type,
                                              AsyncDownloadToken& token) {
        // Ensure context is current
        Runtime::Context::GetInstance().MakeCurrent();

        // Ensure PBOs are initialized
        if (!_initialized) {
            Initialize(_bufferSize);
        }

        // Clear token
        token = AsyncDownloadToken{};

        // Get next available PBO
        PBOEntry& pbo = GetNextPBOForWrite();

        // Bind PBO and initiate transfer
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo.pboId);
        glBindTexture(target, textureId);
        glGetTexImage(target, 0, format, type, nullptr);
        glBindTexture(target, 0);

        // Create fence for tracking
        pbo.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        pbo.isActive = true;

        // Fill token
        token.pboId = pbo.pboId;
        token.fence = pbo.fence;
        token.size = _bufferSize;
        token.mappedPtr = nullptr;
        token.pboIndex = _currentWriteIndex == 0 ? 1 : 0; // Index we just used

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        return true;
    }

    bool TexturePBOManager::IsDownloadComplete(const AsyncDownloadToken& token) const {
        if (!token.fence) {
            return false;
        }

        // Check fence status without waiting
        GLenum status = glClientWaitSync(token.fence, 0, 0);
        return (status == GL_ALREADY_SIGNALED || status == GL_CONDITION_SATISFIED);
    }

    bool TexturePBOManager::WaitForDownload(const AsyncDownloadToken& token, uint32_t timeoutMs) const {
        if (!token.fence) {
            return false;
        }

        GLuint64 timeoutNs = timeoutMs == 0 ? GL_TIMEOUT_IGNORED : 
                            static_cast<GLuint64>(timeoutMs) * 1000000ULL;

        GLenum status = glClientWaitSync(token.fence, GL_SYNC_FLUSH_COMMANDS_BIT, timeoutNs);
        
        return (status == GL_ALREADY_SIGNALED || status == GL_CONDITION_SATISFIED);
    }

    void TexturePBOManager::CompleteDownload(AsyncDownloadToken& token, void* outData) {
        if (!outData || token.pboId == 0) {
            return;
        }

        // Ensure download is complete
        if (token.fence) {
            glClientWaitSync(token.fence, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
        }

        // Bind PBO and copy data
        glBindBuffer(GL_PIXEL_PACK_BUFFER, token.pboId);
        void* mapped = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, token.size, GL_MAP_READ_BIT);
        if (mapped) {
            std::memcpy(outData, mapped, token.size);
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        // Release token resources
        ReleaseToken(token);
    }

    void* TexturePBOManager::MapDownloadBuffer(AsyncDownloadToken& token) {
        if (token.pboId == 0) {
            return nullptr;
        }

        // Ensure download is complete
        if (token.fence) {
            glClientWaitSync(token.fence, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
        }

        // Bind and map PBO
        glBindBuffer(GL_PIXEL_PACK_BUFFER, token.pboId);
        token.mappedPtr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, token.size, GL_MAP_READ_BIT);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        return token.mappedPtr;
    }

    void TexturePBOManager::UnmapDownloadBuffer(AsyncDownloadToken& token) {
        if (token.pboId == 0 || !token.mappedPtr) {
            return;
        }

        glBindBuffer(GL_PIXEL_PACK_BUFFER, token.pboId);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        token.mappedPtr = nullptr;
    }

    void TexturePBOManager::ReleaseToken(AsyncDownloadToken& token) {
        if (token.fence) {
            glDeleteSync(token.fence);
        }

        // Mark the PBO entry as no longer active
        if (token.pboIndex >= 0 && token.pboIndex < 2) {
            if (_pboEntries[token.pboIndex].fence == token.fence) {
                _pboEntries[token.pboIndex].fence = nullptr;
                _pboEntries[token.pboIndex].isActive = false;
            }
        }

        // Clear token
        token = AsyncDownloadToken{};
    }

} // namespace GPU::Runtime
