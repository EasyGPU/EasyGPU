/**
 * Texture.h:
 *      @Descripiton    :   2D Texture for GPU compute shader
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 * 
 * Supports:
 *   - Creating empty texture of specified size
 *   - Creating from raw pixel buffer
 *   - Uploading raw pixel data
 *   - Downloading to raw pixel buffer
 *   - Read/Write access in compute shaders
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

namespace GPU::Runtime {
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
     *   // Upload new data
     *   tex.Upload(newPixels.data());
     *
     *   // Download to buffer
     *   std::vector<uint8_t> result(1024 * 1024 * 4);
     *   tex.Download(result.data());
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
                  _boundBinding(other._boundBinding) {
            other._textureId = 0;
            other._width = 0;
            other._height = 0;
            other._boundBinding = -1;
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
                other._textureId = 0;
                other._width = 0;
                other._height = 0;
                other._boundBinding = -1;
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
        /**
         * Upload raw pixel data to GPU texture
         * @param data Raw pixel data pointer
         */
        void Upload(const void *data) {
            if (_textureId == 0 || data == nullptr) {
                return;
            }

            // Ensure context is current
            Runtime::Context::GetInstance().MakeCurrent();

            auto [internalFormat, format, type] = GetGLPixelFormatInfo(_format);

            glBindTexture(GL_TEXTURE_2D, _textureId);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, format, type, data);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        /**
         * Upload a portion of the texture
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
         * Download texture data to CPU buffer
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
        /**
         * Bind this texture to the current kernel being defined
         * Automatically allocates a binding slot and registers the texture
         * @return TextureRef for DSL access
         */
        [[nodiscard]] IR::Value::TextureRef<Format> Bind() {
            // Get current builder context
            auto *context = IR::Builder::Builder::Get().Context();
            if (!context) {
                throw std::runtime_error("Texture2D::Bind() called outside of Kernel definition");
            }

            // Allocate binding slot
            uint32_t binding = context->AllocateTextureBinding();

            // Generate texture variable name
            std::string textureName = std::format("tex{}", binding);

            // Register texture in context
            context->RegisterTexture(binding, _format, textureName, _width, _height);

            // Register runtime texture handle
            context->BindRuntimeTexture(binding, _textureId);

            // Store binding info
            _boundBinding = static_cast<int>(binding);

            // Return TextureRef for DSL access
            return IR::Value::TextureRef<Format>(textureName, binding, _width, _height);
        }

    public:
        /**
         * Get OpenGL texture handle
         */
        [[nodiscard]] uint32_t GetHandle() const {
            return _textureId;
        }

        /**
         * Get texture width
         */
        [[nodiscard]] uint32_t GetWidth() const {
            return _width;
        }

        /**
         * Get texture height
         */
        [[nodiscard]] uint32_t GetHeight() const {
            return _height;
        }

        /**
         * Get pixel format
         */
        static constexpr PixelFormat GetFormat() {
            return Format;
        }

        /**
         * Get bytes per pixel
         */
        [[nodiscard]] size_t GetBytesPerPixel() const {
            return Runtime::GetBytesPerPixel(_format);
        }

        /**
         * Get total size in bytes
         */
        [[nodiscard]] size_t GetSizeInBytes() const {
            return _width * _height * GetBytesPerPixel();
        }

        /**
         * Get binding slot if bound to a kernel
         * @return The binding slot, or -1 if not bound
         */
        [[nodiscard]] int GetBinding() const {
            return _boundBinding;
        }

    private:
        void CreateTexture(const void *initialData) {
            // Auto-initialize OpenGL context
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

            // Set texture parameters (required for proper operation)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            // Allocate texture storage
            glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, _width, _height, 0, format, type, initialData);

            glBindTexture(GL_TEXTURE_2D, 0);
        }

        void DestroyTexture() {
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
        int _boundBinding = -1;  // -1 means not bound
    };

/**
 * Type aliases for common texture formats
 */
    using TextureRGBA8 = Texture2D<PixelFormat::RGBA8>;
    using TextureRGBA32F = Texture2D<PixelFormat::RGBA32F>;
    using TextureR32F = Texture2D<PixelFormat::R32F>;
    using TextureRG32F = Texture2D<PixelFormat::RG32F>;
    using TextureR8 = Texture2D<PixelFormat::R8>;

} // namespace GPU::Runtime

#endif // EASYGPU_TEXTURE_H
