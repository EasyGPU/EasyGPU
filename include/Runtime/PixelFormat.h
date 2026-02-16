/**
 * PixelFormat.h:
 *      @Descripiton    :   Pixel format definitions for Texture
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_PIXELFORMAT_H
#define EASYGPU_PIXELFORMAT_H

#include <GLAD/glad.h>
#include <string>
#include <tuple>

namespace GPU::Runtime {
    /**
     * Pixel format enumeration for textures
     * Matches common OpenGL internal formats
     */
    enum class PixelFormat {
        // 8-bit unsigned normalized formats
        R8,         // Single channel, 8-bit
        RG8,        // Two channels, 8-bit each
        RGBA8,      // Four channels, 8-bit each - most common

        // 32-bit float formats
        R32F,       // Single channel, 32-bit float
        RG32F,      // Two channels, 32-bit float each
        RGBA32F,    // Four channels, 32-bit float each

        // 16-bit float formats (optional but useful)
        R16F,       // Single channel, 16-bit float
        RG16F,      // Two channels, 16-bit float each
        RGBA16F,    // Four channels, 16-bit float each

        // 32-bit signed integer formats
        R32I,       // Single channel, 32-bit int
        RG32I,      // Two channels, 32-bit int each
        RGBA32I,    // Four channels, 32-bit int each

        // 32-bit unsigned integer formats
        R32UI,      // Single channel, 32-bit uint
        RG32UI,     // Two channels, 32-bit uint each
        RGBA32UI,   // Four channels, 32-bit uint each
    };

    /**
     * Get OpenGL internal format, format, and type for a PixelFormat
     * @return tuple of (internalFormat, format, type)
     */
    inline std::tuple<GLenum, GLenum, GLenum> GetGLPixelFormatInfo(PixelFormat fmt) {
        switch (fmt) {
            // 8-bit normalized
            case PixelFormat::R8:
                return {GL_R8, GL_RED, GL_UNSIGNED_BYTE};
            case PixelFormat::RG8:
                return {GL_RG8, GL_RG, GL_UNSIGNED_BYTE};
            case PixelFormat::RGBA8:
                return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};

                // 32-bit float
            case PixelFormat::R32F:
                return {GL_R32F, GL_RED, GL_FLOAT};
            case PixelFormat::RG32F:
                return {GL_RG32F, GL_RG, GL_FLOAT};
            case PixelFormat::RGBA32F:
                return {GL_RGBA32F, GL_RGBA, GL_FLOAT};

                // 16-bit float
            case PixelFormat::R16F:
                return {GL_R16F, GL_RED, GL_HALF_FLOAT};
            case PixelFormat::RG16F:
                return {GL_RG16F, GL_RG, GL_HALF_FLOAT};
            case PixelFormat::RGBA16F:
                return {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT};

                // 32-bit signed integer
            case PixelFormat::R32I:
                return {GL_R32I, GL_RED_INTEGER, GL_INT};
            case PixelFormat::RG32I:
                return {GL_RG32I, GL_RG_INTEGER, GL_INT};
            case PixelFormat::RGBA32I:
                return {GL_RGBA32I, GL_RGBA_INTEGER, GL_INT};

                // 32-bit unsigned integer
            case PixelFormat::R32UI:
                return {GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT};
            case PixelFormat::RG32UI:
                return {GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT};
            case PixelFormat::RGBA32UI:
                return {GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT};
        }
        return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE}; // default
    }

    /**
     * Get GLSL layout qualifier for image format
     * Used in: layout(rgba8, binding=X) uniform image2D tex;
     */
    inline std::string GetGLSLFormatQualifier(PixelFormat fmt) {
        switch (fmt) {
            case PixelFormat::R8:
                return "r8";
            case PixelFormat::RG8:
                return "rg8";
            case PixelFormat::RGBA8:
                return "rgba8";
            case PixelFormat::R32F:
                return "r32f";
            case PixelFormat::RG32F:
                return "rg32f";
            case PixelFormat::RGBA32F:
                return "rgba32f";
            case PixelFormat::R16F:
                return "r16f";
            case PixelFormat::RG16F:
                return "rg16f";
            case PixelFormat::RGBA16F:
                return "rgba16f";
            case PixelFormat::R32I:
                return "r32i";
            case PixelFormat::RG32I:
                return "rg32i";
            case PixelFormat::RGBA32I:
                return "rgba32i";
            case PixelFormat::R32UI:
                return "r32ui";
            case PixelFormat::RG32UI:
                return "rg32ui";
            case PixelFormat::RGBA32UI:
                return "rgba32ui";
        }
        return "rgba8";
    }

    /**
     * Get bytes per pixel for a format
     */
    inline size_t GetBytesPerPixel(PixelFormat fmt) {
        switch (fmt) {
            case PixelFormat::R8:
                return 1;
            case PixelFormat::RG8:
                return 2;
            case PixelFormat::RGBA8:
                return 4;
            case PixelFormat::R32F:
                return 4;
            case PixelFormat::RG32F:
                return 8;
            case PixelFormat::RGBA32F:
                return 16;
            case PixelFormat::R16F:
                return 2;
            case PixelFormat::RG16F:
                return 4;
            case PixelFormat::RGBA16F:
                return 8;
            case PixelFormat::R32I:
                return 4;
            case PixelFormat::RG32I:
                return 8;
            case PixelFormat::RGBA32I:
                return 16;
            case PixelFormat::R32UI:
                return 4;
            case PixelFormat::RG32UI:
                return 8;
            case PixelFormat::RGBA32UI:
                return 16;
        }
        return 4;
    }

    /**
     * Get number of channels for a format
     */
    inline int GetChannelCount(PixelFormat fmt) {
        switch (fmt) {
            case PixelFormat::R8:
            case PixelFormat::R32F:
            case PixelFormat::R16F:
            case PixelFormat::R32I:
            case PixelFormat::R32UI:
                return 1;
            case PixelFormat::RG8:
            case PixelFormat::RG32F:
            case PixelFormat::RG16F:
            case PixelFormat::RG32I:
            case PixelFormat::RG32UI:
                return 2;
            case PixelFormat::RGBA8:
            case PixelFormat::RGBA32F:
            case PixelFormat::RGBA16F:
            case PixelFormat::RGBA32I:
            case PixelFormat::RGBA32UI:
                return 4;
        }
        return 4;
    }

    /**
     * Check if format uses floating point values
     */
    inline bool IsFloatFormat(PixelFormat fmt) {
        switch (fmt) {
            case PixelFormat::R32F:
            case PixelFormat::RG32F:
            case PixelFormat::RGBA32F:
            case PixelFormat::R16F:
            case PixelFormat::RG16F:
            case PixelFormat::RGBA16F:
                return true;
            default:
                return false;
        }
    }

    /**
     * Check if format uses integer values
     */
    inline bool IsIntegerFormat(PixelFormat fmt) {
        switch (fmt) {
            case PixelFormat::R32I:
            case PixelFormat::RG32I:
            case PixelFormat::RGBA32I:
            case PixelFormat::R32UI:
            case PixelFormat::RG32UI:
            case PixelFormat::RGBA32UI:
                return true;
            default:
                return false;
        }
    }

    /**
     * Check if format is signed integer
     */
    inline bool IsSignedIntegerFormat(PixelFormat fmt) {
        switch (fmt) {
            case PixelFormat::R32I:
            case PixelFormat::RG32I:
            case PixelFormat::RGBA32I:
                return true;
            default:
                return false;
        }
    }

    /**
     * Check if format is unsigned integer
     */
    inline bool IsUnsignedIntegerFormat(PixelFormat fmt) {
        switch (fmt) {
            case PixelFormat::R8:
            case PixelFormat::RG8:
            case PixelFormat::RGBA8:
            case PixelFormat::R32UI:
            case PixelFormat::RG32UI:
            case PixelFormat::RGBA32UI:
                return true;
            default:
                return false;
        }
    }

} // namespace GPU::Runtime

#endif // EASYGPU_PIXELFORMAT_H
