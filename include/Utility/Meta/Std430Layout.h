#pragma once

/**
 * Std430Layout.h:
 *      @Descripiton    :   Automatic std430 layout conversion for GPU buffers
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 * 
 * TRUE AUTOMATIC LAYOUT CONVERSION:
 * Users write natural C++ structs, we handle all the conversion!
 */
#ifndef EASYGPU_STD430_LAYOUT_H
#define EASYGPU_STD430_LAYOUT_H

#include <cstring>
#include <memory>
#include <vector>

// Forward declarations for math types
namespace GPU::Math {
    struct Vec2; struct Vec3; struct Vec4;
    struct IVec2; struct IVec3; struct IVec4;
    struct Mat2; struct Mat3; struct Mat4;
    struct Mat2x3; struct Mat2x4; struct Mat3x2; struct Mat3x4; struct Mat4x2; struct Mat4x3;
}

namespace GPU::Meta {

// Use the full name GPU::Meta::StructMeta directly

/**
 * Layout converter interface
 */
class LayoutConverter {
public:
    virtual ~LayoutConverter() = default;
    virtual void ConvertToGPU(const void* src, void* dst, size_t count) const = 0;
    virtual void ConvertFromGPU(const void* src, void* dst, size_t count) const = 0;
    virtual size_t GetCPPLayoutSize() const = 0;
    virtual size_t GetGPULayoutSize() const = 0;
    virtual bool NeedsConversion() const = 0;
};

/**
 * Function pointer types for field converters
 */
using ToGPUConverter = void(*)(const char* src, char* dst, size_t srcStride, size_t dstStride, size_t count);
using FromGPUConverter = void(*)(const char* src, char* dst, size_t srcStride, size_t dstStride, size_t count);

/**
 * Helper to get CPU size for primitive types
 */
template<typename T>
constexpr size_t GetCPPSize() { return sizeof(T); }

// Forward declarations for math types - defined in Vec.h/Matrix.h
// We need these to avoid including full headers here
template<> constexpr size_t GetCPPSize<GPU::Math::Vec2>() { return 8; }   // 2 * 4
template<> constexpr size_t GetCPPSize<GPU::Math::Vec3>() { return 12; }  // 3 * 4
template<> constexpr size_t GetCPPSize<GPU::Math::Vec4>() { return 16; }  // 4 * 4
template<> constexpr size_t GetCPPSize<GPU::Math::IVec2>() { return 8; }
template<> constexpr size_t GetCPPSize<GPU::Math::IVec3>() { return 12; }
template<> constexpr size_t GetCPPSize<GPU::Math::IVec4>() { return 16; }
template<> constexpr size_t GetCPPSize<GPU::Math::Mat2>() { return 16; }   // 2 * vec2(8) aligned to 16 in CPU? Actually 16 for 2x vec4
template<> constexpr size_t GetCPPSize<GPU::Math::Mat3>() { return 36; }   // 3 * vec3(12)
template<> constexpr size_t GetCPPSize<GPU::Math::Mat4>() { return 64; }   // 4 * vec4(16)
template<> constexpr size_t GetCPPSize<GPU::Math::Mat2x3>() { return 24; } // 2 * vec3(12)
template<> constexpr size_t GetCPPSize<GPU::Math::Mat2x4>() { return 32; } // 2 * vec4(16)
template<> constexpr size_t GetCPPSize<GPU::Math::Mat3x2>() { return 24; } // 3 * vec2(8)
template<> constexpr size_t GetCPPSize<GPU::Math::Mat3x4>() { return 48; } // 3 * vec4(16)
template<> constexpr size_t GetCPPSize<GPU::Math::Mat4x2>() { return 32; } // 4 * vec2(8)
template<> constexpr size_t GetCPPSize<GPU::Math::Mat4x3>() { return 48; } // 4 * vec3(12)

/**
 * Helper to get GPU std430 size for primitive types
 * Vec3 is aligned to 16 bytes in std430 (like vec4)
 */
template<typename T>
constexpr size_t GetStd430SizeHelper() { return sizeof(T); }

template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Vec2>() { return 8; }
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Vec3>() { return 16; }  // vec3 aligned to 16 in std430!
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Vec4>() { return 16; }
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::IVec2>() { return 8; }
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::IVec3>() { return 16; } // ivec3 aligned to 16
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::IVec4>() { return 16; }
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Mat2>() { return 32; }   // 2 columns * 16 (each vec2 aligned to 16)
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Mat3>() { return 48; }   // 3 columns * 16 (each vec3 aligned to 16)
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Mat4>() { return 64; }   // 4 columns * 16
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Mat2x3>() { return 32; } // 2 columns * 16 (vec3 aligned to 16)
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Mat2x4>() { return 32; } // 2 columns * 16
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Mat3x2>() { return 48; } // 3 columns * 16 (vec2 aligned to 16 in matrix column)
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Mat3x4>() { return 48; } // 3 columns * 16
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Mat4x2>() { return 64; } // 4 columns * 16
template<> constexpr size_t GetStd430SizeHelper<GPU::Math::Mat4x3>() { return 64; } // 4 columns * 16 (vec3 aligned to 16)

/**
 * Helper to check if a type needs layout conversion
 * Only Vec3 and IVec3 need conversion (CPU 12 bytes -> GPU 16 bytes)
 * Also matrices may need conversion
 */
template<typename T>
constexpr bool NeedsLayoutConversionHelper() { 
    return GetCPPSize<T>() != GetStd430SizeHelper<T>(); 
}

/**
 * Generic converter using runtime field information
 * Generated by EASYGPU_STRUCT macro
 */
template<typename T>
class Std430Converter : public LayoutConverter {
public:
    Std430Converter() {
        if constexpr (GPU::Meta::StructMeta<T>::isRegistered) {
            // Get conversion info from StructMeta
            _cppSize = StructMeta<T>::GetCPPLayoutSize();
            _gpuSize = StructMeta<T>::GetGPULayoutSize();
            _needsConversion = StructMeta<T>::NeedsLayoutConversion();
            _toGPU = StructMeta<T>::GetToGPUConverter();
            _fromGPU = StructMeta<T>::GetFromGPUConverter();
        } else {
            // For primitive types (including Vec3, IVec3), use helper functions
            _cppSize = GetCPPSize<T>();
            _gpuSize = GetStd430SizeHelper<T>();
            _needsConversion = NeedsLayoutConversionHelper<T>();
            _toGPU = nullptr;
            _fromGPU = nullptr;
        }
    }
    
    void ConvertToGPU(const void* src, void* dst, size_t count) const override {
        if (!_needsConversion) {
            std::memcpy(dst, src, count * _cppSize);
            return;
        }
        // Registered struct path: use generated per-field converter.
        if (_toGPU) {
            _toGPU(static_cast<const char*>(src), static_cast<char*>(dst), _cppSize, _gpuSize, count);
            return;
        }
        // For primitive types with size mismatch (like Vec3), we need stride-based copy
        const char* srcPtr = static_cast<const char*>(src);
        char* dstPtr = static_cast<char*>(dst);
        for (size_t i = 0; i < count; i++) {
            // Copy only the valid data (12 bytes for Vec3), leave padding (4 bytes) as-is
            std::memcpy(dstPtr + i * _gpuSize, srcPtr + i * _cppSize, _cppSize);
        }
    }
    
    void ConvertFromGPU(const void* src, void* dst, size_t count) const override {
        if (!_needsConversion) {
            std::memcpy(dst, src, count * _cppSize);
            return;
        }
        // Registered struct path: use generated per-field converter.
        if (_fromGPU) {
            _fromGPU(static_cast<const char*>(src), static_cast<char*>(dst), _gpuSize, _cppSize, count);
            return;
        }
        // For primitive types with size mismatch, we need stride-based copy
        const char* srcPtr = static_cast<const char*>(src);
        char* dstPtr = static_cast<char*>(dst);
        for (size_t i = 0; i < count; i++) {
            // Copy only the valid data (12 bytes for Vec3), ignore padding
            std::memcpy(dstPtr + i * _cppSize, srcPtr + i * _gpuSize, _cppSize);
        }
    }
    
    size_t GetCPPLayoutSize() const override { return _cppSize; }
    size_t GetGPULayoutSize() const override { return _gpuSize; }
    bool NeedsConversion() const override { return _needsConversion; }

private:
    size_t _cppSize = 0;
    size_t _gpuSize = 0;
    bool _needsConversion = false;
    ToGPUConverter _toGPU = nullptr;
    FromGPUConverter _fromGPU = nullptr;
};

/**
 * Factory function
 */
template<typename T>
std::unique_ptr<LayoutConverter> GetLayoutConverter() {
    return std::make_unique<Std430Converter<T>>();
}

} // namespace GPU::Meta

#endif // EASYGPU_STD430_LAYOUT_H
