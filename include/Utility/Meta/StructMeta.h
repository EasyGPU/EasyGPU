#pragma once

/**
 * StructMeta.h:
 *      @Descripiton    :   Structure reflection meta system for EASYGPU_STRUCT
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_STRUCT_META_H
#define EASYGPU_STRUCT_META_H

#include <string>
#include <string_view>
#include <format>
#include <type_traits>
#include <sstream>
#include <concepts>
#include <cstring>

#include <Utility/Vec.h>
#include <Utility/Matrix.h>

// Include GLAD for OpenGL function declarations used in uniform upload
#include <glad/glad.h>

// ============================================================================
// Part 1: GPU::Meta namespace - defined FIRST
// ============================================================================
namespace GPU::Meta {
    /**
     * Structure meta traits (forward declaration)
     */
    template<typename T>
    struct StructMeta {
        static constexpr bool isRegistered = false;
    };

    /**
     * Check if a type is a registered struct
     */
    template<typename T>
    concept RegisteredStruct = StructMeta<T>::isRegistered;
}

// ============================================================================
// Part 2: GPU::IR::Value namespace - concepts that depend on GPU::Meta
// ============================================================================
namespace GPU::IR::Value {
    // Forward declare type concepts
    template<class Type>
    concept BitableType = std::same_as<Type, float> ||
                          std::same_as<Type, int> ||
                          std::same_as<Type, bool>;

    template<class Type>
    concept CountableType = std::same_as<Type, float> ||
                            std::same_as<Type, int>;

    // ScalarTypeImpl - basic scalar types
    template<class Type>
    concept ScalarTypeImpl = std::same_as<Type, float> ||
                             std::same_as<Type, int> ||
                             std::same_as<Type, bool> ||
                             std::same_as<Type, Math::Vec2> ||
                             std::same_as<Type, Math::Vec3> ||
                             std::same_as<Type, Math::Vec4> ||
                             std::same_as<Type, Math::IVec2> ||
                             std::same_as<Type, Math::IVec3> ||
                             std::same_as<Type, Math::IVec4> ||
                             std::same_as<Type, Math::Mat2> ||
                             std::same_as<Type, Math::Mat3> ||
                             std::same_as<Type, Math::Mat4> ||
                             std::same_as<Type, Math::Mat2x3> ||
                             std::same_as<Type, Math::Mat2x4> ||
                             std::same_as<Type, Math::Mat3x2> ||
                             std::same_as<Type, Math::Mat3x4> ||
                             std::same_as<Type, Math::Mat4x2> ||
                             std::same_as<Type, Math::Mat4x3>;

    // ScalarType - includes registered structs
    template<class Type>
    concept ScalarType = ScalarTypeImpl<Type> || GPU::Meta::RegisteredStruct<Type>;

    // NumericType - numeric types only (excludes bool)
    template<class Type>
    concept NumericType = std::same_as<Type, float> ||
                          std::same_as<Type, int> ||
                          std::same_as<Type, Math::Vec2> ||
                          std::same_as<Type, Math::Vec3> ||
                          std::same_as<Type, Math::Vec4> ||
                          std::same_as<Type, Math::IVec2> ||
                          std::same_as<Type, Math::IVec3> ||
                          std::same_as<Type, Math::IVec4> ||
                          std::same_as<Type, Math::Mat2> ||
                          std::same_as<Type, Math::Mat3> ||
                          std::same_as<Type, Math::Mat4> ||
                          std::same_as<Type, Math::Mat2x3> ||
                          std::same_as<Type, Math::Mat2x4> ||
                          std::same_as<Type, Math::Mat3x2> ||
                          std::same_as<Type, Math::Mat3x4> ||
                          std::same_as<Type, Math::Mat4x2> ||
                          std::same_as<Type, Math::Mat4x3>;
}

// ============================================================================
// Part 3: Forward declarations for other namespaces
// ============================================================================
namespace GPU::IR::Node {
    class LocalVariableNode;

    class LoadLocalVariableNode;

    class StoreNode;
}

namespace GPU::IR::Builder {
    class Builder;
}

namespace GPU::IR::Value {
    // Forward declarations - Expr is now a template, forward declare as such
    template<ScalarType T>
    class Expr;

    class Value;

    class ExprBase;
}

// ============================================================================
// Part 4: GPU::Meta implementation
// ============================================================================
namespace GPU::Meta {
    /**
     * Value to string conversion for scalar types
     */
    template<typename T>
    std::string ValueToString(const T &value) {
        if constexpr (std::is_same_v<T, float>) {
            return std::format("float({})", value);
        } else if constexpr (std::is_same_v<T, int>) {
            return std::format("int({})", value);
        } else if constexpr (std::is_same_v<T, bool>) {
            return value ? "true" : "false";
        } else if constexpr (std::is_same_v<T, Math::Vec2>) {
            return std::format("vec2(float({}), float({}))", static_cast<double>(value.x), static_cast<double>(value.y));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::Vec3>) {
            return std::format("vec3(float({}), float({}), float({}))", static_cast<double>(value.x), static_cast<double>(value.y), static_cast<double>(value.z));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::Vec4>) {
            return std::format("vec4(float({}), float({}), float({}), float({}))", static_cast<double>(value.x), static_cast<double>(value.y), static_cast<double>(value.z), static_cast<double>(value.w));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::IVec2>) {
            return std::format("ivec2(int({}), int({}))", value.x, value.y);
        } else if constexpr (std::is_same_v<T, Math::IVec3>) {
            return std::format("ivec3(int({}), int({}), int({}))", value.x, value.y, value.z);
        } else if constexpr (std::is_same_v<T, Math::IVec4>) {
            return std::format("ivec4(int({}), int({}), int({}), int({}))", value.x, value.y, value.z, value.w);
        } else if constexpr (std::is_same_v<T, Math::Mat2>) {
            return std::format("mat2(float({}), float({}), float({}), float({}))",
                               static_cast<double>(value.m00), static_cast<double>(value.m10), static_cast<double>(value.m01), static_cast<double>(value.m11));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::Mat3>) {
            return std::format("mat3(float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}))",
                               static_cast<double>(value.m00), static_cast<double>(value.m10), static_cast<double>(value.m20), static_cast<double>(value.m01), static_cast<double>(value.m11), static_cast<double>(value.m21), static_cast<double>(value.m02), static_cast<double>(value.m12), static_cast<double>(value.m22));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::Mat4>) {
            return std::format("mat4(float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}))",
                               static_cast<double>(value.m00), static_cast<double>(value.m10), static_cast<double>(value.m20), static_cast<double>(value.m30), static_cast<double>(value.m01), static_cast<double>(value.m11), static_cast<double>(value.m21), static_cast<double>(value.m31),
                               static_cast<double>(value.m02), static_cast<double>(value.m12), static_cast<double>(value.m22), static_cast<double>(value.m32), static_cast<double>(value.m03), static_cast<double>(value.m13), static_cast<double>(value.m23), static_cast<double>(value.m33));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::Mat2x3>) {
            return std::format("mat2x3(float({}), float({}), float({}), float({}), float({}), float({}))",
                               static_cast<double>(value.c0.x), static_cast<double>(value.c0.y), static_cast<double>(value.c0.z), static_cast<double>(value.c1.x), static_cast<double>(value.c1.y), static_cast<double>(value.c1.z));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::Mat3x2>) {
            return std::format("mat3x2(float({}), float({}), float({}), float({}), float({}), float({}))",
                               static_cast<double>(value.c0.x), static_cast<double>(value.c0.y), static_cast<double>(value.c1.x), static_cast<double>(value.c1.y), static_cast<double>(value.c2.x), static_cast<double>(value.c2.y));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::Mat2x4>) {
            return std::format("mat2x4(float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}))",
                               static_cast<double>(value.c0.x), static_cast<double>(value.c0.y), static_cast<double>(value.c0.z), static_cast<double>(value.c0.w), static_cast<double>(value.c1.x), static_cast<double>(value.c1.y), static_cast<double>(value.c1.z), static_cast<double>(value.c1.w));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::Mat4x2>) {
            return std::format("mat4x2(float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}))",
                               static_cast<double>(value.c0.x), static_cast<double>(value.c0.y), static_cast<double>(value.c1.x), static_cast<double>(value.c1.y), static_cast<double>(value.c2.x), static_cast<double>(value.c2.y), static_cast<double>(value.c3.x), static_cast<double>(value.c3.y));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::Mat3x4>) {
            return std::format("mat3x4(float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}))",
                               static_cast<double>(value.c0.x), static_cast<double>(value.c0.y), static_cast<double>(value.c0.z), static_cast<double>(value.c0.w), static_cast<double>(value.c1.x), static_cast<double>(value.c1.y), static_cast<double>(value.c1.z), static_cast<double>(value.c1.w), static_cast<double>(value.c2.x), static_cast<double>(value.c2.y), static_cast<double>(value.c2.z), static_cast<double>(value.c2.w));  // Make GCC13 Happy :)
        } else if constexpr (std::is_same_v<T, Math::Mat4x3>) {
            return std::format("mat4x3(float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}), float({}))",
                               static_cast<double>(value.c0.x), static_cast<double>(value.c0.y), static_cast<double>(value.c0.z), static_cast<double>(value.c1.x), static_cast<double>(value.c1.y), static_cast<double>(value.c1.z), static_cast<double>(value.c2.x), static_cast<double>(value.c2.y), static_cast<double>(value.c2.z), static_cast<double>(value.c3.x), static_cast<double>(value.c3.y), static_cast<double>(value.c3.z));  // Make GCC13 Happy :)
        } else if constexpr (StructMeta<T>::isRegistered) {
            // For nested structs, use the struct's ToGLSLInit method
            return StructMeta<T>::ToGLSLInit(value);
        }
        return "unknown";
    }

    /**
     * Get GLSL type name for scalar types and registered structs
     */
    template<typename T>
    constexpr std::string_view GetGLSLTypeName() {
        if constexpr (std::is_same_v<T, float>) return "float";
        else if constexpr (std::is_same_v<T, int>) return "int";
        else if constexpr (std::is_same_v<T, bool>) return "bool";
        else if constexpr (std::is_same_v<T, Math::Vec2>) return "vec2";
        else if constexpr (std::is_same_v<T, Math::Vec3>) return "vec3";
        else if constexpr (std::is_same_v<T, Math::Vec4>) return "vec4";
        else if constexpr (std::is_same_v<T, Math::IVec2>) return "ivec2";
        else if constexpr (std::is_same_v<T, Math::IVec3>) return "ivec3";
        else if constexpr (std::is_same_v<T, Math::IVec4>) return "ivec4";
        else if constexpr (std::is_same_v<T, Math::Mat2>) return "mat2";
        else if constexpr (std::is_same_v<T, Math::Mat3>) return "mat3";
        else if constexpr (std::is_same_v<T, Math::Mat4>) return "mat4";
        else if constexpr (std::is_same_v<T, Math::Mat2x3>) return "mat2x3";
        else if constexpr (std::is_same_v<T, Math::Mat2x4>) return "mat2x4";
        else if constexpr (std::is_same_v<T, Math::Mat3x2>) return "mat3x2";
        else if constexpr (std::is_same_v<T, Math::Mat3x4>) return "mat3x4";
        else if constexpr (std::is_same_v<T, Math::Mat4x2>) return "mat4x2";
        else if constexpr (std::is_same_v<T, Math::Mat4x3>) return "mat4x3";
            // Handle nested registered structs
        else if constexpr (StructMeta<T>::isRegistered) return StructMeta<T>::glslTypeName;
        else return "unknown";
    }

    /**
     * Helper to register a struct and its dependencies
     * Default: do nothing for non-struct types
     */
    template<typename T>
    void RegisterStructWithDependencies() {}

    /**
     * Get the std430 alignment for a type
     */
    template<typename T>
    constexpr size_t GetStd430Alignment() {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, bool>) {
            return 4;
        } else if constexpr (std::is_same_v<T, Math::Vec2> || std::is_same_v<T, Math::IVec2>) {
            return 8;
        } else if constexpr (std::is_same_v<T, Math::Vec3> || std::is_same_v<T, Math::Vec4> ||
                             std::is_same_v<T, Math::IVec3> || std::is_same_v<T, Math::IVec4>) {
            return 16;
        } else if constexpr (std::is_same_v<T, Math::Mat2>) {
            return 16;  // vec2 columns aligned to 16
        } else if constexpr (std::is_same_v<T, Math::Mat3> || std::is_same_v<T, Math::Mat4> ||
                             std::is_same_v<T, Math::Mat2x3> || std::is_same_v<T, Math::Mat2x4> ||
                             std::is_same_v<T, Math::Mat3x2> || std::is_same_v<T, Math::Mat3x4> ||
                             std::is_same_v<T, Math::Mat4x2> || std::is_same_v<T, Math::Mat4x3>) {
            return 16;  // All matrices use vec2/vec3/vec4 columns, aligned to 16
        } else if constexpr (StructMeta<T>::isRegistered) {
            return 16;  // Nested structs aligned to 16 in std430
        }
        return 4;
    }

    /**
     * Get the std430 size for a type (with trailing alignment)
     */
    template<typename T>
    constexpr size_t GetStd430Size() {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, bool>) {
            return 4;
        } else if constexpr (std::is_same_v<T, Math::Vec2> || std::is_same_v<T, Math::IVec2>) {
            return 8;
        } else if constexpr (std::is_same_v<T, Math::Vec3> || std::is_same_v<T, Math::IVec3>) {
            // std430 struct member: vec3/ivec3 alignment is 16, but payload size is 12.
            return 12;
        } else if constexpr (std::is_same_v<T, Math::Vec4> || std::is_same_v<T, Math::IVec4>) {
            return 16;
        } else if constexpr (std::is_same_v<T, Math::Mat2>) {
            return 32;  // 2 * 16 (vec2 in std430 is 8, but column is aligned to 16)
        } else if constexpr (std::is_same_v<T, Math::Mat3>) {
            return 48;  // 3 * 16
        } else if constexpr (std::is_same_v<T, Math::Mat4>) {
            return 64;  // 4 * 16
        } else if constexpr (std::is_same_v<T, Math::Mat2x3>) {
            return 32;  // 2 columns * 16 (each column is vec3, aligned to 16)
        } else if constexpr (std::is_same_v<T, Math::Mat2x4>) {
            return 32;  // 2 columns * 16
        } else if constexpr (std::is_same_v<T, Math::Mat3x2>) {
            return 48;  // 3 columns * 16 (each column is vec2, but aligned to 16)
        } else if constexpr (std::is_same_v<T, Math::Mat3x4>) {
            return 48;  // 3 columns * 16
        } else if constexpr (std::is_same_v<T, Math::Mat4x2>) {
            return 64;  // 4 columns * 16
        } else if constexpr (std::is_same_v<T, Math::Mat4x3>) {
            return 64;  // 4 columns * 16
        } else if constexpr (StructMeta<T>::isRegistered) {
            // For nested structs, we need to compute at runtime or store in metadata
            // Return 16 as minimum alignment, actual size computed by ExpandedDefinition
            return 16;  // Placeholder - actual structs should use their computed size
        }
        return 4;
    }

    template<typename T>
    constexpr size_t GetCPUFieldCopySize();

    /**
     * Helper to copy a field to GPU buffer
     * For primitive types: direct memcpy
     * For nested structs: recursive conversion
     */
    template<typename T>
    inline void CopyFieldToGPU(const char *srcElem, char *dstElem, size_t &cppOffset, size_t &gpuOffset) {
        if constexpr (StructMeta<T>::isRegistered) {
            // Nested struct: use recursive conversion
            if (auto converter = StructMeta<T>::GetToGPUConverter()) {
                converter(srcElem + cppOffset, dstElem + gpuOffset,
                          StructMeta<T>::GetCPPLayoutSize(),
                          StructMeta<T>::GetGPULayoutSize(), 1);
            } else {
                std::memcpy(dstElem + gpuOffset, srcElem + cppOffset, StructMeta<T>::GetCPPLayoutSize());
            }
            cppOffset += StructMeta<T>::GetCPPLayoutSize();
            gpuOffset += StructMeta<T>::GetGPULayoutSize();
        } else {
            // Primitive type: direct memcpy
            constexpr size_t copySize = GetCPUFieldCopySize<T>();
            std::memcpy(dstElem + gpuOffset, srcElem + cppOffset, copySize);
            cppOffset += copySize;
            gpuOffset += GetStd430Size<T>();
        }
    }

    /**
     * Helper to copy a field from GPU buffer
     * For primitive types: direct memcpy
     * For nested structs: recursive conversion
     */
    template<typename T>
    inline void CopyFieldFromGPU(const char *srcElem, char *dstElem, size_t &cppOffset, size_t &gpuOffset) {
        if constexpr (StructMeta<T>::isRegistered) {
            // Nested struct: use recursive conversion
            if (auto converter = StructMeta<T>::GetFromGPUConverter()) {
                converter(srcElem + gpuOffset, dstElem + cppOffset,
                          StructMeta<T>::GetGPULayoutSize(),
                          StructMeta<T>::GetCPPLayoutSize(), 1);
            } else {
                std::memcpy(dstElem + cppOffset, srcElem + gpuOffset, StructMeta<T>::GetCPPLayoutSize());
            }
            cppOffset += StructMeta<T>::GetCPPLayoutSize();
            gpuOffset += StructMeta<T>::GetGPULayoutSize();
        } else {
            // Primitive type: direct memcpy
            constexpr size_t copySize = GetCPUFieldCopySize<T>();
            std::memcpy(dstElem + cppOffset, srcElem + gpuOffset, copySize);
            cppOffset += copySize;
            gpuOffset += GetStd430Size<T>();
        }
    }

    /**
     * Get GPU layout size for a type (handles both primitives and nested structs)
     */
    template<typename T>
    inline size_t GetFieldGPULayoutSize() {
        if constexpr (StructMeta<T>::isRegistered) {
            return StructMeta<T>::GetGPULayoutSize();
        } else {
            return GetStd430Size<T>();
        }
    }

    template<typename T>
    constexpr size_t GetCPUFieldCopySize() {
        if constexpr (std::is_same_v<T, Math::Vec2> || std::is_same_v<T, Math::IVec2>) return 8;
        else if constexpr (std::is_same_v<T, Math::Vec3> || std::is_same_v<T, Math::IVec3>) return 12;
        else if constexpr (std::is_same_v<T, Math::Vec4> || std::is_same_v<T, Math::IVec4>) return 16;
        else if constexpr (std::is_same_v<T, Math::Mat2>) return 16;
        else if constexpr (std::is_same_v<T, Math::Mat3>) return 36;
        else if constexpr (std::is_same_v<T, Math::Mat4>) return 64;
        else if constexpr (std::is_same_v<T, Math::Mat2x3>) return 24;
        else if constexpr (std::is_same_v<T, Math::Mat2x4>) return 32;
        else if constexpr (std::is_same_v<T, Math::Mat3x2>) return 24;
        else if constexpr (std::is_same_v<T, Math::Mat3x4>) return 48;
        else if constexpr (std::is_same_v<T, Math::Mat4x2>) return 32;
        else if constexpr (std::is_same_v<T, Math::Mat4x3>) return 48;
        else return sizeof(T);
    }

    template<typename StructT, typename FieldT>
    inline void CopyMemberToGPU(const char *srcElem, char *dstElem, size_t &gpuOffset, FieldT StructT::*member) {
        const auto *srcObj = reinterpret_cast<const StructT *>(srcElem);
        const char *srcField = reinterpret_cast<const char *>(&(srcObj->*member));

        size_t gpuAlign = GetStd430Alignment<FieldT>();
        gpuOffset = (gpuOffset + gpuAlign - 1) & ~(gpuAlign - 1);

        if constexpr (StructMeta<FieldT>::isRegistered) {
            if (auto converter = StructMeta<FieldT>::GetToGPUConverter()) {
                converter(srcField, dstElem + gpuOffset,
                          StructMeta<FieldT>::GetCPPLayoutSize(),
                          StructMeta<FieldT>::GetGPULayoutSize(), 1);
            } else {
                std::memcpy(dstElem + gpuOffset, srcField, StructMeta<FieldT>::GetCPPLayoutSize());
            }
        } else {
            constexpr size_t copySize = GetCPUFieldCopySize<FieldT>();
            std::memcpy(dstElem + gpuOffset, srcField, copySize);
        }

        gpuOffset += GetFieldGPULayoutSize<FieldT>();
    }

    template<typename StructT, typename FieldT>
    inline void CopyMemberFromGPU(const char *srcElem, char *dstElem, size_t &gpuOffset, FieldT StructT::*member) {
        auto *dstObj = reinterpret_cast<StructT *>(dstElem);
        char *dstField = reinterpret_cast<char *>(&(dstObj->*member));

        size_t gpuAlign = GetStd430Alignment<FieldT>();
        gpuOffset = (gpuOffset + gpuAlign - 1) & ~(gpuAlign - 1);

        if constexpr (StructMeta<FieldT>::isRegistered) {
            if (auto converter = StructMeta<FieldT>::GetFromGPUConverter()) {
                converter(srcElem + gpuOffset, dstField,
                          StructMeta<FieldT>::GetGPULayoutSize(),
                          StructMeta<FieldT>::GetCPPLayoutSize(), 1);
            } else {
                std::memcpy(dstField, srcElem + gpuOffset, StructMeta<FieldT>::GetCPPLayoutSize());
            }
        } else {
            constexpr size_t copySize = GetCPUFieldCopySize<FieldT>();
            std::memcpy(dstField, srcElem + gpuOffset, copySize);
        }

        gpuOffset += GetFieldGPULayoutSize<FieldT>();
    }
}

/************************************************
 * Variadic macro helpers (support up to 16 members)
 ************************************************/

#define EASYGPU_ARG_COUNT(...) EASYGPU_ARG_COUNT_(__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define EASYGPU_ARG_COUNT_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, N, ...) N

#define EASYGPU_CAT(a, b) a ## b

/************************************************
 * Member declaration macro
 ************************************************/

#define EASYGPU_DECL_1(type, name) type name;

/************************************************
 * GLSL field declaration macro
 ************************************************/

#define EASYGPU_GLSL_1(type, name) \
    /* Note: Using ostringstream instead of std::format to work around clangd bug where std::string arguments cause false positive errors */ \
    { std::ostringstream oss_; oss_ << "    " << GPU::Meta::GetGLSLTypeName<type>() << " " << #name << ";\n"; result += oss_.str(); }

/************************************************
 * Member access function macro
 ************************************************/

#define EASYGPU_ACCESS_1(type, name) \
    [[nodiscard]] GPU::IR::Value::Var<type> name() { \
        /* Note: Using ostringstream instead of std::format to work around clangd bug where std::string arguments cause false positive errors */ \
        std::ostringstream oss_; oss_ << _varNode->VarName() << "." << #name; \
        return GPU::IR::Value::Var<type>(oss_.str()); \
    }

/************************************************
 * GLSL init string generation for CPU struct capture
 ************************************************/

#define EASYGPU_INIT_1(type, name) oss << GPU::Meta::ValueToString(value.name) << ", ";

/************************************************
 * Dependency registration - extracts type and registers if it's a struct
 ************************************************/

#define EASYGPU_REG_DEP_1(type, name) GPU::Meta::RegisterStructWithDependencies<type>();

/************************************************
 * Sequential expansion helpers
 ************************************************/

// Member declarations
#define EASYGPU_SEQ_DECL_1(P1) EASYGPU_DECL_1 P1
#define EASYGPU_SEQ_DECL_2(P1, P2) EASYGPU_SEQ_DECL_1(P1) EASYGPU_DECL_1 P2
#define EASYGPU_SEQ_DECL_3(P1, P2, P3) EASYGPU_SEQ_DECL_2(P1, P2) EASYGPU_DECL_1 P3
#define EASYGPU_SEQ_DECL_4(P1, P2, P3, P4) EASYGPU_SEQ_DECL_3(P1, P2, P3) EASYGPU_DECL_1 P4
#define EASYGPU_SEQ_DECL_5(P1, P2, P3, P4, P5) EASYGPU_SEQ_DECL_4(P1, P2, P3, P4) EASYGPU_DECL_1 P5
#define EASYGPU_SEQ_DECL_6(P1, P2, P3, P4, P5, P6) EASYGPU_SEQ_DECL_5(P1, P2, P3, P4, P5) EASYGPU_DECL_1 P6
#define EASYGPU_SEQ_DECL_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SEQ_DECL_6(P1, P2, P3, P4, P5, P6) EASYGPU_DECL_1 P7
#define EASYGPU_SEQ_DECL_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SEQ_DECL_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_DECL_1 P8
#define EASYGPU_SEQ_DECL_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_SEQ_DECL_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_DECL_1 P9
#define EASYGPU_SEQ_DECL_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_SEQ_DECL_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_DECL_1 P10
#define EASYGPU_SEQ_DECL_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_SEQ_DECL_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_DECL_1 P11
#define EASYGPU_SEQ_DECL_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_SEQ_DECL_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_DECL_1 P12
#define EASYGPU_SEQ_DECL_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_SEQ_DECL_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_DECL_1 P13
#define EASYGPU_SEQ_DECL_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_SEQ_DECL_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_DECL_1 P14
#define EASYGPU_SEQ_DECL_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_SEQ_DECL_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_DECL_1 P15
#define EASYGPU_SEQ_DECL_16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) EASYGPU_SEQ_DECL_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_DECL_1 P16

// GLSL fields
#define EASYGPU_SEQ_GLSL_1(P1) EASYGPU_GLSL_1 P1
#define EASYGPU_SEQ_GLSL_2(P1, P2) EASYGPU_SEQ_GLSL_1(P1) EASYGPU_GLSL_1 P2
#define EASYGPU_SEQ_GLSL_3(P1, P2, P3) EASYGPU_SEQ_GLSL_2(P1, P2) EASYGPU_GLSL_1 P3
#define EASYGPU_SEQ_GLSL_4(P1, P2, P3, P4) EASYGPU_SEQ_GLSL_3(P1, P2, P3) EASYGPU_GLSL_1 P4
#define EASYGPU_SEQ_GLSL_5(P1, P2, P3, P4, P5) EASYGPU_SEQ_GLSL_4(P1, P2, P3, P4) EASYGPU_GLSL_1 P5
#define EASYGPU_SEQ_GLSL_6(P1, P2, P3, P4, P5, P6) EASYGPU_SEQ_GLSL_5(P1, P2, P3, P4, P5) EASYGPU_GLSL_1 P6
#define EASYGPU_SEQ_GLSL_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SEQ_GLSL_6(P1, P2, P3, P4, P5, P6) EASYGPU_GLSL_1 P7
#define EASYGPU_SEQ_GLSL_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SEQ_GLSL_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_GLSL_1 P8
#define EASYGPU_SEQ_GLSL_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_SEQ_GLSL_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_GLSL_1 P9
#define EASYGPU_SEQ_GLSL_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_SEQ_GLSL_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_GLSL_1 P10
#define EASYGPU_SEQ_GLSL_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_SEQ_GLSL_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_GLSL_1 P11
#define EASYGPU_SEQ_GLSL_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_SEQ_GLSL_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_GLSL_1 P12
#define EASYGPU_SEQ_GLSL_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_SEQ_GLSL_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_GLSL_1 P13
#define EASYGPU_SEQ_GLSL_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_SEQ_GLSL_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_GLSL_1 P14
#define EASYGPU_SEQ_GLSL_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_SEQ_GLSL_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_GLSL_1 P15
#define EASYGPU_SEQ_GLSL_16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) EASYGPU_SEQ_GLSL_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_GLSL_1 P16

// Member access functions
#define EASYGPU_SEQ_ACCESS_1(P1) EASYGPU_ACCESS_1 P1
#define EASYGPU_SEQ_ACCESS_2(P1, P2) EASYGPU_SEQ_ACCESS_1(P1) EASYGPU_ACCESS_1 P2
#define EASYGPU_SEQ_ACCESS_3(P1, P2, P3) EASYGPU_SEQ_ACCESS_2(P1, P2) EASYGPU_ACCESS_1 P3
#define EASYGPU_SEQ_ACCESS_4(P1, P2, P3, P4) EASYGPU_SEQ_ACCESS_3(P1, P2, P3) EASYGPU_ACCESS_1 P4
#define EASYGPU_SEQ_ACCESS_5(P1, P2, P3, P4, P5) EASYGPU_SEQ_ACCESS_4(P1, P2, P3, P4) EASYGPU_ACCESS_1 P5
#define EASYGPU_SEQ_ACCESS_6(P1, P2, P3, P4, P5, P6) EASYGPU_SEQ_ACCESS_5(P1, P2, P3, P4, P5) EASYGPU_ACCESS_1 P6
#define EASYGPU_SEQ_ACCESS_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SEQ_ACCESS_6(P1, P2, P3, P4, P5, P6) EASYGPU_ACCESS_1 P7
#define EASYGPU_SEQ_ACCESS_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SEQ_ACCESS_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_ACCESS_1 P8
#define EASYGPU_SEQ_ACCESS_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_SEQ_ACCESS_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_ACCESS_1 P9
#define EASYGPU_SEQ_ACCESS_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_SEQ_ACCESS_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_ACCESS_1 P10
#define EASYGPU_SEQ_ACCESS_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_SEQ_ACCESS_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_ACCESS_1 P11
#define EASYGPU_SEQ_ACCESS_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_SEQ_ACCESS_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_ACCESS_1 P12
#define EASYGPU_SEQ_ACCESS_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_SEQ_ACCESS_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_ACCESS_1 P13
#define EASYGPU_SEQ_ACCESS_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_SEQ_ACCESS_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_ACCESS_1 P14
#define EASYGPU_SEQ_ACCESS_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_SEQ_ACCESS_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_ACCESS_1 P15
#define EASYGPU_SEQ_ACCESS_16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) EASYGPU_SEQ_ACCESS_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_ACCESS_1 P16

// GLSL init generation
#define EASYGPU_SEQ_INIT_1(P1) EASYGPU_INIT_1 P1
#define EASYGPU_SEQ_INIT_2(P1, P2) EASYGPU_SEQ_INIT_1(P1) EASYGPU_INIT_1 P2
#define EASYGPU_SEQ_INIT_3(P1, P2, P3) EASYGPU_SEQ_INIT_2(P1, P2) EASYGPU_INIT_1 P3
#define EASYGPU_SEQ_INIT_4(P1, P2, P3, P4) EASYGPU_SEQ_INIT_3(P1, P2, P3) EASYGPU_INIT_1 P4
#define EASYGPU_SEQ_INIT_5(P1, P2, P3, P4, P5) EASYGPU_SEQ_INIT_4(P1, P2, P3, P4) EASYGPU_INIT_1 P5
#define EASYGPU_SEQ_INIT_6(P1, P2, P3, P4, P5, P6) EASYGPU_SEQ_INIT_5(P1, P2, P3, P4, P5) EASYGPU_INIT_1 P6
#define EASYGPU_SEQ_INIT_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SEQ_INIT_6(P1, P2, P3, P4, P5, P6) EASYGPU_INIT_1 P7
#define EASYGPU_SEQ_INIT_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SEQ_INIT_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_INIT_1 P8
#define EASYGPU_SEQ_INIT_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_SEQ_INIT_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_INIT_1 P9
#define EASYGPU_SEQ_INIT_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_SEQ_INIT_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_INIT_1 P10
#define EASYGPU_SEQ_INIT_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_SEQ_INIT_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_INIT_1 P11
#define EASYGPU_SEQ_INIT_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_SEQ_INIT_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_INIT_1 P12
#define EASYGPU_SEQ_INIT_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_SEQ_INIT_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_INIT_1 P13
#define EASYGPU_SEQ_INIT_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_SEQ_INIT_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_INIT_1 P14
#define EASYGPU_SEQ_INIT_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_SEQ_INIT_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_INIT_1 P15
#define EASYGPU_SEQ_INIT_16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) EASYGPU_SEQ_INIT_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_INIT_1 P16

// Dependency registration
#define EASYGPU_SEQ_REG_1(P1) EASYGPU_REG_DEP_1 P1
#define EASYGPU_SEQ_REG_2(P1, P2) EASYGPU_SEQ_REG_1(P1) EASYGPU_REG_DEP_1 P2
#define EASYGPU_SEQ_REG_3(P1, P2, P3) EASYGPU_SEQ_REG_2(P1, P2) EASYGPU_REG_DEP_1 P3
#define EASYGPU_SEQ_REG_4(P1, P2, P3, P4) EASYGPU_SEQ_REG_3(P1, P2, P3) EASYGPU_REG_DEP_1 P4
#define EASYGPU_SEQ_REG_5(P1, P2, P3, P4, P5) EASYGPU_SEQ_REG_4(P1, P2, P3, P4) EASYGPU_REG_DEP_1 P5
#define EASYGPU_SEQ_REG_6(P1, P2, P3, P4, P5, P6) EASYGPU_SEQ_REG_5(P1, P2, P3, P4, P5) EASYGPU_REG_DEP_1 P6
#define EASYGPU_SEQ_REG_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SEQ_REG_6(P1, P2, P3, P4, P5, P6) EASYGPU_REG_DEP_1 P7
#define EASYGPU_SEQ_REG_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SEQ_REG_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_REG_DEP_1 P8
#define EASYGPU_SEQ_REG_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_SEQ_REG_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_REG_DEP_1 P9
#define EASYGPU_SEQ_REG_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_SEQ_REG_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_REG_DEP_1 P10
#define EASYGPU_SEQ_REG_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_SEQ_REG_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_REG_DEP_1 P11
#define EASYGPU_SEQ_REG_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_SEQ_REG_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_REG_DEP_1 P12
#define EASYGPU_SEQ_REG_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_SEQ_REG_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_REG_DEP_1 P13
#define EASYGPU_SEQ_REG_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_SEQ_REG_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_REG_DEP_1 P14
#define EASYGPU_SEQ_REG_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_SEQ_REG_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_REG_DEP_1 P15
#define EASYGPU_SEQ_REG_16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) EASYGPU_SEQ_REG_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_REG_DEP_1 P16

// Size calculation helpers
#define EASYGPU_SIZE_FIELD(type, name) \
    { \
        size_t align = GPU::Meta::GetStd430Alignment<type>(); \
        size = (size + align - 1) & ~(align - 1); \
        size += GPU::Meta::GetFieldGPULayoutSize<type>(); \
        if (align > maxAlign) maxAlign = align; \
    }

#define EASYGPU_SEQ_SIZE_1(P1) EASYGPU_SIZE_FIELD P1
#define EASYGPU_SEQ_SIZE_2(P1, P2) EASYGPU_SEQ_SIZE_1(P1) EASYGPU_SIZE_FIELD P2
#define EASYGPU_SEQ_SIZE_3(P1, P2, P3) EASYGPU_SEQ_SIZE_2(P1, P2) EASYGPU_SIZE_FIELD P3
#define EASYGPU_SEQ_SIZE_4(P1, P2, P3, P4) EASYGPU_SEQ_SIZE_3(P1, P2, P3) EASYGPU_SIZE_FIELD P4
#define EASYGPU_SEQ_SIZE_5(P1, P2, P3, P4, P5) EASYGPU_SEQ_SIZE_4(P1, P2, P3, P4) EASYGPU_SIZE_FIELD P5
#define EASYGPU_SEQ_SIZE_6(P1, P2, P3, P4, P5, P6) EASYGPU_SEQ_SIZE_5(P1, P2, P3, P4, P5) EASYGPU_SIZE_FIELD P6
#define EASYGPU_SEQ_SIZE_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SEQ_SIZE_6(P1, P2, P3, P4, P5, P6) EASYGPU_SIZE_FIELD P7
#define EASYGPU_SEQ_SIZE_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SEQ_SIZE_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SIZE_FIELD P8
#define EASYGPU_SEQ_SIZE_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_SEQ_SIZE_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SIZE_FIELD P9
#define EASYGPU_SEQ_SIZE_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_SEQ_SIZE_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_SIZE_FIELD P10
#define EASYGPU_SEQ_SIZE_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_SEQ_SIZE_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_SIZE_FIELD P11
#define EASYGPU_SEQ_SIZE_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_SEQ_SIZE_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_SIZE_FIELD P12
#define EASYGPU_SEQ_SIZE_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_SEQ_SIZE_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_SIZE_FIELD P13
#define EASYGPU_SEQ_SIZE_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_SEQ_SIZE_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_SIZE_FIELD P14
#define EASYGPU_SEQ_SIZE_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_SEQ_SIZE_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_SIZE_FIELD P15
#define EASYGPU_SEQ_SIZE_16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) EASYGPU_SEQ_SIZE_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_SIZE_FIELD P16

// Field conversion helpers - now with nested struct support!
#define EASYGPU_CONVERT_FIELD_TO_GPU(type, name) \
    { \
        GPU::Meta::CopyMemberToGPU<_EasyGPU_CurrentStruct, type>( \
            srcElem, dstElem, gpuOffset, &_EasyGPU_CurrentStruct::name); \
    }

#define EASYGPU_CONVERT_FIELD_FROM_GPU(type, name) \
    { \
        GPU::Meta::CopyMemberFromGPU<_EasyGPU_CurrentStruct, type>( \
            srcElem, dstElem, gpuOffset, &_EasyGPU_CurrentStruct::name); \
    }

#define EASYGPU_SEQ_CONVERT_TO_GPU_1(P1) EASYGPU_CONVERT_FIELD_TO_GPU P1
#define EASYGPU_SEQ_CONVERT_TO_GPU_2(P1, P2) EASYGPU_SEQ_CONVERT_TO_GPU_1(P1) EASYGPU_CONVERT_FIELD_TO_GPU P2
#define EASYGPU_SEQ_CONVERT_TO_GPU_3(P1, P2, P3) EASYGPU_SEQ_CONVERT_TO_GPU_2(P1, P2) EASYGPU_CONVERT_FIELD_TO_GPU P3
#define EASYGPU_SEQ_CONVERT_TO_GPU_4(P1, P2, P3, P4) EASYGPU_SEQ_CONVERT_TO_GPU_3(P1, P2, P3) EASYGPU_CONVERT_FIELD_TO_GPU P4
#define EASYGPU_SEQ_CONVERT_TO_GPU_5(P1, P2, P3, P4, P5) EASYGPU_SEQ_CONVERT_TO_GPU_4(P1, P2, P3, P4) EASYGPU_CONVERT_FIELD_TO_GPU P5
#define EASYGPU_SEQ_CONVERT_TO_GPU_6(P1, P2, P3, P4, P5, P6) EASYGPU_SEQ_CONVERT_TO_GPU_5(P1, P2, P3, P4, P5) EASYGPU_CONVERT_FIELD_TO_GPU P6
#define EASYGPU_SEQ_CONVERT_TO_GPU_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SEQ_CONVERT_TO_GPU_6(P1, P2, P3, P4, P5, P6) EASYGPU_CONVERT_FIELD_TO_GPU P7
#define EASYGPU_SEQ_CONVERT_TO_GPU_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SEQ_CONVERT_TO_GPU_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_CONVERT_FIELD_TO_GPU P8
#define EASYGPU_SEQ_CONVERT_TO_GPU_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_SEQ_CONVERT_TO_GPU_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_CONVERT_FIELD_TO_GPU P9
#define EASYGPU_SEQ_CONVERT_TO_GPU_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_SEQ_CONVERT_TO_GPU_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_CONVERT_FIELD_TO_GPU P10
#define EASYGPU_SEQ_CONVERT_TO_GPU_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_SEQ_CONVERT_TO_GPU_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_CONVERT_FIELD_TO_GPU P11
#define EASYGPU_SEQ_CONVERT_TO_GPU_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_SEQ_CONVERT_TO_GPU_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_CONVERT_FIELD_TO_GPU P12
#define EASYGPU_SEQ_CONVERT_TO_GPU_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_SEQ_CONVERT_TO_GPU_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_CONVERT_FIELD_TO_GPU P13
#define EASYGPU_SEQ_CONVERT_TO_GPU_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_SEQ_CONVERT_TO_GPU_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_CONVERT_FIELD_TO_GPU P14
#define EASYGPU_SEQ_CONVERT_TO_GPU_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_SEQ_CONVERT_TO_GPU_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_CONVERT_FIELD_TO_GPU P15
#define EASYGPU_SEQ_CONVERT_TO_GPU_16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) EASYGPU_SEQ_CONVERT_TO_GPU_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_CONVERT_FIELD_TO_GPU P16

#define EASYGPU_SEQ_CONVERT_FROM_GPU_1(P1) EASYGPU_CONVERT_FIELD_FROM_GPU P1
#define EASYGPU_SEQ_CONVERT_FROM_GPU_2(P1, P2) EASYGPU_SEQ_CONVERT_FROM_GPU_1(P1) EASYGPU_CONVERT_FIELD_FROM_GPU P2
#define EASYGPU_SEQ_CONVERT_FROM_GPU_3(P1, P2, P3) EASYGPU_SEQ_CONVERT_FROM_GPU_2(P1, P2) EASYGPU_CONVERT_FIELD_FROM_GPU P3
#define EASYGPU_SEQ_CONVERT_FROM_GPU_4(P1, P2, P3, P4) EASYGPU_SEQ_CONVERT_FROM_GPU_3(P1, P2, P3) EASYGPU_CONVERT_FIELD_FROM_GPU P4
#define EASYGPU_SEQ_CONVERT_FROM_GPU_5(P1, P2, P3, P4, P5) EASYGPU_SEQ_CONVERT_FROM_GPU_4(P1, P2, P3, P4) EASYGPU_CONVERT_FIELD_FROM_GPU P5
#define EASYGPU_SEQ_CONVERT_FROM_GPU_6(P1, P2, P3, P4, P5, P6) EASYGPU_SEQ_CONVERT_FROM_GPU_5(P1, P2, P3, P4, P5) EASYGPU_CONVERT_FIELD_FROM_GPU P6
#define EASYGPU_SEQ_CONVERT_FROM_GPU_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SEQ_CONVERT_FROM_GPU_6(P1, P2, P3, P4, P5, P6) EASYGPU_CONVERT_FIELD_FROM_GPU P7
#define EASYGPU_SEQ_CONVERT_FROM_GPU_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SEQ_CONVERT_FROM_GPU_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_CONVERT_FIELD_FROM_GPU P8
#define EASYGPU_SEQ_CONVERT_FROM_GPU_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_SEQ_CONVERT_FROM_GPU_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_CONVERT_FIELD_FROM_GPU P9
#define EASYGPU_SEQ_CONVERT_FROM_GPU_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_SEQ_CONVERT_FROM_GPU_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_CONVERT_FIELD_FROM_GPU P10
#define EASYGPU_SEQ_CONVERT_FROM_GPU_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_SEQ_CONVERT_FROM_GPU_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_CONVERT_FIELD_FROM_GPU P11
#define EASYGPU_SEQ_CONVERT_FROM_GPU_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_SEQ_CONVERT_FROM_GPU_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_CONVERT_FIELD_FROM_GPU P12
#define EASYGPU_SEQ_CONVERT_FROM_GPU_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_SEQ_CONVERT_FROM_GPU_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_CONVERT_FIELD_FROM_GPU P13
#define EASYGPU_SEQ_CONVERT_FROM_GPU_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_SEQ_CONVERT_FROM_GPU_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_CONVERT_FIELD_FROM_GPU P14
#define EASYGPU_SEQ_CONVERT_FROM_GPU_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_SEQ_CONVERT_FROM_GPU_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_CONVERT_FIELD_FROM_GPU P15
#define EASYGPU_SEQ_CONVERT_FROM_GPU_16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) EASYGPU_SEQ_CONVERT_FROM_GPU_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_CONVERT_FIELD_FROM_GPU P16

// Layout generation helpers
#define EASYGPU_LAYOUT_FIELD(type, name) \
    { \
        size_t cppAlign = alignof(type); \
        size_t gpuAlign = GPU::Meta::GetStd430Alignment<type>(); \
        size_t cppSize = sizeof(type); \
        size_t gpuSize = GPU::Meta::GetStd430Size<type>(); \
        cppOffset = (cppOffset + cppAlign - 1) & ~(cppAlign - 1); \
        gpuOffset = (gpuOffset + gpuAlign - 1) & ~(gpuAlign - 1); \
        meta.fields.push_back({cppOffset, cppSize, gpuOffset, gpuSize, gpuAlign}); \
        cppOffset += cppSize; \
        gpuOffset += gpuSize; \
    }

#define EASYGPU_SEQ_LAYOUT_1(P1) EASYGPU_LAYOUT_FIELD P1
#define EASYGPU_SEQ_LAYOUT_2(P1, P2) EASYGPU_SEQ_LAYOUT_1(P1) EASYGPU_LAYOUT_FIELD P2
#define EASYGPU_SEQ_LAYOUT_3(P1, P2, P3) EASYGPU_SEQ_LAYOUT_2(P1, P2) EASYGPU_LAYOUT_FIELD P3
#define EASYGPU_SEQ_LAYOUT_4(P1, P2, P3, P4) EASYGPU_SEQ_LAYOUT_3(P1, P2, P3) EASYGPU_LAYOUT_FIELD P4
#define EASYGPU_SEQ_LAYOUT_5(P1, P2, P3, P4, P5) EASYGPU_SEQ_LAYOUT_4(P1, P2, P3, P4) EASYGPU_LAYOUT_FIELD P5
#define EASYGPU_SEQ_LAYOUT_6(P1, P2, P3, P4, P5, P6) EASYGPU_SEQ_LAYOUT_5(P1, P2, P3, P4, P5) EASYGPU_LAYOUT_FIELD P6
#define EASYGPU_SEQ_LAYOUT_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SEQ_LAYOUT_6(P1, P2, P3, P4, P5, P6) EASYGPU_LAYOUT_FIELD P7
#define EASYGPU_SEQ_LAYOUT_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SEQ_LAYOUT_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_LAYOUT_FIELD P8
#define EASYGPU_SEQ_LAYOUT_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_SEQ_LAYOUT_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_LAYOUT_FIELD P9
#define EASYGPU_SEQ_LAYOUT_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_SEQ_LAYOUT_9(P1, P2, P3, P4, P5, P6, P7, P8, P9) EASYGPU_LAYOUT_FIELD P10
#define EASYGPU_SEQ_LAYOUT_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_SEQ_LAYOUT_10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) EASYGPU_LAYOUT_FIELD P11
#define EASYGPU_SEQ_LAYOUT_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_SEQ_LAYOUT_11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) EASYGPU_LAYOUT_FIELD P12
#define EASYGPU_SEQ_LAYOUT_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_SEQ_LAYOUT_12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) EASYGPU_LAYOUT_FIELD P13
#define EASYGPU_SEQ_LAYOUT_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_SEQ_LAYOUT_13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) EASYGPU_LAYOUT_FIELD P14
#define EASYGPU_SEQ_LAYOUT_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_SEQ_LAYOUT_14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) EASYGPU_LAYOUT_FIELD P15
#define EASYGPU_SEQ_LAYOUT_16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) EASYGPU_SEQ_LAYOUT_15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) EASYGPU_LAYOUT_FIELD P16

// Dispatcher
#define EASYGPU_DO_DECL(N, ...) EASYGPU_CAT(EASYGPU_SEQ_DECL_, N)(__VA_ARGS__)
#define EASYGPU_DO_GLSL(N, ...) EASYGPU_CAT(EASYGPU_SEQ_GLSL_, N)(__VA_ARGS__)
#define EASYGPU_DO_ACCESS(N, ...) EASYGPU_CAT(EASYGPU_SEQ_ACCESS_, N)(__VA_ARGS__)
#define EASYGPU_DO_INIT(N, ...) EASYGPU_CAT(EASYGPU_SEQ_INIT_, N)(__VA_ARGS__)
#define EASYGPU_DO_REG(N, ...) EASYGPU_CAT(EASYGPU_SEQ_REG_, N)(__VA_ARGS__)
#define EASYGPU_DO_SIZE(N, ...) \
    size_t size = 0; \
    size_t maxAlign = 1; \
    EASYGPU_CAT(EASYGPU_SEQ_SIZE_, N)(__VA_ARGS__)

#define EASYGPU_DO_CONVERT_TO_GPU(N, ...) \
    size_t gpuOffset = 0; \
    EASYGPU_CAT(EASYGPU_SEQ_CONVERT_TO_GPU_, N)(__VA_ARGS__)

#define EASYGPU_DO_CONVERT_FROM_GPU(N, ...) \
    size_t gpuOffset = 0; \
    EASYGPU_CAT(EASYGPU_SEQ_CONVERT_FROM_GPU_, N)(__VA_ARGS__)

// Uniform upload helpers for struct types
#define EASYGPU_UPLOAD_UNIFORM_FLOAT(program, locationPrefix, value, name) \
    { \
        GLint loc = glGetUniformLocation(program, (std::string(locationPrefix) + "." + #name).c_str()); \
        if (loc != -1) glProgramUniform1f(program, loc, value.name); \
    }
#define EASYGPU_UPLOAD_UNIFORM_INT(program, locationPrefix, value, name) \
    { \
        GLint loc = glGetUniformLocation(program, (std::string(locationPrefix) + "." + #name).c_str()); \
        if (loc != -1) glProgramUniform1i(program, loc, value.name); \
    }
#define EASYGPU_UPLOAD_UNIFORM_BOOL(program, locationPrefix, value, name) \
    { \
        GLint loc = glGetUniformLocation(program, (std::string(locationPrefix) + "." + #name).c_str()); \
        if (loc != -1) glProgramUniform1i(program, loc, value.name ? 1 : 0); \
    }
#define EASYGPU_UPLOAD_UNIFORM_VEC2(program, locationPrefix, value, name) \
    { \
        GLint loc = glGetUniformLocation(program, (std::string(locationPrefix) + "." + #name).c_str()); \
        if (loc != -1) glProgramUniform2fv(program, loc, 1, &value.name.x); \
    }
#define EASYGPU_UPLOAD_UNIFORM_VEC3(program, locationPrefix, value, name) \
    { \
        GLint loc = glGetUniformLocation(program, (std::string(locationPrefix) + "." + #name).c_str()); \
        if (loc != -1) glProgramUniform3fv(program, loc, 1, &value.name.x); \
    }
#define EASYGPU_UPLOAD_UNIFORM_VEC4(program, locationPrefix, value, name) \
    { \
        GLint loc = glGetUniformLocation(program, (std::string(locationPrefix) + "." + #name).c_str()); \
        if (loc != -1) glProgramUniform4fv(program, loc, 1, &value.name.x); \
    }

#define EASYGPU_UPLOAD_FIELD(type, name) \
    EASYGPU_UPLOAD_UNIFORM_##type(program, uniformName.c_str(), value, name)

#define EASYGPU_SEQ_UPLOAD_1(P1) EASYGPU_UPLOAD_FIELD P1
#define EASYGPU_SEQ_UPLOAD_2(P1, P2) EASYGPU_SEQ_UPLOAD_1(P1) EASYGPU_UPLOAD_FIELD P2
#define EASYGPU_SEQ_UPLOAD_3(P1, P2, P3) EASYGPU_SEQ_UPLOAD_2(P1, P2) EASYGPU_UPLOAD_FIELD P3
#define EASYGPU_SEQ_UPLOAD_4(P1, P2, P3, P4) EASYGPU_SEQ_UPLOAD_3(P1, P2, P3) EASYGPU_UPLOAD_FIELD P4
#define EASYGPU_SEQ_UPLOAD_5(P1, P2, P3, P4, P5) EASYGPU_SEQ_UPLOAD_4(P1, P2, P3, P4) EASYGPU_UPLOAD_FIELD P5
#define EASYGPU_SEQ_UPLOAD_6(P1, P2, P3, P4, P5, P6) EASYGPU_SEQ_UPLOAD_5(P1, P2, P3, P4, P5) EASYGPU_UPLOAD_FIELD P6
#define EASYGPU_SEQ_UPLOAD_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_SEQ_UPLOAD_6(P1, P2, P3, P4, P5, P6) EASYGPU_UPLOAD_FIELD P7
#define EASYGPU_SEQ_UPLOAD_8(P1, P2, P3, P4, P5, P6, P7, P8) EASYGPU_SEQ_UPLOAD_7(P1, P2, P3, P4, P5, P6, P7) EASYGPU_UPLOAD_FIELD P8

#define EASYGPU_DO_UPLOAD(N, ...) EASYGPU_CAT(EASYGPU_SEQ_UPLOAD_, N)(__VA_ARGS__)

/************************************************
 * Main EASYGPU_STRUCT macro - All-in-one with CPU capture support!
 * Usage: EASYGPU_STRUCT(StructName, (type1, member1), (type2, member2), ...)
 ************************************************/

#define EASYGPU_STRUCT(StructType, ...) \
    /* 1. Define the C++ struct */ \
    struct alignas(16) StructType { \
        EASYGPU_DO_DECL(EASYGPU_ARG_COUNT(__VA_ARGS__), __VA_ARGS__) \
    }; \
    \
    /* 2. Define StructMeta specialization */ \
    namespace GPU::Meta { \
        template<> \
        struct StructMeta<StructType> { \
            static constexpr bool isRegistered = true; \
            static constexpr const char* glslTypeName = #StructType; \
            using _EasyGPU_CurrentStruct = StructType; \
            \
            static std::string ExpandedDefinition() { \
                std::string result = "struct " #StructType " {\n"; \
                EASYGPU_DO_GLSL(EASYGPU_ARG_COUNT(__VA_ARGS__), __VA_ARGS__) \
                result += "};\n"; \
                return result; \
            } \
            \
            /* Get std430 layout definition for buffer usage */ \
            static std::string GetStd430Definition() { \
                return ExpandedDefinition(); \
            } \
            \
            /* Convert CPU struct to GLSL initialization string */ \
            static std::string ToGLSLInit(const StructType& value) { \
                std::ostringstream oss; \
                oss << #StructType << "("; \
                EASYGPU_DO_INIT(EASYGPU_ARG_COUNT(__VA_ARGS__), __VA_ARGS__) \
                /* Remove trailing ", " */ \
                std::string result = oss.str(); \
                if (result.length() >= 2) result = result.substr(0, result.length() - 2); \
                result += ")"; \
                return result; \
            } \
            \
            /* Layout information for automatic conversion */ \
            static size_t GetCPPLayoutSize() { return sizeof(StructType); } \
            static size_t GetGPULayoutSize() { \
                EASYGPU_DO_SIZE(EASYGPU_ARG_COUNT(__VA_ARGS__), __VA_ARGS__) \
                return (size + maxAlign - 1) & ~(maxAlign - 1); \
            } \
            /* \
             * Size equality is insufficient (e.g. vec3 align mismatch can keep total size equal \
             * while field offsets differ). Always use generated field-wise converters. \
             */ \
            static bool NeedsLayoutConversion() { return true; } \
            \
            /* Upload struct members as individual uniforms (for Uniform<T> with struct types) */ \
            static void UploadUniform(uint32_t program, const std::string& uniformName, const StructType& value) { \
                EASYGPU_DO_UPLOAD(EASYGPU_ARG_COUNT(__VA_ARGS__), __VA_ARGS__) \
            } \
            \
            /* Generated converters - performs field-by-field copy with padding */ \
            static void ConvertToGPUImpl(const char* src, char* dst, size_t srcStride, size_t dstStride, size_t count) { \
                for (size_t i = 0; i < count; i++) { \
                    const char* srcElem = src + i * srcStride; \
                    char* dstElem = dst + i * dstStride; \
                    EASYGPU_DO_CONVERT_TO_GPU(EASYGPU_ARG_COUNT(__VA_ARGS__), __VA_ARGS__) \
                } \
            } \
            static void ConvertFromGPUImpl(const char* src, char* dst, size_t srcStride, size_t dstStride, size_t count) { \
                for (size_t i = 0; i < count; i++) { \
                    const char* srcElem = src + i * srcStride; \
                    char* dstElem = dst + i * dstStride; \
                    EASYGPU_DO_CONVERT_FROM_GPU(EASYGPU_ARG_COUNT(__VA_ARGS__), __VA_ARGS__) \
                } \
            } \
            static GPU::Meta::ToGPUConverter GetToGPUConverter() { return ConvertToGPUImpl; } \
            static GPU::Meta::FromGPUConverter GetFromGPUConverter() { return ConvertFromGPUImpl; } \
        };                              \
        /* Register this struct type and its dependencies */ \
        template<> \
        inline void RegisterStructWithDependencies<StructType>() { \
            auto& ctx = *GPU::IR::Builder::Builder::Get().Context(); \
            std::string typeName(GPU::Meta::StructMeta<StructType>::glslTypeName); \
            if (!ctx.HasStructDefinition(typeName)) { \
                /* First register all dependencies */ \
                EASYGPU_DO_REG(EASYGPU_ARG_COUNT(__VA_ARGS__), __VA_ARGS__) \
                /* Then register this struct */ \
                ctx.AddStructDefinition(typeName, GPU::Meta::StructMeta<StructType>::ExpandedDefinition()); \
            } \
        } \
    } \
    \
    /* 3. Forward declare Var<StructType> for Expr's constructor */ \
    namespace GPU::IR::Value { \
        template<> class Var<StructType>; \
    } \
    \
    /* 4. Define Expr<StructType> specialization - minimal implementation */ \
    namespace GPU::IR::Value { \
        template<> \
        class Expr<StructType> : public ExprBase { \
        public: \
            Expr() = default; \
            \
            explicit Expr(std::unique_ptr<Node::Node> node) : ExprBase(std::move(node)) {} \
            \
            Expr(const Expr&) = delete; \
            Expr& operator=(const Expr&) = delete; \
            Expr(Expr&&) = default; \
            Expr& operator=(Expr&&) = default; \
            ~Expr() = default; \
            \
            /* Construct from Var<StructType> */ \
            explicit Expr(const Var<StructType>& var); \
            \
            /* Construct from rvalue Var<StructType> */ \
            explicit Expr(Var<StructType>&& var); \
        }; \
    } \
    \
    /* 5. Define Var<StructType> specialization */ \
    namespace GPU::IR::Value {          \
        template<> \
        class Var<StructType> : public Value { \
        public: \
            /* Default constructor */ \
            Var() { \
                GPU::Meta::RegisterStructWithDependencies<StructType>(); \
                auto name = Builder::Builder::Get().Context()->AssignVarName(); \
                _node = std::make_unique<Node::LocalVariableNode>( \
                    name, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); \
                _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); \
                Builder::Builder::Get().Build(*_varNode, true); \
            } \
            \
            /* From existing variable name (for member access chaining) */ \
            explicit Var(const std::string& varName) { \
                GPU::Meta::RegisterStructWithDependencies<StructType>(); \
                _node = std::make_unique<Node::LocalVariableNode>( \
                    varName, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); \
                _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); \
            }                           \
            \
            /* From existing variable name with IsExternal flag (for uniforms) */ \
            Var(const std::string& varName, bool IsExternal) { \
                GPU::Meta::RegisterStructWithDependencies<StructType>(); \
                _node = std::make_unique<Node::LocalVariableNode>( \
                    varName, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName), IsExternal); \
                _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); \
            }                           \
            /* Copy constructor - like VarBase */ \
            Var(const Var &Other) : Var() { \
                auto rhs = Other.Load(); \
                auto lhs = Load(); \
                auto store = std::make_unique<Node::StoreNode>(std::move(lhs), std::move(rhs)); \
                Builder::Builder::Get().Build(*store, true); \
            } \
            \
            /* From CPU struct (uniform capture) */ \
            Var(const StructType& value) { \
                GPU::Meta::RegisterStructWithDependencies<StructType>(); \
                auto name = Builder::Builder::Get().Context()->AssignVarName(); \
                _node = std::make_unique<Node::LocalVariableNode>( \
                    name, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); \
                _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); \
                Builder::Builder::Get().Build(*_varNode, true); \
                /* Initialize with CPU values */ \
                /* Note: Using ostringstream instead of std::format to work around clangd bug where std::string arguments cause false positive errors */ \
                std::ostringstream initCodeOss_; initCodeOss_ << name << "=" << GPU::Meta::StructMeta<StructType>::ToGLSLInit(value) << ";"; auto initCode = initCodeOss_.str(); \
                Builder::Builder::Get().Context()->PushTranslatedCode(initCode); \
            } \
            /* Member access functions */ \
            EASYGPU_DO_ACCESS(EASYGPU_ARG_COUNT(__VA_ARGS__), __VA_ARGS__) \
            \
            /* Assignment from another Var<StructType> */ \
            Var& operator=(const Var& other) { \
                if (this != &other) { \
                    auto store = std::make_unique<Node::StoreNode>(Load(), other.Load()); \
                    Builder::Builder::Get().Build(*store, true); \
                } \
                return *this; \
            } \
            \
            /* Assignment from CPU struct */ \
            Var& operator=(const StructType& value) { \
                /* Note: Using ostringstream instead of std::format to work around clangd bug where std::string arguments cause false positive errors */ \
                std::ostringstream initCodeOss_; initCodeOss_ << _varNode->VarName() << "=" << GPU::Meta::StructMeta<StructType>::ToGLSLInit(value) << ";"; auto initCode = initCodeOss_.str(); \
                Builder::Builder::Get().Context()->PushTranslatedCode(initCode); \
                return *this; \
            } \
            \
            /* To typed Expr - implicit conversion */ \
            operator Expr<StructType>() const { return Expr<StructType>(Load()); } \
            \
            /* Load the variable */ \
            [[nodiscard]] std::unique_ptr<Node::LoadLocalVariableNode> Load() const { \
                return std::make_unique<Node::LoadLocalVariableNode>(_varNode->VarName()); \
            } \
            \
        private: \
            Node::LocalVariableNode* _varNode = nullptr; \
            \
            friend class Expr<StructType>; \
        }; \
    } \
    \
    /* 6. Implement Expr<StructType> constructors after Var<StructType> definition */ \
    namespace GPU::IR::Value { \
        inline Expr<StructType>::Expr(const Var<StructType>& var) : ExprBase(var.Load()) {} \
        inline Expr<StructType>::Expr(Var<StructType>&& var) : ExprBase(var.Load()) {} \
    }

#endif // EASYGPU_STRUCT_META_H
