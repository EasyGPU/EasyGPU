#pragma once

/**
 * Uniform.h:
 *      @Descripiton    :   The GPU uniform variable for compute shader
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/16/2026
 */
#ifndef EASYGPU_UNIFORM_H
#define EASYGPU_UNIFORM_H

#include <IR/Value/Var.h>
#include <IR/Builder/Builder.h>

#include <string>
#include <type_traits>
#include <format>

// Forward declaration for GLAD
#include <GLAD/glad.h>

namespace GPU::Runtime {
    // Forward declaration
    template<typename T>
    class Uniform;

    /**
     * Type trait to get GLSL type name for uniform
     */
    template<typename T>
    constexpr const char* GetUniformGLSLTypeName() {
        if constexpr (std::same_as<T, float>) return "float";
        else if constexpr (std::same_as<T, int>) return "int";
        else if constexpr (std::same_as<T, bool>) return "bool";
        else if constexpr (std::same_as<T, Math::Vec2>) return "vec2";
        else if constexpr (std::same_as<T, Math::Vec3>) return "vec3";
        else if constexpr (std::same_as<T, Math::Vec4>) return "vec4";
        else if constexpr (std::same_as<T, Math::IVec2>) return "ivec2";
        else if constexpr (std::same_as<T, Math::IVec3>) return "ivec3";
        else if constexpr (std::same_as<T, Math::IVec4>) return "ivec4";
        else if constexpr (std::same_as<T, Math::Mat2>) return "mat2";
        else if constexpr (std::same_as<T, Math::Mat3>) return "mat3";
        else if constexpr (std::same_as<T, Math::Mat4>) return "mat4";
        else if constexpr (std::same_as<T, Math::Mat2x3>) return "mat2x3";
        else if constexpr (std::same_as<T, Math::Mat2x4>) return "mat2x4";
        else if constexpr (std::same_as<T, Math::Mat3x2>) return "mat3x2";
        else if constexpr (std::same_as<T, Math::Mat3x4>) return "mat3x4";
        else if constexpr (std::same_as<T, Math::Mat4x2>) return "mat4x2";
        else if constexpr (std::same_as<T, Math::Mat4x3>) return "mat4x3";
        else if constexpr (GPU::Meta::RegisteredStruct<T>) {
            // For registered structs, use the struct type name
            return GPU::Meta::StructMeta<T>::glslTypeName;
        }
        else return "unknown";
    }

    /**
     * The uniform variable class for GPU compute shaders.
     * 
     * Usage:
     *   Uniform<int> a;
     *   a = 30;
     *   
     *   Kernel1D kernel([&]() {
     *       auto b = a.Load();  // b is Var<int>
     *       // use b...
     *   });
     *   
     *   kernel.Dispatch(16, true);  // a's value (30) is passed to GPU
     */
    template<typename T>
    class Uniform {
    public:
        /**
         * Default constructor - creates an uninitialized uniform
         */
        Uniform() = default;

        /**
         * Constructor with initial value
         * @param value The initial value
         */
        Uniform(T value) : _value(value) {}

        /**
         * Copy constructor
         */
        Uniform(const Uniform& other) : _value(other._value) {}

        /**
         * Assignment operator from value
         * @param value The value to assign
         * @return Reference to this uniform
         */
        Uniform& operator=(T value) {
            _value = value;
            return *this;
        }

        /**
         * Assignment operator from another uniform
         * @param other The other uniform to copy from
         * @return Reference to this uniform
         */
        Uniform& operator=(const Uniform& other) {
            _value = other._value;
            return *this;
        }

        /**
         * Load the uniform in a kernel context.
         * This registers the uniform with the kernel and returns a Var<T>.
         * @return Var<T> representing the uniform value in the shader
         */
        [[nodiscard]] IR::Value::Var<T> Load() {
            // Get current builder context
            auto* context = IR::Builder::Builder::Get().Context();
            if (!context) {
                throw std::runtime_error("Uniform::Load() called outside of Kernel definition");
            }

            // Create upload function for this type
            auto uploadFunc = [](uint32_t program, const std::string& name, void* ptr) {
                Uniform<T>* uniform = static_cast<Uniform<T>*>(ptr);
                GLint location = glGetUniformLocation(program, name.c_str());
                if (location == -1) {
                    // Uniform not found in shader (might be optimized out)
                    return;
                }
                
                T value = uniform->GetValue();
                if constexpr (std::same_as<T, float>) {
                    glProgramUniform1f(program, location, value);
                } else if constexpr (std::same_as<T, int>) {
                    glProgramUniform1i(program, location, value);
                } else if constexpr (std::same_as<T, bool>) {
                    glProgramUniform1i(program, location, value ? 1 : 0);
                } else if constexpr (std::same_as<T, Math::Vec2>) {
                    glProgramUniform2fv(program, location, 1, &value.x);
                } else if constexpr (std::same_as<T, Math::Vec3>) {
                    glProgramUniform3fv(program, location, 1, &value.x);
                } else if constexpr (std::same_as<T, Math::Vec4>) {
                    glProgramUniform4fv(program, location, 1, &value.x);
                } else if constexpr (std::same_as<T, Math::IVec2>) {
                    glProgramUniform2iv(program, location, 1, &value.x);
                } else if constexpr (std::same_as<T, Math::IVec3>) {
                    glProgramUniform3iv(program, location, 1, &value.x);
                } else if constexpr (std::same_as<T, Math::IVec4>) {
                    glProgramUniform4iv(program, location, 1, &value.x);
                } else if constexpr (std::same_as<T, Math::Mat2>) {
                    glProgramUniformMatrix2fv(program, location, 1, GL_FALSE, &value.m00);
                } else if constexpr (std::same_as<T, Math::Mat3>) {
                    glProgramUniformMatrix3fv(program, location, 1, GL_FALSE, &value.m00);
                } else if constexpr (std::same_as<T, Math::Mat4>) {
                    glProgramUniformMatrix4fv(program, location, 1, GL_FALSE, &value.m00);
                } else if constexpr (std::same_as<T, Math::Mat2x3>) {
                    glProgramUniformMatrix2x3fv(program, location, 1, GL_FALSE, &value.m00);
                } else if constexpr (std::same_as<T, Math::Mat2x4>) {
                    glProgramUniformMatrix2x4fv(program, location, 1, GL_FALSE, &value.m00);
                } else if constexpr (std::same_as<T, Math::Mat3x2>) {
                    glProgramUniformMatrix3x2fv(program, location, 1, GL_FALSE, &value.m00);
                } else if constexpr (std::same_as<T, Math::Mat3x4>) {
                    glProgramUniformMatrix3x4fv(program, location, 1, GL_FALSE, &value.m00);
                } else if constexpr (std::same_as<T, Math::Mat4x2>) {
                    glProgramUniformMatrix4x2fv(program, location, 1, GL_FALSE, &value.m00);
                } else if constexpr (std::same_as<T, Math::Mat4x3>) {
                    glProgramUniformMatrix4x3fv(program, location, 1, GL_FALSE, &value.m00);
                } else if constexpr (GPU::Meta::RegisteredStruct<T>) {
                    // For struct types, upload each member individually
                    // The location returned is for the struct, members are accessed with "structName.memberName"
                    GPU::Meta::StructMeta<T>::UploadUniform(program, name, value);
                }
            };

            // Register this uniform with the context
            // This will allocate a uniform name and record this uniform for dispatch
            std::string uniformName = context->RegisterUniform(
                GetUniformGLSLTypeName<T>(),
                this,
                uploadFunc
            );

            // Return a Var<T> using the string constructor with IsExternal=true
            // This creates a Var that references the uniform without declaring it in main()
            return IR::Value::Var<T>(uniformName, true);
        }

        /**
         * Get the current value
         * @return The current value
         */
        [[nodiscard]] T GetValue() const {
            return _value;
        }

        /**
         * Set the value
         * @param value The new value
         */
        void SetValue(T value) {
            _value = value;
        }

        /**
         * Implicit conversion to value type
         */
        operator T() const {
            return _value;
        }

    private:
        T _value{};
    };
}

#endif // EASYGPU_UNIFORM_H
