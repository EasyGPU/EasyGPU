#pragma once

/**
 * Uniform.h:
 *      @Descripiton    :   The GPU uniform variable for compute shader
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/16/2026
 */
#ifndef EASYGPU_UNIFORM_H
#define EASYGPU_UNIFORM_H

#include <Backend/Backend.h>
#include <IR/Builder/Builder.h>
#include <IR/Value/Var.h>
#include <Runtime/Context.h>
#include <Utility/Meta/Std430Layout.h>
#include <Utility/Unref.h>

#include <format>
#include <mutex>
#include <string>
#include <type_traits>

namespace GPU::Runtime {

// Forward declaration
template <typename T> class Uniform;

/**
 * Type trait to get GLSL type name for uniform
 */
template <typename T> constexpr const char *GetUniformGLSLTypeName() {
	if constexpr (std::same_as<T, float>)
		return "float";
	else if constexpr (std::same_as<T, int>)
		return "int";
	else if constexpr (std::same_as<T, bool>)
		return "bool";
	else if constexpr (std::same_as<T, Math::Vec2>)
		return "vec2";
	else if constexpr (std::same_as<T, Math::Vec3>)
		return "vec3";
	else if constexpr (std::same_as<T, Math::Vec4>)
		return "vec4";
	else if constexpr (std::same_as<T, Math::IVec2>)
		return "ivec2";
	else if constexpr (std::same_as<T, Math::IVec3>)
		return "ivec3";
	else if constexpr (std::same_as<T, Math::IVec4>)
		return "ivec4";
	else if constexpr (std::same_as<T, Math::Mat2>)
		return "mat2";
	else if constexpr (std::same_as<T, Math::Mat3>)
		return "mat3";
	else if constexpr (std::same_as<T, Math::Mat4>)
		return "mat4";
	else if constexpr (std::same_as<T, Math::Mat2x3>)
		return "mat2x3";
	else if constexpr (std::same_as<T, Math::Mat2x4>)
		return "mat2x4";
	else if constexpr (std::same_as<T, Math::Mat3x2>)
		return "mat3x2";
	else if constexpr (std::same_as<T, Math::Mat3x4>)
		return "mat3x4";
	else if constexpr (std::same_as<T, Math::Mat4x2>)
		return "mat4x2";
	else if constexpr (std::same_as<T, Math::Mat4x3>)
		return "mat4x3";
	else if constexpr (GPU::Meta::RegisteredStruct<T>) {
		// For registered structs, use the struct type name
		return GPU::Meta::StructMeta<T>::glslTypeName;
	} else
		return "unknown";
}

/**
 * The uniform variable class for GPU compute shaders.
 *
 * Usage:
 *   Uniform<int> a;
 *   a = 30;
 *
 *   Kernel1D kernel([&]() {
 *       auto b = a.Load();  // b is Var<int> (independent copy)
 *       // use b...
 *   });
 *
 *   kernel.Dispatch(16, true);  // a's value (30) is passed to GPU
 */
template <typename T> class Uniform {
public:
	/**
	 * Default constructor - creates an uninitialized uniform
	 */
	Uniform() = default;

	/**
	 * Constructor with initial value
	 * @param value The initial value
	 */
	Uniform(T value) : _value(value) {
	}

	/**
	 * Copy constructor
	 */
	Uniform(const Uniform &other) {
		std::lock_guard<std::mutex> lock(other._mutex);
		_value = other._value;
	}

	/**
	 * Assignment operator from value
	 * @param value The value to assign
	 * @return Reference to this uniform
	 */
	Uniform &operator=(T value) {
		std::lock_guard<std::mutex> lock(_mutex);
		_value = value;
		return *this;
	}

	/**
	 * Assignment operator from another uniform
	 * @param other The other uniform to copy from
	 * @return Reference to this uniform
	 */
	Uniform &operator=(const Uniform &other) {
		if (this != &other) {
			std::scoped_lock lock(_mutex, other._mutex);
			_value = other._value;
		}
		return *this;
	}

	/**
	 * Load the uniform in a kernel context.
	 * This registers the uniform with the kernel and returns an independent Var<T> copy.
	 *
	 * The returned Var is a copy of the uniform value, not a reference to it.
	 * This means you can safely modify the returned Var without causing
	 * "assignment to uniform" shader compilation errors.
	 *
	 * @return Var<T> representing an independent copy of the uniform value
	 */
	[[nodiscard]] IR::Value::Var<T> Load() {
		// Get the uniform reference (external variable)
		auto uniformRef = LoadRef();

		// Return an independent copy using Unref to avoid reference semantics
		// This ensures the returned Var can be safely modified
		return GPU::Utility::Unref(uniformRef);
	}

	/**
	 * Load the uniform as a reference in a kernel context.
	 *
	 * WARNING: This returns a Var that directly references the uniform.
	 * Any assignment to the returned Var will cause "assignment to uniform"
	 * shader compilation errors. Only use this when you need read-only access
	 * and want to avoid the overhead of copying.
	 *
	 * @return Var<T> referencing the uniform directly (read-only)
	 */
	[[nodiscard]] IR::Value::Var<T> LoadRef() {
		// Get current builder context
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("Uniform::LoadRef() called outside of Kernel definition");
		}

		// Create upload function for this type
		// We read the current value through the Uniform pointer at dispatch time,
		// rather than capturing a stale value at kernel definition time.
		auto uploadFunc = [](uint32_t program, const std::string &name, void *ptr) {
			T value = ptr ? static_cast<Uniform<T> *>(ptr)->GetValue() : T{};

			// For struct types, upload each member individually (struct itself has no location)
			if constexpr (GPU::Meta::RegisteredStruct<T>) {
				GPU::Meta::StructMeta<T>::UploadUniform(program, name, value);
				return;
			}

			// Get the backend and set uniform through backend interface
			// For now, we use the program handle directly as pipeline handle
			// This is a simplified approach - real implementation may need
			// to cache uniform locations or use backend-specific uniform setting
			auto *backend = Context::GetBackend();
			if (!backend) {
				return;
			}

			// Convert to void pointer for the backend
			if constexpr (std::same_as<T, float> || std::same_as<T, int> || std::same_as<T, bool>) {
				backend->SetUniform(program, name, GetUniformGLSLTypeName<T>(), &value);
			} else if constexpr (std::same_as<T, Math::Vec2> || std::same_as<T, Math::Vec3> ||
								 std::same_as<T, Math::Vec4> || std::same_as<T, Math::IVec2> ||
								 std::same_as<T, Math::IVec3> || std::same_as<T, Math::IVec4>) {
				backend->SetUniform(program, name, GetUniformGLSLTypeName<T>(), &value.x);
			} else if constexpr (std::same_as<T, Math::Mat2> || std::same_as<T, Math::Mat3> ||
								 std::same_as<T, Math::Mat4> || std::same_as<T, Math::Mat2x3> ||
								 std::same_as<T, Math::Mat2x4> || std::same_as<T, Math::Mat3x2> ||
								 std::same_as<T, Math::Mat3x4> || std::same_as<T, Math::Mat4x2> ||
								 std::same_as<T, Math::Mat4x3>) {
				backend->SetUniform(program, name, GetUniformGLSLTypeName<T>(), &value.m00);
			}
		};

		auto packFunc = [](void *dst, void *ptr) {
			T							  value = ptr ? static_cast<Uniform<T> *>(ptr)->GetValue() : T{};

			GPU::Meta::Std430Converter<T> converter;
			converter.ConvertToGPU(&value, dst, 1);
		};

		// Register this uniform with the context
		// This will allocate a uniform name and record this uniform for dispatch
		std::string uniformName =
			context->RegisterUniform(GetUniformGLSLTypeName<T>(), this, GPU::Meta::GetStd430Size<T>(),
									 GPU::Meta::GetStd430Alignment<T>(), uploadFunc, packFunc);

		// Return a Var<T> using the string constructor with IsExternal=true
		// This creates a Var that references the uniform without declaring it in main()
		return IR::Value::Var<T>(uniformName, true);
	}

	/**
	 * Get the current value
	 * @return The current value
	 */
	[[nodiscard]] T GetValue() const {
		std::lock_guard<std::mutex> lock(_mutex);
		return _value;
	}

	/**
	 * Set the value
	 * @param value The new value
	 */
	void SetValue(T value) {
		std::lock_guard<std::mutex> lock(_mutex);
		_value = value;
	}

	/**
	 * Implicit conversion to value type
	 */
	operator T() const {
		std::lock_guard<std::mutex> lock(_mutex);
		return _value;
	}

private:
	mutable std::mutex _mutex;
	T				   _value{};
};

} // namespace GPU::Runtime

#endif // EASYGPU_UNIFORM_H
