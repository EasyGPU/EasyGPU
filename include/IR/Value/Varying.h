#pragma once

/**
 * @file Varying.h
 * @brief Varying<T> — interpolated variable between vertex and fragment shader stages.
 */

#ifndef EASYGPU_VARYING_H
#define EASYGPU_VARYING_H

#include <IR/Builder/Builder.h>
#include <IR/Value/Var.h>

#include <atomic>
#include <string>
#include <vector>

namespace GPU::IR::Value {

/** @brief Auto-generate a unique varying name. */
inline std::string NextVaryingName() {
	static std::atomic<int> counter{0};
	return "_v" + std::to_string(counter++);
}

/**
 * @brief Global registry for pending varying declarations.
 *
 * Varying<T> instances are constructed outside the DSL context (before any
 * kernel/pipeline is built), so they can't register directly with the
 * current BuilderContext. Instead, they register here, and the
 * GraphicsBuildContext drains this registry during code generation.
 */
struct VaryingRegistryEntry {
	std::string name;
	std::string glslType;
};

inline std::vector<VaryingRegistryEntry> &GetVaryingRegistry() {
	static std::vector<VaryingRegistryEntry> registry;
	return registry;
}

inline void RegisterPendingVarying(const std::string &name, const std::string &glslType) {
	auto &reg = GetVaryingRegistry();
	for (const auto &e : reg) {
		if (e.name == name)
			return; // Already registered
	}
	reg.push_back({name, glslType});
}

inline std::vector<VaryingRegistryEntry> DrainVaryingRegistry() {
	auto &reg	 = GetVaryingRegistry();
	auto  result = std::move(reg);
	reg.clear();
	return result;
}

/**
 * @brief A varying variable shared between vertex and fragment shader stages.
 *
 * Usage (auto-named):
 *   Varying<Vec3> vWorldPos;   // auto-named "_v0"
 *   Varying<Vec2> vUV;         // auto-named "_v1"
 *
 *   // In vertex shader:  vWorldPos = someValue;   // writes out variable
 *   // In fragment shader: Vec3 wp = vWorldPos;     // reads interpolated value
 *
 * The name is assigned automatically. An explicit-name constructor is
 * available for debugging but is not required.
 *
 * @tparam T The GLSL scalar type (float, int, Vec2, Vec3, Vec4, etc.)
 */
template <ScalarType T> class Varying {
public:
	/** @brief Construct a varying with an auto-generated unique name. */
	Varying() : _var(NextVaryingName(), true) {
		RegisterPendingVarying(_var.VarName(), TypeShaderName<T>());
	}

	/**
	 * @brief Construct a varying with an explicit GLSL variable name.
	 *
	 * Useful for debugging — the name appears in generated GLSL.
	 */
	explicit Varying(const std::string &name) : _var(name, true) {
		RegisterPendingVarying(name, TypeShaderName<T>());
	}

	Varying(const Varying &)			= delete;
	Varying &operator=(const Varying &) = delete;
	Varying(Varying &&)					= delete;
	Varying &operator=(Varying &&)		= delete;

			 operator Var<T> &() {
		return _var;
	}
	operator const Var<T> &() const {
		return _var;
	}

	/** @brief Assignment from Var<T> — generates varying write in VS. */
	Varying &operator=(const Var<T> &rhs) {
		_var = rhs;
		return *this;
	}

	std::string Name() const {
		return _var.VarName();
	}
	Var<T> GetVar() const {
		return _var;
	}

private:
	Var<T> _var;
};

} // namespace GPU::IR::Value

#endif
