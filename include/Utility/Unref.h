#pragma once

/**
 * Unref.h:
 *      @Description   :   Helper function to create an independent copy of a GPU variable
 *      @Author        :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date          :   3/7/2026
 *
 * This function solves the "reference semantics" issue when initializing a Var from
 * buffer elements or other variables. Due to move constructor optimizations, direct
 * initialization like `Int val = buf[i]` may cause `val` to become an alias to `buf[i]`.
 *
 * Usage:
 *   Var<GameObject> obj = Unref(data[id]);  // Creates independent copy
 */

#include <IR/Value/Var.h>
#include <type_traits>

namespace GPU::Utility {

namespace Detail {
/**
 * Internal helper: create a true local copy from any Var, even if the source
 * is an external reference (uniform, buffer element, function parameter, etc.).
 * The copy constructor short-circuits for external variables (shares the name),
 * so we must explicitly create a new local variable and emit IR load/store.
 */
template <typename T> [[nodiscard]] inline GPU::IR::Value::Var<T> DeepCopyVar(const GPU::IR::Value::Var<T> &var) {
	GPU::IR::Value::Var<T> result;
	result = var; // VarBase::operator= always emits load/store
	return result;
}
} // namespace Detail

/**
 * Create an independent copy of a GPU variable.
 *
 * This function forces a copy instead of move semantics or reference sharing,
 * ensuring the result is a new independent variable with its own storage,
 * rather than a reference to the original variable.
 *
 * @tparam T The scalar type of the variable
 * @param var The source variable (typically from buffer access like buf[i] or Uniform::Load())
 * @return A new independent Var<T> with copied value
 *
 * Example:
 *   // Without Unref - may create reference to buf[i]
 *   Int val = buf[i];
 *   val = 5;  // May unexpectedly modify buf[i]!
 *
 *   // With Unref - creates independent copy
 *   Int val = Unref(buf[i]);
 *   val = 5;  // Only modifies val, NOT buf[i]
 */
template <typename T> [[nodiscard]] inline GPU::IR::Value::Var<T> Unref(const GPU::IR::Value::Var<T> &var) {
	if (var.IsExternal()) {
		return Detail::DeepCopyVar(var);
	}
	return GPU::IR::Value::Var<T>(var);
}

/**
 * Overload for rvalue references - same effect, forces copy instead of move
 */
template <typename T> [[nodiscard]] inline GPU::IR::Value::Var<T> Unref(GPU::IR::Value::Var<T> &&var) {
	if (var.IsExternal()) {
		return Detail::DeepCopyVar(var);
	}
	return GPU::IR::Value::Var<T>(var);
}

} // namespace GPU::Utility
