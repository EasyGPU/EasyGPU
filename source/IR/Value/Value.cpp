/**
 * @file Value.cpp
 * @brief Implementation of base GPU value types.
 */

#include <IR/Value/Value.h>

namespace GPU::IR::Value {
Value::Value(Value &&other) noexcept : _node(std::move(other._node)) {
}

Value &Value::operator=(Value &&other) noexcept {
	if (this != &other) {
		_node = std::move(other._node);
	}
	return *this;
}
} // namespace GPU::IR::Value