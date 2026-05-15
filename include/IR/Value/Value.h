#pragma once

/**
 * @file Value.h
 * @brief The value class.
 */

#ifndef EASYGPU_VALUE_H
#define EASYGPU_VALUE_H

#include <IR/Node/Node.h>
#include <memory>

namespace GPU::IR::Value {
/**
 * @brief Base class for GPU IR values that owns a node pointer
 */
class Value {
public:
	Value()							= default;

	Value(const Value &)			= delete;
	Value &operator=(const Value &) = delete;

	Value(Value &&other) noexcept;
	Value &operator=(Value &&other) noexcept;

	~Value() = default;

public:
	/**
	 * @brief Release ownership of the node
	 * @return The owned node as unique_ptr
	 */
	[[nodiscard]] std::unique_ptr<Node::Node> Release() noexcept {
		return std::move(_node);
	}

protected:
	std::unique_ptr<Node::Node> _node;
};
} // namespace GPU::IR::Value

#endif // EASYGPU_VALUE_H