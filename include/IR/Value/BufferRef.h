#pragma once

/**
 * @file BufferRef.h
 * @brief The buffer reference for DSL access - Full IR integration.
 */

#ifndef EASYGPU_BUFFERREF_H
#define EASYGPU_BUFFERREF_H

#include <IR/Builder/Builder.h>
#include <IR/Value/Expr.h>
#include <IR/Value/Var.h>

#include <format>

// Forward declaration
namespace GPU::Runtime {
template <typename T> class Buffer;
enum class BufferMode;
} // namespace GPU::Runtime

namespace GPU::IR::Value {
// Forward declaration for element type
template <typename T> class BufferElement;

/**
 * @brief Buffer reference for DSL read/write access
 *
 * Usage:
 *   auto buf = buffer.Bind();
 *   Var<float> v = buf[id];        // Read
 *   buf[id] = value;               // Write
 *   buf[id] = buf[i] * 2.0f;       // Expression
 *
 * @tparam T The element type of the buffer
 */
template <typename T> class BufferRef {
public:
	BufferRef(std::string bufferName, uint32_t binding) : _bufferName(std::move(bufferName)), _binding(binding) {
	}

	/** @brief Get the binding index of this buffer reference */
	[[nodiscard]] uint32_t GetBinding() const {
		return _binding;
	}
	/** @brief Get the name of the underlying buffer */
	[[nodiscard]] const std::string &GetBufferName() const {
		return _bufferName;
	}

	/**
	 * @brief Array access returning a Var<T> that can be read or written
	 */
	[[nodiscard]] Var<T> operator[](const Var<int> &index) const;
	[[nodiscard]] Var<T> operator[](const Expr<int> &index) const;
	[[nodiscard]] Var<T> operator[](int index) const;

private:
	std::string _bufferName;
	uint32_t	_binding;
};

// =============================================================================
// Implementation of BufferRef::operator[]
// =============================================================================

template <typename T> [[nodiscard]] Var<T> BufferRef<T>::operator[](const Var<int> &index) const {
	return Var<T>(std::format("{}[{}]", GetBufferName(), Builder::Builder::Get().BuildNode(*index.Load().get())));
}

template <typename T> [[nodiscard]] Var<T> BufferRef<T>::operator[](const Expr<int> &index) const {
	return Var<T>(std::format("{}[{}]", GetBufferName(), Builder::Builder::Get().BuildNode(*index.Node())));
}

template <typename T> [[nodiscard]] Var<T> BufferRef<T>::operator[](int index) const {
	return Var<T>(std::format("{}[{}]", GetBufferName(), std::to_string(index)));
}

/** @brief Convenience type alias for BufferRef */
template <typename T> using buffer = BufferRef<T>;

} // namespace GPU::IR::Value

#endif // EASYGPU_BUFFERREF_H
