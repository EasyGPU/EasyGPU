#pragma once

/**
 * @file Tensor.h
 * @brief Multi-dimensional Tensor abstraction on top of Buffer<T>.
 *
 * Tensor<T, Dims...> wraps a Buffer<T> and provides:
 *   - Multi-dimensional CPU indexing: W(i, j)
 *   - DSL kernel integration via Bind() → TensorRef
 *   - Batch parameter registration via ForEachParam()
 *
 * Usage:
 *   Tensor<float, 128, 64> W(data);
 *   // In kernel:
 *   auto W_ref = W.Bind();
 *   auto w = W_ref(i, j);               // Var<float>
 *   W_ref.ForEachParam([](auto& w) { AD::Param(w); });
 *
 *   // Training loop:
 *   W.Upload();  // push CPU data to GPU
 *   W.Download(); // pull GPU data to CPU
 */

#ifndef EASYGPU_NN_TENSOR_H
#define EASYGPU_NN_TENSOR_H

#include <IR/Value/BufferRef.h>
#include <IR/Value/Expr.h>
#include <IR/Value/Var.h>
#include <Runtime/Buffer.h>
#include <Utility/Helpers.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace GPU::NN {

// =============================================================================
// Stride computation (compile-time)
// =============================================================================
namespace detail {

template <size_t I, size_t... Dims>
struct StrideAt;

template <size_t I, size_t D0, size_t... Rest>
struct StrideAt<I, D0, Rest...> : StrideAt<I - 1, Rest...> {};

template <size_t D0, size_t... Rest>
struct StrideAt<0, D0, Rest...> {
	static constexpr size_t value = (1 * ... * Rest);
};

template <size_t D0>
struct StrideAt<0, D0> {
	static constexpr size_t value = 1;
};

template <size_t I>
struct StrideAt<I> {
	static constexpr size_t value = 1;
};

/** Convert an index to Expr<int> for DSL arithmetic. */
inline IR::Value::Expr<int> ToExprIdx(int v) {
	return MakeInt(v);
}

inline IR::Value::Expr<int> ToExprIdx(const IR::Value::Var<int> &v) {
	return IR::Value::Expr<int>(v);
}

inline const IR::Value::Expr<int> &ToExprIdx(const IR::Value::Expr<int> &e) {
	return e;
}

} // namespace detail

// =============================================================================
// TensorRef — DSL-side multi-dimensional buffer reference
// =============================================================================

template <typename T, size_t... Dims>
class TensorRef {
	static constexpr size_t NumDims = sizeof...(Dims);
	static constexpr size_t TotalSize = (Dims * ...);

public:
	TensorRef() = default;
	explicit TensorRef(IR::Value::BufferRef<T> ref) : ref_(std::move(ref)) {}

	/** Direct flat index access (same as BufferRef). */
	[[nodiscard]] IR::Value::Var<T> operator[](int index) const {
		return ref_[index];
	}
	[[nodiscard]] IR::Value::Var<T> operator[](const IR::Value::Var<int> &index) const {
		return ref_[index];
	}
	[[nodiscard]] IR::Value::Var<T> operator[](const IR::Value::Expr<int> &index) const {
		return ref_[index];
	}

	/**
	 * Multi-dimensional indexing.
	 *   W(i, j)    for 2D → ref_[i * stride0 + j * stride1]
	 *   W(i, j, k) for 3D → ref_[i * stride0 + j * stride1 + k * stride2]
	 *
	 * Supports int, Var<int>, and Expr<int> indices.
	 */
	template <typename... Indices>
	[[nodiscard]] IR::Value::Var<T> operator()(Indices... indices) const {
		static_assert(sizeof...(Indices) == NumDims,
					  "TensorRef::operator() argument count must match tensor dimension");
		return ref_[ComputeFlat<0>(indices...)];
	}

	/**
	 * Register all elements as AD parameters.
	 * Calls f(Var<T>) for each flat-indexed element in order.
	 * Compile-time unrolled for fixed-shape tensors.
	 */
	template <typename F>
	void ForEachParam(F &&f) {
		ForEachImpl(std::forward<F>(f), std::make_index_sequence<TotalSize>());
	}

	[[nodiscard]] const IR::Value::BufferRef<T> &GetBufferRef() const { return ref_; }
	static constexpr size_t Size() { return TotalSize; }

private:
	/** Recursive flat-index computation: idx0*stride0 + idx1*stride1 + ... */
	template <size_t Dim, typename Head, typename... Tail>
	[[nodiscard]] static IR::Value::Expr<int> ComputeFlat(Head head, Tail... tail) {
		constexpr int stride = static_cast<int>(detail::StrideAt<Dim, Dims...>::value);
		IR::Value::Expr<int> term = detail::ToExprIdx(head) * stride;
		if constexpr (sizeof...(Tail) == 0)
			return term;
		else
			return term + ComputeFlat<Dim + 1>(tail...);
	}

	/** Base case for ComputeFlat (unreachable since sizeof...(Indices) == NumDims). */
	template <size_t Dim>
	[[nodiscard]] static IR::Value::Expr<int> ComputeFlat() {
		return MakeInt(0);
	}

	template <typename F, size_t... Is>
	void ForEachImpl(F &&f, std::index_sequence<Is...>) {
		([&] { auto elem = ref_[static_cast<int>(Is)]; f(elem); }(), ...);
	}

	IR::Value::BufferRef<T> ref_;
};

// =============================================================================
// Tensor — CPU+GPU multi-dimensional array
// =============================================================================

template <typename T, size_t... Dims>
class Tensor {
	static constexpr size_t TotalSize = (Dims * ...);

public:
	/** Construct a zero-initialized tensor. */
	Tensor()
		: buffer_(TotalSize, Runtime::BufferMode::ReadWrite), data_(TotalSize, T{}) {
		buffer_.Upload(data_.data(), TotalSize);
	}

	/**
	 * Construct a tensor from existing CPU data.
	 * The data is uploaded to the GPU buffer immediately.
	 */
	explicit Tensor(const std::vector<T> &data,
					Runtime::BufferMode mode = Runtime::BufferMode::ReadWrite)
		: buffer_(TotalSize, mode), data_(data) {
		if (data.size() != TotalSize)
			throw std::invalid_argument("Tensor: data size does not match tensor shape");
		buffer_.Upload(data_.data(), TotalSize);
	}

	/** Move constructor. */
	Tensor(Tensor &&other) noexcept
		: buffer_(std::move(other.buffer_)), data_(std::move(other.data_)) {}

	/** Move assignment. */
	Tensor &operator=(Tensor &&other) noexcept {
		if (this != &other) {
			buffer_ = std::move(other.buffer_);
			data_ = std::move(other.data_);
		}
		return *this;
	}

	Tensor(const Tensor &) = delete;
	Tensor &operator=(const Tensor &) = delete;

	// ---- CPU data access ----

	T *Data() { return data_.data(); }
	const T *Data() const { return data_.data(); }

	/** Multi-dimensional CPU indexing. */
	template <typename... Indices>
	T &operator()(Indices... indices) {
		static_assert(sizeof...(Indices) == sizeof...(Dims),
					  "Tensor::operator() argument count must match tensor dimension");
		return data_[FlatIndex(indices...)];
	}

	template <typename... Indices>
	const T &operator()(Indices... indices) const {
		static_assert(sizeof...(Indices) == sizeof...(Dims),
					  "Tensor::operator() argument count must match tensor dimension");
		return data_[FlatIndex(indices...)];
	}

	static constexpr size_t Size() { return TotalSize; }

	// ---- GPU synchronization ----

	void Upload() { buffer_.Upload(data_.data(), TotalSize); }
	void Download() { buffer_.Download(data_); }

	// ---- DSL binding ----

	/** Bind this tensor for use inside a kernel lambda. */
	[[nodiscard]] TensorRef<T, Dims...> Bind() {
		return TensorRef<T, Dims...>(buffer_.Bind());
	}

	// ---- Underlying buffer access ----

	Runtime::Buffer<T> &GetBuffer() { return buffer_; }
	const Runtime::Buffer<T> &GetBuffer() const { return buffer_; }

private:
	/** Flat-index computation for CPU-side indexing (all indices are int). */
	template <typename... Indices>
	static size_t FlatIndex(Indices... indices) {
		return FlatIndexImpl<0>(static_cast<size_t>(indices)...);
	}

	template <size_t Dim, typename Head, typename... Tail>
	static size_t FlatIndexImpl(Head head, Tail... tail) {
		size_t stride = detail::StrideAt<Dim, Dims...>::value;
		if constexpr (sizeof...(Tail) == 0)
			return head * stride;
		else
			return head * stride + FlatIndexImpl<Dim + 1>(tail...);
	}

	template <size_t Dim>
	static size_t FlatIndexImpl() {
		return 0;
	}

	Runtime::Buffer<T> buffer_;
	std::vector<T> data_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_TENSOR_H
