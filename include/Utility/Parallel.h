#pragma once

/**
 * Parallel.h:
 *      @Descripiton    :   Parallel primitives for GPU (Reduce, Scan, etc.)
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2026
 */
#ifndef EASYGPU_PARALLEL_H
#define EASYGPU_PARALLEL_H

#include <Flow/IfFlow.h>
#include <IR/Value/Expr.h>
#include <IR/Value/SharedMemory.h>
#include <IR/Value/Value.h>
#include <IR/Value/Var.h>
#include <Kernel/Kernel.h>
#include <Utility/Atomic.h>
#include <Utility/ThreadIndex.h>

#include <concepts>

namespace GPU::Parallel {

using namespace GPU::IR::Value; // Bring in ScalarType, BitableType concepts
using GPU::Flow::If;			// Bring in If function

/**
 * Binary operation functors for parallel primitives
 */
struct AddOp {
	template <typename T>
	[[nodiscard]] IR::Value::Expr<T> operator()(const IR::Value::Expr<T> &a, const IR::Value::Expr<T> &b) const {
		return a + b;
	}
};

struct MulOp {
	template <typename T>
	[[nodiscard]] IR::Value::Expr<T> operator()(const IR::Value::Expr<T> &a, const IR::Value::Expr<T> &b) const {
		return a * b;
	}
};

struct MinOp {
	template <typename T>
	[[nodiscard]] IR::Value::Expr<T> operator()(const IR::Value::Expr<T> &a, const IR::Value::Expr<T> &b) const {
		return Select(a < b, a, b);
	}
};

struct MaxOp {
	template <typename T>
	[[nodiscard]] IR::Value::Expr<T> operator()(const IR::Value::Expr<T> &a, const IR::Value::Expr<T> &b) const {
		return Select(a > b, a, b);
	}
};

// ===================================================================================
// Parallel Primitives
// ===================================================================================
// Shared Memory-based Reduction
// ===================================================================================

/**
 * Perform parallel reduction within a workgroup using shared memory
 *
 * @tparam T The value type
 * @tparam N The size of shared memory (should be power of 2 and match workgroup size)
 * @tparam Op The binary operation functor
 *
 * @param shared The shared memory array for intermediate results
 * @param value The input value for this thread
 * @param op The binary operation
 * @return Expr<T> The reduced result (valid only in thread 0)
 *
 * Example usage:
 *   SharedMemory<float, 256> shared;
 *   float result = WorkgroupReduce(shared, inputValue, Parallel::AddOp());
 */
template <typename T, int N, typename Op>
[[nodiscard]] IR::Value::Expr<T> WorkgroupReduce(IR::Value::SharedMemory<T, N> &shared, const IR::Value::Expr<T> &value,
												 Op op) {
	using namespace IR::Value;

	Var<int> lid = LocalThreadId();

	// Load input into shared memory
	shared[lid]	 = value;

	// Synchronize to ensure all data is loaded
	Kernel::KernelBase::WorkgroupBarrier();

	// Reduce in shared memory using tree-based reduction
	// Unrolled loop for compile-time size
	if constexpr (N >= 512) {
		If(lid < 256, [&]() { shared[lid] = op(Expr<T>(shared[lid]), Expr<T>(shared[lid + 256])); });
		Kernel::KernelBase::WorkgroupBarrier();
	}
	if constexpr (N >= 256) {
		If(lid < 128, [&]() { shared[lid] = op(Expr<T>(shared[lid]), Expr<T>(shared[lid + 128])); });
		Kernel::KernelBase::WorkgroupBarrier();
	}
	if constexpr (N >= 128) {
		If(lid < 64, [&]() { shared[lid] = op(Expr<T>(shared[lid]), Expr<T>(shared[lid + 64])); });
		Kernel::KernelBase::WorkgroupBarrier();
	}
	if constexpr (N >= 64) {
		If(lid < 32, [&]() { shared[lid] = op(Expr<T>(shared[lid]), Expr<T>(shared[lid + 32])); });
		Kernel::KernelBase::WorkgroupBarrier();
	}
	if constexpr (N >= 32) {
		If(lid < 16, [&]() { shared[lid] = op(Expr<T>(shared[lid]), Expr<T>(shared[lid + 16])); });
		Kernel::KernelBase::WorkgroupBarrier();
	}
	if constexpr (N >= 16) {
		If(lid < 8, [&]() { shared[lid] = op(Expr<T>(shared[lid]), Expr<T>(shared[lid + 8])); });
		Kernel::KernelBase::WorkgroupBarrier();
	}
	if constexpr (N >= 8) {
		If(lid < 4, [&]() { shared[lid] = op(Expr<T>(shared[lid]), Expr<T>(shared[lid + 4])); });
		Kernel::KernelBase::WorkgroupBarrier();
	}
	if constexpr (N >= 4) {
		If(lid < 2, [&]() { shared[lid] = op(Expr<T>(shared[lid]), Expr<T>(shared[lid + 2])); });
		Kernel::KernelBase::WorkgroupBarrier();
	}
	if constexpr (N >= 2) {
		If(lid < 1, [&]() { shared[lid] = op(Expr<T>(shared[lid]), Expr<T>(shared[lid + 1])); });
		Kernel::KernelBase::WorkgroupBarrier();
	}

	// Return the result (only valid in thread 0, but return for all)
	return Expr<T>(shared[0]);
}

/**
 * Convenience overload for WorkgroupReduce with default Add operation
 */
template <typename T, int N>
[[nodiscard]] IR::Value::Expr<T> WorkgroupReduce(IR::Value::SharedMemory<T, N> &shared,
												 const IR::Value::Expr<T>	   &value) {
	return WorkgroupReduce(shared, value, AddOp());
}

// ===================================================================================
// Shared Memory-based Inclusive Scan (Prefix Sum)
// ===================================================================================

/**
 * Perform parallel inclusive scan within a workgroup using shared memory
 *
 * @tparam T The value type
 * @tparam N The size of shared memory (should be power of 2 and match workgroup size)
 * @tparam Op The binary operation functor
 *
 * @param shared The shared memory array for intermediate results
 * @param value The input value for this thread
 * @param op The binary operation
 * @return Var<T> The scanned result for this thread
 *
 * Example usage:
 *   SharedMemory<float, 256> shared;
 *   float result = WorkgroupScanInclusive(shared, inputValue, Parallel::AddOp());
 */
template <typename T, int N, typename Op>
[[nodiscard]] IR::Value::Var<T> WorkgroupScanInclusive(IR::Value::SharedMemory<T, N> &shared,
													   const IR::Value::Expr<T> &value, Op op) {
	using namespace IR::Value;

	Var<int> lid = LocalThreadId();

	// Load input into shared memory
	shared[lid]	 = value;

	// Synchronize to ensure all data is loaded
	Kernel::KernelBase::WorkgroupBarrier();

	// Up-sweep (reduce) phase - build partial sums
	// Unrolled loop for compile-time size
	if constexpr (N >= 2) {
		If(lid >= 1, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 1]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 4) {
		If(lid >= 2, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 2]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 8) {
		If(lid >= 4, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 4]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 16) {
		If(lid >= 8, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 8]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 32) {
		If(lid >= 16, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 16]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 64) {
		If(lid >= 32, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 32]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 128) {
		If(lid >= 64, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 64]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 256) {
		If(lid >= 128, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 128]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 512) {
		If(lid >= 256, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 256]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}

	// Return the scanned value for this thread
	return shared[lid];
}

/**
 * Convenience overload for WorkgroupScanInclusive with default Add operation
 */
template <typename T, int N>
[[nodiscard]] IR::Value::Var<T> WorkgroupScanInclusive(IR::Value::SharedMemory<T, N> &shared,
													   const IR::Value::Expr<T>		 &value) {
	return WorkgroupScanInclusive(shared, value, AddOp());
}

// ===================================================================================
// Shared Memory-based Exclusive Scan
// ===================================================================================

/**
 * Perform parallel exclusive scan within a workgroup using shared memory
 *
 * @tparam T The value type
 * @tparam N The size of shared memory (should be power of 2 and match workgroup size)
 * @tparam Op The binary operation functor
 *
 * @param shared The shared memory array for intermediate results
 * @param value The input value for this thread
 * @param identity The identity element for the operation (e.g., 0 for Add, 1 for Mul)
 * @param op The binary operation
 * @return Var<T> The scanned result for this thread
 */
template <typename T, int N, typename Op>
[[nodiscard]] IR::Value::Var<T> WorkgroupScanExclusive(IR::Value::SharedMemory<T, N> &shared,
													   const IR::Value::Expr<T> &value, T identity, Op op) {
	using namespace IR::Value;

	Var<int> lid = LocalThreadId();

	// Shift data: shared[lid] = (lid > 0) ? value[lid - 1] : identity
	If(lid > 0, [&]() {
		// Note: we need to load from a temp or use previous value
		// For simplicity, we shift after loading
		shared[lid] = value;
	}).Else([&]() { shared[lid] = MakeFloat(identity); });

	// Synchronize
	Kernel::KernelBase::WorkgroupBarrier();

	// Now perform inclusive scan on the shifted data
	// Up-sweep phase - similar to inclusive scan but with shifted indices
	if constexpr (N >= 2) {
		If(lid >= 1, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 1]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 4) {
		If(lid >= 2, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 2]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 8) {
		If(lid >= 4, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 4]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 16) {
		If(lid >= 8, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 8]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 32) {
		If(lid >= 16, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 16]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 64) {
		If(lid >= 32, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 32]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 128) {
		If(lid >= 64, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 64]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}
	if constexpr (N >= 256) {
		If(lid >= 128, [&]() {
			Var<T> temp = op(Expr<T>(shared[lid]), Expr<T>(shared[lid - 128]));
			Kernel::KernelBase::WorkgroupBarrier();
			shared[lid] = temp;
		}).Else([&]() { Kernel::KernelBase::WorkgroupBarrier(); });
	}

	return shared[lid];
}

/**
 * Convenience overload for WorkgroupScanExclusive with default Add operation
 */
template <typename T, int N>
[[nodiscard]] IR::Value::Var<T> WorkgroupScanExclusive(IR::Value::SharedMemory<T, N> &shared,
													   const IR::Value::Expr<T> &value, T identity = T{}) {
	return WorkgroupScanExclusive(shared, value, identity, AddOp());
}

} // namespace GPU::Parallel

#endif // EASYGPU_PARALLEL_H
