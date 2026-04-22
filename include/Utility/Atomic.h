#pragma once

/**
 * Atomic.h:
 *      @Descripiton    :   Atomic operations for GPU
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2026
 */
#ifndef EASYGPU_ATOMIC_H
#define EASYGPU_ATOMIC_H

#include <IR/Builder/Builder.h>
#include <IR/Node/AtomicOp.h>
#include <IR/Node/RawCode.h>
#include <IR/Value/Expr.h>
#include <IR/Value/Value.h>
#include <IR/Value/Var.h>
#include <Utility/Helpers.h>

namespace GPU {

namespace AtomicDetail {
// Helper to build atomic operation as raw code
inline IR::Value::Expr<int> BuildAtomicIntOp(const std::string &opName, const std::string &targetExpr,
											 const std::string &valueExpr) {
	std::string atomicCode = opName + "(" + targetExpr + ", " + valueExpr + ")";
	return IR::Value::Expr<int>(std::make_unique<IR::Node::RawCodeNode>(atomicCode));
}

inline IR::Value::Expr<float> BuildAtomicFloatOp(const std::string &opName, const std::string &targetExpr,
												 const std::string &valueExpr) {
	std::string atomicCode = opName + "(" + targetExpr + ", " + valueExpr + ")";
	return IR::Value::Expr<float>(std::make_unique<IR::Node::RawCodeNode>(atomicCode));
}
} // namespace AtomicDetail

/**
 * Atomic add operation for int
 * @param target The target memory location
 * @param value The value to add
 * @return Expr<int> The original value at the memory location before the addition
 */
[[nodiscard]] inline IR::Value::Expr<int> AtomicAdd(const IR::Value::Expr<int> &target,
													const IR::Value::Expr<int> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicIntOp("atomicAdd", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<int> AtomicAdd(const IR::Value::Expr<int> &target, int value) {
	return AtomicAdd(target, MakeInt(value));
}

/**
 * Atomic add operation for float
 */
[[nodiscard]] inline IR::Value::Expr<float> AtomicAdd(const IR::Value::Expr<float> &target,
													  const IR::Value::Expr<float> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicFloatOp("atomicAdd", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<float> AtomicAdd(const IR::Value::Expr<float> &target, float value) {
	return AtomicAdd(target, MakeFloat(value));
}

/**
 * Atomic subtract operation for int
 * Implemented via atomicAdd with negated value, as GLSL does not provide atomicSub.
 */
[[nodiscard]] inline IR::Value::Expr<int> AtomicSub(const IR::Value::Expr<int> &target,
													const IR::Value::Expr<int> &value) {
	return AtomicAdd(target, -value);
}

[[nodiscard]] inline IR::Value::Expr<int> AtomicSub(const IR::Value::Expr<int> &target, int value) {
	return AtomicSub(target, MakeInt(value));
}

/**
 * Atomic min operation for int
 */
[[nodiscard]] inline IR::Value::Expr<int> AtomicMin(const IR::Value::Expr<int> &target,
													const IR::Value::Expr<int> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicIntOp("atomicMin", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<int> AtomicMin(const IR::Value::Expr<int> &target, int value) {
	return AtomicMin(target, MakeInt(value));
}

/**
 * Atomic min operation for float
 */
[[nodiscard]] inline IR::Value::Expr<float> AtomicMin(const IR::Value::Expr<float> &target,
													  const IR::Value::Expr<float> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicFloatOp("atomicMin", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<float> AtomicMin(const IR::Value::Expr<float> &target, float value) {
	return AtomicMin(target, MakeFloat(value));
}

/**
 * Atomic max operation for int
 */
[[nodiscard]] inline IR::Value::Expr<int> AtomicMax(const IR::Value::Expr<int> &target,
													const IR::Value::Expr<int> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicIntOp("atomicMax", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<int> AtomicMax(const IR::Value::Expr<int> &target, int value) {
	return AtomicMax(target, MakeInt(value));
}

/**
 * Atomic max operation for float
 */
[[nodiscard]] inline IR::Value::Expr<float> AtomicMax(const IR::Value::Expr<float> &target,
													  const IR::Value::Expr<float> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicFloatOp("atomicMax", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<float> AtomicMax(const IR::Value::Expr<float> &target, float value) {
	return AtomicMax(target, MakeFloat(value));
}

/**
 * Atomic AND operation for int
 */
[[nodiscard]] inline IR::Value::Expr<int> AtomicAnd(const IR::Value::Expr<int> &target,
													const IR::Value::Expr<int> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicIntOp("atomicAnd", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<int> AtomicAnd(const IR::Value::Expr<int> &target, int value) {
	return AtomicAnd(target, MakeInt(value));
}

/**
 * Atomic OR operation for int
 */
[[nodiscard]] inline IR::Value::Expr<int> AtomicOr(const IR::Value::Expr<int> &target,
												   const IR::Value::Expr<int> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicIntOp("atomicOr", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<int> AtomicOr(const IR::Value::Expr<int> &target, int value) {
	return AtomicOr(target, MakeInt(value));
}

/**
 * Atomic XOR operation for int
 */
[[nodiscard]] inline IR::Value::Expr<int> AtomicXor(const IR::Value::Expr<int> &target,
													const IR::Value::Expr<int> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicIntOp("atomicXor", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<int> AtomicXor(const IR::Value::Expr<int> &target, int value) {
	return AtomicXor(target, MakeInt(value));
}

/**
 * Atomic exchange operation for int
 */
[[nodiscard]] inline IR::Value::Expr<int> AtomicExchange(const IR::Value::Expr<int> &target,
														 const IR::Value::Expr<int> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicIntOp("atomicExchange", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<int> AtomicExchange(const IR::Value::Expr<int> &target, int value) {
	return AtomicExchange(target, MakeInt(value));
}

/**
 * Atomic exchange operation for float
 */
[[nodiscard]] inline IR::Value::Expr<float> AtomicExchange(const IR::Value::Expr<float> &target,
														   const IR::Value::Expr<float> &value) {
	std::string targetStr = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string valueStr  = IR::Builder::Builder::Get().BuildNode(*value.Node());
	return AtomicDetail::BuildAtomicFloatOp("atomicExchange", targetStr, valueStr);
}

[[nodiscard]] inline IR::Value::Expr<float> AtomicExchange(const IR::Value::Expr<float> &target, float value) {
	return AtomicExchange(target, MakeFloat(value));
}

/**
 * Atomic compare-and-swap operation for int
 */
[[nodiscard]] inline IR::Value::Expr<int> AtomicCompSwap(const IR::Value::Expr<int> &target,
														 const IR::Value::Expr<int> &compare,
														 const IR::Value::Expr<int> &value) {
	std::string targetStr  = IR::Builder::Builder::Get().BuildNode(*target.Node());
	std::string compareStr = IR::Builder::Builder::Get().BuildNode(*compare.Node());
	std::string valueStr   = IR::Builder::Builder::Get().BuildNode(*value.Node());
	std::string atomicCode = "atomicCompSwap(" + targetStr + ", " + compareStr + ", " + valueStr + ")";
	return IR::Value::Expr<int>(std::make_unique<IR::Node::RawCodeNode>(atomicCode));
}

[[nodiscard]] inline IR::Value::Expr<int> AtomicCompSwap(const IR::Value::Expr<int> &target, int compare, int value) {
	return AtomicCompSwap(target, MakeInt(compare), MakeInt(value));
}

} // namespace GPU

#endif // EASYGPU_ATOMIC_H
