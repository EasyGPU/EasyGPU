#pragma once

/**
 * @file Atomic.h
 * @brief Atomic operations for GPU.
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

// Build float atomic via CAS loop on int-aliased SSBO for universal GPU compatibility.
// Uses atomicCompSwap on int[] SSBO declared at same binding as the float[] target.
// This approach works on all GLSL 4.30+ hardware without requiring any extensions.
inline IR::Value::Expr<float> BuildAtomicFloatOp(const std::string &opName, const std::string &targetExpr,
												 const std::string &valueExpr) {
	auto bracketPos = targetExpr.find('[');
	if (bracketPos == std::string::npos) {
		// Non-SSBO target (local/shared variable): emit native atomic call
		std::string code = opName + "(" + targetExpr + ", " + valueExpr + ")";
		return IR::Value::Expr<float>(std::make_unique<IR::Node::RawCodeNode>(code));
	}

	std::string bufName = targetExpr.substr(0, bracketPos);
	std::string indexExpr = targetExpr.substr(bracketPos);

	auto *ctx = IR::Builder::Builder::Get().Context();
	if (ctx) {
		ctx->RegisterFloatAtomicBuffer(bufName);
	}

	std::string intTargetExpr = bufName + "_int" + indexExpr;

	// Build the CAS-loop new-value expression depending on the operation
	std::string casNewValExpr;
	if (opName == "atomicAdd") {
		casNewValExpr = "intBitsToFloat(_atomic_old) + (" + valueExpr + ")";
	} else if (opName == "atomicMin") {
		casNewValExpr = "min(intBitsToFloat(_atomic_old), " + valueExpr + ")";
	} else if (opName == "atomicMax") {
		casNewValExpr = "max(intBitsToFloat(_atomic_old), " + valueExpr + ")";
	} else if (opName == "atomicExchange") {
		casNewValExpr = valueExpr;
	} else {
		return IR::Value::Expr<float>(std::make_unique<IR::Node::RawCodeNode>(
			opName + "(" + targetExpr + ", " + valueExpr + ")"));
	}

	std::string code;
	code += "    {{\n";
	code += "        int _atomic_old = floatBitsToInt(" + targetExpr + ");\n";
	code += "        int _atomic_new;\n";
	code += "        while (true) {\n";
	code += "            float _atomic_f = " + casNewValExpr + ";\n";
	code += "            _atomic_new = floatBitsToInt(_atomic_f);\n";
	code += "            int _atomic_prev = atomicCompSwap(" + intTargetExpr + ", _atomic_old, _atomic_new);\n";
	code += "            if (_atomic_prev == _atomic_old) break;\n";
	code += "            _atomic_old = _atomic_prev;\n";
	code += "        }\n";
	code += "    }}";

	return IR::Value::Expr<float>(std::make_unique<IR::Node::RawCodeNode>(code));
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
