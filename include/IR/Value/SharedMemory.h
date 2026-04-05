#pragma once

/**
 * SharedMemory.h:
 *      @Descripiton    :   The shared memory API for users
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2026
 */
#ifndef EASYGPU_SHAREDMEMORY_VALUE_H
#define EASYGPU_SHAREDMEMORY_VALUE_H

#include <IR/Node/SharedMemory.h>
#include <IR/Value/Expr.h>
#include <IR/Value/Var.h>

#include <format>

namespace GPU::IR::Value {
/**
 * Shared memory array API for users
 * @tparam Type The scalar type supported by GPU
 * @tparam N The size of the shared memory array
 *
 * Example usage:
 *   SharedMemory<float, 256> shared;
 *   shared[localId] = value;
 *   Kernel1D::WorkgroupBarrier();
 *   float val = shared[otherId];
 */
template <ScalarType Type, int N> class SharedMemory {
public:
	/**
	 * Create a shared memory array
	 * This declares a workgroup-local array that is shared among all threads in a workgroup.
	 * Maps to GLSL: shared Type Name[N];
	 */
	SharedMemory() {
		auto name = Builder::Builder::Get().ContextChecked()->AssignVarName();

		_node	  = std::make_unique<Node::SharedMemoryNode>(name, TypeShaderName<Type>(), N);

		// Declare the shared memory at global scope (outside main)
		Builder::Builder::Get().ContextChecked()->PushSharedMemoryDeclaration(
			std::format("shared {} {}[{}];", TypeShaderName<Type>(), name, N));
	}

public:
	/**
	 * Access shared memory element by index
	 * @param Index The index (can be Var<int>, Expr<int>, or int)
	 * @return Var<Type> that references the shared memory location
	 */
	template <CountableType T> Var<Type> operator[](T Index) {
		return Var<Type>(std::format("{}[{}]", _node->VarName(), ValueToString(Index)));
	}

	Var<Type> operator[](ExprBase Index) {
		std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
		return Var<Type>(std::format("{}[{}]", _node->VarName(), exprStr));
	}

	template <ScalarType IndexT> Var<Type> operator[](Expr<IndexT> Index) {
		std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
		return Var<Type>(std::format("{}[{}]", _node->VarName(), exprStr));
	}

public:
	/**
	 * Get the name of the shared memory array
	 * @return The variable name
	 */
	[[nodiscard]] std::string GetName() const {
		return _node->VarName();
	}

	/**
	 * Get the size of the shared memory array
	 * @return The array size
	 */
	[[nodiscard]] static constexpr int GetSize() {
		return N;
	}

private:
	std::unique_ptr<Node::SharedMemoryNode> _node;
};
} // namespace GPU::IR::Value

#endif // EASYGPU_SHAREDMEMORY_VALUE_H
