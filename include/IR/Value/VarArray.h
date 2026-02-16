#pragma once

/**
 * VarArray.h:
 *      @Descripiton    :   The array for variables
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_VARARRAY_H
#define EASYGPU_VARARRAY_H

#include <IR/Value/Var.h>
#include <IR/Value/Expr.h>
#include <IR/Node/LocalVariableArray.h>
#include <IR/Node/LoadLocalArray.h>

#include <format>
#include <array>

namespace GPU::IR::Value {
    template<ScalarType Type, int N>
    std::string ArrayToString(std::array<Type, N> Array) {
        std::ostringstream oss;
        oss << TypeShaderName<Type>() << "[](" << ValueToString<Type>(Array[0]);
        for (auto index = 1; index < N; ++index) {
            oss << "," << ValueToString<Type>(Array[index]);
        }
        oss << ")";

        return oss.str();
    }

    /**
     * Fixed size array API for users
     * @tparam Type THe scalar type supported by GPU
     * @tparam N The size of the array
     */
    template<ScalarType Type, int N>
    class VarArray {
    public:
        /**
         * Create an empty array
         */
        VarArray() {
            auto name = Builder::Builder::Get().Context()->AssignVarName();

            _node = std::make_unique<Node::LocalVariableArrayNode>(name, TypeShaderName<Type>(), N);

            Builder::Builder::Get().Build(*_node, true);
        }

        /**
         * Loading array from the CPU side
         * @param array The array to be loaded
         */
        VarArray(std::array<Type, N> array) {
            auto name = Builder::Builder::Get().Context()->AssignVarName();

            _node = std::make_unique<Node::LocalVariableArrayNode>(name, TypeShaderName<Type>(), N);

            auto uniform = std::make_unique<Node::LoadUniformNode>(ArrayToString<Type, N>(array));
            auto load    = Load();
            auto store   = std::make_unique<Node::StoreNode>(std::move(load), std::move(uniform));

            // The array definition is truly the statement
            Builder::Builder::Get().Build(*_node, true);

            // Building the store node
            Builder::Builder::Get().Build(*store, true);
        }

    public:
        template<CountableType T>
        Var<Type> operator[](T Index) {
            return Var<Type>(std::format("{}[{}]", _node->VarName(), ValueToString(Index)));
        }
        
        Var<Type> operator[](ExprBase Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return Var<Type>(std::format("{}[{}]", _node->VarName(), exprStr));
        }

        template<ScalarType IndexT>
        Var<Type> operator[](Expr<IndexT> Index) {
            std::string exprStr = Builder::Builder::Get().BuildNode(*Index.Node());
            return Var<Type>(std::format("{}[{}]", _node->VarName(), exprStr));
        }

    public:
        /**
         * Loading the variable array to the IR node
         * @return The load node of this var
         */
        [[nodiscard]] std::unique_ptr<Node::LoadLocalArrayNode> Load() const {
            return std::make_unique<Node::LoadLocalArrayNode>(_node->VarName());
        }

    private:
        std::unique_ptr<Node::LocalVariableArrayNode> _node;
    };
}

#endif //EASYGPU_VARARRAY_H
