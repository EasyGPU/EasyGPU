/**
 * Return.h:
 *      @Descripiton    :   The return statement API for user-defined callable functions
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_FLOW_RETURN_H
#define EASYGPU_FLOW_RETURN_H

#include <IR/Value/Expr.h>
#include <IR/Node/Return.h>
#include <IR/Builder/Builder.h>

namespace GPU::Flow {
    /**
     * Return a value from a callable function
     * This generates a GLSL return statement with the given expression
     * 
     * Example usage:
     *   Callable<float(float)> MyFunc = [](Var<float> x) {
     *       Return(x * 2.0f);
     *   };
     * 
     * @tparam T The type of the value to return
     * @param value The expression to return
     */
    template<IR::Value::ScalarType T>
    inline void Return(const IR::Value::Expr<T> &value) {
        auto returnNode = std::make_unique<IR::Node::ReturnNode>(IR::Value::CloneNode(value));
        IR::Builder::Builder::Get().Build(*returnNode, true);
    }

    /**
     * Return a value from a callable function (rvalue overload)
     * @tparam T The type of the value to return
     * @param value The expression to return
     */
    template<IR::Value::ScalarType T>
    inline void Return(IR::Value::Expr<T> &&value) {
        auto returnNode = std::make_unique<IR::Node::ReturnNode>(value.Release());
        IR::Builder::Builder::Get().Build(*returnNode, true);
    }

    /**
     * Return a literal value from a callable function
     * @tparam T The type of the value to return
     * @param value The literal value to return
     */
    template<IR::Value::ScalarType T>
    inline void Return(T value) {
        IR::Value::Expr<T> expr(value);
        auto returnNode = std::make_unique<IR::Node::ReturnNode>(expr.Release());
        IR::Builder::Builder::Get().Build(*returnNode, true);
    }

    /**
     * Return a Var value from a callable function
     * @tparam T The type of the value to return
     * @param var The variable to return
     */
    template<IR::Value::ScalarType T>
    inline void Return(const IR::Value::Var<T> &var) {
        auto returnNode = std::make_unique<IR::Node::ReturnNode>(var.Load());
        IR::Builder::Builder::Get().Build(*returnNode, true);
    }
}

#endif //EASYGPU_FLOW_RETURN_H
