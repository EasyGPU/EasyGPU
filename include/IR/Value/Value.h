/**
 * Value.h:
 *      @Descripiton    :   The value class
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */
#ifndef EASYGPU_VALUE_H
#define EASYGPU_VALUE_H

#include <IR/Node/Node.h>
#include <memory>

namespace GPU::IR::Value {
    /**
     * The value class which maintains node pointer
     */
    class Value {
    public:
        Value() = default;

        Value(const Value &) = delete;
        Value &operator=(const Value &) = delete;

        Value(Value &&other) noexcept;
        Value &operator=(Value &&other) noexcept;

        ~Value() = default;

    public:
        /**
         * Release ownership of the node
         * @return The owned node as unique_ptr
         */
        [[nodiscard]] std::unique_ptr<Node::Node> Release() noexcept {
            return std::move(_node);
        }

    protected:
        std::unique_ptr<Node::Node> _node;
    };
}

#endif //EASYGPU_VALUE_H