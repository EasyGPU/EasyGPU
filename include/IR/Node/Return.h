#pragma once

/**
 * Return.h:
 *      @Descripiton    :   The node for return statement in functions
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_RETURN_H
#define EASYGPU_RETURN_H

#include <IR/Node/Node.h>

#include <memory>

namespace GPU::IR::Node {
    /**
     * The node for return statement
     * Used to return values from callable functions
     */
    class ReturnNode : public Node {
    public:
        /**
         * Construct a return node with a value expression
         * @param Value The expression to return
         */
        explicit ReturnNode(std::unique_ptr<Node> Value);

        ~ReturnNode() override = default;

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Get the value expression being returned
         * @return The value expression node
         */
        [[nodiscard]] const Node *Value() const;

        /**
         * Clone this node and its value
         * @return A deep copy of this node
         */
        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::unique_ptr<Node> _value;
    };
}

#endif //EASYGPU_RETURN_H
