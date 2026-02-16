#pragma once

/**
 * While.h:
 *      @Descripiton    :   The node for while loop control flow
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_WHILE_H
#define EASYGPU_WHILE_H

#include <IR/Node/Node.h>

#include <vector>
#include <memory>

namespace GPU::IR::Node {
    /**
     * The node for while loop control flow
     * Structure: while (condition) { body }
     */
    class WhileNode : public Node {
    public:
        /**
         * Constructor for while node
         * @param Condition The loop condition
         * @param Body The loop body statements
         */
        WhileNode(std::unique_ptr<Node> &Condition, std::vector<std::unique_ptr<Node>> &Body);

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Getting the loop condition
         * @return The condition of the while loop
         */
        [[nodiscard]] const std::unique_ptr<Node> &Condition() const;

        /**
         * Getting the loop body
         * @return The node list of the loop body
         */
        [[nodiscard]] const std::vector<std::unique_ptr<Node>> &Body() const;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::unique_ptr<Node> _condition;
        std::vector<std::unique_ptr<Node>> _body;
    };
}

#endif //EASYGPU_WHILE_H
