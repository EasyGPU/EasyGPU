#pragma once

/**
 * ContinueNode.h:
 *      @Descripiton    :   The node for continue statement in loops
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/15/2026
 */
#ifndef EASYGPU_CONTINUE_NODE_H
#define EASYGPU_CONTINUE_NODE_H

#include <IR/Node/Node.h>

namespace GPU::IR::Node {
    /**
     * The node for continue statement
     * Used to skip to the next iteration in loops (for, while, do-while)
     */
    class ContinueNode : public Node {
    public:
        /**
         * Default constructor for continue node
         */
        ContinueNode() = default;

    public:
        [[nodiscard]] NodeType Type() const override;

        /**
         * Clone this node
         * @return A deep copy of this node
         */
        [[nodiscard]] std::unique_ptr<Node> Clone() const override;
    };
}

#endif //EASYGPU_CONTINUE_NODE_H
