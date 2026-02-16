#pragma once

/**
 * ArrayAccess.h:
 *      @Descripiton    :   The node for array access
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_ARRAYACCESS_H
#define EASYGPU_ARRAYACCESS_H

#include <IR/Node/Node.h>

namespace GPU::IR::Node {
    /**
     * The node for array access
     */
    class ArrayAccessNode : public Node {
    public:
        ArrayAccessNode(std::unique_ptr<Node> Target, std::unique_ptr<Node> Index);

        ~ArrayAccessNode() override = default;

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Getting the target node of the node
         * @return The target node of the node
         */
        [[nodiscard]] const Node * Target() const;

        /**
         * Getting the index node of the node
         * @return The index node of the node
         */
        [[nodiscard]] const Node * Index() const;

        /**
         * Clone this node and its children
         * @return A deep copy of this node
         */
        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::unique_ptr<Node> _target;
        std::unique_ptr<Node> _index;
    };
}

#endif //EASYGPU_ARRAYACCESS_H