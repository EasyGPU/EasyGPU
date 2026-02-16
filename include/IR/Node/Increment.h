#pragma once

/**
 * Increment.h:
 *      @Descripiton    :   The node for increment/decrement
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_INCREMENT_H
#define EASYGPU_INCREMENT_H

#include <IR/Node/Node.h>

namespace GPU::IR::Node {
    /**
     * Increment/decrement node (++, --)
     */
    enum class IncrementDirection {
        Increment, // ++
        Decrement  // --
    };

    class IncrementNode : public Node {
    public:
        /**
         * Construct increment/decrement node
         * @param Direction  Increment or decrement operation
         * @param Target     Addressable storage location (must be LoadLocalVariableNode)
         * @param IsPrefix   True for prefix (++a), false for postfix (a++) - affects codegen semantics
         */
        IncrementNode(IncrementDirection    Direction,
                      std::unique_ptr<Node> Target,
                      bool                  IsPrefix = true);

        /**
         * Getting the type of the node
         * @return The type of the node
         */
        [[nodiscard]] NodeType Type() const override;

        /**
         * Getting the direction of the increment/decrement node
         * @return The direction of the increment/decrement node
         */
        [[nodiscard]] IncrementDirection Direction() const;

        /**
         * Getting the target of the node
         * @return The target of the node
         */
        [[nodiscard]] const Node *Target() const;

        /**
         * Figuring out whether this node is prefixed or not
         * @return If the value is true, it will be prefixed or not prefixed
         */
        [[nodiscard]] bool IsPrefix() const;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        IncrementDirection    _direction;
        std::unique_ptr<Node> _target;
        bool                  _isPrefix;
    };
}

#endif //EASYGPU_INCREMENT_H
