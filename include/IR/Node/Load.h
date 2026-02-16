#pragma once

/**
 * Load.h:
 *      @Descripiton    :   The node for any address loading
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */
#ifndef EASYGPU_LOAD_H
#define EASYGPU_LOAD_H

#include <IR/Node/Node.h>

#include <memory>
#include <string>

namespace GPU::IR::Node {
    /**
     * The base class node for any address loading
     */
    class LoadNode : public Node {
    public:
        LoadNode() = default;

        ~LoadNode() override = default;

    public:
        [[nodiscard]] NodeType Type() const override;

        /**
         * Clone this node and its children
         * @return A deep copy of this node
         */
        [[nodiscard]] std::unique_ptr<Node> Clone() const override = 0;

    public:
        /**
         * Unwarping the address reference the load instruction holding
         * @return The unwarped string
         */
        virtual std::string Unwarp() const = 0;

    protected:
        size_t _variableId = 0;
    };
}

#endif //EASYGPU_LOAD_H
