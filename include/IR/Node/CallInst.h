#pragma once

/**
 * CallInst.h:
 *      @Descripiton    :   The intrinsic call node
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */
#ifndef EASYGPU_CALLINST_H
#define EASYGPU_CALLINST_H

#include <IR/Node/Node.h>

#include <memory>
#include <string>
#include <vector>

namespace GPU::IR::Node {
    /**
     * The intrinsic call node
     */
    class IntrinsicCallNode : public Node {
    public:
        IntrinsicCallNode(std::string Name, std::vector<std::unique_ptr<Node>> Parameter);

        ~IntrinsicCallNode() override = default;

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Getting the name of the call
         * @return The name of the calling
         */
        [[nodiscard]] std::string_view Name() const;

        /**
         * Get the parameters of the calling
         * @return The parameters of the calling
         */
        [[nodiscard]] const std::vector<std::unique_ptr<Node>> &Parameter() const;

        /**
         * Clone this node and its arguments
         * @return A deep copy of this node
         */
        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::string                        _name;
        std::vector<std::unique_ptr<Node>> _parameter;
    };
}

#endif //EASYGPU_CALLINST_H
