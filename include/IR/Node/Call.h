#pragma once

/**
 * Call.h:
 *      @Descripiton    :   The node for calling user-defined callable functions
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_CALL_H
#define EASYGPU_CALL_H

#include <IR/Node/Node.h>

#include <memory>
#include <string>
#include <vector>

namespace GPU::IR::Node {
    /**
     * The node for calling user-defined callable functions
     */
    class CallNode : public Node {
    public:
        /**
         * Construct a call node
         * @param FuncName The name of the function to call
         * @param Arguments The argument expressions
         */
        CallNode(std::string FuncName, std::vector<std::unique_ptr<Node>> Arguments);

        ~CallNode() override = default;

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Get the function name
         * @return The function name
         */
        [[nodiscard]] const std::string &FuncName() const;

        /**
         * Get the arguments
         * @return The argument nodes
         */
        [[nodiscard]] const std::vector<std::unique_ptr<Node>> &Arguments() const;

        /**
         * Clone this node and its arguments
         * @return A deep copy of this node
         */
        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::string _funcName;
        std::vector<std::unique_ptr<Node>> _arguments;
    };
}

#endif //EASYGPU_CALL_H
