#pragma once

/**
 * For.h:
 *      @Descripiton    :   The node for for loop control flow
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_FOR_H
#define EASYGPU_FOR_H

#include <IR/Node/Node.h>

#include <vector>
#include <memory>
#include <string>

namespace GPU::IR::Node {
    /**
     * The node for for loop control flow
     * Structure: for (int varName = start; varName < end; varName += step) { body }
     */
    class ForNode : public Node {
    public:
        /**
         * Constructor for for node
         * @param VarName The loop variable name
         * @param Start The loop start value (inclusive)
         * @param End The loop end value (exclusive)
         * @param Step The loop step value
         * @param Body The loop body statements
         */
        ForNode(const std::string &VarName, int Start, int End, int Step,
                std::vector<std::unique_ptr<Node>> &Body);

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Getting the loop variable name
         * @return The name of the loop variable
         */
        [[nodiscard]] const std::string &VarName() const;

        /**
         * Getting the loop start value
         * @return The start value (inclusive)
         */
        [[nodiscard]] int Start() const;

        /**
         * Getting the loop end value
         * @return The end value (exclusive)
         */
        [[nodiscard]] int End() const;

        /**
         * Getting the loop step value
         * @return The step value
         */
        [[nodiscard]] int Step() const;

        /**
         * Getting the loop body
         * @return The node list of the loop body
         */
        [[nodiscard]] const std::vector<std::unique_ptr<Node>> &Body() const;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::string _varName;
        int _start;
        int _end;
        int _step;
        std::vector<std::unique_ptr<Node>> _body;
    };
}

#endif //EASYGPU_FOR_H
