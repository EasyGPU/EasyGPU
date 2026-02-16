#pragma once

/**
 * RawCode.h:
 *      @Descripiton    :   The node for raw GLSL code (used in control flow)
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_RAWCODE_H
#define EASYGPU_RAWCODE_H

#include <IR/Node/Node.h>

#include <string>

namespace GPU::IR::Node {
    /**
     * The node for storing raw GLSL code
     * Used when collecting code from lambda for control flow statements
     */
    class RawCodeNode : public Node {
    public:
        /**
         * Constructor from raw GLSL code string
         * @param Code The GLSL code string
         */
        explicit RawCodeNode(std::string Code);

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Getting the raw code
         * @return The stored GLSL code
         */
        [[nodiscard]] const std::string &Code() const;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::string _code;
    };
}

#endif //EASYGPU_RAWCODE_H
