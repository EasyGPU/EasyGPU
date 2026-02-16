#pragma once

/**
 * LoadLocalVariable.h:
 *      @Descripiton    :   The load node for local variable
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */
#ifndef EASYGPU_LOADLOCALVARIABLE_H
#define EASYGPU_LOADLOCALVARIABLE_H

#include <IR/Node/Load.h>

namespace GPU::IR::Node {
    /**
     * The load node for local variable
     */
    class LoadLocalVariableNode : public LoadNode {
    public:
        LoadLocalVariableNode(std::string Name);

    public:
        [[nodiscard]] std::string Unwarp() const override;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::string _name;
    };
}

#endif //EASYGPU_LOADLOCALVARIABLE_H