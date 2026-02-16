#pragma once

/**
 * LoadLocalArray.h:
 *      @Descripiton    :   The load node for local array
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */
#ifndef EASYGPU_LOADLOCALARRAY_H
#define EASYGPU_LOADLOCALARRAY_H

#include <IR/Node/Load.h>

namespace GPU::IR::Node {
    /**
     * The load node for local array
     */
    class LoadLocalArrayNode : public LoadNode {
    public:
        LoadLocalArrayNode(std::string Name);

    public:
        [[nodiscard]] std::string Unwarp() const override;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::string _name;
    };
}

#endif //EASYGPU_LOADLOCALARRAY_H