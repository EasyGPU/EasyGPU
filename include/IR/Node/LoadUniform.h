/**
 * LoadUniform.h:
 *      @Descripiton    :   The uniform load node
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_LOADUNIFORM_H
#define EASYGPU_LOADUNIFORM_H

#include <IR/Node/Load.h>

namespace GPU::IR::Node {
    /**
     * The uniform node is the node to load a constant which may be captured by API from C++ side
     */
    class LoadUniformNode : public LoadNode {
    public:
        LoadUniformNode(std::string Uniform);

    public:
        [[nodiscard]] std::string Unwarp() const override;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::string _uniform;
    };
}

#endif //EASYGPU_LOADUNIFORM_H