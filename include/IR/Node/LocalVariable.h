/**
 * LocalVariable.h:
 *      @Descripiton    :   The node for the local variable definition
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */
#ifndef EASYGPU_LOCALVARIABLE_H
#define EASYGPU_LOCALVARIABLE_H

#include <IR/Node/Node.h>

#include <string>

namespace GPU::IR::Node {
    /**
     * The node for the local variable definition
     */
    class LocalVariableNode : public Node {
    public:
        LocalVariableNode(std::string Name, std::string Type);

    public:
        NodeType Type() const override;

    public:
        /**
         * Getting name of the variable
         * @return The name of the local variable
         */
        [[nodiscard]] std::string VarName() const;
        /**
         * Getting type of the variable
         * @return The type of the local variable
         */
        [[nodiscard]] std::string VarType() const;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::string _name;
        std::string _type;
    };
}

#endif //EASYGPU_LOCALVARIABLE_H