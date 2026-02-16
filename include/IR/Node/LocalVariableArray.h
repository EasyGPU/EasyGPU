/**
 * LocalVariableArray.h:
 *      @Descripiton    :   The node for the local variable array definition
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_LOCALVARIABLEARRAY_H
#define EASYGPU_LOCALVARIABLEARRAY_H

#include <IR/Node/Node.h>

#include <string>

namespace GPU::IR::Node {
    /**
     * The node for the local variable array definition
     */
    class LocalVariableArrayNode : public Node {
    public:
        LocalVariableArrayNode(std::string Name, std::string Type, int Size);

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

        /**
         * Getting the size of the array
         * @return The size of the array
         */
        [[nodiscard]] int Size() const;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::string _name;
        std::string _type;
        int         _size;
    };
}

#endif //EASYGPU_LOCALVARIABLEARRAY_H
