/**
 * Break.h:
 *      @Descripiton    :   The node for break statement in loops
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_BREAK_H
#define EASYGPU_BREAK_H

#include <IR/Node/Node.h>

namespace GPU::IR::Node {
    /**
     * The node for break statement
     * Used to exit from loops (for, while, do-while)
     */
    class BreakNode : public Node {
    public:
        /**
         * Default constructor for break node
         */
        BreakNode() = default;

    public:
        [[nodiscard]] NodeType Type() const override;

        /**
         * Clone this node
         * @return A deep copy of this node
         */
        [[nodiscard]] std::unique_ptr<Node> Clone() const override;
    };
}

#endif //EASYGPU_BREAK_H
