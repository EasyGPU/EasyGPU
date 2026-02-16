/**
 * DoWhile.h:
 *      @Descripiton    :   The node for do-while loop control flow
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_DOWHILE_H
#define EASYGPU_DOWHILE_H

#include <IR/Node/Node.h>

#include <vector>
#include <memory>

namespace GPU::IR::Node {
    /**
     * The node for do-while loop control flow
     * Structure: do { body } while (condition);
     */
    class DoWhileNode : public Node {
    public:
        /**
         * Constructor for do-while node
         * @param Body The loop body statements
         * @param Condition The loop condition
         */
        DoWhileNode(std::vector<std::unique_ptr<Node>> &Body, std::unique_ptr<Node> &Condition);

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Getting the loop body
         * @return The node list of the loop body
         */
        const std::vector<std::unique_ptr<Node>> &Body() const;

        /**
         * Getting the loop condition
         * @return The condition of the do-while loop
         */
        const std::unique_ptr<Node> &Condition() const;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::vector<std::unique_ptr<Node>> _body;
        std::unique_ptr<Node> _condition;
    };
}

#endif //EASYGPU_DOWHILE_H
