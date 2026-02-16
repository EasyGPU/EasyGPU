/**
 * If.h:
 *      @Descripiton    :   The node for if-elif-else control flow
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_IF_H
#define EASYGPU_IF_H

#include <IR/Node/Node.h>

#include <vector>
#include <memory>
#include <utility>

namespace GPU::IR::Node {
    /**
     * The node for if-elif-else control flow
     * Structure: if (condition) { do } elif (condition) { do } ... else { else }
     */
    class IfNode : public Node {
    public:
        /**
         * Constructor for if-elif-else node
         * @param Do The main if branch statements
         * @param Condition The main if condition
         * @param Elifs Vector of (condition, statements) pairs for elif branches
         * @param Else The else branch statements (can be empty)
         */
        IfNode(std::vector<std::unique_ptr<Node> > &Do, std::unique_ptr<Node> &Condition,
               std::vector<std::pair<std::unique_ptr<Node>, std::vector<std::unique_ptr<Node> > > > &Elifs,
               std::vector<std::unique_ptr<Node> > &Else);

    public:
        [[nodiscard]] NodeType Type() const override;

    public:
        /**
         * Getting the main if branch
         * @return The node list of the main if branch
         */
        [[nodiscard]] const std::vector<std::unique_ptr<Node> > &Do() const;

        /**
         * Getting the main if condition
         * @return The condition of the main if
         */
        [[nodiscard]] const std::unique_ptr<Node> &Condition() const;

        /**
         * Getting the elif branches
         * @return Vector of (condition, statements) pairs
         */
        [[nodiscard]] const std::vector<std::pair<std::unique_ptr<Node>, std::vector<std::unique_ptr<Node> > > > &
        Elifs() const;

        /**
         * Getting the else branch
         * @return The node list of the else code block, an empty container meaning no else branch
         */
        [[nodiscard]] const std::vector<std::unique_ptr<Node> > &Else() const;

        [[nodiscard]] std::unique_ptr<Node> Clone() const override;

    private:
        std::vector<std::unique_ptr<Node> > _do;
        std::unique_ptr<Node> _condition;
        std::vector<std::pair<std::unique_ptr<Node>, std::vector<std::unique_ptr<Node> > > > _elifs;
        std::vector<std::unique_ptr<Node> > _else;
    };
}

#endif //EASYGPU_IF_H
