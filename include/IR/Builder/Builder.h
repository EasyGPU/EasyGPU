/**
 * Builder.h:
 *      @Descripiton    :   The builder for the DSL
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */
#ifndef EASYGPU_BUILDER_H
#define EASYGPU_BUILDER_H

#include <IR/Builder/BuilderContext.h>

#include <IR/Node/Node.h>
#include <IR/Node/CallInst.h>
#include <IR/Node/Operation.h>
#include <IR/Node/LocalVariable.h>
#include <IR/Node/LocalVariableArray.h>
#include <IR/Node/CompoundAssignment.h>
#include <IR/Node/Increment.h>
#include <IR/Node/Load.h>
#include <IR/Node/If.h>
#include <IR/Node/While.h>
#include <IR/Node/DoWhile.h>
#include <IR/Node/For.h>
#include <IR/Node/Break.h>
#include <IR/Node/Continue.h>
#include <IR/Node/Return.h>
#include <IR/Node/Call.h>
#include <IR/Node/RawCode.h>
#include <IR/Node/Store.h>
#include <IR/Node/ArrayAccess.h>
#include <IR/Node/MemberAccess.h>

namespace GPU::IR::Builder {
    /**
     * The builder for the DSL, mainly takes charge of the node translating.
     * The builder obeys the singleton pattern.
     */
    class Builder {
    public:
        /**
         * Getting the global builder for kernel function to bind
         * @return The global builder for kernel function to bind
         */
        static Builder &Get();

    public:
        /**
         * Binding the builder to a builder context
         * @param Context The context to be bound
         */
        void Bind(BuilderContext &Context);

        /**
         * Unbinding the builder from current context
         * This is called when Kernel construction completes to release the context.
         */
        void Unbind();

        /**
         * Getting the context the builder now binding
         * @return The context the builder now is binding
         */
        BuilderContext *Context();

    public:
        /**
         * Building a node and pushing it to the code string stream
         * @param Node The node to be built
         * @param IsStatement Whether this node is a statement or a expression
         */
        void Build(const Node::Node &Node, bool IsStatement);

    public:
        /**
         * Building a node to the code string
         * @param Node The node to be built
         * @return The built string, if the node is invalid, it will return an empty string
         */
        std::string BuildNode(const Node::Node &Node);

        /**
         * Building an intrinsic call node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildCallInst(const Node::IntrinsicCallNode &Node);

        /**
         * Building an operation node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildOperation(const Node::OperationNode &Node);

        /**
         * Building a local variable node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildLocalVariable(const Node::LocalVariableNode &Node);

        /**
         * Building a local variable array node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildLocalVariableArray(const Node::LocalVariableArrayNode &Node);

        /**
         * Building a load node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildLoad(const Node::LoadNode &Node);

        /**
         * Building a store node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildStore(const Node::StoreNode &Node);

        /**
         * Building a array access node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildArrayAccess(const Node::ArrayAccessNode &Node);

        /**
         * Building a compound assigment node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildCompoundAssignment(const Node::CompoundAssignmentNode &Node);

        /**
         * Building an increment assigment node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildIncrement(const Node::IncrementNode &Node);

        /**
         * Building a member access node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildMemberAccess(const Node::MemberAccessNode &Node);

        std::string BuildIf(const Node::IfNode &Node);

        /**
         * Building a while node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildWhile(const Node::WhileNode &Node);

        /**
         * Building a do-while node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildDoWhile(const Node::DoWhileNode &Node);

        /**
         * Building a for node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildFor(const Node::ForNode &Node);

        /**
         * Building a break node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildBreak(const Node::BreakNode &Node);

        /**
         * Building a continue node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildContinue(const Node::ContinueNode &Node);

        /**
         * Building a return node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildReturn(const Node::ReturnNode &Node);

        /**
         * Building a call node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildCall(const Node::CallNode &Node);

        /**
         * Building a raw code node
         * @param Node The node to be built
         * @return The built string
         */
        std::string BuildRawCode(const Node::RawCodeNode &Node);

    private:
        Builder() = default;

    private:
        BuilderContext *_context = nullptr;
    };
}

#endif //EASYGPU_BUILDER_H
