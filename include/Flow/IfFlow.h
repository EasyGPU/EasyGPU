#pragma once

/**
 * If.h:
 *      @Descripiton    :   The if-elif-else control flow API for users
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_FLOW_IF_H
#define EASYGPU_FLOW_IF_H

#include <Flow/CodeCollectContext.h>

#include <IR/Node/If.h>
#include <IR/Node/RawCode.h>
#include <IR/Value/Expr.h>
#include <IR/Builder/Builder.h>

#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <utility>

namespace GPU::Flow {
    // Forward declaration
    class IfChain;

    /**
     * RAII guard for temporarily switching to a code collection context
     */
    class ScopedCodeCollect {
    public:
        /**
         * Constructor - switches to collection context
         * @param collectContext The context to collect code into
         */
        explicit ScopedCodeCollect(CodeCollectContext& collectContext);

        /**
         * Destructor - restores original context
         */
        ~ScopedCodeCollect();

        // Disable copy
        ScopedCodeCollect(const ScopedCodeCollect&) = delete;
        ScopedCodeCollect& operator=(const ScopedCodeCollect&) = delete;

    private:
        CodeCollectContext& _collectContext;
        IR::Builder::BuilderContext* _originalContext;
    };

    /**
     * Convert collected code lines to IR nodes
     * @param codeLines Vector of collected GLSL code lines
     * @return Vector of RawCodeNode unique_ptrs
     */
    std::vector<std::unique_ptr<IR::Node::Node>> CollectedCodeToNodes(const std::vector<std::string>& codeLines);

    /**
     * The If chain class for building elif-else chains
     */
    class IfChain {
    public:
        /**
         * Internal constructor used by If()
         * @param condition The if condition node (ownership transferred)
         * @param ifCode Collected code for if branch
         * @param originalContext The original builder context to emit code to
         */
        IfChain(std::unique_ptr<IR::Node::Node> condition,
                std::vector<std::string> ifCode,
                IR::Builder::BuilderContext* originalContext);

        /**
         * Add an elif branch
         * @param condition The elif condition expression
         * @param body The lambda containing the body code
         * @return Reference to this for chaining
         */
        IfChain& Elif(IR::Value::Expr<bool>&& condition, const std::function<void()>& body);

        /**
         * Add an else branch (terminal)
         * @param body The lambda containing the body code
         */
        void Else(const std::function<void()>& body);

        /**
         * Destructor - emits the if statement if not already done
         */
        ~IfChain();

        // Disable copy, allow move
        IfChain(const IfChain&) = delete;
        IfChain& operator=(const IfChain&) = delete;
        IfChain(IfChain&& other) noexcept;
        IfChain& operator=(IfChain&& other) noexcept;

    private:
        /**
         * Emit the complete if statement to the original context
         */
        void EmitIfStatement();

        /**
         * Build the GLSL if statement string (for direct emission)
         * @return The complete GLSL if statement
         */
        std::string BuildGLSLIf();

    private:
        std::vector<std::unique_ptr<IR::Node::Node>> _conditions;
        std::vector<std::vector<std::string>> _codeBlocks;
        std::vector<std::string> _elseCode;
        IR::Builder::BuilderContext* _originalContext;
        bool _emitted;
    };

    /**
     * Start an if statement chain
     * @param condition The if condition expression
     * @param body The lambda containing the if body code
     * @return IfChain for continuing with elif/else
     *
     * Usage:
     *   If(cond1, [&]() { ... })
     *       .Elif(cond2, [&]() { ... })
     *       .Elif(cond3, [&]() { ... })
     *       .Else([&]() { ... });
     */
    IfChain If(IR::Value::Expr<bool> condition, const std::function<void()>& body);
}

#endif //EASYGPU_FLOW_IF_H
