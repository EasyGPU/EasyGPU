#pragma once

/**
 * SideEffectToken.h:
 *      @Descripiton    :   Token for ensuring side effects are recorded in the IR
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   3/6/2026
 */
#ifndef EASYGPU_SIDE_EFFECT_TOKEN_H
#define EASYGPU_SIDE_EFFECT_TOKEN_H

#include <IR/Value/Value.h>
#include <IR/Builder/Builder.h>

#include <memory>
#include <utility>

namespace GPU::IR::Value {
    /**
     * SideEffectToken ensures that void-returning callable side effects
     * are properly recorded in the IR even when the expression result is unused.
     * 
     * The token commits the side effect (by building the call node) upon destruction
     * unless it has been explicitly consumed (via dismiss() or bool conversion).
     * 
     * Usage:
     *   Callable<void(int&)> A = [](Int& a) { a = 20; };
     *   A(b);  // SideEffectToken created and destroyed at semicolon,
     *          // automatically committing the side effect
     */
    class SideEffectToken {
    public:
        /**
         * Construct a token that will commit the given node on destruction
         * @param node The call node to commit (ownership transferred)
         */
        explicit SideEffectToken(std::unique_ptr<Node::Node> node);
        
        /**
         * Destructor commits the side effect if not already dismissed
         */
        ~SideEffectToken();
        
        // Move operations allowed
        SideEffectToken(SideEffectToken&& other) noexcept;
        SideEffectToken& operator=(SideEffectToken&& other) noexcept;
        
        // Copy operations disabled (nodes are unique)
        SideEffectToken(const SideEffectToken&) = delete;
        SideEffectToken& operator=(const SideEffectToken&) = delete;
        
    public:
        /**
         * Dismiss the token, preventing the side effect from being committed
         * on destruction. This is called automatically when the token is
         * converted to bool.
         */
        void dismiss() const;
        
        /**
         * Convert to bool, dismissing the token.
         * This allows usage like: if (A(b)) { ... }
         * or: (void)A(b);
         * @return true
         */
        explicit operator bool() const;
        
    private:
        mutable std::unique_ptr<Node::Node> _node;
        mutable bool _dismissed;
    };
}

#endif // EASYGPU_SIDE_EFFECT_TOKEN_H
