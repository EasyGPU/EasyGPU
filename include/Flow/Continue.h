/**
 * Continue.h:
 *      @Descripiton    :   The continue control flow API for users
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/15/2026
 */
#ifndef EASYGPU_FLOW_CONTINUE_H
#define EASYGPU_FLOW_CONTINUE_H

#include <IR/Node/Continue.h>
#include <IR/Builder/Builder.h>

namespace GPU::Flow {
    /**
     * Emit a continue statement to skip to the next iteration of the current loop
     * 
     * Usage:
     *   For(0, 10, [&](Var<int>& i) {
     *       If(i % 2 == 0, [&]() {
     *           Continue();  // Skip even numbers
     *       });
     *       // Process odd numbers only
     *   });
     */
    inline void Continue() {
        auto continueNode = std::make_unique<IR::Node::ContinueNode>();
        IR::Builder::Builder::Get().Build(*continueNode, true);
    }
}

#endif //EASYGPU_FLOW_CONTINUE_H
