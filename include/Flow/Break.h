/**
 * Break.h:
 *      @Descripiton    :   The break control flow API for users
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_FLOW_BREAK_H
#define EASYGPU_FLOW_BREAK_H

#include <IR/Node/Break.h>
#include <IR/Builder/Builder.h>

namespace GPU::Flow {
    /**
     * Emit a break statement to exit from the current loop
     * 
     * Usage:
     *   For(0, 10, [&](Var<int>& i) {
     *       If(i == 5, [&]() {
     *           Break();  // Exit the loop when i equals 5
     *       });
     *   });
     */
    inline void Break() {
        auto breakNode = std::make_unique<IR::Node::BreakNode>();
        IR::Builder::Builder::Get().Build(*breakNode, true);
    }
}

#endif //EASYGPU_FLOW_BREAK_H
