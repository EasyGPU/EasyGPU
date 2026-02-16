/**
 * While.h:
 *      @Descripiton    :   The while loop control flow API for users
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/15/2026
 */
#ifndef EASYGPU_FLOW_WHILE_H
#define EASYGPU_FLOW_WHILE_H

#include <Flow/CodeCollectContext.h>

#include <IR/Value/Expr.h>
#include <IR/Builder/Builder.h>

#include <format>
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace GPU::Flow {

    /**
     * While loop API
     * Usage:
     *   Var<int> i = 0;
     *   While(i < 10, [&]() {
     *       // loop body
     *       i = i + 1;
     *   });
     */
    inline void While(GPU::IR::Value::Expr<bool> condition,
                      const std::function<void()>& body) {
        auto* originalContext = GPU::IR::Builder::Builder::Get().Context();
        if (!originalContext) {
            throw std::runtime_error("While() called outside of Kernel definition");
        }

        // Build condition string
        std::string condStr = GPU::IR::Builder::Builder::Get().BuildNode(*condition.Node());

        // Collect code for loop body
        CodeCollectContext collectContext;
        {
            ScopedCodeCollect guard(collectContext);
            body();
        }

        // Build while code
        std::string whileCode = std::format("while ({}) {{\n", condStr);
        for (const auto& line : collectContext.GetCollectedCode()) {
            whileCode += "    " + line;
        }
        whileCode += "}\n";

        originalContext->PushTranslatedCode(whileCode);
    }

}

#endif //EASYGPU_FLOW_WHILE_H
