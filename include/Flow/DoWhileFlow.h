#pragma once

/**
 * DoWhile.h:
 *      @Descripiton    :   The do-while loop control flow API for users
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_FLOW_DOWHILE_H
#define EASYGPU_FLOW_DOWHILE_H

#include <Flow/CodeCollectContext.h>
#include <Flow/IfFlow.h>

#include <IR/Node/DoWhile.h>
#include <IR/Node/RawCode.h>
#include <IR/Value/Expr.h>
#include <IR/Builder/Builder.h>

#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace GPU::Flow {

    /**
     * Do-while loop API
     * Usage:
     *   DoWhile([&]() {
     *       // loop body (executed at least once)
     *   }, condition);
     */
    inline void DoWhile(const std::function<void()>& body, 
                        GPU::IR::Value::Expr<bool> condition) {
        auto* originalContext = GPU::IR::Builder::Builder::Get().Context();
        if (!originalContext) {
            throw std::runtime_error("DoWhile() called outside of Kernel definition");
        }

        // Collect code for loop body
        CodeCollectContext collectContext;
        {
            ScopedCodeCollect guard(collectContext);
            body();
        }

        // Build condition string
        std::string condStr = GPU::IR::Builder::Builder::Get().BuildNode(*condition.Node());

        // Build do-while code
        std::string doWhileCode = "do {\n";
        for (const auto& line : collectContext.GetCollectedCode()) {
            doWhileCode += "    " + line;
        }
        doWhileCode += std::format("}} while ({});\n", condStr);

        originalContext->PushTranslatedCode(doWhileCode);
    }

}

#endif //EASYGPU_FLOW_DOWHILE_H
