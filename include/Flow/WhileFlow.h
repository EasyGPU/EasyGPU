#pragma once

/**
 * @file WhileFlow.h
 * @brief The while loop control flow API for users.
 */

#ifndef EASYGPU_FLOW_WHILE_H
#define EASYGPU_FLOW_WHILE_H

#include <Flow/CodeCollectContext.h>

#include <IR/Builder/Builder.h>
#include <IR/Value/Expr.h>

#include <format>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace GPU::Flow {

/**
 * @brief While loop control flow.
 * @param condition The loop condition expression (Expr<bool>).
 * @param body The lambda containing the loop body code.
 *
 * Usage:
 *   Var<int> i = 0;
 *   While(i < 10, [&]() {
 *       // loop body
 *       i = i + 1;
 *   });
 */
inline void While(GPU::IR::Value::Expr<bool> condition, const std::function<void()> &body) {
	auto *originalContext = GPU::IR::Builder::Builder::Get().Context();
	if (!originalContext) {
		throw std::runtime_error("While() called outside of Kernel definition");
	}

	// Build condition string
	std::string		   condStr = GPU::IR::Builder::Builder::Get().BuildNode(*condition.Node());

	// Collect code for loop body
	CodeCollectContext collectContext;
	{
		ScopedCodeCollect guard(collectContext);
		body();
	}

	// Build while code
	std::string whileCode = std::format("while ({}) {{\n", condStr);
	for (const auto &line : collectContext.GetCollectedCode()) {
		whileCode += "    " + line;
	}
	whileCode += "}\n";

	originalContext->PushTranslatedCode(whileCode);
}

} // namespace GPU::Flow

#endif // EASYGPU_FLOW_WHILE_H
