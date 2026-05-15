#pragma once

/**
 * @file DoWhileFlow.h
 * @brief The do-while loop control flow API for users.
 */

#ifndef EASYGPU_FLOW_DOWHILE_H
#define EASYGPU_FLOW_DOWHILE_H

#include <Flow/CodeCollectContext.h>
#include <Flow/IfFlow.h>

#include <IR/Builder/Builder.h>
#include <IR/Node/DoWhile.h>
#include <IR/Node/RawCode.h>
#include <IR/Value/Expr.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace GPU::Flow {

/**
 * @brief Do-while loop control flow (body executes at least once).
 * @param body The lambda containing the loop body code.
 * @param condition The loop condition expression (Expr<bool>).
 *
 * Usage:
 *   DoWhile([&]() {
 *       // loop body (executed at least once)
 *   }, condition);
 */
inline void DoWhile(const std::function<void()> &body, GPU::IR::Value::Expr<bool> condition) {
	auto *originalContext = GPU::IR::Builder::Builder::Get().Context();
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
	std::string condStr		= GPU::IR::Builder::Builder::Get().BuildNode(*condition.Node());

	// Build do-while code
	std::string doWhileCode = "do {\n";
	for (const auto &line : collectContext.GetCollectedCode()) {
		doWhileCode += "    " + line;
	}
	doWhileCode += "} while (" + condStr + ");\n";

	originalContext->PushTranslatedCode(doWhileCode);
}

} // namespace GPU::Flow

#endif // EASYGPU_FLOW_DOWHILE_H
