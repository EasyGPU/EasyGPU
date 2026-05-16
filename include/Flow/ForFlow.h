#pragma once

/**
 * @file ForFlow.h
 * @brief The for loop control flow API for users.
 */

#ifndef EASYGPU_FLOW_FOR_H
#define EASYGPU_FLOW_FOR_H

#include <AD/GradientTape.h>

#include <Flow/CodeCollectContext.h>
#include <Flow/IfFlow.h>

#include <IR/Builder/Builder.h>
#include <IR/Node/For.h>
#include <IR/Node/LoadLocalVariable.h>
#include <IR/Node/LocalVariable.h>
#include <IR/Node/RawCode.h>
#include <IR/Value/Expr.h>
#include <IR/Value/Var.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace GPU::Flow {

/**
 * @brief Internal implementation of for loop.
 *
 * Takes Expr<int> for all bounds (Var<int> and int convert to Expr<int> implicitly).
 * @param start The loop start value.
 * @param end The loop end value (exclusive).
 * @param step The iteration step value.
 * @param body The lambda receiving the loop variable as Var<int>&.
 */
inline void ForImpl(GPU::IR::Value::Expr<int> &&start, GPU::IR::Value::Expr<int> &&end,
					GPU::IR::Value::Expr<int> &&step, const std::function<void(GPU::IR::Value::Var<int> &)> &body) {
	auto *originalContext = GPU::IR::Builder::Builder::Get().Context();
	if (!originalContext) {
		throw std::runtime_error("For() called outside of Kernel definition");
	}

	// Get variable name for loop variable and step variable
	std::string				 varName  = originalContext->AssignVarName();
	std::string				 stepVar  = originalContext->AssignVarName();

	// Build bound expressions from Expr nodes
	std::string				 startStr = GPU::IR::Builder::Builder::Get().BuildNode(*start.Node());
	std::string				 endStr	  = GPU::IR::Builder::Builder::Get().BuildNode(*end.Node());
	std::string				 stepStr  = GPU::IR::Builder::Builder::Get().BuildNode(*step.Node());

	// Create loop variable
	GPU::IR::Value::Var<int> loopVar(varName);

	// Record for loop markers on the gradient tape
	auto &builder = GPU::IR::Builder::Builder::Get();
	if (auto *tape = builder.GetGradientTape()) {
		tape->BeginForLoop(varName, startStr, endStr, stepStr);
	}

	// Collect code for loop body
	CodeCollectContext		 collectContext;
	{
		ScopedCodeCollect guard(collectContext);
		body(loopVar);
	}

	if (auto *tape = builder.GetGradientTape()) {
		tape->EndForLoop();
	}

	// Build for loop code with support for both positive and negative step values.
	// The condition (s > 0 && v < end) || (s < 0 && v > end) correctly handles
	// positive step (forward) and negative step (backward), and safely skips
	// the loop when step is zero.
	std::string forCode =
		std::format("for (int {} = {}, {} = {}; ({} > 0 && {} < {}) || ({} < 0 && {} > {}); {} += {}) {{\n", varName,
					startStr, stepVar, stepStr, stepVar, varName, endStr, stepVar, varName, endStr, varName, stepVar);
	for (const auto &line : collectContext.GetCollectedCode()) {
		forCode += "    " + line;
	}
	forCode += "}\n";

	originalContext->PushTranslatedCode(forCode);
}

/**
 * @brief For loop with explicit step value.
 *
 * Accepts: int, Var<int>, or Expr<int> for all parameters.
 * Var<int> implicitly converts to Expr<int>; int constructs Expr<int> implicitly.
 * @param start The loop start value.
 * @param end The loop end value (exclusive).
 * @param step The iteration step value (positive or negative).
 * @param body The lambda receiving the loop variable as Var<int>&.
 */
inline void For(GPU::IR::Value::Expr<int> start, GPU::IR::Value::Expr<int> end, GPU::IR::Value::Expr<int> step,
				const std::function<void(GPU::IR::Value::Var<int> &)> &body) {
	ForImpl(std::move(start), std::move(end), std::move(step), body);
}

/**
 * @brief For loop with default step = 1.
 * @param start The loop start value.
 * @param end The loop end value (exclusive).
 * @param body The lambda receiving the loop variable as Var<int>&.
 */
inline void For(GPU::IR::Value::Expr<int> start, GPU::IR::Value::Expr<int> end,
				const std::function<void(GPU::IR::Value::Var<int> &)> &body) {
	ForImpl(std::move(start), std::move(end), GPU::IR::Value::Expr<int>(1), body);
}

} // namespace GPU::Flow

#endif // EASYGPU_FLOW_FOR_H
