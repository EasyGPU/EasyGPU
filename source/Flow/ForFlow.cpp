/**
 * @file ForFlow.cpp
 * @brief Implementation of the for loop control flow API.
 */

#include <Flow/ForFlow.h>

#include <AD/GradientTape.h>
#include <Flow/CodeCollectContext.h>
#include <Flow/IfFlow.h>
#include <IR/Builder/Builder.h>

#include <format>
#include <stdexcept>
#include <string>
#include <utility>

namespace GPU::Flow {

void ForImpl(GPU::IR::Value::Expr<int> &&start, GPU::IR::Value::Expr<int> &&end, GPU::IR::Value::Expr<int> &&step,
			 const std::function<void(GPU::IR::Value::Var<int> &)> &body) {
	auto *originalContext = GPU::IR::Builder::Builder::Get().Context();
	if (!originalContext) {
		throw std::runtime_error("For() called outside of Kernel definition");
	}

	// Get variable name for loop variable and step variable
	std::string varName	 = originalContext->AssignVarName();
	std::string stepVar	 = originalContext->AssignVarName();

	// Build bound expressions from Expr nodes
	std::string startStr = GPU::IR::Builder::Builder::Get().BuildNode(*start.Node());
	std::string endStr	 = GPU::IR::Builder::Builder::Get().BuildNode(*end.Node());
	std::string stepStr	 = GPU::IR::Builder::Builder::Get().BuildNode(*step.Node());
	GPU::IR::Builder::Builder::Get().ValidateGeneratedCode(startStr, "for-loop start expression");
	GPU::IR::Builder::Builder::Get().ValidateGeneratedCode(endStr, "for-loop end expression");
	GPU::IR::Builder::Builder::Get().ValidateGeneratedCode(stepStr, "for-loop step expression");

	// Create loop variable
	GPU::IR::Value::Var<int> loopVar(varName);

	// Record for loop markers on the gradient tape
	auto					&builder = GPU::IR::Builder::Builder::Get();
	if (auto *tape = builder.GetGradientTape()) {
		tape->BeginForLoop(varName, startStr, endStr, stepStr);
	}

	// Collect code for loop body
	CodeCollectContext collectContext;
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

void For(GPU::IR::Value::Expr<int> start, GPU::IR::Value::Expr<int> end, GPU::IR::Value::Expr<int> step,
		 const std::function<void(GPU::IR::Value::Var<int> &)> &body) {
	ForImpl(std::move(start), std::move(end), std::move(step), body);
}

void For(GPU::IR::Value::Expr<int> start, GPU::IR::Value::Expr<int> end,
		 const std::function<void(GPU::IR::Value::Var<int> &)> &body) {
	ForImpl(std::move(start), std::move(end), GPU::IR::Value::Expr<int>(1), body);
}

} // namespace GPU::Flow
