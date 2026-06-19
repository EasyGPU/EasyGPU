#pragma once

/**
 * @file ForFlow.h
 * @brief The for loop control flow API for users.
 */

#ifndef EASYGPU_FLOW_FOR_H
#define EASYGPU_FLOW_FOR_H

#include <IR/Value/Expr.h>
#include <IR/Value/Var.h>

#include <functional>

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
void ForImpl(GPU::IR::Value::Expr<int> &&start, GPU::IR::Value::Expr<int> &&end, GPU::IR::Value::Expr<int> &&step,
			 const std::function<void(GPU::IR::Value::Var<int> &)> &body);

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
void For(GPU::IR::Value::Expr<int> start, GPU::IR::Value::Expr<int> end, GPU::IR::Value::Expr<int> step,
		 const std::function<void(GPU::IR::Value::Var<int> &)> &body);

/**
 * @brief For loop with default step = 1.
 * @param start The loop start value.
 * @param end The loop end value (exclusive).
 * @param body The lambda receiving the loop variable as Var<int>&.
 */
void For(GPU::IR::Value::Expr<int> start, GPU::IR::Value::Expr<int> end,
		 const std::function<void(GPU::IR::Value::Var<int> &)> &body);

} // namespace GPU::Flow

#endif // EASYGPU_FLOW_FOR_H
