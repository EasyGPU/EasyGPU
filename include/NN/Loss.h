#pragma once

/**
 * @file Loss.h
 * @brief Loss function utilities for EasyGPU AD training.
 *
 * Usage:
 *   // Inside kernel, after computing predictions:
 *   Var<float> loss = MSELoss(predBuf, targetBuf, id, outputDim);
 *   AD::Loss(loss);
 */

#ifndef EASYGPU_NN_LOSS_H
#define EASYGPU_NN_LOSS_H

#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>
#include <Utility/Helpers.h>

namespace GPU::NN {

/**
 * Mean Squared Error loss for a single sample.
 *
 *   loss = sum_i (pred[threadId * outputDim + i] - target[threadId * outputDim + i])^2
 *
 * For a single-output regression, pass outputDim=1.
 *
 * @param predBuf   Buffer containing predictions [N x outputDim]
 * @param targetBuf Buffer containing targets [N x outputDim]
 * @param threadId  Current thread index (sample index)
 * @param outputDim Number of output dimensions per sample
 * @return Var<float> sum of squared errors for this sample
 */
inline IR::Value::Var<float> MSELoss(const IR::Value::BufferRef<float> &predBuf,
									  const IR::Value::BufferRef<float> &targetBuf,
									  const IR::Value::Var<int> &threadId,
									  int outputDim) {
	if (outputDim <= 0) return MakeFloat(0.0f);

	IR::Value::Var<float> loss = MakeFloat(0.0f);
	for (int d = 0; d < outputDim; d++) {
		IR::Value::Var<float> diff = predBuf[threadId * outputDim + d] - targetBuf[threadId * outputDim + d];
		loss = loss + diff * diff;
	}
	return loss;
}

/**
 * Mean Squared Error loss for two already-computed Var<float> tensors.
 * Simple scalar version: loss = (pred - target)^2
 */
inline IR::Value::Var<float> MSELoss(const IR::Value::Var<float> &pred,
									  const IR::Value::Var<float> &target) {
	IR::Value::Var<float> diff = pred - target;
	return diff * diff;
}

/**
 * L1 loss (Mean Absolute Error): |pred - target|
 */
inline IR::Value::Var<float> L1Loss(const IR::Value::Var<float> &pred,
									 const IR::Value::Var<float> &target) {
	IR::Value::Var<float> diff = pred - target;
	return GPU::Math::Abs(diff);
}

} // namespace GPU::NN

#endif // EASYGPU_NN_LOSS_H
