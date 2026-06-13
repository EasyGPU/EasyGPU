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
#include <Utility/Math.h>

#include <Flow/ForFlow.h>

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
									 const IR::Value::BufferRef<float> &targetBuf, const IR::Value::Var<int> &threadId,
									 int outputDim) {
	if (outputDim <= 0)
		return MakeFloat(0.0f);

	IR::Value::Var<float> loss = MakeFloat(0.0f);
	for (int d = 0; d < outputDim; d++) {
		IR::Value::Var<float> diff	= predBuf[threadId * outputDim + d] - targetBuf[threadId * outputDim + d];
		IR::Value::Var<float> diff2 = diff * diff;
		loss						= loss + diff2;
	}
	return loss;
}

/**
 * Mean Squared Error loss for two already-computed Var<float> tensors.
 * Simple scalar version: loss = (pred - target)^2
 */
inline IR::Value::Var<float> MSELoss(const IR::Value::Var<float> &pred, const IR::Value::Var<float> &target) {
	IR::Value::Var<float> diff = pred - target;
	return diff * diff;
}

/**
 * L1 loss (Mean Absolute Error): |pred - target|
 */
inline IR::Value::Var<float> L1Loss(const IR::Value::Var<float> &pred, const IR::Value::Var<float> &target) {
	IR::Value::Var<float> diff = pred - target;
	return GPU::Math::Abs(diff);
}

/**
 * Cross-entropy loss for classification.
 *
 *   loss = -log(softmax(logits)[target])
 *
 * Uses the log-sum-exp trick for numerical stability:
 *   loss = -(logits[target] - maxLogit - log(sum(exp(logits[i] - maxLogit))))
 *
 * @param logits     Buffer containing logits [numClasses] at current offset
 * @param numClasses Number of output classes
 * @param targetId   Index of the correct class
 * @return Var<float> scalar cross-entropy loss
 */
inline IR::Value::Var<float> CrossEntropyLoss(const IR::Value::BufferRef<float> &logits, int numClasses,
											  const IR::Value::Var<int> &targetId) {
	// Stable log-softmax: find max logit first
	IR::Value::Var<float> maxLogit = MakeFloat(-1e9f);
	GPU::Flow::For(MakeInt(0), MakeInt(numClasses),
				   [&](IR::Value::Var<int> &i) { maxLogit = GPU::Math::Max(maxLogit, logits[i]); });

	// Sum of exp(logit - max)
	IR::Value::Var<float> sumExp = MakeFloat(0.0f);
	GPU::Flow::For(MakeInt(0), MakeInt(numClasses), [&](IR::Value::Var<int> &i) {
		IR::Value::Var<float> diff	 = logits[i] - maxLogit;
		IR::Value::Var<float> expVal = GPU::Math::Exp(diff);
		sumExp						 = sumExp + expVal;
	});

	// Negative log-likelihood for the target class (broken into simple ops)
	IR::Value::Var<float> diff		= logits[targetId] - maxLogit;
	IR::Value::Var<float> logSum	= GPU::Math::Log(sumExp);
	IR::Value::Var<float> lossInput = diff - logSum;
	IR::Value::Var<float> loss		= -lossInput;
	return loss;
}

/** Overload with explicit buffer offset. */
inline IR::Value::Var<float> CrossEntropyLoss(const IR::Value::BufferRef<float> &logits, int numClasses,
											  const IR::Value::Var<int> &targetId, const IR::Value::Expr<int> &offset) {
	IR::Value::Var<float> maxLogit = MakeFloat(-1e9f);
	GPU::Flow::For(MakeInt(0), MakeInt(numClasses),
				   [&](IR::Value::Var<int> &i) { maxLogit = GPU::Math::Max(maxLogit, logits[offset + i]); });
	IR::Value::Var<float> sumExp = MakeFloat(0.0f);
	GPU::Flow::For(MakeInt(0), MakeInt(numClasses), [&](IR::Value::Var<int> &i) {
		IR::Value::Var<float> diff	 = logits[offset + i] - maxLogit;
		IR::Value::Var<float> expVal = GPU::Math::Exp(diff);
		sumExp						 = sumExp + expVal;
	});
	IR::Value::Var<float> diff		= logits[offset + targetId] - maxLogit;
	IR::Value::Var<float> logSum	= GPU::Math::Log(sumExp);
	IR::Value::Var<float> lossInput = diff - logSum;
	IR::Value::Var<float> loss		= -lossInput;
	return loss;
}

} // namespace GPU::NN

#endif // EASYGPU_NN_LOSS_H
