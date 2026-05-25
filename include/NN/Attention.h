#pragma once

/**
 * @file Attention.h
 * @brief Causal multi-head self-attention for EasyGPU GPT-style transformers.
 *
 *   CausalSelfAttention<T, EmbedDim, NumHeads>
 *
 * Uses three-pass softmax (max reduction, exp-sum, weighted sum).
 * No biases (GPT-2 style). K and V are cached in dedicated buffers.
 */

#ifndef EASYGPU_NN_ATTENTION_H
#define EASYGPU_NN_ATTENTION_H

#include <AD/ADKernel.h>
#include <NN/Tensor.h>

#include <Flow/ForFlow.h>
#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>
#include <Utility/Helpers.h>
#include <Utility/Math.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace GPU::NN {

template <typename T, size_t EmbedDim, size_t NumHeads>
class CausalSelfAttention {
	static_assert(std::is_same_v<T, float>, "CausalSelfAttention only supports float");
	static_assert(EmbedDim % NumHeads == 0, "EmbedDim must be divisible by NumHeads");
	static constexpr size_t HeadDim = EmbedDim / NumHeads;

public:
	CausalSelfAttention(unsigned initSeed = 42) {
		std::vector<T> wQ(EmbedDim * EmbedDim);
		std::vector<T> wK(EmbedDim * EmbedDim);
		std::vector<T> wV(EmbedDim * EmbedDim);
		std::vector<T> wO(EmbedDim * EmbedDim);

		unsigned sQ = initSeed, sK = initSeed + 1, sV = initSeed + 2, sO = initSeed + 3;
		float range = std::sqrt(6.0f / static_cast<float>(2 * EmbedDim));

		for (size_t j = 0; j < EmbedDim; j++) {
			for (size_t i = 0; i < EmbedDim; i++) {
				size_t idx = j * EmbedDim + i;
				sQ = sQ * 1664525u + 1013904223u;
				wQ[idx] = (static_cast<float>(sQ) / UINT32_MAX * 2.0f - 1.0f) * range;
				sK = sK * 1664525u + 1013904223u;
				wK[idx] = (static_cast<float>(sK) / UINT32_MAX * 2.0f - 1.0f) * range;
				sV = sV * 1664525u + 1013904223u;
				wV[idx] = (static_cast<float>(sV) / UINT32_MAX * 2.0f - 1.0f) * range;
				sO = sO * 1664525u + 1013904223u;
				wO[idx] = (static_cast<float>(sO) / UINT32_MAX * 2.0f - 1.0f) * range;
			}
		}
		wq_ = Tensor<T, EmbedDim, EmbedDim>(wQ);
		wk_ = Tensor<T, EmbedDim, EmbedDim>(wK);
		wv_ = Tensor<T, EmbedDim, EmbedDim>(wV);
		wo_ = Tensor<T, EmbedDim, EmbedDim>(wO);
	}

	void Setup() {
		wqRef_ = wq_.Bind();
		wkRef_ = wk_.Bind();
		wvRef_ = wv_.Bind();
		woRef_ = wo_.Bind();
		wqRef_.ForEachParam([](auto &p) { AD::Param(p); });
		wkRef_.ForEachParam([](auto &p) { AD::Param(p); });
		wvRef_.ForEachParam([](auto &p) { AD::Param(p); });
		woRef_.ForEachParam([](auto &p) { AD::Param(p); });
	}

	/**
	 * Forward pass for a single position in a sequence.
	 *
	 * @param x       Residual stream, read-only at pos [batchSize * embedDim * blockSize]
	 * @param kBuf    Key cache buffer, same layout — write K at pos, read K[0..pos]
	 * @param vBuf    Value cache buffer, same layout — write V at pos, read V[0..pos]
	 * @param attnOut Attention output buffer, write at pos
	 * @param pos     Current position (0..blockSize-1)
	 * @param offset  Base offset for this batch = batchIdx * blockSize * embedDim
	 */
	void Forward(const IR::Value::BufferRef<T> &x,
				 const IR::Value::BufferRef<T> &kBuf,
				 const IR::Value::BufferRef<T> &vBuf,
				 const IR::Value::BufferRef<T> &attnOut,
				 const IR::Value::Var<int> &pos,
				 const IR::Value::Expr<int> &offset) {
		constexpr int E = static_cast<int>(EmbedDim);
		constexpr int H = static_cast<int>(NumHeads);
		constexpr int HD = static_cast<int>(HeadDim);

		IR::Value::Expr<int> po = offset + pos * E;

		// --- Q/K/V projections for current position ---
		// Q stored to local buffer (reuse attnOut as temp at pos)
		// K, V stored to kBuf, vBuf
		GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &o) {
			IR::Value::Var<T> sq = MakeFloat(0.0f);
			IR::Value::Var<T> sk = MakeFloat(0.0f);
			IR::Value::Var<T> sv = MakeFloat(0.0f);
			GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &i) {
				IR::Value::Var<T> xi = x[po + i];
				sq = sq + wqRef_(o, i) * xi;
				sk = sk + wkRef_(o, i) * xi;
				sv = sv + wvRef_(o, i) * xi;
			});
			kBuf[po + o] = sk;
			vBuf[po + o] = sv;
			attnOut[po + o] = sq; // temp store Q in attnOut
		});

		// --- Attention per head ---
		IR::Value::Expr<int> endPos = pos + MakeInt(1);

		GPU::Flow::For(MakeInt(0), MakeInt(H), [&](IR::Value::Var<int> &h) {
			IR::Value::Expr<int> hs = h * MakeInt(HD);

			// Pass 1: max logit over 0..pos
			IR::Value::Var<T> maxLogit = MakeFloat(-1e9f);
			GPU::Flow::For(MakeInt(0), endPos, [&](IR::Value::Var<int> &t) {
				IR::Value::Expr<int> to = offset + t * E;
				IR::Value::Var<T> dot = MakeFloat(0.0f);
				GPU::Flow::For(MakeInt(0), MakeInt(HD), [&](IR::Value::Var<int> &d) {
					dot = dot + attnOut[po + hs + d] * kBuf[to + hs + d];
				});
				dot = dot / MakeFloat(std::sqrt(static_cast<float>(HD)));
				maxLogit = GPU::Math::Max(maxLogit, dot);
			});

			// Pass 2: sum exp(logit - max)
			IR::Value::Var<T> sumExp = MakeFloat(0.0f);
			GPU::Flow::For(MakeInt(0), endPos, [&](IR::Value::Var<int> &t) {
				IR::Value::Expr<int> to = offset + t * E;
				IR::Value::Var<T> dot = MakeFloat(0.0f);
				GPU::Flow::For(MakeInt(0), MakeInt(HD), [&](IR::Value::Var<int> &d) {
					dot = dot + attnOut[po + hs + d] * kBuf[to + hs + d];
				});
				dot = dot / MakeFloat(std::sqrt(static_cast<float>(HD)));
				sumExp = sumExp + GPU::Math::Exp(dot - maxLogit);
			});

			// Pass 3: weighted sum of values
			GPU::Flow::For(MakeInt(0), MakeInt(HD), [&](IR::Value::Var<int> &d) {
				IR::Value::Var<T> sumV = MakeFloat(0.0f);
				GPU::Flow::For(MakeInt(0), endPos, [&](IR::Value::Var<int> &t) {
					IR::Value::Expr<int> to = offset + t * E;
					IR::Value::Var<T> dot = MakeFloat(0.0f);
					GPU::Flow::For(MakeInt(0), MakeInt(HD), [&](IR::Value::Var<int> &dd) {
						dot = dot + attnOut[po + hs + dd] * kBuf[to + hs + dd];
					});
					dot = dot / MakeFloat(std::sqrt(static_cast<float>(HD)));
					IR::Value::Var<T> weight = GPU::Math::Exp(dot - maxLogit) / sumExp;
					sumV = sumV + weight * vBuf[to + hs + d];
				});
				attnOut[po + hs + d] = sumV; // replace Q with attention output
			});
		});

		// --- Output projection ---
		GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &o) {
			IR::Value::Var<T> sum = MakeFloat(0.0f);
			GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &i) {
				sum = sum + woRef_(o, i) * attnOut[po + i];
			});
			attnOut[po + o] = sum;
		});
	}

	Tensor<T, EmbedDim, EmbedDim> &Wq() { return wq_; }
	Tensor<T, EmbedDim, EmbedDim> &Wk() { return wk_; }
	Tensor<T, EmbedDim, EmbedDim> &Wv() { return wv_; }
	Tensor<T, EmbedDim, EmbedDim> &Wo() { return wo_; }
	static constexpr size_t TotalSize = 4 * EmbedDim * EmbedDim;

private:
	Tensor<T, EmbedDim, EmbedDim> wq_, wk_, wv_, wo_;
	TensorRef<T, EmbedDim, EmbedDim> wqRef_, wkRef_, wvRef_, woRef_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_ATTENTION_H
