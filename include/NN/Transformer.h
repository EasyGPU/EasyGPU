#pragma once

/**
 * @file Transformer.h
 * @brief Transformer block (attention + MLP) with RMSNorm and residual connections.
 *
 *   TransformerBlock<T, BlockSize, EmbedDim, NumHeads>
 *
 * Pre-norm architecture (GPT-2 style): RMSNorm → Attention → +residual
 *                                     → RMSNorm → MLP (ReLU) → +residual
 *
 * Uses a single scratch buffer internally and keeps the FFN inside the
 * Transformer block kernel, so GPT does not route through a standalone
 * Sequential/Linear MLP dispatch chain.
 */

#ifndef EASYGPU_NN_TRANSFORMER_H
#define EASYGPU_NN_TRANSFORMER_H

#include <AD/ADKernel.h>
#include <NN/Attention.h>
#include <NN/Normalization.h>
#include <NN/Tensor.h>

#include <Flow/ForFlow.h>
#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>
#include <Runtime/Buffer.h>
#include <Utility/Helpers.h>
#include <Utility/Math.h>

#include <cstddef>
#include <random>
#include <vector>

namespace GPU::NN {

template <typename T, size_t BlockSize, size_t EmbedDim, size_t NumHeads> class TransformerBlock {
	static_assert(std::is_same_v<T, float>, "TransformerBlock only supports float");
	static_assert(EmbedDim % NumHeads == 0, "EmbedDim must be divisible by NumHeads");
	static constexpr size_t MLPDim	 = 4 * EmbedDim;

	// Per-batch scratch region sizes (in floats)
	static constexpr size_t K_REGION = BlockSize * EmbedDim;
	static constexpr size_t V_REGION = BlockSize * EmbedDim;
	static constexpr size_t A_REGION = BlockSize * EmbedDim;
	static constexpr size_t M_REGION = BlockSize * MLPDim;
	static constexpr size_t REGIONS	 = 7; // K(1) + V(1) + A(1) + M(4) = 7 × BE

public:
	TransformerBlock(size_t batchSize, unsigned seed = 42)
		: batchSize_(batchSize), attn_(seed), norm1_(), norm2_(),
		  scratchBuf_(batchSize * REGIONS * BlockSize * EmbedDim, Runtime::BufferMode::ReadWrite) {
		// Compute region base offsets within scratch buffer
		int bs = static_cast<int>(batchSize);
		int be = static_cast<int>(BlockSize * EmbedDim);
		kBase_ = 0;
		vBase_ = bs * be;
		aBase_ = 2 * bs * be;
		mBase_ = 3 * bs * be;

		std::vector<T> d1(MLPDim * EmbedDim);
		std::vector<T> d2(EmbedDim * MLPDim);

		float		   range = std::sqrt(6.0f / static_cast<float>(EmbedDim + MLPDim));
		unsigned	   s1 = seed + 10, s2 = seed + 11;

		for (size_t j = 0; j < MLPDim; j++)
			for (size_t i = 0; i < EmbedDim; i++) {
				s1					 = s1 * 1664525u + 1013904223u;
				d1[j * EmbedDim + i] = (static_cast<float>(static_cast<double>(s1) / UINT32_MAX) * 2.0f - 1.0f) * range;
			}
		for (size_t j = 0; j < EmbedDim; j++)
			for (size_t i = 0; i < MLPDim; i++) {
				s2				   = s2 * 1664525u + 1013904223u;
				d2[j * MLPDim + i] = (static_cast<float>(static_cast<double>(s2) / UINT32_MAX) * 2.0f - 1.0f) * range;
			}

		fc1W_ = Tensor<T, MLPDim, EmbedDim>(d1);
		fc2W_ = Tensor<T, EmbedDim, MLPDim>(d2);
	}

	void Setup() {
		attn_.Setup();
		norm1_.Setup();
		norm2_.Setup();

		fc1Ref_ = fc1W_.Bind();
		fc2Ref_ = fc2W_.Bind();
		fc1Ref_.RegisterAsParam();
		fc2Ref_.RegisterAsParam();

		scratchRef_ = scratchBuf_.Bind();
	}

	/**
	 * Forward pass for a single position.
	 *
	 * @param x      Residual stream buffer [batchSize * blockSize * embedDim]
	 * @param pos    Current position (0..blockSize-1)
	 * @param offset Base offset for embed-dim buffers = batchIdx * blockSize * embedDim
	 */
	void Forward(const IR::Value::BufferRef<T> &x, const IR::Value::Var<int> &pos, const IR::Value::Expr<int> &offset) {
		constexpr int		 E	  = static_cast<int>(EmbedDim);
		constexpr int		 M	  = static_cast<int>(MLPDim);

		IR::Value::Expr<int> po	  = offset + pos * E;
		IR::Value::Expr<int> poM  = po * MakeInt(static_cast<int>(MLPDim / EmbedDim));

		// Region base offsets as DSL expressions
		IR::Value::Expr<int> kOff = MakeInt(kBase_);
		IR::Value::Expr<int> vOff = MakeInt(vBase_);
		IR::Value::Expr<int> aOff = MakeInt(aBase_);
		IR::Value::Expr<int> mOff = MakeInt(mBase_);

		// --- RMSNorm → Attention → residual add ---
		norm1_.Forward(x, scratchRef_, aOff + po);
		attn_.Forward(scratchRef_, aOff, kOff, vOff, aOff, mOff, pos, offset);
		GPU::Flow::For(MakeInt(0), MakeInt(E),
					   [&](IR::Value::Var<int> &d) { x[po + d] = x[po + d] + scratchRef_[aOff + po + d]; });

		// --- RMSNorm → MLP → residual add ---
		norm2_.Forward(x, scratchRef_, aOff + po);

		// Fused block FFN: FC1 + ReLU + FC2 live inside the same Transformer
		// kernel. The hidden vector is materialized once in scratch to keep the
		// generated AD shader compact enough for real training runs.
		GPU::Flow::For(MakeInt(0), MakeInt(M), [&](IR::Value::Var<int> &h) {
			IR::Value::Var<T> hidden = MakeFloat(0.0f);
			GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &i) {
				hidden = hidden + fc1Ref_(h, i) * scratchRef_[aOff + po + i];
			});
			scratchRef_[mOff + poM + h] = GPU::Math::Max(hidden, MakeFloat(0.0f));
		});

		GPU::Flow::For(MakeInt(0), MakeInt(E), [&](IR::Value::Var<int> &o) {
			IR::Value::Var<T> sum = MakeFloat(0.0f);
			GPU::Flow::For(MakeInt(0), MakeInt(M),
						   [&](IR::Value::Var<int> &h) { sum = sum + fc2Ref_(o, h) * scratchRef_[mOff + poM + h]; });
			x[po + o] = x[po + o] + sum;
		});
	}

	CausalSelfAttention<T, EmbedDim, NumHeads> &Attention() {
		return attn_;
	}
	Tensor<T, MLPDim, EmbedDim> &FC1() {
		return fc1W_;
	}
	Tensor<T, EmbedDim, MLPDim> &FC2() {
		return fc2W_;
	}
	static constexpr size_t ParamCount =
		CausalSelfAttention<T, EmbedDim, NumHeads>::TotalSize + MLPDim * EmbedDim + EmbedDim * MLPDim;

private:
	size_t									   batchSize_;
	int										   kBase_, vBase_, aBase_, mBase_;
	CausalSelfAttention<T, EmbedDim, NumHeads> attn_;
	RMSNorm<T, EmbedDim>					   norm1_, norm2_;

	Tensor<T, MLPDim, EmbedDim>				   fc1W_;
	Tensor<T, EmbedDim, MLPDim>				   fc2W_;
	TensorRef<T, MLPDim, EmbedDim>			   fc1Ref_;
	TensorRef<T, EmbedDim, MLPDim>			   fc2Ref_;

	Runtime::Buffer<T>						   scratchBuf_;
	IR::Value::BufferRef<T>					   scratchRef_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_TRANSFORMER_H
