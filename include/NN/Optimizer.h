#pragma once

/**
 * @file Optimizer.h
 * @brief Built-in optimizers for EasyGPU AD training.
 *
 * Adam, SGD, and RMSprop with weight decay, gradient clipping, and
 * automatic gradient aggregation (mean across GPU threads).
 *
 * Usage:
 *   Adam optimizer(0.001f);
 *   optimizer.AddTensor(W);
 *   // In training loop:
 *   for (int step = 0; step < 1000; step++) {
 *       kernel.Backward(groups, true);
 *       optimizer.Step(kernel);
 *   }
 */

#ifndef EASYGPU_NN_OPTIMIZER_H
#define EASYGPU_NN_OPTIMIZER_H

#include <AD/ADKernel.h>
#include <NN/Tensor.h>
#include <Runtime/Buffer.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace GPU::NN {

namespace detail {

inline void ApplyAdamUpdate(float *weight, const float *grad, float *m, float *v,
                            size_t size, float lr, float beta1, float beta2,
                            float eps, int step, float weightDecay, float gradClip) {
    const double biasCorr1 = 1.0 - std::pow(beta1, step);
    const double biasCorr2 = 1.0 - std::pow(beta2, step);

    for (size_t i = 0; i < size; ++i) {
        double g = static_cast<double>(grad[i]);
        if (weightDecay > 0.0f) g += 2.0 * weightDecay * static_cast<double>(weight[i]);
        if (gradClip > 0.0f) g = std::clamp(g, -static_cast<double>(gradClip), static_cast<double>(gradClip));

        m[i] = beta1 * m[i] + (1.0f - beta1) * static_cast<float>(g);
        v[i] = beta2 * v[i] + (1.0f - beta2) * static_cast<float>(g * g);

        const double mHat = static_cast<double>(m[i]) / biasCorr1;
        const double vHat = static_cast<double>(v[i]) / biasCorr2;
        weight[i] -= static_cast<float>(lr * mHat / (std::sqrt(vHat) + eps));
    }
}

inline void ApplySGDUpdate(float *weight, const float *grad, float *m,
                           size_t size, float lr, float momentum,
                           float weightDecay, float gradClip) {
    for (size_t i = 0; i < size; ++i) {
        double g = static_cast<double>(grad[i]);
        if (weightDecay > 0.0f) g += 2.0 * weightDecay * static_cast<double>(weight[i]);
        if (gradClip > 0.0f) g = std::clamp(g, -static_cast<double>(gradClip), static_cast<double>(gradClip));

        m[i] = momentum * m[i] + static_cast<float>(g);
        weight[i] -= lr * m[i];
    }
}

inline void ApplyRMSpropUpdate(float *weight, const float *grad, float *m,
                               size_t size, float lr, float beta, float eps,
                               float weightDecay, float gradClip) {
    for (size_t i = 0; i < size; ++i) {
        double g = static_cast<double>(grad[i]);
        if (weightDecay > 0.0f) g += 2.0 * weightDecay * static_cast<double>(weight[i]);
        if (gradClip > 0.0f) g = std::clamp(g, -static_cast<double>(gradClip), static_cast<double>(gradClip));

        m[i] = beta * m[i] + (1.0f - beta) * static_cast<float>(g * g);
        weight[i] -= lr * static_cast<float>(g) / (std::sqrt(m[i] + eps));
    }
}

} // namespace detail

// =============================================================================
// Parameter state
// =============================================================================

struct ParamSlot {
	std::vector<float> m;   // first moment (Adam) / velocity (SGD+momentum) / sq_avg (RMSprop)
	std::vector<float> v;   // second moment (Adam only; unused in SGD/RMSprop but kept for uniform layout)
	float *data = nullptr;  // pointer to CPU weight array
	size_t size   = 0;      // number of weight elements
	Runtime::Buffer<float> *buffer = nullptr;  // GPU buffer to upload to (can be nullptr)
};

// =============================================================================
// Adam
// =============================================================================

class Adam {
public:
	/**
	 * @param lr    Learning rate
	 * @param beta1 First moment decay (default 0.9)
	 * @param beta2 Second moment decay (default 0.999)
	 * @param eps   Epsilon for numerical stability (default 1e-8)
	 */
	Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
		: lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {}

	void SetWeightDecay(float wd) { weightDecay_ = wd; }
	void SetGradClip(float clip) { gradClip_ = clip; }

	/** Register a scalar weight array as a trainable parameter. */
	void AddParameter(float *data, size_t size, Runtime::Buffer<float> *buf = nullptr) {
		ParamSlot ps;
		ps.m.resize(size, 0.0f);
		ps.v.resize(size, 0.0f);
		ps.data   = data;
		ps.size   = size;
		ps.buffer = buf;
		params_.push_back(std::move(ps));
	}

	/** Register all elements of a tensor as trainable parameters. */
	template <size_t... Dims>
	void AddTensor(Tensor<float, Dims...> &tensor) {
		AddParameter(tensor.Data(), tensor.Size(), &tensor.GetBuffer());
	}

	/**
	 * Execute one optimization step.
	 * Downloads gradients from the AD kernel, averages across threads,
	 * applies weight decay + clipping + Adam update, and uploads weights.
	 */
	void Step(AD::ADKernel1D &kernel) {
		step_++;

		auto allGrads = kernel.DownloadAllGradients();

		int paramIdx = 0;
		for (auto &ps : params_) {
			std::vector<float> meanGrad(ps.size, 0.0f);

			for (size_t j = 0; j < ps.size; j++) {
				if (paramIdx >= static_cast<int>(allGrads.size())) {
					throw std::runtime_error("Adam gradient count does not match registered parameters");
				}

				const auto &grad = allGrads[paramIdx++];
				double g = 0.0;
				for (float sampleGrad : grad) {
					g += static_cast<double>(sampleGrad);
				}
				meanGrad[j] = grad.empty() ? 0.0f : static_cast<float>(g / static_cast<double>(grad.size()));
			}

			detail::ApplyAdamUpdate(ps.data, meanGrad.data(), ps.m.data(), ps.v.data(),
				ps.size, lr_, beta1_, beta2_, eps_, step_, weightDecay_, gradClip_);

			if (ps.buffer)
				ps.buffer->Upload(ps.data, ps.size);
		}
	}

	int GetStep() const { return step_; }
	size_t ParameterCount() const { return params_.size(); }

private:
	float lr_, beta1_, beta2_, eps_;
	float weightDecay_ = 0.0f;
	float gradClip_    = 0.0f;
	int   step_        = 0;
	std::vector<ParamSlot> params_;
};

// =============================================================================
// SGD with momentum
// =============================================================================

class SGD {
public:
	/**
	 * @param lr       Learning rate
	 * @param momentum Momentum coefficient (0 = vanilla SGD)
	 */
	SGD(float lr = 0.01f, float momentum = 0.0f)
		: lr_(lr), momentum_(momentum) {}

	void SetWeightDecay(float wd) { weightDecay_ = wd; }
	void SetGradClip(float clip) { gradClip_ = clip; }

	void AddParameter(float *data, size_t size, Runtime::Buffer<float> *buf = nullptr) {
		ParamSlot ps;
		ps.m.resize(size, 0.0f);
		ps.data   = data;
		ps.size   = size;
		ps.buffer = buf;
		params_.push_back(std::move(ps));
	}

	template <size_t... Dims>
	void AddTensor(Tensor<float, Dims...> &tensor) {
		AddParameter(tensor.Data(), tensor.Size(), &tensor.GetBuffer());
	}

	void Step(AD::ADKernel1D &kernel) {
		step_++;

		auto allGrads = kernel.DownloadAllGradients();

		int paramIdx = 0;
		for (auto &ps : params_) {
			for (size_t j = 0; j < ps.size; j++) {
				const auto &grad = allGrads[paramIdx];

				double meanGrad = 0.0;
				for (size_t g = 0; g < grad.size(); g++)
					meanGrad += static_cast<double>(grad[g]);
				meanGrad /= static_cast<double>(grad.size());

				double g = meanGrad;
				if (weightDecay_ > 0.0f)
					g += 2.0 * weightDecay_ * static_cast<double>(ps.data[j]);
				if (gradClip_ > 0.0f)
					g = std::clamp(g, -static_cast<double>(gradClip_),
					                  static_cast<double>(gradClip_));

				// SGD with momentum
				ps.m[j] = momentum_ * ps.m[j] + static_cast<float>(g);
				ps.data[j] -= lr_ * ps.m[j];

				paramIdx++;
			}

			if (ps.buffer)
				ps.buffer->Upload(ps.data, ps.size);
		}
	}

	int GetStep() const { return step_; }
	size_t ParameterCount() const { return params_.size(); }

private:
	float lr_, momentum_;
	float weightDecay_ = 0.0f;
	float gradClip_    = 0.0f;
	int   step_        = 0;
	std::vector<ParamSlot> params_;
};

// =============================================================================
// RMSprop
// =============================================================================

class RMSprop {
public:
	/**
	 * @param lr   Learning rate
	 * @param beta Moving average decay for squared gradients (default 0.9)
	 * @param eps  Epsilon for numerical stability (default 1e-8)
	 */
	RMSprop(float lr = 0.001f, float beta = 0.9f, float eps = 1e-8f)
		: lr_(lr), beta_(beta), eps_(eps) {}

	void SetWeightDecay(float wd) { weightDecay_ = wd; }
	void SetGradClip(float clip) { gradClip_ = clip; }

	void AddParameter(float *data, size_t size, Runtime::Buffer<float> *buf = nullptr) {
		ParamSlot ps;
		ps.m.resize(size, 0.0f);
		ps.data   = data;
		ps.size   = size;
		ps.buffer = buf;
		params_.push_back(std::move(ps));
	}

	template <size_t... Dims>
	void AddTensor(Tensor<float, Dims...> &tensor) {
		AddParameter(tensor.Data(), tensor.Size(), &tensor.GetBuffer());
	}

	void Step(AD::ADKernel1D &kernel) {
		step_++;

		auto allGrads = kernel.DownloadAllGradients();

		int paramIdx = 0;
		for (auto &ps : params_) {
			for (size_t j = 0; j < ps.size; j++) {
				const auto &grad = allGrads[paramIdx];

				double meanGrad = 0.0;
				for (size_t g = 0; g < grad.size(); g++)
					meanGrad += static_cast<double>(grad[g]);
				meanGrad /= static_cast<double>(grad.size());

				double g = meanGrad;
				if (weightDecay_ > 0.0f)
					g += 2.0 * weightDecay_ * static_cast<double>(ps.data[j]);
				if (gradClip_ > 0.0f)
					g = std::clamp(g, -static_cast<double>(gradClip_),
					                  static_cast<double>(gradClip_));

				// RMSprop update
				ps.m[j] = beta_ * ps.m[j] + (1.0f - beta_) * static_cast<float>(g * g);
				ps.data[j] -= lr_ * static_cast<float>(g) /
				              (std::sqrt(ps.m[j] + eps_));

				paramIdx++;
			}

			if (ps.buffer)
				ps.buffer->Upload(ps.data, ps.size);
		}
	}

	int GetStep() const { return step_; }
	size_t ParameterCount() const { return params_.size(); }

private:
	float lr_, beta_, eps_;
	float weightDecay_ = 0.0f;
	float gradClip_    = 0.0f;
	int   step_        = 0;
	std::vector<ParamSlot> params_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_OPTIMIZER_H
