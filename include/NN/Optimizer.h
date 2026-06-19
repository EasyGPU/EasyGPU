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
#include <Backend/Backend.h>
#include <NN/Tensor.h>
#include <Runtime/Buffer.h>
#include <Runtime/Context.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace GPU::NN {

class GPUAdam;
class GPUSGD;
class GPURMSprop;

// =============================================================================
// Parameter state
// =============================================================================

struct ParamSlot {
	std::vector<float>		m;
	std::vector<float>		v;
	float				   *data   = nullptr;
	size_t					size   = 0;
	Runtime::Buffer<float> *buffer = nullptr;
};

// =============================================================================
// Adam
// =============================================================================

class Adam {
public:
	Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
		: lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
	}
	~Adam();

	void SetWeightDecay(float wd) {
		weightDecay_ = wd;
	}
	void SetGradClip(float clip) {
		gradClip_ = clip;
	}
	void SetLearningRate(float lr) {
		lr_ = lr;
	}
	void SetBetas(float beta1, float beta2) {
		beta1_ = beta1;
		beta2_ = beta2;
	}
	void SetEps(float eps) {
		eps_ = eps;
	}

	void AddParameter(float *data, size_t size, Runtime::Buffer<float> *buf = nullptr) {
		ParamSlot ps;
		ps.m.resize(size, 0.0f);
		ps.v.resize(size, 0.0f);
		ps.data	  = data;
		ps.size	  = size;
		ps.buffer = buf;
		params_.push_back(std::move(ps));
	}

	template <size_t... Dims> void AddTensor(Tensor<float, Dims...> &tensor) {
		AddParameter(tensor.Data(), tensor.Size(), &tensor.GetBuffer());
	}

	void Step(AD::ADKernel1D &kernel);

	int	 GetStep() const {
		return step_;
	}
	size_t ParameterCount() const {
		return params_.size();
	}

private:
	float					 lr_, beta1_, beta2_, eps_;
	float					 weightDecay_ = 0.0f;
	float					 gradClip_	  = 0.0f;
	int						 step_		  = 0;
	std::vector<ParamSlot>	 params_;
	std::unique_ptr<GPUAdam> gpu_;
};

// =============================================================================
// GPUAdam
// =============================================================================

class GPUAdam {
public:
	GPUAdam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
		: lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
	}

	~GPUAdam() {
		ReleasePipelines();
	}

	GPUAdam(const GPUAdam &)			= delete;
	GPUAdam &operator=(const GPUAdam &) = delete;

	void	 SetWeightDecay(float wd) {
		weightDecay_ = wd;
	}
	void SetGradClip(float clip) {
		gradClip_ = clip;
	}
	void SetLearningRate(float lr) {
		lr_ = lr;
	}
	void SetBetas(float beta1, float beta2) {
		beta1_ = beta1;
		beta2_ = beta2;
	}
	void SetEps(float eps) {
		eps_ = eps;
	}

	template <size_t... Dims> void AddTensor(Tensor<float, Dims...> &tensor) {
		AddParameter(tensor.Size(), tensor.GetBuffer().GetHandle());
	}

	void AddParameter(size_t size, Backend::BufferHandle weightHandle);

	void Step(AD::ADKernel1D &kernel, bool sync = false);

	int	 GetStep() const {
		return step_;
	}
	size_t ParameterCount() const {
		return params_.size();
	}

private:
	struct ParamSlotGPU {
		size_t									size		 = 0;
		Backend::BufferHandle					weightHandle = Backend::INVALID_BUFFER_HANDLE;
		std::unique_ptr<Runtime::Buffer<float>> m;
		std::unique_ptr<Runtime::Buffer<float>> v;
		std::unique_ptr<Runtime::Buffer<float>> hyper;
		Backend::ShaderHandle					shader			   = Backend::INVALID_SHADER_HANDLE;
		Backend::PipelineHandle					pipeline		   = Backend::INVALID_PIPELINE_HANDLE;
		size_t									compiledSamples	   = 0;
		int										compiledGradOffset = 0;
		int										compiledGradStride = 1;
	};

	struct CombinedSlot {
		size_t				  size		   = 0;
		size_t				  base		   = 0;
		size_t				  sampleCount  = 0;
		int					  gradOffset   = 0;
		int					  gradStride   = 1;
		Backend::BufferHandle weightHandle = Backend::INVALID_BUFFER_HANDLE;
		Backend::BufferHandle gradHandle   = Backend::INVALID_BUFFER_HANDLE;
	};

	std::vector<CombinedSlot> BuildCombinedSlots(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
												 size_t												  &totalSize) const;

	std::string				  CombinedSignature(const std::vector<CombinedSlot> &slots, size_t totalSize) const;

	static std::string		  BuildCombinedReduceShader(const std::vector<CombinedSlot> &slots, size_t totalSize);
	static std::string		  BuildCombinedAdamShader(const std::vector<CombinedSlot> &slots, size_t totalSize);

	void					  StepCombined(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams, bool sync);

	void					  EnsureCombinedPipeline(const std::vector<CombinedSlot> &slots, size_t totalSize);
	void					  EnsureCombinedReducePipeline(const std::vector<CombinedSlot> &slots, size_t totalSize);
	void					  UploadCombinedHyperParams(size_t totalSize, size_t sampleCount);
	void					  DispatchCombinedReduce(const std::vector<CombinedSlot> &slots, size_t totalSize);
	void					  DispatchCombined(const std::vector<CombinedSlot> &slots, size_t totalSize, bool sync);

	static std::string		  BuildAdamShader(size_t paramSize, size_t sampleCount, int gradOffset, int gradStride);

	void					  EnsurePipeline(ParamSlotGPU &ps, const AD::ADKernel1D::GradientParamInfo &first);
	void					  UploadHyperParams(ParamSlotGPU &ps, size_t sampleCount);
	void					  DispatchSlot(ParamSlotGPU &ps, Backend::BufferHandle gradHandle, bool sync);
	void					  ReleasePipelines();

	float					  lr_, beta1_, beta2_, eps_;
	float					  weightDecay_ = 0.0f;
	float					  gradClip_	   = 0.0f;
	int						  step_		   = 0;
	std::vector<ParamSlotGPU> params_;
	std::unique_ptr<Runtime::Buffer<float>> _flatM;
	std::unique_ptr<Runtime::Buffer<float>> _flatV;
	std::unique_ptr<Runtime::Buffer<float>> _meanGrad;
	std::unique_ptr<Runtime::Buffer<float>> _combinedHyper;
	size_t									_flatMSize		  = 0;
	Backend::ShaderHandle					_combinedShader	  = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle					_combinedPipeline = Backend::INVALID_PIPELINE_HANDLE;
	Backend::ShaderHandle					_combinedReduceShader	= Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle					_combinedReducePipeline = Backend::INVALID_PIPELINE_HANDLE;
	std::string								_combinedSignature;
	std::string								_combinedReduceSignature;
};

// =============================================================================
// GPUSGD
// =============================================================================

class GPUSGD {
public:
	GPUSGD(float lr = 0.01f, float momentum = 0.0f) : lr_(lr), momentum_(momentum) {
	}

	~GPUSGD() {
		ReleasePipelines();
	}

	GPUSGD(const GPUSGD &)			  = delete;
	GPUSGD &operator=(const GPUSGD &) = delete;

	void	SetWeightDecay(float wd) {
		weightDecay_ = wd;
	}
	void SetGradClip(float clip) {
		gradClip_ = clip;
	}
	void SetLearningRate(float lr) {
		lr_ = lr;
	}
	void SetMomentum(float momentum) {
		momentum_ = momentum;
	}

	template <size_t... Dims> void AddTensor(Tensor<float, Dims...> &tensor) {
		AddParameter(tensor.Size(), tensor.GetBuffer().GetHandle());
	}

	void AddParameter(size_t size, Backend::BufferHandle weightHandle);

	void Step(AD::ADKernel1D &kernel, bool sync = false);

	int	 GetStep() const {
		return step_;
	}
	size_t ParameterCount() const {
		return params_.size();
	}

private:
	struct ParamSlotGPU {
		size_t									size		 = 0;
		Backend::BufferHandle					weightHandle = Backend::INVALID_BUFFER_HANDLE;
		std::unique_ptr<Runtime::Buffer<float>> velocity;
		std::unique_ptr<Runtime::Buffer<float>> hyper;
		Backend::ShaderHandle					shader			   = Backend::INVALID_SHADER_HANDLE;
		Backend::PipelineHandle					pipeline		   = Backend::INVALID_PIPELINE_HANDLE;
		size_t									compiledSamples	   = 0;
		int										compiledGradOffset = 0;
		int										compiledGradStride = 1;
	};

	struct CombinedSlot {
		size_t				  size		   = 0;
		size_t				  base		   = 0;
		size_t				  sampleCount  = 0;
		int					  gradOffset   = 0;
		int					  gradStride   = 1;
		Backend::BufferHandle weightHandle = Backend::INVALID_BUFFER_HANDLE;
		Backend::BufferHandle gradHandle   = Backend::INVALID_BUFFER_HANDLE;
	};

	static void ValidateGradientGroup(const char										   *name,
									  const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
									  size_t paramBase, size_t size, const AD::ADKernel1D::GradientParamInfo &first);

	std::vector<CombinedSlot> BuildCombinedSlots(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
												 size_t												  &totalSize) const;

	std::string				  CombinedSignature(const std::vector<CombinedSlot> &slots, size_t totalSize) const;

	static std::string		  BuildCombinedShader(const std::vector<CombinedSlot> &slots, size_t totalSize);

	void					  StepCombined(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams, bool sync);

	void					  EnsureCombinedPipeline(const std::vector<CombinedSlot> &slots, size_t totalSize);
	void					  UploadCombinedHyperParams(size_t sampleCount);
	void					  DispatchCombined(const std::vector<CombinedSlot> &slots, size_t totalSize, bool sync);

	static std::string		  BuildShader(size_t paramSize, size_t sampleCount, int gradOffset, int gradStride);

	void					  EnsurePipeline(ParamSlotGPU &ps, const AD::ADKernel1D::GradientParamInfo &first);
	void					  UploadHyperParams(ParamSlotGPU &ps, size_t sampleCount);
	void					  DispatchSlot(ParamSlotGPU &ps, Backend::BufferHandle gradHandle, bool sync);
	void					  ReleasePipelines();

	float					  lr_, momentum_;
	float					  weightDecay_ = 0.0f;
	float					  gradClip_	   = 0.0f;
	int						  step_		   = 0;
	std::vector<ParamSlotGPU> params_;
	std::unique_ptr<Runtime::Buffer<float>> _flatVelocity;
	std::unique_ptr<Runtime::Buffer<float>> _combinedHyper;
	size_t									_flatVelocitySize = 0;
	Backend::ShaderHandle					_combinedShader	  = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle					_combinedPipeline = Backend::INVALID_PIPELINE_HANDLE;
	std::string								_combinedSignature;
};

// =============================================================================
// GPURMSprop
// =============================================================================

class GPURMSprop {
public:
	GPURMSprop(float lr = 0.001f, float beta = 0.9f, float eps = 1e-8f) : lr_(lr), beta_(beta), eps_(eps) {
	}

	~GPURMSprop() {
		ReleasePipelines();
	}

	GPURMSprop(const GPURMSprop &)			  = delete;
	GPURMSprop &operator=(const GPURMSprop &) = delete;

	void		SetWeightDecay(float wd) {
		weightDecay_ = wd;
	}
	void SetGradClip(float clip) {
		gradClip_ = clip;
	}
	void SetLearningRate(float lr) {
		lr_ = lr;
	}
	void SetBeta(float beta) {
		beta_ = beta;
	}
	void SetEps(float eps) {
		eps_ = eps;
	}

	template <size_t... Dims> void AddTensor(Tensor<float, Dims...> &tensor) {
		AddParameter(tensor.Size(), tensor.GetBuffer().GetHandle());
	}

	void AddParameter(size_t size, Backend::BufferHandle weightHandle);

	void Step(AD::ADKernel1D &kernel, bool sync = false);

	int	 GetStep() const {
		return step_;
	}
	size_t ParameterCount() const {
		return params_.size();
	}

private:
	struct ParamSlotGPU {
		size_t									size		 = 0;
		Backend::BufferHandle					weightHandle = Backend::INVALID_BUFFER_HANDLE;
		std::unique_ptr<Runtime::Buffer<float>> squareAvg;
		std::unique_ptr<Runtime::Buffer<float>> hyper;
		Backend::ShaderHandle					shader			   = Backend::INVALID_SHADER_HANDLE;
		Backend::PipelineHandle					pipeline		   = Backend::INVALID_PIPELINE_HANDLE;
		size_t									compiledSamples	   = 0;
		int										compiledGradOffset = 0;
		int										compiledGradStride = 1;
	};

	struct CombinedSlot {
		size_t				  size		   = 0;
		size_t				  base		   = 0;
		size_t				  sampleCount  = 0;
		int					  gradOffset   = 0;
		int					  gradStride   = 1;
		Backend::BufferHandle weightHandle = Backend::INVALID_BUFFER_HANDLE;
		Backend::BufferHandle gradHandle   = Backend::INVALID_BUFFER_HANDLE;
	};

	static void ValidateGradientGroup(const char										   *name,
									  const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
									  size_t paramBase, size_t size, const AD::ADKernel1D::GradientParamInfo &first);

	std::vector<CombinedSlot> BuildCombinedSlots(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
												 size_t												  &totalSize) const;

	std::string				  CombinedSignature(const std::vector<CombinedSlot> &slots, size_t totalSize) const;

	static std::string		  BuildCombinedShader(const std::vector<CombinedSlot> &slots, size_t totalSize);

	void					  StepCombined(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams, bool sync);

	void					  EnsureCombinedPipeline(const std::vector<CombinedSlot> &slots, size_t totalSize);
	void					  UploadCombinedHyperParams(size_t sampleCount);
	void					  DispatchCombined(const std::vector<CombinedSlot> &slots, size_t totalSize, bool sync);

	static std::string		  BuildShader(size_t paramSize, size_t sampleCount, int gradOffset, int gradStride);

	void					  EnsurePipeline(ParamSlotGPU &ps, const AD::ADKernel1D::GradientParamInfo &first);
	void					  UploadHyperParams(ParamSlotGPU &ps, size_t sampleCount);
	void					  DispatchSlot(ParamSlotGPU &ps, Backend::BufferHandle gradHandle, bool sync);
	void					  ReleasePipelines();

	float					  lr_, beta_, eps_;
	float					  weightDecay_ = 0.0f;
	float					  gradClip_	   = 0.0f;
	int						  step_		   = 0;
	std::vector<ParamSlotGPU> params_;
	std::unique_ptr<Runtime::Buffer<float>> _flatSquareAvg;
	std::unique_ptr<Runtime::Buffer<float>> _combinedHyper;
	size_t									_flatSquareAvgSize = 0;
	Backend::ShaderHandle					_combinedShader	   = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle					_combinedPipeline  = Backend::INVALID_PIPELINE_HANDLE;
	std::string								_combinedSignature;
};

// =============================================================================
// SGD with momentum
// =============================================================================

class SGD {
public:
	SGD(float lr = 0.01f, float momentum = 0.0f) : lr_(lr), momentum_(momentum) {
	}
	~SGD() = default;

	void SetWeightDecay(float wd) {
		weightDecay_ = wd;
	}
	void SetGradClip(float clip) {
		gradClip_ = clip;
	}
	void SetLearningRate(float lr) {
		lr_ = lr;
	}
	void SetMomentum(float momentum) {
		momentum_ = momentum;
	}

	void AddParameter(float *data, size_t size, Runtime::Buffer<float> *buf = nullptr) {
		ParamSlot ps;
		ps.m.resize(size, 0.0f);
		ps.data	  = data;
		ps.size	  = size;
		ps.buffer = buf;
		params_.push_back(std::move(ps));
	}

	template <size_t... Dims> void AddTensor(Tensor<float, Dims...> &tensor) {
		AddParameter(tensor.Data(), tensor.Size(), &tensor.GetBuffer());
	}

	void Step(AD::ADKernel1D &kernel);

	int	 GetStep() const {
		return step_;
	}
	size_t ParameterCount() const {
		return params_.size();
	}

private:
	float					lr_, momentum_;
	float					weightDecay_ = 0.0f;
	float					gradClip_	 = 0.0f;
	int						step_		 = 0;
	std::vector<ParamSlot>	params_;
	std::unique_ptr<GPUSGD> gpu_;
};

// =============================================================================
// RMSprop
// =============================================================================

class RMSprop {
public:
	RMSprop(float lr = 0.001f, float beta = 0.9f, float eps = 1e-8f) : lr_(lr), beta_(beta), eps_(eps) {
	}
	~RMSprop() = default;

	void SetWeightDecay(float wd) {
		weightDecay_ = wd;
	}
	void SetGradClip(float clip) {
		gradClip_ = clip;
	}
	void SetLearningRate(float lr) {
		lr_ = lr;
	}
	void SetBeta(float beta) {
		beta_ = beta;
	}
	void SetEps(float eps) {
		eps_ = eps;
	}

	void AddParameter(float *data, size_t size, Runtime::Buffer<float> *buf = nullptr) {
		ParamSlot ps;
		ps.m.resize(size, 0.0f);
		ps.data	  = data;
		ps.size	  = size;
		ps.buffer = buf;
		params_.push_back(std::move(ps));
	}

	template <size_t... Dims> void AddTensor(Tensor<float, Dims...> &tensor) {
		AddParameter(tensor.Data(), tensor.Size(), &tensor.GetBuffer());
	}

	void Step(AD::ADKernel1D &kernel);

	int	 GetStep() const {
		return step_;
	}
	size_t ParameterCount() const {
		return params_.size();
	}

private:
	float						lr_, beta_, eps_;
	float						weightDecay_ = 0.0f;
	float						gradClip_	 = 0.0f;
	int							step_		 = 0;
	std::vector<ParamSlot>		params_;
	std::unique_ptr<GPURMSprop> gpu_;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_OPTIMIZER_H
