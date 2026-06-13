#pragma once

/**
 * @file FusedMLPTraining.h
 * @brief Raw-GLSL specialized fused MLP training kernels.
 */

#ifndef EASYGPU_NN_FUSED_MLP_TRAINING_H
#define EASYGPU_NN_FUSED_MLP_TRAINING_H

#include <Backend/Backend.h>
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

namespace detail {

constexpr bool IsFusedWidth(size_t n) {
	return n == 16 || n == 32 || n == 64;
}

inline float FusedTrainXavier(unsigned &seed, size_t fanIn, size_t fanOut) {
	seed		= seed * 1664525u + 1013904223u;
	float u		= static_cast<float>(seed) / static_cast<float>(UINT32_MAX);
	float range = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
	return (u * 2.0f - 1.0f) * range;
}

inline void ReplaceAll(std::string &s, const std::string &from, const std::string &to) {
	size_t pos = 0;
	while ((pos = s.find(from, pos)) != std::string::npos) {
		s.replace(pos, from.size(), to);
		pos += to.size();
	}
}

} // namespace detail

/**
 * @brief Specialized two-layer MLP trainer for widths 16, 32, and 64.
 *
 * This path bypasses the generic DSL/AD tape and emits raw GLSL for:
 *
 *   forward -> MSE loss -> backward gradient accumulation -> Adam update
 *
 * Matrix-vector math is statically unrolled at C++ codegen time. Each workgroup
 * stages W1/W2/biases into shared memory, then each invocation keeps activations
 * and adjoints in scalar locals. Gradients are accumulated with GPU atomics and
 * Adam state is stored in flat GPU buffers.
 */
template <typename T, size_t InFeatures, size_t HiddenFeatures, size_t OutFeatures> class FusedMLP2Trainer {
	static_assert(std::is_same_v<T, float>, "FusedMLP2Trainer only supports float");
	static_assert(detail::IsFusedWidth(InFeatures), "InFeatures must be 16, 32, or 64");
	static_assert(detail::IsFusedWidth(HiddenFeatures), "HiddenFeatures must be 16, 32, or 64");
	static_assert(detail::IsFusedWidth(OutFeatures), "OutFeatures must be 16, 32, or 64");

	static constexpr size_t	  W1Size		= HiddenFeatures * InFeatures;
	static constexpr size_t	  B1Size		= HiddenFeatures;
	static constexpr size_t	  W2Size		= OutFeatures * HiddenFeatures;
	static constexpr size_t	  B2Size		= OutFeatures;
	static constexpr size_t	  TotalParams	= W1Size + B1Size + W2Size + B2Size;
	static constexpr uint32_t WorkgroupSize = 128;

public:
	explicit FusedMLP2Trainer(unsigned seed = 42)
		: w1_(W1Size, Runtime::BufferMode::ReadWrite), b1_(B1Size, Runtime::BufferMode::ReadWrite),
		  w2_(W2Size, Runtime::BufferMode::ReadWrite), b2_(B2Size, Runtime::BufferMode::ReadWrite),
		  gw1_(W1Size, Runtime::BufferMode::ReadWrite), gb1_(B1Size, Runtime::BufferMode::ReadWrite),
		  gw2_(W2Size, Runtime::BufferMode::ReadWrite), gb2_(B2Size, Runtime::BufferMode::ReadWrite),
		  m_(TotalParams, Runtime::BufferMode::ReadWrite), v_(TotalParams, Runtime::BufferMode::ReadWrite),
		  loss_(std::vector<float>{0.0f}, Runtime::BufferMode::ReadWrite),
		  hyper_(std::vector<float>(7, 0.0f), Runtime::BufferMode::Read) {
		Reset(seed);
		ZeroOptimizerState();
	}

	~FusedMLP2Trainer() {
		Release();
	}

	FusedMLP2Trainer(const FusedMLP2Trainer &)					= delete;
	FusedMLP2Trainer	   &operator=(const FusedMLP2Trainer &) = delete;

	Runtime::Buffer<float> &W1() {
		return w1_;
	}
	Runtime::Buffer<float> &B1() {
		return b1_;
	}
	Runtime::Buffer<float> &W2() {
		return w2_;
	}
	Runtime::Buffer<float> &B2() {
		return b2_;
	}
	Runtime::Buffer<float> &LossBuffer() {
		return loss_;
	}

	static constexpr size_t ParameterCount() {
		return TotalParams;
	}

	void Reset(unsigned seed = 42) {
		std::vector<float> w1(W1Size), b1(B1Size, 0.0f), w2(W2Size), b2(B2Size, 0.0f);
		unsigned		   s = seed;
		for (size_t h = 0; h < HiddenFeatures; h++)
			for (size_t i = 0; i < InFeatures; i++)
				w1[h * InFeatures + i] = detail::FusedTrainXavier(s, InFeatures, HiddenFeatures);
		for (size_t o = 0; o < OutFeatures; o++)
			for (size_t h = 0; h < HiddenFeatures; h++)
				w2[o * HiddenFeatures + h] = detail::FusedTrainXavier(s, HiddenFeatures, OutFeatures);
		SetWeights(w1, b1, w2, b2);
	}

	void SetWeights(const std::vector<float> &w1, const std::vector<float> &b1, const std::vector<float> &w2,
					const std::vector<float> &b2) {
		if (w1.size() != W1Size || b1.size() != B1Size || w2.size() != W2Size || b2.size() != B2Size)
			throw std::invalid_argument("FusedMLP2Trainer::SetWeights size mismatch");
		w1_.Upload(w1);
		b1_.Upload(b1);
		w2_.Upload(w2);
		b2_.Upload(b2);
	}

	std::vector<float> DownloadW1() {
		return DownloadVector(w1_, W1Size);
	}
	std::vector<float> DownloadB1() {
		return DownloadVector(b1_, B1Size);
	}
	std::vector<float> DownloadW2() {
		return DownloadVector(w2_, W2Size);
	}
	std::vector<float> DownloadB2() {
		return DownloadVector(b2_, B2Size);
	}
	float DownloadLoss() {
		std::vector<float> out(1, 0.0f);
		loss_.Download(out);
		return out[0];
	}

	/**
	 * @brief Run forward inference: output shape is [batch, OutFeatures].
	 */
	void Forward(Runtime::Buffer<float> &input, Runtime::Buffer<float> &output, size_t batch, bool sync = false) {
		if (input.GetCount() < batch * InFeatures || output.GetCount() < batch * OutFeatures)
			throw std::out_of_range("FusedMLP2Trainer::Forward buffer is too small");
		EnsureForwardPipeline(batch);
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			throw std::runtime_error("FusedMLP2Trainer backend not available");

		Backend::ResourceBinding bindings[6] = {};
		Bind(bindings[0], 0, input.GetHandle(), true);
		Bind(bindings[1], 1, output.GetHandle(), false);
		Bind(bindings[2], 2, w1_.GetHandle(), true);
		Bind(bindings[3], 3, b1_.GetHandle(), true);
		Bind(bindings[4], 4, w2_.GetHandle(), true);
		Bind(bindings[5], 5, b2_.GetHandle(), true);

		backend->BindPipeline(forwardPipeline_);
		backend->BindResources(bindings, 6);
		backend->Dispatch(static_cast<uint32_t>((batch + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
		backend->MemoryBarrier(Backend::BarrierType::Buffer);
		if (sync)
			backend->Finish();
	}

	/**
	 * @brief Run one fused MSE training step.
	 *
	 * The step uses three dispatches: clear loss/grad buffers, accumulate
	 * unrolled MLP gradients, then update all parameters with one Adam shader.
	 */
	void TrainMSE(Runtime::Buffer<float> &input, Runtime::Buffer<float> &target, size_t batch, float lr = 0.001f,
				  float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, bool sync = false) {
		if (batch == 0)
			throw std::invalid_argument("FusedMLP2Trainer::TrainMSE batch must be > 0");
		if (input.GetCount() < batch * InFeatures || target.GetCount() < batch * OutFeatures)
			throw std::out_of_range("FusedMLP2Trainer::TrainMSE buffer is too small");

		step_++;
		UploadHyper(batch, lr, beta1, beta2, eps);
		EnsureClearPipeline();
		EnsureTrainPipeline(batch);
		EnsureUpdatePipeline();

		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			throw std::runtime_error("FusedMLP2Trainer backend not available");

		DispatchClear(sync);
		DispatchTrain(input, target, batch, sync);
		DispatchUpdate(sync);
	}

	static std::string ForwardShaderSource(size_t batch) {
		return BuildForwardShader(batch);
	}
	static std::string TrainingShaderSource(size_t batch) {
		return BuildTrainShader(batch);
	}
	static std::string UpdateShaderSource() {
		return BuildUpdateShader();
	}

private:
	static std::vector<float> DownloadVector(Runtime::Buffer<float> &buffer, size_t count) {
		std::vector<float> out(count, 0.0f);
		buffer.Download(out);
		return out;
	}

	void ZeroOptimizerState() {
		std::vector<float> zeros(TotalParams, 0.0f);
		m_.Upload(zeros);
		v_.Upload(zeros);
		std::vector<float> zw1(W1Size, 0.0f), zb1(B1Size, 0.0f), zw2(W2Size, 0.0f), zb2(B2Size, 0.0f);
		gw1_.Upload(zw1);
		gb1_.Upload(zb1);
		gw2_.Upload(zw2);
		gb2_.Upload(zb2);
	}

	static void Bind(Backend::ResourceBinding &rb, uint32_t binding, Backend::BufferHandle handle, bool readOnly) {
		rb.binding	= binding;
		rb.type		= Backend::BindingType::Buffer;
		rb.buffer	= handle;
		rb.readOnly = readOnly;
	}

	static std::string Header(uint32_t workgroupSize = WorkgroupSize) {
		return std::format(R"GLSL(#version 430
layout(local_size_x = {}) in;

)GLSL",
						   workgroupSize);
	}

	static std::string AtomicFloatCAS() {
		return R"GLSL(
#define ATOMIC_ADD_FLOAT(target, value) do { \
    int _oldBits = target; \
    int _assumedBits; \
    do { \
        _assumedBits = _oldBits; \
        float _newValue = intBitsToFloat(_assumedBits) + (value); \
        _oldBits = atomicCompSwap(target, _assumedBits, floatBitsToInt(_newValue)); \
    } while (_oldBits != _assumedBits); \
} while (false)

)GLSL";
	}

	static void AppendShared(std::string &src) {
		src += std::format("shared float shW1[{}];\n", W1Size);
		src += std::format("shared float shB1[{}];\n", B1Size);
		src += std::format("shared float shW2[{}];\n", W2Size);
		src += std::format("shared float shB2[{}];\n\n", B2Size);
	}

	static void AppendSharedLoad(std::string &src) {
		src += "    uint lid = gl_LocalInvocationID.x;\n";
		src += std::format("    for (uint k = lid; k < {}u; k += {}u) shW1[k] = w1[k];\n", W1Size, WorkgroupSize);
		src += std::format("    for (uint k = lid; k < {}u; k += {}u) shB1[k] = b1[k];\n", B1Size, WorkgroupSize);
		src += std::format("    for (uint k = lid; k < {}u; k += {}u) shW2[k] = w2[k];\n", W2Size, WorkgroupSize);
		src += std::format("    for (uint k = lid; k < {}u; k += {}u) shB2[k] = b2[k];\n", B2Size, WorkgroupSize);
		src += "    barrier();\n\n";
	}

	static void AppendForwardMath(std::string &src, const std::string &inputBase, const std::string &outputBase,
								  bool writeOutput) {
		for (size_t i = 0; i < InFeatures; i++)
			src += std::format("    float x{} = inputData[{} + {}u];\n", i, inputBase, i);
		src += "\n";

		for (size_t h = 0; h < HiddenFeatures; h++) {
			src += std::format("    float pre{} = shB1[{}]", h, h);
			for (size_t i = 0; i < InFeatures; i++)
				src += std::format(" + shW1[{}] * x{}", h * InFeatures + i, i);
			src += ";\n";
			src += std::format("    float act{} = max(pre{}, 0.0);\n", h, h);
		}
		src += "\n";

		for (size_t o = 0; o < OutFeatures; o++) {
			src += std::format("    float y{} = shB2[{}]", o, o);
			for (size_t h = 0; h < HiddenFeatures; h++)
				src += std::format(" + shW2[{}] * act{}", o * HiddenFeatures + h, h);
			src += ";\n";
			if (writeOutput)
				src += std::format("    outData[{} + {}u] = y{};\n", outputBase, o, o);
		}
	}

	static std::string BuildForwardShader(size_t batch) {
		std::string src	 = Header();
		src				+= "layout(std430, binding = 0) readonly buffer InputBuf { float inputData[]; };\n";
		src				+= "layout(std430, binding = 1) buffer OutputBuf { float outData[]; };\n";
		src				+= "layout(std430, binding = 2) readonly buffer W1Buf { float w1[]; };\n";
		src				+= "layout(std430, binding = 3) readonly buffer B1Buf { float b1[]; };\n";
		src				+= "layout(std430, binding = 4) readonly buffer W2Buf { float w2[]; };\n";
		src				+= "layout(std430, binding = 5) readonly buffer B2Buf { float b2[]; };\n";
		AppendShared(src);
		src += "void main() {\n";
		AppendSharedLoad(src);
		src += "    uint sid = gl_GlobalInvocationID.x;\n";
		src += std::format("    if (sid >= {}u) return;\n", batch);
		src += std::format("    uint inBase = sid * {}u;\n", InFeatures);
		src += std::format("    uint outBase = sid * {}u;\n\n", OutFeatures);
		AppendForwardMath(src, "inBase", "outBase", true);
		src += "}\n";
		return src;
	}

	static std::string BuildTrainShader(size_t batch) {
		std::string src	 = Header();
		src				+= "layout(std430, binding = 0) readonly buffer InputBuf { float inputData[]; };\n";
		src				+= "layout(std430, binding = 1) readonly buffer TargetBuf { float target[]; };\n";
		src				+= "layout(std430, binding = 2) readonly buffer W1Buf { float w1[]; };\n";
		src				+= "layout(std430, binding = 3) readonly buffer B1Buf { float b1[]; };\n";
		src				+= "layout(std430, binding = 4) readonly buffer W2Buf { float w2[]; };\n";
		src				+= "layout(std430, binding = 5) readonly buffer B2Buf { float b2[]; };\n";
		src				+= "layout(std430, binding = 6) buffer GW1BufInt { int gw1_i[]; };\n";
		src				+= "layout(std430, binding = 7) buffer GB1BufInt { int gb1_i[]; };\n";
		src				+= "layout(std430, binding = 8) buffer GW2BufInt { int gw2_i[]; };\n";
		src				+= "layout(std430, binding = 9) buffer GB2BufInt { int gb2_i[]; };\n";
		src				+= "layout(std430, binding = 10) buffer LossBufInt { int loss_i[]; };\n";
		src				+= "layout(std430, binding = 11) readonly buffer HyperBuf { float h[]; };\n";
		src				+= AtomicFloatCAS();
		AppendShared(src);
		src += "void main() {\n";
		AppendSharedLoad(src);
		src += "    uint sid = gl_GlobalInvocationID.x;\n";
		src += std::format("    if (sid >= {}u) return;\n", batch);
		src += std::format("    uint inBase = sid * {}u;\n", InFeatures);
		src += std::format("    uint outBase = sid * {}u;\n\n", OutFeatures);
		AppendForwardMath(src, "inBase", "outBase", false);
		src += "\n";

		for (size_t h = 0; h < HiddenFeatures; h++)
			src += std::format("    float dAct{} = 0.0;\n", h);
		src += "\n";

		for (size_t o = 0; o < OutFeatures; o++) {
			src += std::format("    float diff{} = y{} - target[outBase + {}u];\n", o, o, o);
			src += std::format("    ATOMIC_ADD_FLOAT(loss_i[0], 0.5 * diff{} * diff{} * h[6]);\n", o, o);
			src += std::format("    float dY{} = diff{} * h[6];\n", o, o);
			src += std::format("    ATOMIC_ADD_FLOAT(gb2_i[{}], dY{});\n", o, o);
			for (size_t h = 0; h < HiddenFeatures; h++) {
				src += std::format("    ATOMIC_ADD_FLOAT(gw2_i[{}], dY{} * act{});\n", o * HiddenFeatures + h, o, h);
				src += std::format("    dAct{} += dY{} * shW2[{}];\n", h, o, o * HiddenFeatures + h);
			}
		}
		src += "\n";

		for (size_t h = 0; h < HiddenFeatures; h++) {
			src += std::format("    float dPre{} = pre{} > 0.0 ? dAct{} : 0.0;\n", h, h, h);
			src += std::format("    ATOMIC_ADD_FLOAT(gb1_i[{}], dPre{});\n", h, h);
			for (size_t i = 0; i < InFeatures; i++)
				src += std::format("    ATOMIC_ADD_FLOAT(gw1_i[{}], dPre{} * x{});\n", h * InFeatures + i, h, i);
		}

		src += "}\n";
		return src;
	}

	static std::string BuildClearShader() {
		std::string src	 = Header();
		src				+= "layout(std430, binding = 0) buffer GW1Buf { float gw1[]; };\n";
		src				+= "layout(std430, binding = 1) buffer GB1Buf { float gb1[]; };\n";
		src				+= "layout(std430, binding = 2) buffer GW2Buf { float gw2[]; };\n";
		src				+= "layout(std430, binding = 3) buffer GB2Buf { float gb2[]; };\n";
		src				+= "layout(std430, binding = 4) buffer LossBuf { float loss[]; };\n";
		src				+= "void main() {\n";
		src				+= "    uint i = gl_GlobalInvocationID.x;\n";
		src				+= std::format("    if (i < {}u) gw1[i] = 0.0;\n", W1Size);
		src				+= std::format("    if (i < {}u) gb1[i] = 0.0;\n", B1Size);
		src				+= std::format("    if (i < {}u) gw2[i] = 0.0;\n", W2Size);
		src				+= std::format("    if (i < {}u) gb2[i] = 0.0;\n", B2Size);
		src				+= "    if (i == 0u) loss[0] = 0.0;\n";
		src				+= "}\n";
		return src;
	}

	static std::string BuildUpdateShader() {
		std::string src	 = Header();
		src				+= "layout(std430, binding = 0) buffer W1Buf { float w1[]; };\n";
		src				+= "layout(std430, binding = 1) buffer B1Buf { float b1[]; };\n";
		src				+= "layout(std430, binding = 2) buffer W2Buf { float w2[]; };\n";
		src				+= "layout(std430, binding = 3) buffer B2Buf { float b2[]; };\n";
		src				+= "layout(std430, binding = 4) buffer GW1Buf { float gw1[]; };\n";
		src				+= "layout(std430, binding = 5) buffer GB1Buf { float gb1[]; };\n";
		src				+= "layout(std430, binding = 6) buffer GW2Buf { float gw2[]; };\n";
		src				+= "layout(std430, binding = 7) buffer GB2Buf { float gb2[]; };\n";
		src				+= "layout(std430, binding = 8) buffer M { float m[]; };\n";
		src				+= "layout(std430, binding = 9) buffer V { float v[]; };\n";
		src				+= "layout(std430, binding = 10) readonly buffer HyperBuf { float h[]; };\n";
		src				+= R"GLSL(
void adam(inout float weight, inout float grad, uint idx) {
    float g = grad;
    m[idx] = h[1] * m[idx] + (1.0 - h[1]) * g;
    v[idx] = h[2] * v[idx] + (1.0 - h[2]) * g * g;
    float mh = m[idx] / h[4];
    float vh = v[idx] / h[5];
    weight -= h[0] * mh / (sqrt(vh) + h[3]);
    grad = 0.0;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= TOTAL_PARAMSu) return;
)GLSL";
		detail::ReplaceAll(src, "TOTAL_PARAMS", std::to_string(TotalParams));
		src += std::format("    if (i < {}u) {{ adam(w1[i], gw1[i], i); return; }}\n", W1Size);
		src += std::format("    if (i < {}u) {{ uint j = i - {}u; adam(b1[j], gb1[j], i); return; }}\n",
						   W1Size + B1Size, W1Size);
		src += std::format("    if (i < {}u) {{ uint j = i - {}u; adam(w2[j], gw2[j], i); return; }}\n",
						   W1Size + B1Size + W2Size, W1Size + B1Size);
		src += std::format("    uint j = i - {}u; adam(b2[j], gb2[j], i);\n", W1Size + B1Size + W2Size);
		src += "}\n";
		return src;
	}

	void UploadHyper(size_t batch, float lr, float beta1, float beta2, float eps) {
		const double	   bc1 = 1.0 - std::pow(beta1, step_);
		const double	   bc2 = 1.0 - std::pow(beta2, step_);
		std::vector<float> h   = {lr,
								  beta1,
								  beta2,
								  eps,
								  static_cast<float>(bc1),
								  static_cast<float>(bc2),
								  1.0f / static_cast<float>(batch * OutFeatures)};
		hyper_.Upload(h);
	}

	Backend::ShaderHandle CreateShader(const std::string &source) {
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			throw std::runtime_error("FusedMLP2Trainer backend not available");
		Backend::ShaderDesc shaderDesc;
		shaderDesc.type		  = Backend::ShaderType::Compute;
		shaderDesc.sourceCode = source;
		return backend->CreateShader(shaderDesc);
	}

	Backend::PipelineHandle CreatePipeline(Backend::ShaderHandle shader, uint32_t bindings, uint32_t readOnlyMask) {
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			throw std::runtime_error("FusedMLP2Trainer backend not available");
		Backend::PipelineDesc pipelineDesc;
		pipelineDesc.computeShader	= shader;
		pipelineDesc.workGroupSizeX = WorkgroupSize;
		for (uint32_t i = 0; i < bindings; i++)
			pipelineDesc.resources.push_back(
				{i, Backend::BindingType::Buffer, Backend::PixelFormat::RGBA8, (readOnlyMask & (1u << i)) != 0});
		auto pipeline = backend->CreatePipeline(pipelineDesc);
		if (pipeline == Backend::INVALID_PIPELINE_HANDLE)
			throw std::runtime_error("FusedMLP2Trainer failed to create pipeline");
		return pipeline;
	}

	void EnsureForwardPipeline(size_t batch) {
		if (forwardPipeline_ != Backend::INVALID_PIPELINE_HANDLE && forwardBatch_ == batch)
			return;
		Runtime::Context::GetInstance().MakeCurrent();
		auto *backend = Runtime::Context::GetBackend();
		if (forwardPipeline_ != Backend::INVALID_PIPELINE_HANDLE)
			backend->DestroyPipeline(forwardPipeline_);
		if (forwardShader_ != Backend::INVALID_SHADER_HANDLE)
			backend->DestroyShader(forwardShader_);
		forwardShader_	 = CreateShader(BuildForwardShader(batch));
		forwardPipeline_ = CreatePipeline(forwardShader_, 6, 0b111101u);
		forwardBatch_	 = batch;
	}

	void EnsureClearPipeline() {
		if (clearPipeline_ != Backend::INVALID_PIPELINE_HANDLE)
			return;
		Runtime::Context::GetInstance().MakeCurrent();
		clearShader_   = CreateShader(BuildClearShader());
		clearPipeline_ = CreatePipeline(clearShader_, 5, 0u);
	}

	void EnsureTrainPipeline(size_t batch) {
		if (trainPipeline_ != Backend::INVALID_PIPELINE_HANDLE && trainBatch_ == batch)
			return;
		Runtime::Context::GetInstance().MakeCurrent();
		auto *backend = Runtime::Context::GetBackend();
		if (trainPipeline_ != Backend::INVALID_PIPELINE_HANDLE)
			backend->DestroyPipeline(trainPipeline_);
		if (trainShader_ != Backend::INVALID_SHADER_HANDLE)
			backend->DestroyShader(trainShader_);
		trainShader_   = CreateShader(BuildTrainShader(batch));
		trainPipeline_ = CreatePipeline(trainShader_, 12, 0b100000111111u);
		trainBatch_	   = batch;
	}

	void EnsureUpdatePipeline() {
		if (updatePipeline_ != Backend::INVALID_PIPELINE_HANDLE)
			return;
		Runtime::Context::GetInstance().MakeCurrent();
		updateShader_	= CreateShader(BuildUpdateShader());
		updatePipeline_ = CreatePipeline(updateShader_, 11, 1u << 10);
	}

	void DispatchClear(bool sync) {
		auto					*backend	 = Runtime::Context::GetBackend();
		Backend::ResourceBinding bindings[5] = {};
		Bind(bindings[0], 0, gw1_.GetHandle(), false);
		Bind(bindings[1], 1, gb1_.GetHandle(), false);
		Bind(bindings[2], 2, gw2_.GetHandle(), false);
		Bind(bindings[3], 3, gb2_.GetHandle(), false);
		Bind(bindings[4], 4, loss_.GetHandle(), false);
		backend->BindPipeline(clearPipeline_);
		backend->BindResources(bindings, 5);
		backend->Dispatch(
			static_cast<uint32_t>((std::max({W1Size, B1Size, W2Size, B2Size}) + WorkgroupSize - 1) / WorkgroupSize), 1,
			1);
		backend->MemoryBarrier(Backend::BarrierType::Buffer);
		if (sync)
			backend->Finish();
	}

	void DispatchTrain(Runtime::Buffer<float> &input, Runtime::Buffer<float> &target, size_t batch, bool sync) {
		auto					*backend	  = Runtime::Context::GetBackend();
		Backend::ResourceBinding bindings[12] = {};
		Bind(bindings[0], 0, input.GetHandle(), true);
		Bind(bindings[1], 1, target.GetHandle(), true);
		Bind(bindings[2], 2, w1_.GetHandle(), true);
		Bind(bindings[3], 3, b1_.GetHandle(), true);
		Bind(bindings[4], 4, w2_.GetHandle(), true);
		Bind(bindings[5], 5, b2_.GetHandle(), true);
		Bind(bindings[6], 6, gw1_.GetHandle(), false);
		Bind(bindings[7], 7, gb1_.GetHandle(), false);
		Bind(bindings[8], 8, gw2_.GetHandle(), false);
		Bind(bindings[9], 9, gb2_.GetHandle(), false);
		Bind(bindings[10], 10, loss_.GetHandle(), false);
		Bind(bindings[11], 11, hyper_.GetHandle(), true);
		backend->BindPipeline(trainPipeline_);
		backend->BindResources(bindings, 12);
		backend->Dispatch(static_cast<uint32_t>((batch + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
		backend->MemoryBarrier(Backend::BarrierType::Buffer);
		if (sync)
			backend->Finish();
	}

	void DispatchUpdate(bool sync) {
		auto					*backend	  = Runtime::Context::GetBackend();
		Backend::ResourceBinding bindings[11] = {};
		Bind(bindings[0], 0, w1_.GetHandle(), false);
		Bind(bindings[1], 1, b1_.GetHandle(), false);
		Bind(bindings[2], 2, w2_.GetHandle(), false);
		Bind(bindings[3], 3, b2_.GetHandle(), false);
		Bind(bindings[4], 4, gw1_.GetHandle(), false);
		Bind(bindings[5], 5, gb1_.GetHandle(), false);
		Bind(bindings[6], 6, gw2_.GetHandle(), false);
		Bind(bindings[7], 7, gb2_.GetHandle(), false);
		Bind(bindings[8], 8, m_.GetHandle(), false);
		Bind(bindings[9], 9, v_.GetHandle(), false);
		Bind(bindings[10], 10, hyper_.GetHandle(), true);
		backend->BindPipeline(updatePipeline_);
		backend->BindResources(bindings, 11);
		backend->Dispatch(static_cast<uint32_t>((TotalParams + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
		backend->MemoryBarrier(Backend::BarrierType::Buffer);
		if (sync)
			backend->Finish();
	}

	void Release() {
		auto *backend = Runtime::Context::GetBackend();
		if (!backend)
			return;
		auto destroy = [&](Backend::ShaderHandle &shader, Backend::PipelineHandle &pipeline) {
			if (pipeline != Backend::INVALID_PIPELINE_HANDLE) {
				backend->DestroyPipeline(pipeline);
				pipeline = Backend::INVALID_PIPELINE_HANDLE;
			}
			if (shader != Backend::INVALID_SHADER_HANDLE) {
				backend->DestroyShader(shader);
				shader = Backend::INVALID_SHADER_HANDLE;
			}
		};
		destroy(forwardShader_, forwardPipeline_);
		destroy(clearShader_, clearPipeline_);
		destroy(trainShader_, trainPipeline_);
		destroy(updateShader_, updatePipeline_);
	}

	Runtime::Buffer<float>	w1_, b1_, w2_, b2_;
	Runtime::Buffer<float>	gw1_, gb1_, gw2_, gb2_;
	Runtime::Buffer<float>	m_, v_, loss_, hyper_;
	int						step_			 = 0;

	size_t					forwardBatch_	 = 0;
	size_t					trainBatch_		 = 0;
	Backend::ShaderHandle	forwardShader_	 = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle forwardPipeline_ = Backend::INVALID_PIPELINE_HANDLE;
	Backend::ShaderHandle	clearShader_	 = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle clearPipeline_	 = Backend::INVALID_PIPELINE_HANDLE;
	Backend::ShaderHandle	trainShader_	 = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle trainPipeline_	 = Backend::INVALID_PIPELINE_HANDLE;
	Backend::ShaderHandle	updateShader_	 = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle updatePipeline_	 = Backend::INVALID_PIPELINE_HANDLE;
};

} // namespace GPU::NN

#endif // EASYGPU_NN_FUSED_MLP_TRAINING_H
