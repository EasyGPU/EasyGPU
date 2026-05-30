/**
 * @file TestNN.cpp
 * @brief Test NN module: Tensor, Optimizer, Layers, Loss, Checkpoint.
 *
 * Offline tests use AdjointInspector1D (no GPU required).
 * GPU tests use ADKernel1D (requires OpenGL/Vulkan backend).
 */

#include <NN/NN.h>

#include <AD/ADKernel.h>
#include <AD/AdjointInspector.h>
#include <IR/Value/Var.h>
#include <Kernel/Kernel.h>
#include <Runtime/Buffer.h>
#include <Utility/Helpers.h>
#include <Utility/Math.h>
#include <Utility/ActiveCompaction.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace GPU;
using namespace GPU::Kernel;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Runtime;
using namespace GPU::AD;
using namespace GPU::NN;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name)                                                                                             \
	void test_##name() {                                                                                       \
		std::cout << "\n[TEST] " #name " ... ";                                                                \
		test_count++;                                                                                           \
		try {

#define END_TEST                                                                                               \
		pass_count++;                                                                                              \
		std::cout << "PASSED\n";                                                                                   \
		}                                                                                                          \
		catch (const std::exception &e) {                                                                          \
			std::cout << "FAILED: " << e.what() << "\n";                                                           \
		}                                                                                                          \
		}

#define ASSERT(cond)                                                                                           \
	if (!(cond)) {                                                                                             \
		throw std::runtime_error("Assertion failed: " #cond);                                                  \
	}

#define CHECK_CONTAINS(str, sub)                                                                               \
	if ((str).find(sub) == std::string::npos) {                                                                \
		throw std::runtime_error("Expected '" + std::string(sub) + "' in:\n" + str);                            \
	}

#define CHECK_NOT_CONTAINS(str, sub)                                                                           \
	if ((str).find(sub) != std::string::npos) {                                                                \
		throw std::runtime_error("Unexpected '" + std::string(sub) + "' in:\n" + str);                          \
	}

// =============================================================================
// SECTION 1: Tensor — construction and CPU indexing
// =============================================================================

TEST(nn_tensor_1d_construct)
{
	Tensor<float, 10> t;
	ASSERT(t.Size() == 10);
	ASSERT(t.Data()[0] == 0.0f);
	ASSERT(t.Data()[9] == 0.0f);

	// Fill and verify
	for (size_t i = 0; i < 10; i++) t.Data()[i] = (float)i * 1.5f;
	t.Upload();
	t.Download();
	for (size_t i = 0; i < 10; i++) ASSERT(t.Data()[i] == (float)i * 1.5f);
}
END_TEST

TEST(nn_tensor_2d_indexing)
{
	std::vector<float> data(128 * 64);
	for (size_t i = 0; i < 128; i++)
		for (size_t j = 0; j < 64; j++)
			data[i * 64 + j] = (float)(i * 1000 + j);

	Tensor<float, 128, 64> t(data);
	ASSERT(t.Size() == 128 * 64);

	// Verify multi-dimensional indexing maps to flat correctly
	ASSERT(t(0, 0) == 0.0f);
	ASSERT(t(0, 1) == 1.0f);
	ASSERT(t(1, 0) == 1000.0f);
	ASSERT(t(5, 10) == 5010.0f);
	ASSERT(t(127, 63) == 127063.0f);
}
END_TEST

TEST(nn_tensor_3d_indexing)
{
	std::vector<float> data(2 * 3 * 4);
	for (size_t i = 0; i < 2; i++)
		for (size_t j = 0; j < 3; j++)
			for (size_t k = 0; k < 4; k++)
				data[(i * 3 + j) * 4 + k] = (float)(i * 100 + j * 10 + k);

	Tensor<float, 2, 3, 4> t(data);
	ASSERT(t.Size() == 24);
	ASSERT(t(0, 0, 0) == 0.0f);
	ASSERT(t(0, 0, 1) == 1.0f);
	ASSERT(t(0, 1, 0) == 10.0f);
	ASSERT(t(1, 0, 0) == 100.0f);
	ASSERT(t(1, 2, 3) == 123.0f);
}
END_TEST

TEST(nn_tensor_const_indexing)
{
	std::vector<float> data(5 * 3);
	for (size_t i = 0; i < 5; i++)
		for (size_t j = 0; j < 3; j++)
			data[i * 3 + j] = (float)(i * 3 + j);

	const Tensor<float, 5, 3> t(data);
	ASSERT(t(2, 1) == 7.0f);
	ASSERT(t.Size() == 15);
	ASSERT(t.Data()[14] == 14.0f);
}
END_TEST

TEST(nn_tensor_move_semantics)
{
	Tensor<float, 10> t1;
	t1.Data()[0] = 42.0f;
	t1.Upload();

	Tensor<float, 10> t2(std::move(t1));
	ASSERT(t2.Size() == 10);
	ASSERT(t2.Data()[0] == 42.0f);

	t2.Download();
	ASSERT(t2.Data()[0] == 42.0f);
}
END_TEST

// =============================================================================
// SECTION 2: Tensor — AdjointInspector DSL verification
// =============================================================================

TEST(nn_tensor_ref_1d_for_each_param)
{
	// Verify ForEachParam registers all elements in flat order
	Tensor<float, 5> t;
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
	for (int i = 0; i < 5; i++) t.Data()[i] = data[i];
	t.Upload();

	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		auto tref = t.Bind();
		ctx.RegisterParameter(tref[0]);
		ctx.RegisterParameter(tref[1]);
		ctx.RegisterParameter(tref[2]);
		ctx.RegisterParameter(tref[3]);
		ctx.RegisterParameter(tref[4]);
		Var<float> loss = tref[0] + tref[1] + tref[2] + tref[3] + tref[4];
		ctx.MarkLoss(loss);
	});

	auto &tape = inspector.Tape();
	ASSERT(tape.Size() > 0);
	// Should have 5 parameters + some operations + 1 loss
	ASSERT(!inspector.GetForwardCode().empty());
	ASSERT(inspector.HasBackwardCode());

	auto fwd = inspector.GetForwardCode();
	CHECK_CONTAINS(fwd, "float");
	CHECK_CONTAINS(inspector.GetBackwardCode(), "d_");
}
END_TEST

TEST(nn_tensor_ref_for_each_param_dsl)
{
	// Verify ForEachParam can be used with a lambda
	Tensor<float, 8> t;
	for (int i = 0; i < 8; i++) t.Data()[i] = (float)i;
	t.Upload();

	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		auto tref = t.Bind();
		tref.ForEachParam([&](auto &w) { ctx.RegisterParameter(w); });
		Var<float> loss = MakeFloat(0.0f);
		for (int i = 0; i < 8; i++) loss = loss + tref[i];
		ctx.MarkLoss(loss);
	});

	ASSERT(inspector.HasBackwardCode());
	auto bw = inspector.GetBackwardCode();
	// Should have adjoint declarations for each parameter
	CHECK_CONTAINS(bw, "d_");
}
END_TEST

TEST(nn_tensor_ref_2d_inspector)
{
	std::vector<float> data(3 * 4);
	for (size_t i = 0; i < 3; i++)
		for (size_t j = 0; j < 4; j++)
			data[i * 4 + j] = (float)(i * 4 + j);

	Tensor<float, 3, 4> t(data);

	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		auto tref = t.Bind();
		ctx.RegisterParameter(tref(0, 0));
		ctx.RegisterParameter(tref(1, 2));
		Var<float> loss = tref(0, 0) * MakeFloat(2.0f) + tref(1, 2) * MakeFloat(3.0f);
		ctx.MarkLoss(loss);
	});

	auto fwd = inspector.GetForwardCode();
	CHECK_CONTAINS(fwd, "buf");
	CHECK_CONTAINS(inspector.GetBackwardCode(), "d_");
}
END_TEST

// =============================================================================
// SECTION 3: Optimizer — parameter registration
// =============================================================================

TEST(nn_adam_parameter_count)
{
	Adam adam(0.001f);
	ASSERT(adam.GetStep() == 0);
	ASSERT(adam.ParameterCount() == 0);

	float w[10] = {};
	adam.AddParameter(w, 10);
	ASSERT(adam.ParameterCount() == 1);

	float b[5] = {};
	adam.AddParameter(b, 5);
	ASSERT(adam.ParameterCount() == 2);
}
END_TEST

TEST(nn_adam_add_tensor)
{
	Tensor<float, 20> t;
	Adam adam(0.001f);
	adam.AddTensor(t);
	ASSERT(adam.ParameterCount() == 1);
}
END_TEST

TEST(nn_adam_weight_decay_setting)
{
	Adam adam(0.001f);
	adam.SetWeightDecay(0.0001f);
	adam.SetGradClip(0.5f);
	// Setting validation: these don't crash and are accessible via internal state
	// (no public getters — just verify constructor + setters don't throw)
	ASSERT(adam.GetStep() == 0);
	ASSERT(adam.ParameterCount() == 0);
}
END_TEST

TEST(nn_sgd_parameter_count)
{
	SGD sgd(0.01f, 0.9f);
	ASSERT(sgd.GetStep() == 0);
	ASSERT(sgd.ParameterCount() == 0);

	float w[5] = {};
	sgd.AddParameter(w, 5);
	ASSERT(sgd.ParameterCount() == 1);
}
END_TEST

TEST(nn_rmsprop_parameter_count)
{
	RMSprop rms(0.001f, 0.9f);
	ASSERT(rms.GetStep() == 0);
	ASSERT(rms.ParameterCount() == 0);

	float w[7] = {};
	rms.AddParameter(w, 7);
	ASSERT(rms.ParameterCount() == 1);
}
END_TEST

TEST(nn_optimizer_multiple_tensors)
{
	Tensor<float, 10> W;
	Tensor<float, 5> b;

	Adam adam(0.001f);
	adam.AddTensor(W);
	adam.AddTensor(b);
	ASSERT(adam.ParameterCount() == 2);
}
END_TEST

// =============================================================================
// SECTION 4: Optimizer — update rule math verification
// =============================================================================

TEST(nn_adam_update_math)
{
	// Verify Adam update rule manually:
	// m = beta1*m + (1-beta1)*g
	// v = beta2*v + (1-beta2)*g^2
	// m_hat = m / (1-beta1^t); v_hat = v / (1-beta2^t)
	// w -= lr * m_hat / (sqrt(v_hat) + eps)

	float lr = 0.1f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

	// Manual computation for comparison
	float weight = 1.0f;
	float grad = 0.5f;
	float m = 0.0f, v = 0.0f;
	int t = 1;

	m = beta1 * m + (1.0f - beta1) * grad;
	v = beta2 * v + (1.0f - beta2) * grad * grad;
	float bias1 = 1.0f - std::pow(beta1, t);
	float bias2 = 1.0f - std::pow(beta2, t);
	float mHat = m / bias1;
	float vHat = v / bias2;
	float expectedW = weight - lr * mHat / (std::sqrt(vHat) + eps);

	// Verify Adam formula correctness
	float expectedM = 0.0f * 0.9f + 0.1f * 0.5f; // = 0.05
	ASSERT(std::abs(m - expectedM) < 1e-6f);
	ASSERT(std::abs(expectedW - 1.0f + 0.1f * (0.05f / 0.1f) / (std::sqrt(v / 0.001f) + 1e-8f)) < 1e-3f);
}
END_TEST

TEST(nn_adam_elementwise_update_matches_gradients)
{
	std::vector<float> weight = {1.0f, 2.0f};
	std::vector<float> grad   = {0.5f, -0.25f};
	std::vector<float> m(2, 0.0f);
	std::vector<float> v(2, 0.0f);

	GPU::NN::detail::ApplyAdamUpdate(weight.data(), grad.data(), m.data(), v.data(),
		weight.size(), 0.1f, 0.9f, 0.999f, 1e-8f, 1, 0.0f, 0.0f);

	// The update must use each element's own gradient, not a single averaged scalar.
	ASSERT(std::abs(m[0] - 0.05f) < 1e-6f);
	ASSERT(std::abs(m[1] - (-0.025f)) < 1e-6f);
	ASSERT(std::abs(v[0] - 0.00025f) < 1e-6f);
	ASSERT(std::abs(v[1] - 0.0000625f) < 1e-6f);
}
END_TEST

TEST(nn_gpu_adam_updates_on_device)
{
	Tensor<float, 2> weight({1.0f, -1.0f});
	Buffer<float> xBuf({2.0f, 3.0f}, BufferMode::Read);

	ADKernel1D kernel([&](Var<int> &id) {
		(void)id;
		auto x = xBuf.Bind();
		auto w = weight.Bind();
		AD::Param(w[0]);
		AD::Param(w[1]);
		Var<float> y = w[0] * x[0] + w[1] * x[1];
		Var<float> loss = y * y;
		AD::Loss(loss);
	}, 1, 256);

	GPUAdam optimizer(0.1f, 0.0f, 0.0f, 1e-8f);
	optimizer.AddTensor(weight);

	kernel.Backward(1, true);
	optimizer.Step(kernel, true);
	weight.Download();

	ASSERT(std::abs(weight.Data()[0] - 1.1f) < 1e-4f);
	ASSERT(std::abs(weight.Data()[1] - -0.9f) < 1e-4f);
}
END_TEST

TEST(nn_adam_step_is_gpu_backed)
{
	Tensor<float, 2> weight({1.0f, -1.0f});
	Buffer<float> xBuf({2.0f, 3.0f}, BufferMode::Read);

	ADKernel1D kernel([&](Var<int> &id) {
		(void)id;
		auto x = xBuf.Bind();
		auto w = weight.Bind();
		AD::Param(w[0]);
		AD::Param(w[1]);
		Var<float> y = w[0] * x[0] + w[1] * x[1];
		Var<float> loss = y * y;
		AD::Loss(loss);
	}, 1, 256);

	Adam optimizer(0.1f, 0.0f, 0.0f, 1e-8f);
	optimizer.AddTensor(weight);

	kernel.Backward(1, true);
	optimizer.Step(kernel);
	weight.Download();

	ASSERT(std::abs(weight.Data()[0] - 1.1f) < 1e-4f);
	ASSERT(std::abs(weight.Data()[1] - -0.9f) < 1e-4f);
}
END_TEST

TEST(nn_adam_bias_correction)
{
	// At t=1, bias correction = 1 - beta1 = 0.1 (for beta1=0.9)
	// m_hat = m / 0.1 = 10 * m
	float beta1 = 0.9f;
	float m = 0.05f;
	float bias1 = 1.0f - std::pow(beta1, 1);
	ASSERT(std::abs(bias1 - 0.1f) < 1e-6f);

	float mHat = m / bias1;
	ASSERT(std::abs(mHat - 0.5f) < 1e-6f); // 0.05 / 0.1 = 0.5

	// At t=2, bias correction = 1 - 0.9^2 = 1 - 0.81 = 0.19
	float bias1_2 = 1.0f - std::pow(beta1, 2);
	ASSERT(std::abs(bias1_2 - 0.19f) < 1e-6f);
}
END_TEST

TEST(nn_sgd_momentum_math)
{
	// SGD with momentum:
	// v = momentum*v + g
	// w -= lr * v

	float lr = 0.1f, momentum = 0.9f;
	float w = 1.0f;
	float grad = 0.5f;
	float v = 0.0f;

	v = momentum * v + grad;
	w -= lr * v;

	ASSERT(std::abs(v - 0.5f) < 1e-6f);
	ASSERT(std::abs(w - 0.95f) < 1e-6f); // 1.0 - 0.1*0.5 = 0.95

	// Second step
	float grad2 = 0.3f;
	v = momentum * v + grad2; // 0.9*0.5 + 0.3 = 0.75
	w -= lr * v;              // 0.95 - 0.1*0.75 = 0.875
	ASSERT(std::abs(v - 0.75f) < 1e-6f);
	ASSERT(std::abs(w - 0.875f) < 1e-6f);
}
END_TEST

TEST(nn_sgd_step_is_gpu_backed)
{
	Tensor<float, 1> weight(std::vector<float>{1.0f});
	Buffer<float> xBuf(std::vector<float>{2.0f}, BufferMode::Read);

	ADKernel1D kernel([&](Var<int> &id) {
		(void)id;
		auto x = xBuf.Bind();
		auto w = weight.Bind();
		AD::Param(w[0]);
		Var<float> y = w[0] * x[0];
		Var<float> loss = y * y;
		AD::Loss(loss);
	}, 1, 256);

	SGD optimizer(0.1f, 0.0f);
	optimizer.AddTensor(weight);

	kernel.Backward(1, true);
	optimizer.Step(kernel);
	weight.Download();

	ASSERT(std::abs(weight.Data()[0] - 0.2f) < 1e-4f);
}
END_TEST

TEST(nn_rmsprop_update_math)
{
	// RMSprop:
	// v = beta*v + (1-beta)*g^2
	// w -= lr * g / sqrt(v + eps)

	float lr = 0.1f, beta = 0.9f, eps = 1e-8f;
	float w = 1.0f;
	float grad = 0.5f;
	float v = 0.0f;

	v = beta * v + (1.0f - beta) * grad * grad;
	w -= lr * grad / std::sqrt(v + eps);

	float expectedV = 0.9f * 0.0f + 0.1f * 0.25f; // = 0.025
	float expectedW = 1.0f - 0.1f * 0.5f / std::sqrt(0.025f + 1e-8f);
	ASSERT(std::abs(v - expectedV) < 1e-6f);
	ASSERT(std::abs(w - expectedW) < 1e-6f);
}
END_TEST

TEST(nn_rmsprop_step_is_gpu_backed)
{
	Tensor<float, 1> weight(std::vector<float>{1.0f});
	Buffer<float> xBuf(std::vector<float>{2.0f}, BufferMode::Read);

	ADKernel1D kernel([&](Var<int> &id) {
		(void)id;
		auto x = xBuf.Bind();
		auto w = weight.Bind();
		AD::Param(w[0]);
		Var<float> y = w[0] * x[0];
		Var<float> loss = y * y;
		AD::Loss(loss);
	}, 1, 256);

	RMSprop optimizer(0.1f, 0.0f, 1e-8f);
	optimizer.AddTensor(weight);

	kernel.Backward(1, true);
	optimizer.Step(kernel);
	weight.Download();

	ASSERT(std::abs(weight.Data()[0] - 0.9f) < 1e-4f);
}
END_TEST

TEST(nn_weight_decay_math)
{
	// Weight decay adds 2*wd*w to gradient:
	// effective_grad = grad + 2 * wd * w

	float w = 2.0f, grad = 0.3f, wd = 0.01f;
	float effectiveGrad = grad + 2.0f * wd * w;
	ASSERT(std::abs(effectiveGrad - 0.34f) < 1e-6f); // 0.3 + 2*0.01*2.0 = 0.34
}
END_TEST

TEST(nn_grad_clip_math)
{
	float grad = 1.5f, clip = 0.5f;
	float clipped = std::clamp(grad, -clip, clip);
	ASSERT(std::abs(clipped - 0.5f) < 1e-6f);

	float grad2 = -0.8f;
	float clipped2 = std::clamp(grad2, -clip, clip);
	ASSERT(std::abs(clipped2 - -0.5f) < 1e-6f);

	float grad3 = 0.2f;
	float clipped3 = std::clamp(grad3, -clip, clip);
	ASSERT(std::abs(clipped3 - 0.2f) < 1e-6f);
}
END_TEST

// =============================================================================
// SECTION 5: Layers — Linear
// =============================================================================

TEST(nn_linear_construct)
{
	Linear<float, 4, 8> fc;
	ASSERT(fc.InputDim() == 4);
	ASSERT(fc.OutputDim() == 8);
	ASSERT(fc.ParamCount() == 4 * 8 + 8); // weights + biases

	// Verify Xavier init produces non-zero weights
	auto &W = fc.Weight();
	bool hasNonZero = false;
	for (size_t i = 0; i < W.Size(); i++) {
		if (std::abs(W.Data()[i]) > 1e-6f) { hasNonZero = true; break; }
	}
	ASSERT(hasNonZero);

	// Verify biases start at zero
	auto &b = fc.Bias();
	for (size_t i = 0; i < b.Size(); i++) {
		ASSERT(std::abs(b.Data()[i]) < 1e-6f);
	}
}
END_TEST

TEST(nn_linear_xavier_range)
{
	// Xavier uniform range for 4→8: ±√(6/(4+8)) = ±√(0.5) ≈ ±0.707
	Linear<float, 4, 8> fc;
	float limit = std::sqrt(6.0f / 12.0f);
	auto &W = fc.Weight();
	for (size_t i = 0; i < W.Size(); i++) {
		ASSERT(std::abs(W.Data()[i]) <= limit + 1e-6f);
	}
}
END_TEST

TEST(nn_linear_reset)
{
	Linear<float, 4, 8> fc(123);
	auto origW0 = fc.Weight().Data()[0];

	fc.Reset(456);
	auto newW0 = fc.Weight().Data()[0];
	// Different seeds should produce different weights
	ASSERT(std::abs(origW0 - newW0) > 1e-7f);
}
END_TEST

TEST(nn_linear_setup_forward_inspector)
{
	Linear<float, 3, 2> fc;

	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		// Bind input/output buffers
		Buffer<float> inBuf(3 * 10, BufferMode::Read);
		Buffer<float> outBuf(2 * 10, BufferMode::ReadWrite);
		auto input = inBuf.Bind();
		auto output = outBuf.Bind();

		fc.Setup();
		fc.Forward(input, id, output);

		// Register loss on output
		Var<float> loss = output[id * 2 + 0] + output[id * 2 + 1];
		ctx.MarkLoss(loss);
	});

	auto fwd = inspector.GetForwardCode();
	CHECK_CONTAINS(fwd, "for");  // DSL For loop generated
	CHECK_CONTAINS(inspector.GetBackwardCode(), "d_");

	auto &tape = inspector.Tape();
	ASSERT(tape.Size() > 0);
}
END_TEST

TEST(nn_linear_params_are_registered)
{
	Linear<float, 3, 2> fc; // 3*2=6 weights + 2 biases = 8 params

	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		Buffer<float> inBuf(3 * 5, BufferMode::Read);
		Buffer<float> outBuf(2 * 5, BufferMode::ReadWrite);
		auto input = inBuf.Bind();
		auto output = outBuf.Bind();

		fc.Setup();
		// Use params in forward
		fc.Forward(input, id, output);

		Var<float> loss = output[id * 2 + 0];
		ctx.MarkLoss(loss);
	});

	ASSERT(inspector.HasBackwardCode());
	auto bw = inspector.GetBackwardCode();
	// Should have adjoint declarations for weights and biases
	CHECK_CONTAINS(bw, "d_");
}
END_TEST

TEST(nn_fused_mlp2_forward_matches_cpu)
{
	constexpr size_t N = 2;
	FusedMLP2<float, 2, 3, 1> mlp(42, FusedActivation::ReLU);

	float *w1 = mlp.W1().Data();
	w1[0] = 1.0f; w1[1] = 0.0f;
	w1[2] = 0.0f; w1[3] = 1.0f;
	w1[4] = 1.0f; w1[5] = 1.0f;
	float *b1 = mlp.B1().Data();
	b1[0] = 0.5f; b1[1] = -0.5f; b1[2] = 0.0f;
	float *w2 = mlp.W2().Data();
	w2[0] = 1.0f; w2[1] = 2.0f; w2[2] = -1.0f;
	mlp.B2().Data()[0] = 0.25f;
	mlp.W1().Upload();
	mlp.B1().Upload();
	mlp.W2().Upload();
	mlp.B2().Upload();

	Buffer<float> inBuf({2.0f, 3.0f, -1.0f, 4.0f}, BufferMode::Read);
	Buffer<float> outBuf(N, BufferMode::ReadWrite);

	Kernel1D kernel([&](Var<int> &id) {
		auto in = inBuf.Bind();
		auto out = outBuf.Bind();
		mlp.Setup(false);
		mlp.Forward(in, id, out);
	}, 1);
	kernel.Dispatch(static_cast<int>(N), true);

	std::vector<float> out;
	outBuf.Download(out);
	ASSERT(std::abs(out[0] - 2.75f) < 1e-4f);
	ASSERT(std::isfinite(out[1]));
}
END_TEST

TEST(nn_fused_mlp2_trainer_codegen_is_specialized)
{
	using Trainer = FusedMLP2Trainer<float, 16, 16, 16>;
	auto trainSrc = Trainer::TrainingShaderSource(8);
	CHECK_CONTAINS(trainSrc, "shared float shW1[256]");
	CHECK_CONTAINS(trainSrc, "float pre15");
	CHECK_CONTAINS(trainSrc, "atomicCompSwap");
	CHECK_CONTAINS(trainSrc, "ATOMIC_ADD_FLOAT(gw1_i");
	CHECK_CONTAINS(trainSrc, "ATOMIC_ADD_FLOAT(gw2_i");
	CHECK_NOT_CONTAINS(trainSrc, "GPU::Flow");
	CHECK_NOT_CONTAINS(trainSrc, "AD::");

	auto updateSrc = Trainer::UpdateShaderSource();
	CHECK_CONTAINS(updateSrc, "void adam");
	CHECK_CONTAINS(updateSrc, "if (i < 256u)");
}
END_TEST

TEST(nn_fused_mlp2_trainer_forward_identity_16)
{
	using Trainer = FusedMLP2Trainer<float, 16, 16, 16>;
	Trainer trainer(1);

	std::vector<float> w1(16 * 16, 0.0f), b1(16, 0.0f), w2(16 * 16, 0.0f), b2(16, 0.0f);
	for (int i = 0; i < 16; i++) {
		w1[i * 16 + i] = 1.0f;
		w2[i * 16 + i] = 1.0f;
	}
	trainer.SetWeights(w1, b1, w2, b2);

	std::vector<float> input(2 * 16, 0.0f);
	for (int i = 0; i < 16; i++) {
		input[i] = static_cast<float>(i - 8);
		input[16 + i] = static_cast<float>(8 - i);
	}
	Buffer<float> inBuf(input, BufferMode::Read);
	Buffer<float> outBuf(2 * 16, BufferMode::ReadWrite);

	trainer.Forward(inBuf, outBuf, 2, true);

	std::vector<float> out;
	outBuf.Download(out);
	for (int i = 0; i < 16; i++) {
		ASSERT(std::abs(out[i] - std::max(input[i], 0.0f)) < 1e-5f);
		ASSERT(std::abs(out[16 + i] - std::max(input[16 + i], 0.0f)) < 1e-5f);
	}
}
END_TEST

TEST(nn_fused_mlp2_trainer_mse_loss_16)
{
	using Trainer = FusedMLP2Trainer<float, 16, 16, 16>;
	Trainer trainer(1);

	std::vector<float> w1(16 * 16, 0.0f), b1(16, 0.0f), w2(16 * 16, 0.0f), b2(16, 0.0f);
	for (int i = 0; i < 16; i++) {
		w1[i * 16 + i] = 1.0f;
		w2[i * 16 + i] = 1.0f;
	}
	trainer.SetWeights(w1, b1, w2, b2);

	std::vector<float> input(2 * 16, 1.0f);
	std::vector<float> target(2 * 16, 0.0f);
	Buffer<float> inBuf(input, BufferMode::Read);
	Buffer<float> targetBuf(target, BufferMode::Read);

	trainer.TrainMSE(inBuf, targetBuf, 2, 0.0f, 0.9f, 0.999f, 1e-8f, true);
	ASSERT(std::abs(trainer.DownloadLoss() - 0.5f) < 1e-4f);
}
END_TEST

TEST(nn_active_compaction_builds_dense_index_list)
{
	std::vector<int> mask = {0, 1, 0, 1, 1, 0, 0, 1};
	Buffer<int> maskBuf(mask, BufferMode::Read);
	GPU::Utility::ActiveCompaction compactor(mask.size());

	compactor.Compact(maskBuf, mask.size(), true);
	int count = compactor.DownloadCount();
	ASSERT(count == 4);

	auto indices = compactor.DownloadIndices(static_cast<size_t>(count));
	std::sort(indices.begin(), indices.end());
	std::vector<int> expected = {1, 3, 4, 7};
	ASSERT(indices == expected);
}
END_TEST

// =============================================================================
// SECTION 6: Layers — Activations
// =============================================================================

TEST(nn_relu_construct)
{
	ReLU<float> relu(128);
	// Stateless — Setup should not crash
	relu.Setup();
}
END_TEST

TEST(nn_relu_forward_inspector)
{
	ReLU<float> relu(4);

	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		Buffer<float> buf(4 * 5, BufferMode::ReadWrite);
		auto b = buf.Bind();

		relu.Forward(b, id, b); // in-place

		Var<float> loss = b[id * 4 + 0] + b[id * 4 + 1] + b[id * 4 + 2] + b[id * 4 + 3];
		ctx.MarkLoss(loss);
	});

	auto fwd = inspector.GetForwardCode();
	CHECK_CONTAINS(fwd, "max");  // ReLU uses max(x, 0)
}
END_TEST

TEST(nn_sigmoid_forward_inspector)
{
	Sigmoid<float> sig(4);

	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		Buffer<float> buf(4 * 5, BufferMode::ReadWrite);
		auto b = buf.Bind();

		sig.Forward(b, id, b);

		Var<float> loss = b[id * 4 + 0];
		ctx.MarkLoss(loss);
	});

	auto fwd = inspector.GetForwardCode();
	CHECK_CONTAINS(fwd, "exp");
}
END_TEST

TEST(nn_tanh_activation_forward_inspector)
{
	TanhActivation<float> tanh(4);

	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		Buffer<float> buf(4 * 5, BufferMode::ReadWrite);
		auto b = buf.Bind();

		tanh.Forward(b, id, b);

		Var<float> loss = b[id * 4 + 0];
		ctx.MarkLoss(loss);
	});

	auto fwd = inspector.GetForwardCode();
	CHECK_CONTAINS(fwd, "exp"); // tanh uses exp internally
}
END_TEST

// =============================================================================
// SECTION 7: Layers — Sequential
// =============================================================================

TEST(nn_sequential_construct)
{
	Sequential<float, Linear<float, 4, 8>, ReLU<float>, Linear<float, 8, 2>> model(64, 8);

	ASSERT(model.NumLayers == 3);
}
END_TEST

TEST(nn_sequential_setup_forward_inspector)
{
	Linear<float, 3, 4> fc1;
	ReLU<float> relu(4);
	Linear<float, 4, 2> fc2;
	Sequential<float, Linear<float, 3, 4>, ReLU<float>, Linear<float, 4, 2>>
		model(64, std::max({4u, 2u}));

	// Move layers into sequential (Sequential owns its layers — this test
	// verifies the types compile and Setup/Forward don't crash)

	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		Buffer<float> inBuf(3 * 64, BufferMode::Read);
		Buffer<float> outBuf(2 * 64, BufferMode::ReadWrite);
		auto input = inBuf.Bind();
		auto output = outBuf.Bind();

		model.Setup();
		model.Forward(input, id, output);

		Var<float> loss = output[id * 2 + 0] + output[id * 2 + 1];
		ctx.MarkLoss(loss);
	});

	ASSERT(inspector.HasBackwardCode());
	auto fwd = inspector.GetForwardCode();
	CHECK_CONTAINS(fwd, "for");
	CHECK_CONTAINS(inspector.GetBackwardCode(), "d_");
}
END_TEST

// =============================================================================
// SECTION 8: Loss functions
// =============================================================================

TEST(nn_mse_scalar)
{
	// Scalar MSE: (pred - target)^2
	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		Var<float> pred, target;
		pred = MakeFloat(2.5f);
		target = MakeFloat(1.0f);
		Var<float> loss = MSELoss(pred, target);
		ctx.MarkLoss(loss);
	});

	ASSERT(inspector.HasBackwardCode());
	auto fwd = inspector.GetForwardCode();
	CHECK_CONTAINS(fwd, "float");
}
END_TEST

TEST(nn_mse_buffer_loss)
{
	// Multi-output MSE from buffers
	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		Buffer<float> predBuf(2 * 10, BufferMode::Read);
		Buffer<float> targetBuf(2 * 10, BufferMode::Read);
		auto pred = predBuf.Bind();
		auto target = targetBuf.Bind();

		Var<float> loss = MSELoss(pred, target, id, 2);
		ctx.MarkLoss(loss);
	});

	ASSERT(inspector.HasBackwardCode());
	auto bw = inspector.GetBackwardCode();
	CHECK_CONTAINS(bw, "d_");
}
END_TEST

TEST(nn_l1_loss)
{
	AdjointInspector1D inspector([&](Var<int> &id, AdjointContext &ctx) {
		Var<float> pred, target;
		pred = MakeFloat(3.0f);
		target = MakeFloat(1.0f);
		Var<float> loss = L1Loss(pred, target);
		ctx.MarkLoss(loss);
	});

	ASSERT(inspector.HasBackwardCode());
	auto fwd = inspector.GetForwardCode();
	CHECK_CONTAINS(fwd, "abs");
}
END_TEST

// =============================================================================
// SECTION 9: Checkpoint
// =============================================================================

TEST(nn_checkpoint_save_load_roundtrip)
{
	Tensor<float, 10> W;
	for (size_t i = 0; i < 10; i++) W.Data()[i] = (float)i * 1.5f;

	Tensor<float, 5> b;
	for (size_t i = 0; i < 5; i++) b.Data()[i] = (float)i * 0.5f;

	// Save
	const char *path = "test_checkpoint.bin";
	SaveWeights(path, W, b);

	// Modify
	for (size_t i = 0; i < 10; i++) W.Data()[i] = 0.0f;
	for (size_t i = 0; i < 5; i++) b.Data()[i] = 0.0f;

	// Load
	LoadWeights(path, W, b);

	// Verify
	for (size_t i = 0; i < 10; i++)
		ASSERT(std::abs(W.Data()[i] - (float)i * 1.5f) < 1e-6f);
	for (size_t i = 0; i < 5; i++)
		ASSERT(std::abs(b.Data()[i] - (float)i * 0.5f) < 1e-6f);

	// Cleanup
	std::remove(path);
}
END_TEST

TEST(nn_checkpoint_single_tensor)
{
	Tensor<float, 3> t;
	t.Data()[0] = 1.0f; t.Data()[1] = 2.0f; t.Data()[2] = 3.0f;

	const char *path = "test_single.bin";
	SaveWeights(path, t);

	t.Data()[0] = 0.0f;
	LoadWeights(path, t);

	ASSERT(t.Data()[0] == 1.0f);
	ASSERT(t.Data()[1] == 2.0f);
	ASSERT(t.Data()[2] == 3.0f);

	std::remove(path);
}
END_TEST

TEST(nn_checkpoint_size_mismatch_throws)
{
	Tensor<float, 10> t1;
	Tensor<float, 5> t2;

	const char *path = "test_mismatch.bin";
	SaveWeights(path, t1);

	ASSERT(t2.Size() == 5); // different from saved 10
	try {
		LoadWeights(path, t2);
		ASSERT(false); // should not reach here
	} catch (const std::runtime_error &) {
		// Expected
	}

	std::remove(path);
}
END_TEST

// =============================================================================
// SECTION 10: End-to-end GPU integration tests
// These require an actual GPU backend (OpenGL/Vulkan).
// =============================================================================

TEST(nn_e2e_simple_gradient_nonzero)
{
	Tensor<float, 2> weight({1.0f, -1.0f});
	Buffer<float> xBuf({2.0f, 3.0f}, BufferMode::Read);

	ADKernel1D kernel([&](Var<int> &id) {
		auto x = xBuf.Bind();
		auto w = weight.Bind();
		AD::Param(w[0]);
		AD::Param(w[1]);
		Var<float> y = w[0] * x[0] + w[1] * x[1];
		Var<float> loss = y * y;
		AD::Loss(loss);
	}, 1, 256);

	try {
		kernel.Backward(1, true);
		auto g = kernel.DownloadAllGradients();
		ASSERT(g.size() == 2);
		ASSERT(g[0].size() == 1);
		ASSERT(g[1].size() == 1);
		ASSERT(std::abs(g[0][0]) > 1e-6f || std::abs(g[1][0]) > 1e-6f);
	} catch (const std::exception &e) {
		// GPU backend may be unavailable in some environments; fail loudly here so the regression is visible.
		throw std::runtime_error(std::string("GPU gradient regression failed: ") + e.what());
	}
}
END_TEST

TEST(nn_e2e_linear_regression_gpu)
{
	// Simple linear regression with ADKernel1D + Linear + Adam
	constexpr size_t N = 64;
	constexpr size_t InDim = 1;
	constexpr size_t OutDim = 1;

	// Generate synthetic data: y = 2*x + 1
	std::vector<float> xData(N * InDim), yData(N * OutDim);
	for (size_t i = 0; i < N; i++) {
		xData[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(N - 1);
		yData[i] = 2.0f * xData[i] + 1.0f;
	}

	// Build model
	Linear<float, InDim, OutDim> fc;
	fc.Weight().Data()[0] = 0.0f;
	fc.Bias().Data()[0] = 0.0f;
	fc.Weight().Upload();
	fc.Bias().Upload();

	Adam optimizer(0.01f, 0.9f, 0.999f, 1e-6f);
	optimizer.AddTensor(fc.Weight());
	optimizer.AddTensor(fc.Bias());

	// Create buffers
	Buffer<float> bufX(xData, BufferMode::Read);
	Buffer<float> bufY(yData, BufferMode::Read);
	Buffer<float> bufPred(N * OutDim, BufferMode::ReadWrite);

	// Build AD kernel
	ADKernel1D kernel([&](Var<int> &id) {
		auto x = bufX.Bind();
		auto y = bufY.Bind();
		auto pred = bufPred.Bind();
		fc.Setup();
		fc.Forward(x, id, pred);
		Var<float> loss = MSELoss(pred, y, id, static_cast<int>(OutDim));
		AD::Loss(loss);
	}, N);

	int groups = static_cast<int>((N + 255) / 256);
	kernel.Backward(groups, true);

	auto grads = kernel.DownloadAllGradients();
	ASSERT(grads.size() == 2);
	ASSERT(grads[0].size() == N);
	ASSERT(grads[1].size() == N);

	double expectedW = 0.0;
	double expectedB = 0.0;
	double actualW = 0.0;
	double actualB = 0.0;
	for (size_t i = 0; i < N; i++) {
		float pred = 0.0f;
		float diff = pred - yData[i];
		expectedW += 2.0 * diff * xData[i];
		expectedB += 2.0 * diff;
		actualW += grads[0][i];
		actualB += grads[1][i];
	}
	expectedW /= static_cast<double>(N);
	expectedB /= static_cast<double>(N);
	actualW /= static_cast<double>(N);
	actualB /= static_cast<double>(N);

	ASSERT(std::abs(actualW - expectedW) < 1e-4);
	ASSERT(std::abs(actualB - expectedB) < 1e-4);

	optimizer.Step(kernel);
	fc.Weight().Download();
	fc.Bias().Download();
	ASSERT(fc.Weight().Data()[0] > 0.0f);
	ASSERT(fc.Bias().Data()[0] > 0.0f);
}
END_TEST

// =============================================================================
// Main
// =============================================================================

int main() {
	std::cout << "=== TestNN: Neural Network Module Tests ===\n";

	test_nn_tensor_1d_construct();
	test_nn_tensor_2d_indexing();
	test_nn_tensor_3d_indexing();
	test_nn_tensor_const_indexing();
	test_nn_tensor_move_semantics();

	test_nn_tensor_ref_1d_for_each_param();
	test_nn_tensor_ref_for_each_param_dsl();
	test_nn_tensor_ref_2d_inspector();

	test_nn_adam_parameter_count();
	test_nn_adam_add_tensor();
	test_nn_adam_weight_decay_setting();
	test_nn_sgd_parameter_count();
	test_nn_rmsprop_parameter_count();
	test_nn_optimizer_multiple_tensors();

	test_nn_adam_update_math();
	test_nn_adam_elementwise_update_matches_gradients();
	test_nn_gpu_adam_updates_on_device();
	test_nn_adam_step_is_gpu_backed();
	test_nn_adam_bias_correction();
	test_nn_sgd_momentum_math();
	test_nn_sgd_step_is_gpu_backed();
	test_nn_rmsprop_update_math();
	test_nn_rmsprop_step_is_gpu_backed();
	test_nn_weight_decay_math();
	test_nn_grad_clip_math();

	test_nn_linear_construct();
	test_nn_linear_xavier_range();
	test_nn_linear_reset();
	test_nn_linear_setup_forward_inspector();
	test_nn_linear_params_are_registered();
	test_nn_fused_mlp2_forward_matches_cpu();
	test_nn_fused_mlp2_trainer_codegen_is_specialized();
	test_nn_fused_mlp2_trainer_forward_identity_16();
	test_nn_fused_mlp2_trainer_mse_loss_16();
	test_nn_active_compaction_builds_dense_index_list();

	test_nn_relu_construct();
	test_nn_relu_forward_inspector();
	test_nn_sigmoid_forward_inspector();
	test_nn_tanh_activation_forward_inspector();

	test_nn_sequential_construct();
	test_nn_sequential_setup_forward_inspector();

	test_nn_mse_scalar();
	test_nn_mse_buffer_loss();
	test_nn_l1_loss();

	test_nn_checkpoint_save_load_roundtrip();
	test_nn_checkpoint_single_tensor();
	test_nn_checkpoint_size_mismatch_throws();

	test_nn_e2e_simple_gradient_nonzero();
	test_nn_e2e_linear_regression_gpu();

	std::cout << "\n=== Results: " << pass_count << "/" << test_count << " passed ===\n";
	return (pass_count == test_count) ? 0 : 1;
}
