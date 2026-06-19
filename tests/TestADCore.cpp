/**
 * @file TestADCore.cpp
 * @brief Test automatic differentiation: tape recording and backward pass generation.
 */

#include <AD/ADCore.h>
#include <Callable/Callable.h>
#include <Flow/ForFlow.h>
#include <Flow/IfFlow.h>
#include <Flow/ReturnFlow.h>
#include <IR/Builder/Builder.h>
#include <IR/Value/Var.h>
#include <Kernel/Kernel.h>
#include <Runtime/Buffer.h>
#include <Utility/Helpers.h>
#include <Utility/Math.h>
#include <Utility/Vec.h>

#include <cassert>
#include <cmath>
#include <format>
#include <iostream>
#include <string>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Flow;
using namespace GPU::Callables;
using namespace GPU::Runtime;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name)                                                                                                     \
	void test_##name() {                                                                                               \
		std::cout << "\n[TEST] " #name " ... ";                                                                        \
		test_count++;                                                                                                  \
		try {

#define END_TEST                                                                                                       \
	pass_count++;                                                                                                      \
	std::cout << "PASSED\n";                                                                                           \
	}                                                                                                                  \
	catch (const std::exception &e) {                                                                                  \
		std::cout << "FAILED: " << e.what() << "\n";                                                                   \
	}                                                                                                                  \
	}

#define ASSERT(cond)                                                                                                   \
	if (!(cond)) {                                                                                                     \
		throw std::runtime_error("Assertion failed: " #cond);                                                          \
	}

#define CHECK_CONTAINS(str, sub)                                                                                       \
	if ((str).find(sub) == std::string::npos) {                                                                        \
		throw std::runtime_error("Expected '" + std::string(sub) + "' in:\n" + str);                                   \
	}

// =============================================================================
// Helper: record a kernel and generate backward GLSL
// =============================================================================

struct ADTestResult {
	std::string forwardCode;
	std::string backwardCode;
	std::string tapeSummary;
};

template <typename Func> ADTestResult RunADTest(Func &&kernelFunc) {
	ADTestResult		  result;

	GPU::AD::GradientTape tape;

	GPU::IR::Builder::Builder::Get().SetGradientTape(&tape);

	GPU::Kernel::InspectorKernel1D kernel([&](Var<int> &id) { kernelFunc(id, tape); });

	GPU::IR::Builder::Builder::Get().SetGradientTape(nullptr);

	result.forwardCode = kernel.GetCode();

	for (size_t i = 0; i < tape.Size(); ++i) {
		const auto &e		= tape[i];
		result.tapeSummary += std::format("[{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
		if (!e.intrinsicName.empty())
			result.tapeSummary += " fn=" + e.intrinsicName;
		result.tapeSummary += " ins:";
		for (const auto &in : e.inputs)
			result.tapeSummary += in.name + ",";
		result.tapeSummary += "\n";
	}

	GPU::AD::AdjointGenerator gen;
	result.backwardCode = gen.Generate(tape, false);

	return result;
}

// =============================================================================
// Helper: run AD test with registered parameters
// =============================================================================

struct ADParamResult {
	std::string backwardCode;
	std::string tapeSummary;
};

template <typename Func> ADParamResult RunADParamTest(Func &&kernelFunc) {
	ADParamResult		  result;

	GPU::AD::GradientTape tape;

	GPU::IR::Builder::Builder::Get().SetGradientTape(&tape);

	GPU::Kernel::InspectorKernel1D kernel([&](Var<int> &id) { kernelFunc(id, tape); });

	GPU::IR::Builder::Builder::Get().SetGradientTape(nullptr);

	for (size_t i = 0; i < tape.Size(); ++i) {
		const auto &e		= tape[i];
		result.tapeSummary += std::format("[{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
		if (!e.intrinsicName.empty())
			result.tapeSummary += " fn=" + e.intrinsicName;
		result.tapeSummary += " ins:";
		for (const auto &in : e.inputs)
			result.tapeSummary += in.name + ",";
		result.tapeSummary += "\n";
	}

	GPU::AD::AdjointGenerator gen;
	result.backwardCode = gen.Generate(tape, false);

	return result;
}

// =============================================================================
// Helper for Callable AD tests — keeps tape active during GetCode()
// (unlike RunADTest/RunADParamTest which unset the tape too early)
// =============================================================================

template <typename Func> ADTestResult RunADCallableTest(Func &&kernelFunc) {
	ADTestResult		  result;

	GPU::AD::GradientTape tape;

	GPU::IR::Builder::Builder::Get().SetGradientTape(&tape);

	GPU::Kernel::InspectorKernel1D kernel([&](Var<int> &id) { kernelFunc(id, tape); });

	// Must call GetCode() BEFORE unsetting the tape so that callable body
	// generators run while the tape/sub-tape stack is still active.
	result.forwardCode = kernel.GetCode();

	GPU::IR::Builder::Builder::Get().SetGradientTape(nullptr);

	// Build tape summary including sub-tapes
	for (size_t i = 0; i < tape.Size(); ++i) {
		const auto &e		= tape[i];
		result.tapeSummary += std::format("[{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
		if (!e.intrinsicName.empty())
			result.tapeSummary += " fn=" + e.intrinsicName;
		if (!e.callableFuncName.empty())
			result.tapeSummary += " call=" + e.callableFuncName;
		result.tapeSummary += " ins:";
		for (const auto &in : e.inputs)
			result.tapeSummary += in.name + ",";
		result.tapeSummary += "\n";
	}

	// Add sub-tape info
	for (size_t si = 0; si < tape.SubTapeCount(); si++) {
		result.tapeSummary += std::format("-- sub-tape[{}]:\n", si);
		const auto &sub		= tape.SubTape(si);
		for (size_t i = 0; i < sub.Size(); i++) {
			const auto &e		= sub[i];
			result.tapeSummary += std::format("  [{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
			if (!e.intrinsicName.empty())
				result.tapeSummary += " fn=" + e.intrinsicName;
			if (e.kind == GPU::AD::TapeOpKind::Return)
				result.tapeSummary += " [RETURN]";
			result.tapeSummary += " ins:";
			for (const auto &in : e.inputs)
				result.tapeSummary += in.name + ",";
			result.tapeSummary += "\n";
		}
	}

	GPU::AD::AdjointGenerator gen;
	result.backwardCode = gen.Generate(tape, false);

	return result;
}

// =============================================================================
// SECTION 1: Scalar arithmetic
// =============================================================================

TEST(ad_tape_add)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 2.0f;
	Var<float> b;
	b			 = 3.0f;
	Var<float> c = a + b;
	tape.MarkLoss(c.VarName(), "float");
});
ASSERT(!r.forwardCode.empty());
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "d_v");
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_tape_mul)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 2.0f;
	Var<float> b;
	b			 = 3.0f;
	Var<float> c = a * b;
	tape.MarkLoss(c.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "*");
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_tape_chain)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 2.0f;
	Var<float> b;
	b = 3.0f;
	Var<float> c;
	c			 = 1.0f;
	Var<float> d = a * b;
	Var<float> e = d + c;
	tape.MarkLoss(e.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
CHECK_CONTAINS(r.backwardCode, "d_v");
END_TEST

TEST(ad_tape_div)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 4.0f;
	Var<float> b;
	b			 = 2.0f;
	Var<float> c = a / b;
	tape.MarkLoss(c.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_tape_sub)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 5.0f;
	Var<float> b;
	b			 = 2.0f;
	Var<float> c = a - b;
	tape.MarkLoss(c.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "-");
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_tape_neg)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a			 = 3.0f;
	Var<float> b = -a;
	tape.MarkLoss(b.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "-(");
END_TEST

// =============================================================================
// SECTION 2: Single-parameter intrinsics
// =============================================================================

TEST(ad_tape_sin)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 1.0f;
	Var<float> y = Sin(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "cos");
END_TEST

TEST(ad_tape_cos)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 1.0f;
	Var<float> y = Cos(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "sin");
END_TEST

TEST(ad_tape_exp)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 1.0f;
	Var<float> y = Exp(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "exp");
END_TEST

TEST(ad_tape_log)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 2.0f;
	Var<float> y = Log(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_tape_sqrt)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 4.0f;
	Var<float> y = Sqrt(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "sqrt");
END_TEST

TEST(ad_tape_abs)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = -2.0f;
	Var<float> y = Abs(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "sign");
END_TEST

TEST(ad_tape_tan)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 0.5f;
	Var<float> y = Tan(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "tan");
END_TEST

TEST(ad_tape_tanh)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 0.5f;
	Var<float> y = Tanh(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "tanh");
END_TEST

TEST(ad_tape_asinh)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 0.5f;
	Var<float> y = Asinh(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "sqrt");
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_tape_acosh)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 2.0f;
	Var<float> y = Acosh(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "sqrt");
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_tape_atanh)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 0.5f;
	Var<float> y = Atanh(x);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

// =============================================================================
// SECTION 3: Two-parameter intrinsics
// =============================================================================

TEST(ad_tape_pow)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 2.0f;
	Var<float> b;
	b			 = 3.0f;
	Var<float> c = Pow(a, b);
	tape.MarkLoss(c.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "pow");
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_tape_min)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 2.0f;
	Var<float> b;
	b			 = 5.0f;
	Var<float> c = Min(a, b);
	tape.MarkLoss(c.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "step");
END_TEST

TEST(ad_tape_max)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 2.0f;
	Var<float> b;
	b			 = 5.0f;
	Var<float> c = Max(a, b);
	tape.MarkLoss(c.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "step");
END_TEST

TEST(ad_tape_atan2)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> y;
	y = 1.0f;
	Var<float> x;
	x			 = 1.0f;
	Var<float> z = Atan2(y, x);
	tape.MarkLoss(z.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

// =============================================================================
// SECTION 4: Three-parameter intrinsics
// =============================================================================

TEST(ad_tape_clamp)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x = 0.5f;
	Var<float> lo;
	lo = 0.0f;
	Var<float> hi;
	hi			 = 1.0f;
	Var<float> y = Clamp(x, lo, hi);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "step");
END_TEST

TEST(ad_tape_mix)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 1.0f;
	Var<float> b;
	b = 2.0f;
	Var<float> t;
	t			 = 0.5f;
	Var<float> y = Mix(a, b, t);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

// =============================================================================
// SECTION 5: Sigmoid and activation chains
// =============================================================================

TEST(ad_tape_l2_loss)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x			 = 3.0f;
	Var<float> y = x * x;
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_tape_sigmoid_chain)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x				 = 1.0f;
	Var<float> neg_x = Expr<float>(-1.0f) * x;
	Var<float> e	 = Exp(neg_x);
	Var<float> one;
	one				 = 1.0f;
	Var<float> denom = e + one;
	Var<float> y	 = one / denom;
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
CHECK_CONTAINS(r.backwardCode, "exp");
END_TEST

TEST(ad_tape_tanh_activation)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> w;
	w = 2.0f;
	Var<float> x;
	x			 = 0.5f;
	Var<float> z = w * x;
	Var<float> y = Tanh(z);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "tanh");
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_tape_relu_subgradient)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x = 0.5f;
	Var<float> zero;
	zero		 = 0.0f;
	Var<float> y = Max(x, zero);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "step");
END_TEST

// =============================================================================
// SECTION 6: Vector operations
// =============================================================================

TEST(ad_vec3_add)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<Vec3> a = MakeFloat3(1.0f, 2.0f, 3.0f);
	Var<Vec3> b = MakeFloat3(4.0f, 5.0f, 6.0f);
	Var<Vec3> c = a + b;
	tape.MarkLoss(c.VarName(), "vec3");
});
CHECK_CONTAINS(r.backwardCode, "+=");
CHECK_CONTAINS(r.backwardCode, "d_v");
END_TEST

TEST(ad_vec3_scalar_mul)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<Vec3>  v = MakeFloat3(1.0f, 2.0f, 3.0f);
	Var<float> s;
	s				 = 2.0f;
	Var<Vec3> result = v * s;
	tape.MarkLoss(result.VarName(), "vec3");
});
CHECK_CONTAINS(r.backwardCode, "+=");
CHECK_CONTAINS(r.backwardCode, "d_v");
END_TEST

TEST(ad_vec3_scalar_mul_backward_types)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<Vec3>  v = MakeFloat3(1.0f, 2.0f, 3.0f);
	Var<float> s;
	s				 = 2.0f;
	Var<Vec3> result = v * s;
	tape.MarkLoss(result.VarName(), "vec3");
});
CHECK_CONTAINS(r.backwardCode, "vec3 _adj");
CHECK_CONTAINS(r.backwardCode, "dot(");
if (r.backwardCode.find("float _adj0_ = d_v") != std::string::npos) {
	throw std::runtime_error("Vector adjoint temporary was declared as float:\n" + r.backwardCode);
}
END_TEST

TEST(ad_vec3_scalar_expression_gradient_type_recording)
GPU::AD::GradientTape tape;
GPU::IR::Builder::Builder::Get().SetGradientTape(&tape);
GPU::Kernel::InspectorKernel1D kernel([&](Var<int> &id) {
	Var<float> s;
	s			= 2.0f;
	Var<Vec3> v = MakeFloat3(1.0f, 2.0f, 3.0f);
	Var<Vec3> y = (s * v) + v;
	tape.MarkLoss(y.VarName(), "vec3");
});
GPU::IR::Builder::Builder::Get().SetGradientTape(nullptr);

bool sawVectorCoeffForScalarInput = false;
for (size_t i = 0; i < tape.Size(); ++i) {
	const auto &entry = tape[i];
	if (entry.kind != GPU::AD::TapeOpKind::ExpressionGradient)
		continue;
	for (size_t j = 0; j < entry.inputs.size() && j < entry.inputGradTypes.size(); ++j) {
		if (entry.inputs[j].glslType == "float" && entry.inputGradTypes[j] == "vec3") {
			sawVectorCoeffForScalarInput = true;
		}
	}
}
ASSERT(sawVectorCoeffForScalarInput);

GPU::AD::AdjointGenerator gen;
std::string				  backwardCode = gen.Generate(tape, false);
CHECK_CONTAINS(backwardCode, "dot(");
END_TEST

TEST(ad_vec3_dot)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<Vec3>  a = MakeFloat3(1.0f, 0.0f, 0.0f);
	Var<Vec3>  b = MakeFloat3(0.0f, 1.0f, 0.0f);
	Var<float> d = Dot(a, b);
	tape.MarkLoss(d.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
CHECK_CONTAINS(r.backwardCode, "*");
END_TEST

TEST(ad_vec3_cross)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<Vec3> a = MakeFloat3(1.0f, 0.0f, 0.0f);
	Var<Vec3> b = MakeFloat3(0.0f, 1.0f, 0.0f);
	Var<Vec3> c = Cross(a, b);
	tape.MarkLoss(c.VarName(), "vec3");
});
CHECK_CONTAINS(r.backwardCode, "cross");
END_TEST

TEST(ad_vec3_length)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<Vec3>  v   = MakeFloat3(3.0f, 4.0f, 0.0f);
	Var<float> len = Length(v);
	tape.MarkLoss(len.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "length");
END_TEST

TEST(ad_vec3_normalize)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<Vec3> v = MakeFloat3(3.0f, 4.0f, 0.0f);
	Var<Vec3> n = Normalize(v);
	tape.MarkLoss(n.VarName(), "vec3");
});
CHECK_CONTAINS(r.backwardCode, "length");
CHECK_CONTAINS(r.backwardCode, "dot");
END_TEST

// =============================================================================
// SECTION 7: Compound assignments
// =============================================================================

TEST(ad_compound_add_assign)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> acc;
	acc = 0.0f;
	Var<float> x;
	x	 = 3.0f;
	acc += x;
	tape.MarkLoss(acc.VarName(), "float");
});
// Compound assign: acc += x → dx += d_acc
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_accumulate_chain)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> sum;
	sum = 0.0f;
	Var<float> a;
	a = 1.0f;
	Var<float> b;
	b = 2.0f;
	Var<float> c;
	c	 = 3.0f;
	sum += a;
	sum += b;
	sum += c;
	tape.MarkLoss(sum.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

// =============================================================================
// SECTION 8: MLP-like chains
// =============================================================================

TEST(ad_mlp_neuron)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	// y = sigmoid(w1*x1 + w2*x2 + b)
	Var<float> w1;
	w1 = 0.5f;
	Var<float> w2;
	w2 = -0.3f;
	Var<float> x1;
	x1 = 1.0f;
	Var<float> x2;
	x2 = 2.0f;
	Var<float> b;
	b			  = 0.1f;

	Var<float> t1 = w1 * x1;
	Var<float> t2 = w2 * x2;
	Var<float> z  = t1 + t2 + b;
	Var<float> y  = Tanh(z);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "tanh");
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_linear_layer_l2)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	// L = (w*x + b)^2
	Var<float> w;
	w = 2.0f;
	Var<float> x;
	x = 3.0f;
	Var<float> b;
	b			 = 1.0f;
	Var<float> z = w * x + b;
	Var<float> L = z * z;
	tape.MarkLoss(L.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
CHECK_CONTAINS(r.backwardCode, "d_v");
END_TEST

// =============================================================================
// SECTION 9: Parameter registration and active propagation
// =============================================================================

TEST(ad_params_multiple)
GPU::AD::GradientTape tape;
tape.RegisterParameter("w", "float");
tape.RegisterParameter("b", "float");
tape.RegisterParameter("x", "float");
ASSERT(tape.IsParameter("w"));
ASSERT(tape.IsParameter("b"));
ASSERT(tape.IsParameter("x"));
ASSERT(!tape.IsParameter("y"));
ASSERT(tape.Parameters().size() == 3);
END_TEST

TEST(ad_adjoint_table)
GPU::AD::AdjointTable table;
std::string			  adj0		= table.GetOrCreate("v0", "float");
std::string			  adj1		= table.GetOrCreate("v1", "vec3");
std::string			  adj0Again = table.GetOrCreate("v0", "float");
ASSERT(adj0 == adj0Again);
ASSERT(adj0 != adj1);
ASSERT(table.Has("v0"));
ASSERT(!table.Has("v9"));
auto decls = table.AllDeclarations();
ASSERT(decls.size() == 2);
END_TEST

TEST(ad_adjoint_table_clear_resets_array_sizes)
GPU::AD::AdjointTable table;
std::string			  adj = table.GetOrCreate("buf0[0]", "float");
table.SetArraySize(adj, 4);
ASSERT(table.GetArraySize(adj) == 4);
table.Clear();
ASSERT(table.GetArraySize(adj) == 0);
std::string adjAfterClear = table.GetOrCreate("buf0", "float");
auto		decls		  = table.AllDeclarations();
ASSERT(adjAfterClear == "d_buf0");
ASSERT(decls.size() == 1);
ASSERT(decls[0].second == "float");
END_TEST

TEST(ad_active_propagation)
GPU::AD::GradientTape tape2;
tape2.RegisterParameter("p0", "float");
tape2.MarkLoss("loss", "float");
ASSERT(tape2.IsActive("p0"));
ASSERT(tape2.IsActive("loss"));
ASSERT(!tape2.IsActive("v99"));
END_TEST

// =============================================================================
// SECTION 10: Parameter-based gradient test
// =============================================================================

TEST(ad_param_gradient)
auto r = RunADParamTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	tape.RegisterParameter("v1", "float");
	tape.RegisterParameter("v2", "float");

	Var<float> w;
	w = 2.0f;
	Var<float> b;
	b			 = 1.0f;
	Var<float> z = w * b;
	tape.MarkLoss(z.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

// =============================================================================
// SECTION 11: AdjointInspector1D user API
// =============================================================================

TEST(ad_inspector_basic)
GPU::AD::AdjointInspector1D inspector([](Var<int> &id, GPU::AD::AdjointContext &ctx) {
	Var<float> a;
	a = 2.0f;
	Var<float> b;
	b			 = 3.0f;
	Var<float> c = a + b;
	ctx.MarkLoss<float>(c.VarName());
});
ASSERT(!inspector.GetForwardCode().empty());
ASSERT(!inspector.GetBackwardCode().empty());
CHECK_CONTAINS(inspector.GetBackwardCode(), "d_v");
CHECK_CONTAINS(inspector.GetBackwardCode(), "+=");
ASSERT(inspector.HasBackwardCode());
END_TEST

TEST(ad_inspector_sigmoid)
GPU::AD::AdjointInspector1D inspector([](Var<int> &id, GPU::AD::AdjointContext &ctx) {
	Var<float> x;
	x				 = 1.0f;
	Var<float> neg_x = Expr<float>(-1.0f) * x;
	Var<float> e	 = Exp(neg_x);
	Var<float> one;
	one				 = 1.0f;
	Var<float> denom = e + one;
	Var<float> y	 = one / denom;
	ctx.MarkLoss<float>(y.VarName());
});
CHECK_CONTAINS(inspector.GetBackwardCode(), "exp");
CHECK_CONTAINS(inspector.GetBackwardCode(), "+=");
ASSERT(inspector.Tape().Size() > 0);
END_TEST

TEST(ad_inspector_params)
GPU::AD::AdjointInspector1D inspector([](Var<int> &id, GPU::AD::AdjointContext &ctx) {
	Var<float> w;
	w = 2.0f;
	Var<float> x;
	x = 3.0f;
	Var<float> b;
	b = 1.0f;

	ctx.RegisterParameter(w.VarName(), "float");
	ctx.RegisterParameter(b.VarName(), "float");

	Var<float> z = w * x + b;
	Var<float> L = z * z;
	ctx.MarkLoss<float>(L.VarName());
});
CHECK_CONTAINS(inspector.GetBackwardCode(), "d_v");
ASSERT(inspector.Tape().Parameters().size() == 2);
END_TEST

// =============================================================================
// SECTION 12: Control flow — If/Else
// =============================================================================

TEST(ad_if_else_basic)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x = 1.0f;
	Var<float> y;
	y			   = 0.0f;
	Var<bool> cond = x > Expr<float>(0.0f);

	If(cond, [&]() { y = x * Expr<float>(2.0f); }).Else([&]() { y = x * Expr<float>(3.0f); });

	Var<float> loss = y * y;
	tape.MarkLoss(loss.VarName(), "float");
});
ASSERT(!r.forwardCode.empty());
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "if (");
CHECK_CONTAINS(r.backwardCode, "else");
END_TEST

TEST(ad_if_no_else)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x = 1.0f;
	Var<float> y;
	y			   = 0.0f;
	Var<bool> cond = x > Expr<float>(0.0f);

	If(cond, [&]() { y = x * Expr<float>(2.0f); });

	Var<float> loss = y * y;
	tape.MarkLoss(loss.VarName(), "float");
});
ASSERT(!r.forwardCode.empty());
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "if (");
END_TEST

TEST(ad_if_elif_else)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 1.0f;
	Var<float> b;
	b = 2.0f;
	Var<float> c;
	c = 3.0f;
	Var<float> y;
	y			 = 0.0f;
	Var<bool> c1 = a > Expr<float>(0.0f);
	Var<bool> c2 = b > Expr<float>(3.0f);

	If(c1, [&]() { y = a; }).Elif(std::move(c2), [&]() { y = b; }).Else([&]() { y = c; });

	tape.MarkLoss(y.VarName(), "float");
});
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "if (");
CHECK_CONTAINS(r.backwardCode, "else if (");
CHECK_CONTAINS(r.backwardCode, "else");
END_TEST

TEST(ad_if_gradient_structure)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x = 1.0f;
	Var<float> a;
	a			   = 0.0f;
	Var<bool> cond = x > Expr<float>(0.0f);

	If(cond, [&]() { a = x; });

	tape.MarkLoss(a.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "if (");
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

// =============================================================================
// SECTION 13: Control flow — For loops
// =============================================================================

TEST(ad_for_loop_basic)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> sum;
	sum = 0.0f;
	Var<float> x;
	x = 1.0f;

	For(0, 3, [&](Var<int> &i) { sum += x; });

	tape.MarkLoss(sum.VarName(), "float");
});
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "for (");
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_for_loop_mul)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> sum;
	sum = 0.0f;
	Var<float> w;
	w = 2.0f;
	Var<float> x;
	x = 3.0f;

	For(0, 4, [&](Var<int> &i) {
		Var<float> prod	 = w * x;
		sum				+= prod;
	});

	tape.MarkLoss(sum.VarName(), "float");
});
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "for (");
CHECK_CONTAINS(r.backwardCode, "+=");
CHECK_CONTAINS(r.backwardCode, "*");
END_TEST

TEST(ad_for_loop_variable_bound)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<int> N;
	N = 5;
	Var<float> sum;
	sum = 0.0f;
	Var<float> x;
	x = 1.0f;

	For(0, N, [&](Var<int> &i) { sum += x; });

	tape.MarkLoss(sum.VarName(), "float");
});
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "for (");
END_TEST

// =============================================================================
// SECTION 14: Control flow — Nested
// =============================================================================

TEST(ad_if_inside_for)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> sum;
	sum = 0.0f;
	Var<float> x;
	x = 1.0f;
	Var<float> threshold;
	threshold = 0.5f;

	For(0, 3, [&](Var<int> &i) {
		Var<bool> cond = x > threshold;
		If(cond, [&]() { sum += x; });
	});

	tape.MarkLoss(sum.VarName(), "float");
});
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "for (");
CHECK_CONTAINS(r.backwardCode, "if (");
END_TEST

// =============================================================================
// Main
// =============================================================================

// =============================================================================
// SECTION 15: AdjointKernel1D — combined forward+backward GLSL
// =============================================================================

TEST(ad_kernel_merged_forward_backward)
// Test merge with synthetic adjoint body on real forward code
auto				 r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 2.0f;
	Var<float> b;
	b			 = 3.0f;
	Var<float> c = a * b;
	tape.MarkLoss(c.VarName(), "float");
});
// Build a synthetic adjoint body
GPU::AD::AdjointBody body;
body.declarations.push_back({"d_test", "float"});
body.lines.push_back("d_test += float(1.0);");
std::vector<GPU::AD::GradBuffer> grads;
std::string						 combined = GPU::AD::MergeForwardBackward(r.forwardCode, body, grads);
CHECK_CONTAINS(combined, "void main()");
CHECK_CONTAINS(combined, "d_test");
CHECK_CONTAINS(combined, "+=");
END_TEST

TEST(ad_kernel_merged_with_gradbufs)
auto				 r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> w;
	w = 2.0f;
	Var<float> x;
	x			 = 3.0f;
	Var<float> y = w * x;
	tape.MarkLoss(y.VarName(), "float");
});
GPU::AD::AdjointBody body;
body.declarations.push_back({"d_w", "float"});
body.declarations.push_back({"d_x", "float"});
body.lines.push_back("d_w += d_y * x;");
body.writebacks.push_back({"v1", "d_w"});
body.writebacks.push_back({"v2", "d_x"});
std::vector<GPU::AD::GradBuffer> grads;
GPU::AD::GradBuffer				 gb;
gb.paramName = "v1";
gb.glslType	 = "float";
gb.binding	 = 10;
grads.push_back(gb);
gb.paramName = "v2";
gb.glslType	 = "float";
gb.binding	 = 11;
grads.push_back(gb);
std::string combined = GPU::AD::MergeForwardBackward(r.forwardCode, body, grads);
CHECK_CONTAINS(combined, "grad_v1");
CHECK_CONTAINS(combined, "grad_v2");
CHECK_CONTAINS(combined, "_grad_v1_data");
END_TEST

TEST(ad_kernel_1d_api)
GPU::AD::AdjointKernel1D kernel([](GPU::IR::Value::Var<int> &id, GPU::AD::AdjointContext &ctx) {
	Var<float> a;
	a = 2.0f;
	Var<float> b;
	b			 = 3.0f;
	Var<float> c = a * b;
	ctx.MarkLoss<float>(c.VarName());
});
ASSERT(!kernel.GetForwardCode().empty());
ASSERT(kernel.HasCombinedCode());
CHECK_CONTAINS(kernel.GetCombinedCode(), "void main()");
CHECK_CONTAINS(kernel.GetCombinedCode(), "+=");
CHECK_CONTAINS(kernel.GetCombinedCode(), "d_v");
ASSERT(kernel.Tape().Size() > 0);
END_TEST

TEST(ad_kernel_1d_sigmoid)
GPU::AD::AdjointKernel1D kernel([](GPU::IR::Value::Var<int> &id, GPU::AD::AdjointContext &ctx) {
	Var<float> x;
	x				 = 1.0f;
	Var<float> neg_x = Expr<float>(-1.0f) * x;
	Var<float> e	 = Exp(neg_x);
	Var<float> one;
	one				 = 1.0f;
	Var<float> denom = e + one;
	Var<float> y	 = one / denom;
	ctx.MarkLoss<float>(y.VarName());
});
ASSERT(kernel.HasCombinedCode());
CHECK_CONTAINS(kernel.GetCombinedCode(), "exp");
CHECK_CONTAINS(kernel.GetCombinedCode(), "+=");
END_TEST

TEST(ad_kernel_1d_grad_bindings_follow_forward_bindings)
GPU::Runtime::Buffer<float> a(8);
GPU::Runtime::Buffer<float> b(8);
GPU::Runtime::Buffer<float> w(8);
GPU::AD::ADKernel1D kernel(
	[&](GPU::IR::Value::Var<int> &id) {
		auto		  ar = a.Bind();
		auto		  br = b.Bind();
		auto		  wr = w.Bind();
		Var<float> x	= ar[id] + br[id];
		Var<float> p	= wr[id];
		Var<float> loss = x * p;
		GPU::AD::Param(p);
		GPU::AD::Loss(loss);
	},
	8);
std::string code = kernel.CombinedCode();
CHECK_CONTAINS(code, "binding = 3");
CHECK_CONTAINS(code, "_ad_grad_");
END_TEST

TEST(ad_expression_gradient_uses_temporaries_for_long_coefficients)
auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 1.1f;
	Var<float> b;
	b = 1.2f;
	Var<float> c;
	c = 1.3f;
	Var<float> d;
	d = 1.4f;
	Var<float> e;
	e = 1.5f;
	Var<float> f;
	f = 1.6f;
	Var<float> g;
	g = 1.7f;
	Var<float> h;
	h = 1.8f;
	Var<float> i;
	i = 1.9f;
	Var<float> j;
	j = 2.0f;
	Var<float> k;
	k = 2.1f;
	Var<float> l;
	l = 2.2f;
	Var<float> m;
	m = 2.3f;
	Var<float> n;
	n = 2.4f;
	Var<float> o;
	o = 2.5f;
	tape.RegisterParameter(a.VarName(), "float");
	Var<float> y = ((((((((((((((Expr<float>(a) * b) * c) * d) * e) * f) * g) * h) * i) * j) * k) * l) * m) * n) * o);
	tape.MarkLoss(y.VarName(), "float");
});
CHECK_CONTAINS(r.backwardCode, "_ad_tmp");
END_TEST

// =============================================================================
// SECTION 16: AdjointInspector3D and AdjointKernel2D
// =============================================================================

TEST(ad_inspector_3d_basic)
GPU::AD::AdjointInspector3D inspector(
	[](Var<int> &idX, Var<int> &idY, Var<int> &idZ, GPU::AD::AdjointContext &ctx) {
		Var<float> a;
		a = 2.0f;
		Var<float> b;
		b			 = 3.0f;
		Var<float> c = a + b;
		ctx.MarkLoss<float>(c.VarName());
	},
	8, 8, 4);
ASSERT(!inspector.GetForwardCode().empty());
ASSERT(!inspector.GetBackwardCode().empty());
CHECK_CONTAINS(inspector.GetBackwardCode(), "d_v");
CHECK_CONTAINS(inspector.GetBackwardCode(), "+=");
ASSERT(inspector.HasBackwardCode());
END_TEST

TEST(ad_inspector_3d_vector)
GPU::AD::AdjointInspector3D inspector(
	[](Var<int> &idX, Var<int> &idY, Var<int> &idZ, GPU::AD::AdjointContext &ctx) {
		Var<Vec3>  v   = MakeFloat3(1.0f, 2.0f, 3.0f);
		Var<float> len = Length(v);
		ctx.MarkLoss<float>(len.VarName());
	},
	4, 4, 4);
CHECK_CONTAINS(inspector.GetBackwardCode(), "length");
CHECK_CONTAINS(inspector.GetBackwardCode(), "+=");
END_TEST

TEST(ad_kernel_2d_basic)
GPU::AD::AdjointKernel2D kernel(
	[](Var<int> &idX, Var<int> &idY, GPU::AD::AdjointContext &ctx) {
		Var<float> a;
		a = 2.0f;
		Var<float> b;
		b			 = 3.0f;
		Var<float> c = a + b;
		ctx.MarkLoss<float>(c.VarName());
	},
	8, 8);
ASSERT(!kernel.GetForwardCode().empty());
ASSERT(kernel.HasCombinedCode());
CHECK_CONTAINS(kernel.GetCombinedCode(), "void main()");
CHECK_CONTAINS(kernel.GetCombinedCode(), "+=");
END_TEST

// =============================================================================
// SECTION 17: Callable AD
// =============================================================================

TEST(ad_callable_mul)
Callable<float(float, float)> mulOp([](Var<float> a, Var<float> b) {
	Var<float> r = a * b;
	Return(r);
});

auto						  r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> x;
	x = 2.0f;
	Var<float> y;
	y			 = 3.0f;
	Var<float> z = mulOp(x, y);
	tape.MarkLoss(z.VarName(), "float");
});

ASSERT(!r.forwardCode.empty());
ASSERT(!r.backwardCode.empty());
// The backward pass should have gradient accumulation
CHECK_CONTAINS(r.backwardCode, "+=");
// The tape should have a Call entry
CHECK_CONTAINS(r.tapeSummary, "kind=" + std::to_string((int)GPU::AD::TapeOpKind::Call));
END_TEST

TEST(ad_callable_add)
Callable<float(float, float)> addOp([](Var<float> a, Var<float> b) {
	Var<float> r = a + b;
	Return(r);
});

auto						  r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a = 1.0f;
	Var<float> b;
	b			 = 2.0f;
	Var<float> c = addOp(a, b);
	tape.MarkLoss(c.VarName(), "float");
});

ASSERT(!r.forwardCode.empty());
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(ad_callable_chain)
Callable<float(float)> square([](Var<float> x) {
	Var<float> r = x * x;
	Return(r);
});

auto				   r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> a;
	a			  = 3.0f;
	Var<float> a2 = square(a);
	Var<float> y  = a2 * Expr<float>(2.0f);
	tape.MarkLoss(y.VarName(), "float");
});

ASSERT(!r.forwardCode.empty());
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "+=");
CHECK_CONTAINS(r.backwardCode, "*");
END_TEST

TEST(ad_callable_two_params_with_param_reg)
Callable<float(float, float)> mulOp2([](Var<float> a, Var<float> b) {
	Var<float> r = a * b;
	Return(r);
});

auto						  r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
	Var<float> w;
	w = 2.0f;
	Var<float> x;
	x = 3.0f;
	tape.RegisterParameter(w.VarName(), "float");
	tape.RegisterParameter(x.VarName(), "float");
	Var<float> z = mulOp2(w, x);
	tape.MarkLoss(z.VarName(), "float");
});

ASSERT(!r.forwardCode.empty());
ASSERT(!r.backwardCode.empty());
CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

// Section 16 calls in main() — appended below

int main() {
	std::cout << "=== EasyGPU Automatic Differentiation Tests ===\n";

	// Section 1: Scalar arithmetic
	test_ad_tape_add();
	test_ad_tape_mul();
	test_ad_tape_chain();
	test_ad_tape_div();
	test_ad_tape_sub();
	test_ad_tape_neg();

	// Section 2: Single-parameter intrinsics
	test_ad_tape_sin();
	test_ad_tape_cos();
	test_ad_tape_exp();
	test_ad_tape_log();
	test_ad_tape_sqrt();
	test_ad_tape_abs();
	test_ad_tape_tan();
	test_ad_tape_tanh();
	test_ad_tape_asinh();
	test_ad_tape_acosh();
	test_ad_tape_atanh();

	// Section 3: Two-parameter intrinsics
	test_ad_tape_pow();
	test_ad_tape_min();
	test_ad_tape_max();
	test_ad_tape_atan2();

	// Section 4: Three-parameter intrinsics
	test_ad_tape_clamp();
	test_ad_tape_mix();

	// Section 5: Sigmoid and activation chains
	test_ad_tape_l2_loss();
	test_ad_tape_sigmoid_chain();
	test_ad_tape_tanh_activation();
	test_ad_tape_relu_subgradient();

	// Section 6: Vector operations
	test_ad_vec3_add();
	test_ad_vec3_scalar_mul();
	test_ad_vec3_scalar_mul_backward_types();
	test_ad_vec3_scalar_expression_gradient_type_recording();
	test_ad_vec3_dot();
	test_ad_vec3_cross();
	test_ad_vec3_length();
	test_ad_vec3_normalize();

	// Section 7: Compound assignments
	test_ad_compound_add_assign();
	test_ad_accumulate_chain();

	// Section 8: MLP-like chains
	test_ad_mlp_neuron();
	test_ad_linear_layer_l2();

	// Section 9: Parameter and adjoint table
	test_ad_params_multiple();
	test_ad_adjoint_table();
	test_ad_adjoint_table_clear_resets_array_sizes();
	test_ad_active_propagation();

	// Section 10: Parameter gradient
	test_ad_param_gradient();

	// Section 11: AdjointInspector1D user API
	test_ad_inspector_basic();
	test_ad_inspector_sigmoid();
	test_ad_inspector_params();

	// Section 12: Control flow — If/Else
	test_ad_if_else_basic();
	test_ad_if_no_else();
	test_ad_if_elif_else();
	test_ad_if_gradient_structure();

	// Section 13: Control flow — For loops
	test_ad_for_loop_basic();
	test_ad_for_loop_mul();
	test_ad_for_loop_variable_bound();

	// Section 14: Control flow — Nested
	test_ad_if_inside_for();
	test_ad_kernel_merged_forward_backward();
	test_ad_kernel_merged_with_gradbufs();
	test_ad_kernel_1d_api();
	test_ad_kernel_1d_sigmoid();
	test_ad_kernel_1d_grad_bindings_follow_forward_bindings();
	test_ad_expression_gradient_uses_temporaries_for_long_coefficients();
	test_ad_inspector_3d_basic();
	test_ad_inspector_3d_vector();
	test_ad_kernel_2d_basic();

	// Section 17: Callable AD
	test_ad_callable_mul();
	test_ad_callable_add();
	test_ad_callable_chain();
	test_ad_callable_two_params_with_param_reg();

	std::cout << "\n=== Results: " << pass_count << "/" << test_count << " passed ===\n";
	return pass_count == test_count ? 0 : 1;
}
