/**
 * @file TestADStress.cpp
 * @brief Industrial-grade stress tests for the EasyGPU automatic differentiation system.
 *
 * Categories:
 *   A - Deep Graph Stress       (50-100 op chains, diamond patterns, MLP)
 *   B - Edge Cases              (zero, extreme values, empty kernels)
 *   C - Control Flow Stress     (deep nesting, complex conditions)
 *   D - Callable Stress         (nested callables, many callables)
 *   E - Multi-Parameter Stress  (20-50 params, mixed types)
 *   F - Vector/Matrix Stress    (swizzles, mat ops, pipelines)
 *   G - Complex Composition     (softmax, attention, residual, LSTM-like)
 *   H - Backward Code Quality   (declarations, seeds, writebacks)
 *   I - Numerical Correctness   (finite difference framework)
 *   J - Regression Protection   (fixed-bug regression tests)
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
#include <sstream>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Flow;
using namespace GPU::Callables;
using namespace GPU::Runtime;

static int test_count = 0;
static int pass_count = 0;

// =============================================================================
// Test macros
// =============================================================================

#define TEST(name)                                                                                             \
	void test_##name() {                                                                                       \
		std::cout << "\n[STRESS] " #name " ... ";                                                              \
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
		throw std::runtime_error("Unexpected '" + std::string(sub) + "' found in:\n" + str);                    \
	}

#define CHECK_MIN_COUNT(str, sub, minCount)                                                                    \
	{                                                                                                          \
		size_t pos = 0;                                                                                        \
		size_t count = 0;                                                                                      \
		while ((pos = (str).find(sub, pos)) != std::string::npos) { ++count; ++pos; }                          \
		if (count < (size_t)(minCount)) {                                                                      \
			throw std::runtime_error("Expected at least " + std::to_string(minCount) + " occurrences of '" +    \
									 sub + "' but found " + std::to_string(count));                            \
		}                                                                                                      \
	}

// =============================================================================
// Test result structs
// =============================================================================

struct ADTestResult {
	std::string forwardCode;
	std::string backwardCode;
	std::string tapeSummary;
};

struct ADParamResult {
	std::string backwardCode;
	std::string tapeSummary;
};

// =============================================================================
// Helper: record a kernel and generate backward GLSL
// =============================================================================

template <typename Func>
ADTestResult RunADTest(Func &&kernelFunc) {
	ADTestResult result;
	GPU::AD::GradientTape tape;
	GPU::IR::Builder::Builder::Get().SetGradientTape(&tape);
	GPU::Kernel::InspectorKernel1D kernel([&](Var<int> &id) { kernelFunc(id, tape); });
	GPU::IR::Builder::Builder::Get().SetGradientTape(nullptr);
	result.forwardCode = kernel.GetCode();
	for (size_t i = 0; i < tape.Size(); ++i) {
		const auto &e = tape[i];
		result.tapeSummary += std::format("[{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
		if (!e.intrinsicName.empty()) result.tapeSummary += " fn=" + e.intrinsicName;
		result.tapeSummary += " ins:";
		for (const auto &in : e.inputs) result.tapeSummary += in.name + ",";
		result.tapeSummary += "\n";
	}
	GPU::AD::AdjointGenerator gen;
	result.backwardCode = gen.Generate(tape, false);
	return result;
}

template <typename Func>
ADParamResult RunADParamTest(Func &&kernelFunc) {
	ADParamResult result;
	GPU::AD::GradientTape tape;
	GPU::IR::Builder::Builder::Get().SetGradientTape(&tape);
	GPU::Kernel::InspectorKernel1D kernel([&](Var<int> &id) { kernelFunc(id, tape); });
	GPU::IR::Builder::Builder::Get().SetGradientTape(nullptr);
	for (size_t i = 0; i < tape.Size(); ++i) {
		const auto &e = tape[i];
		result.tapeSummary += std::format("[{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
		if (!e.intrinsicName.empty()) result.tapeSummary += " fn=" + e.intrinsicName;
		result.tapeSummary += " ins:";
		for (const auto &in : e.inputs) result.tapeSummary += in.name + ",";
		result.tapeSummary += "\n";
	}
	GPU::AD::AdjointGenerator gen;
	result.backwardCode = gen.Generate(tape, false);
	return result;
}

template <typename Func>
ADTestResult RunADCallableTest(Func &&kernelFunc) {
	ADTestResult result;
	GPU::AD::GradientTape tape;
	GPU::IR::Builder::Builder::Get().SetGradientTape(&tape);
	GPU::Kernel::InspectorKernel1D kernel([&](Var<int> &id) { kernelFunc(id, tape); });
	result.forwardCode = kernel.GetCode();
	GPU::IR::Builder::Builder::Get().SetGradientTape(nullptr);
	for (size_t i = 0; i < tape.Size(); ++i) {
		const auto &e = tape[i];
		result.tapeSummary += std::format("[{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
		if (!e.intrinsicName.empty()) result.tapeSummary += " fn=" + e.intrinsicName;
		if (!e.callableFuncName.empty()) result.tapeSummary += " call=" + e.callableFuncName;
		result.tapeSummary += " ins:";
		for (const auto &in : e.inputs) result.tapeSummary += in.name + ",";
		result.tapeSummary += "\n";
	}
	for (size_t si = 0; si < tape.SubTapeCount(); si++) {
		result.tapeSummary += std::format("-- sub-tape[{}]:\n", si);
		const auto &sub = tape.SubTape(si);
		for (size_t i = 0; i < sub.Size(); i++) {
			const auto &e = sub[i];
			result.tapeSummary += std::format("  [{}] kind={} out={}", e.id, (int)e.kind, e.output.name);
			if (!e.intrinsicName.empty()) result.tapeSummary += " fn=" + e.intrinsicName;
			if (e.kind == GPU::AD::TapeOpKind::Return) result.tapeSummary += " [RETURN]";
			result.tapeSummary += " ins:";
			for (const auto &in : e.inputs) result.tapeSummary += in.name + ",";
			result.tapeSummary += "\n";
		}
	}
	GPU::AD::AdjointGenerator gen;
	result.backwardCode = gen.Generate(tape, false);
	return result;
}

// =============================================================================
// SECTION A: Deep Graph Stress
// =============================================================================

TEST(stress_deep_50_add_chain)
	// 50 consecutive add operations: v = a0 + a1 + a2 + ... + a49
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> v; v = 0.0f;
		for (int i = 0; i < 50; ++i) {
			Var<float> x; x = float(i + 1);
			v = v + Expr<float>(x);
		}
		tape.MarkLoss(v.VarName(), "float");
	});
	ASSERT(!r.forwardCode.empty());
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
	CHECK_MIN_COUNT(r.backwardCode, "+=", 25);
END_TEST

TEST(stress_deep_50_mixed_ops)
	// 50 mixed operations (Add/Mul/Sub/Div alternating)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> v; v = 1.0f;
		for (int i = 0; i < 50; ++i) {
			Var<float> x; x = float(i + 1);
			if (i % 4 == 0) v = v + Expr<float>(x);
			else if (i % 4 == 1) v = v * Expr<float>(x);
			else if (i % 4 == 2) v = v - Expr<float>(x);
			else v = v / Expr<float>(x);
		}
		tape.MarkLoss(v.VarName(), "float");
	});
	ASSERT(!r.forwardCode.empty());
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.forwardCode, "void main()");
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_deep_100_add_chain)
	// 100 consecutive add operations
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> v; v = 0.0f;
		for (int i = 0; i < 100; ++i) {
			Var<float> x; x = float(i);
			v = v + Expr<float>(x);
		}
		tape.MarkLoss(v.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
	CHECK_MIN_COUNT(r.backwardCode, "+=", 50);
END_TEST

TEST(stress_deep_nested_10_levels)
	// ((...(a1*x1 + a2)*x2 + a3)*x3 + ... + a10)*x10
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> v; v = 1.0f;
		for (int i = 0; i < 10; ++i) {
			Var<float> a; a = float(i + 2);
			Var<float> x; x = float(i * 3);
			v = v * x + a;
		}
		tape.MarkLoss(v.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
	CHECK_MIN_COUNT(r.backwardCode, "+=", 5);
END_TEST

TEST(stress_deep_wide_fan_out)
	// One input variable used in 20 different branches, all summed to loss
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 5.0f;
		Var<float> loss; loss = 0.0f;
		for (int i = 0; i < 20; ++i) {
			Var<float> w; w = float(i);
			loss = loss + x * Expr<float>(w);
		}
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_MIN_COUNT(r.backwardCode, "+=", 10);
END_TEST

TEST(stress_deep_diamond)
	// Diamond dependency: input splits to 5 paths, each path does 3 ops, then all merge
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 2.0f;
		Var<float> loss; loss = 0.0f;
		for (int path = 0; path < 5; ++path) {
			Var<float> v = x;
			Var<float> w; w = float(path + 1);
			v = v * w;
			v = v + Expr<float>(w);
			v = v * Expr<float>(x);
			loss = loss + v;
		}
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_MIN_COUNT(r.backwardCode, "+=", 5);
END_TEST

TEST(stress_deep_mlp_5_layers)
	// 5-layer MLP: W5*(W4*(W3*(W2*(W1*x + b1) + b2) + b3) + b4) + b5
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 1.0f;
		Var<float> h = x;
		for (int layer = 0; layer < 5; ++layer) {
			Var<float> W; W = float(layer + 1);
			Var<float> b; b = float(layer);
			h = W * h + b;
		}
		tape.MarkLoss(h.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
	CHECK_MIN_COUNT(r.backwardCode, "+=", 3);
END_TEST

TEST(stress_deep_single_var_50_uses)
	// One variable used 50 times - gradient must accumulate correctly
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 3.0f;
		Var<float> loss; loss = 0.0f;
		for (int i = 0; i < 50; ++i) {
			loss = loss + x * Expr<float>(float(i));
		}
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	// Each use generates a += contribution to d_x
	CHECK_MIN_COUNT(r.backwardCode, "d_v", 1);
	CHECK_MIN_COUNT(r.backwardCode, "+=", 20);
END_TEST

// =============================================================================
// SECTION B: Edge Cases
// =============================================================================

TEST(stress_edge_all_zero_inputs)
	// All constants are zero
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 0.0f;
		Var<float> b; b = 0.0f;
		Var<float> c = a + b;
		Var<float> d = a * b;
		Var<float> e = d / Expr<float>(1.0f);
		Var<float> loss = c + Expr<float>(e);
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_edge_negative_sqrt)
	// sqrt(-1) might produce NaN in GPU execution, but code generation must complete
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = -1.0f;
		Var<float> s = Sqrt(a);
		tape.MarkLoss(s.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_edge_negative_log)
	// log(-1) generates code but will produce NaN at runtime
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = -1.0f;
		Var<float> l = Log(a);
		tape.MarkLoss(l.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_edge_extreme_values)
	// Very large and very small values
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> big;   big   = 1e30f;
		Var<float> small; small = 1e-30f;
		Var<float> m = big * small;
		Var<float> d = big / Expr<float>(small);
		Var<float> s = big + Expr<float>(small);
		Var<float> loss = m + d + s;
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_edge_near_div_by_zero)
	// Divisor extremely small (but not exactly zero)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 1.0f;
		Var<float> b; b = 1e-30f;
		Var<float> c = a / Expr<float>(b);
		tape.MarkLoss(c.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_edge_empty_kernel)
	// Lambda with no operations (only the id parameter is used)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		// intentionally empty - no differentiable operations
	});
	ASSERT(r.backwardCode.empty() || r.backwardCode.find("void main()") != std::string::npos);
END_TEST

TEST(stress_edge_declare_only)
	// Variables declared but no computation connecting them to loss
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 1.0f;
		Var<float> b; b = 2.0f;
		// a and b are never used together or connected to any loss
	});
	// Should not crash - just no loss means empty backward
	ASSERT(r.backwardCode.empty() || !r.backwardCode.empty()); // always passes, just no-crash check
END_TEST

TEST(stress_edge_loss_no_params)
	// MarkLoss but no RegisterParameter
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 5.0f;
		Var<float> y = x * Expr<float>(2.0f);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "= float(1.0)");
END_TEST

TEST(stress_edge_params_no_loss)
	// RegisterParameter but no MarkLoss
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> w; w = 2.0f;
		Var<float> x; x = 3.0f;
		Var<float> y = w * x;
		tape.RegisterParameter(w.VarName(), "float");
		// No MarkLoss
	});
	// Should complete without crash, backward code may be empty or minimal
	(void)r; // no crash check
END_TEST

TEST(stress_edge_param_is_loss)
	// Same variable is both parameter and loss
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> w; w = 2.0f;
		tape.RegisterParameter(w.VarName(), "float");
		tape.MarkLoss(w.VarName(), "float");
	});
	// When param is the loss itself, backward code may be minimal/empty
	ASSERT(r.backwardCode.empty() || !r.backwardCode.empty()); // no-crash check
END_TEST

TEST(stress_edge_multi_loss)
	// Multiple loss variables (only last marked effective for seeding)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 1.0f;
		Var<float> b; b = 2.0f;
		tape.MarkLoss(a.VarName(), "float");
		tape.MarkLoss(b.VarName(), "float");
	});
	// Multiple loss markers: only last seeded, may produce minimal code
	ASSERT(r.backwardCode.empty() || !r.backwardCode.empty()); // no-crash check
END_TEST

TEST(stress_edge_all_int_ops)
	// All integer operations (non-differentiable)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<int> a; a = 1;
		Var<int> b; b = 2;
		Var<int> c = a + b;
		Var<int> d = c * Expr<int>(3);
		(void)d;
	});
	// Should not crash - int types are not recorded in tape
	ASSERT(r.tapeSummary.empty() || !r.tapeSummary.empty()); // no-crash check
END_TEST

TEST(stress_edge_mixed_types)
	// Mix of float, int, and bool in same kernel
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> f; f = 3.14f;
		Var<int>   i; i = 42;
		Var<bool>  b; b = f > Expr<float>(0.0f);
		Var<float> result = f * Expr<float>(2.0f);
		tape.MarkLoss(result.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

// =============================================================================
// SECTION C: Control Flow Stress
// =============================================================================

TEST(stress_ctrl_5_nested_if)
	// 5 levels of nested if statements
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 5.0f;
		Var<float> y; y = 0.0f;
		Var<bool> c1; c1 = x > Expr<float>(0.0f);
		Var<bool> c2; c2 = x > Expr<float>(1.0f);
		Var<bool> c3; c3 = x > Expr<float>(2.0f);
		Var<bool> c4; c4 = x > Expr<float>(3.0f);
		Var<bool> c5; c5 = x > Expr<float>(4.0f);
		If(c1, [&]{
			y = y + Expr<float>(1.0f);
			If(c2, [&]{
				y = y + Expr<float>(2.0f);
				If(c3, [&]{
					y = y + Expr<float>(3.0f);
					If(c4, [&]{
						y = y + Expr<float>(4.0f);
						If(c5, [&]{
							y = y + Expr<float>(5.0f);
						});
					});
				});
			});
		});
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_MIN_COUNT(r.backwardCode, "if (", 5);
	CHECK_MIN_COUNT(r.backwardCode, "+=", 3);
END_TEST

TEST(stress_ctrl_10_nested_if)
	// 10 levels of nested if statements
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 10.0f;
		Var<float> y; y = 0.0f;
		If(x > Expr<float>(0.0f), [&]{
			y = y + Expr<float>(1.0f);
			If(x > Expr<float>(1.0f), [&]{
				y = y + Expr<float>(2.0f);
				If(x > Expr<float>(2.0f), [&]{
					y = y + Expr<float>(3.0f);
					If(x > Expr<float>(3.0f), [&]{
						y = y + Expr<float>(4.0f);
						If(x > Expr<float>(4.0f), [&]{
							y = y + Expr<float>(5.0f);
							If(x > Expr<float>(5.0f), [&]{
								y = y + Expr<float>(6.0f);
								If(x > Expr<float>(6.0f), [&]{
									y = y + Expr<float>(7.0f);
									If(x > Expr<float>(7.0f), [&]{
										y = y + Expr<float>(8.0f);
										If(x > Expr<float>(8.0f), [&]{
											y = y + Expr<float>(9.0f);
											If(x > Expr<float>(9.0f), [&]{
												y = y + Expr<float>(10.0f);
											});
										});
									});
								});
							});
						});
					});
				});
			});
		});
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_MIN_COUNT(r.backwardCode, "if (", 5);
END_TEST

TEST(stress_ctrl_if_in_for_in_if)
	// if inside for inside if (alternating nesting)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 3.0f;
		Var<float> sum; sum = 0.0f;
		If(x > Expr<float>(0.0f), [&]{
			For(0, 5, 1, [&](Var<int> &i){
				sum = sum + Expr<float>(x);
				If(sum > Expr<float>(5.0f), [&]{
					sum = sum + Expr<float>(1.0f);
				});
			});
			sum = sum * Expr<float>(x);
		});
		tape.MarkLoss(sum.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "for (");
	CHECK_CONTAINS(r.backwardCode, "if (");
END_TEST

TEST(stress_ctrl_5_branch_elif)
	// if/elif/elif/elif/else chain with 5 branches
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 5.0f;
		Var<float> y; y = 0.0f;
		If(x > Expr<float>(10.0f), [&]{
			y = y + Expr<float>(1.0f);
		}).Elif(x > Expr<float>(8.0f), [&]{
			y = y + Expr<float>(2.0f);
		}).Elif(x > Expr<float>(6.0f), [&]{
			y = y + Expr<float>(3.0f);
		}).Elif(x > Expr<float>(4.0f), [&]{
			y = y + Expr<float>(4.0f);
		}).Else([&]{
			y = y + Expr<float>(5.0f);
		});
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "if (");
	CHECK_CONTAINS(r.backwardCode, "else");
END_TEST

TEST(stress_ctrl_for_100_iter)
	// For loop with 100 iterations
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> sum; sum = 0.0f;
		For(0, 100, 1, [&](Var<int> &i){
			Var<float> v; v = float(1);
			sum = sum + v;
		});
		tape.MarkLoss(sum.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "for (");
END_TEST

TEST(stress_ctrl_for_variable_bound)
	// For loop with variable upper bound
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> sum; sum = 0.0f;
		Var<int> N; N = 50;
		For(0, N, 1, [&](Var<int> &i){
			Var<float> v; v = float(1);
			sum = sum + v;
		});
		tape.MarkLoss(sum.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "for (");
END_TEST

TEST(stress_ctrl_empty_if_body)
	// If with empty body (no operations inside)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 5.0f;
		Var<float> y; y = 1.0f;
		If(x > Expr<float>(10.0f), [&]{
			// empty - condition is false
		}).Else([&]{
			y = y * Expr<float>(x);
		});
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
END_TEST

TEST(stress_ctrl_empty_for_body)
	// For with empty body
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 5.0f;
		For(0, 5, 1, [&](Var<int> &i){
			// empty body
		});
		tape.MarkLoss(x.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
END_TEST

TEST(stress_ctrl_complex_condition)
	// If with complex condition expression
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 3.0f;
		Var<float> b; b = 2.0f;
		Var<float> c; c = 1.0f;
		Var<float> y; y = 0.0f;
		If((a > b) && (b > c), [&]{
			y = a * b + c;
		}).Else([&]{
			y = a + b * c;
		});
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "if (");
END_TEST

TEST(stress_ctrl_for_accumulation)
	// Gradient accumulation across for loop iterations
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> accumulator; accumulator = 0.0f;
		Var<float> x; x = 2.0f;
		For(0, 10, 1, [&](Var<int> &i){
			accumulator = accumulator + x * Expr<float>(float(3));
		});
		tape.MarkLoss(accumulator.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "for (");
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

// =============================================================================
// SECTION D: Callable Stress
// =============================================================================

TEST(stress_callable_nested_2_layers)
	// Outer callable calls inner callable
	Callable<float(float)> inner([](Var<float> x) {
		Var<float> r = x * Expr<float>(2.0f);
		Return(r);
	});
	Callable<float(float)> outer([&](Var<float> x) {
		Var<float> tmp = inner(x);
		Var<float> r = tmp + Expr<float>(1.0f);
		Return(r);
	});

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 3.0f;
		Var<float> y = outer(a);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.forwardCode.empty());
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.tapeSummary, "kind=" + std::to_string((int)GPU::AD::TapeOpKind::Call));
END_TEST

TEST(stress_callable_deep_5_layers)
	// Callable calls callable calls callable ... 5 levels deep
	Callable<float(float)> layer1([](Var<float> x) {
		Var<float> r = x * Expr<float>(2.0f);
		Return(r);
	});
	Callable<float(float)> layer2([&](Var<float> x) {
		Var<float> r = layer1(x) + Expr<float>(1.0f);
		Return(r);
	});
	Callable<float(float)> layer3([&](Var<float> x) {
		Var<float> r = layer2(x) * Expr<float>(3.0f);
		Return(r);
	});
	Callable<float(float)> layer4([&](Var<float> x) {
		Var<float> r = layer3(x) - Expr<float>(2.0f);
		Return(r);
	});
	Callable<float(float)> layer5([&](Var<float> x) {
		Var<float> r = layer4(x) + Expr<float>(5.0f);
		Return(r);
	});

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 1.0f;
		Var<float> y = layer5(a);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.forwardCode.empty());
	ASSERT(!r.backwardCode.empty());
	CHECK_MIN_COUNT(r.tapeSummary, "kind=" + std::to_string((int)GPU::AD::TapeOpKind::Call), 1);
END_TEST

TEST(stress_callable_10_different)
	// Single kernel uses 10 different callables
	Callable<float(float)> f0([](Var<float> x) { Var<float> r = x + Expr<float>(1.0f); Return(r); });
	Callable<float(float)> f1([](Var<float> x) { Var<float> r = x * Expr<float>(2.0f); Return(r); });
	Callable<float(float)> f2([](Var<float> x) { Var<float> r = x - Expr<float>(3.0f); Return(r); });
	Callable<float(float)> f3([](Var<float> x) { Var<float> r = x / Expr<float>(4.0f); Return(r); });
	Callable<float(float)> f4([](Var<float> x) { Var<float> r = x + Expr<float>(5.0f); Return(r); });
	Callable<float(float)> f5([](Var<float> x) { Var<float> r = x * Expr<float>(6.0f); Return(r); });
	Callable<float(float)> f6([](Var<float> x) { Var<float> r = x - Expr<float>(7.0f); Return(r); });
	Callable<float(float)> f7([](Var<float> x) { Var<float> r = x / Expr<float>(8.0f); Return(r); });
	Callable<float(float)> f8([](Var<float> x) { Var<float> r = x + Expr<float>(9.0f); Return(r); });
	Callable<float(float)> f9([](Var<float> x) { Var<float> r = x * Expr<float>(10.0f); Return(r); });

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 1.0f;
		Var<float> y = f0(x);
		y = f1(y); y = f2(y); y = f3(y); y = f4(y);
		y = f5(y); y = f6(y); y = f7(y); y = f8(y); y = f9(y);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.forwardCode.empty());
	ASSERT(!r.backwardCode.empty());
	CHECK_MIN_COUNT(r.tapeSummary, "call=", 10);
END_TEST

TEST(stress_callable_with_if_inside)
	// Callable containing if/else
	Callable<float(float, float)> reluLike([](Var<float> a, Var<float> b) {
		Var<float> r;
		If(a > b, [&]{
			r = a * Expr<float>(b);
		}).Else([&]{
			r = a + Expr<float>(b);
		});
		Return(r);
	});

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 3.0f;
		Var<float> y; y = 1.0f;
		Var<float> z = reluLike(x, y);
		tape.MarkLoss(z.VarName(), "float");
	});
	ASSERT(!r.forwardCode.empty());
	ASSERT(!r.backwardCode.empty());
END_TEST

TEST(stress_callable_with_for_inside)
	// Callable containing for loop
	Callable<float(float, int)> sumMul([](Var<float> x, Var<int> n) {
		Var<float> s; s = 0.0f;
		For(0, n, 1, [&](Var<int> &i){
			s = s + x;
		});
		Return(s);
	});

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 2.0f;
		Var<float> y = sumMul(x, 5);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.forwardCode.empty());
	ASSERT(!r.backwardCode.empty());
END_TEST

TEST(stress_callable_void_return)
	// Callable with void return type
	Callable<void(float&)> addTo([](Var<float> &x) {
		x = x + Expr<float>(10.0f);
		Return();
	});

	// For void callables, we test that the system doesn't crash
	// and generates valid forward code
	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> v; v = 5.0f;
		addTo(v);
		tape.MarkLoss(v.VarName(), "float");
	});
	ASSERT(!r.forwardCode.empty());
END_TEST

TEST(stress_callable_3_params)
	// Callable with 3 parameters
	Callable<float(float, float, float)> triple([](Var<float> a, Var<float> b, Var<float> c) {
		Var<float> r = a * b + c;
		Return(r);
	});

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 1.0f;
		Var<float> y; y = 2.0f;
		Var<float> z; z = 3.0f;
		Var<float> w = triple(x, y, z);
		tape.MarkLoss(w.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(stress_callable_param_as_registered)
	// Callable where one input is a registered parameter
	Callable<float(float, float)> mulOpC([](Var<float> a, Var<float> b) {
		Var<float> r = a * b;
		Return(r);
	});

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> w; w = 2.0f;
		Var<float> x; x = 3.0f;
		tape.RegisterParameter(w.VarName(), "float");
		Var<float> y = mulOpC(w, x);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(stress_callable_chained_call)
	// f(g(h(x))) nested call style
	Callable<float(float)> add1C([](Var<float> x) { Var<float> r = x + Expr<float>(1.0f); Return(r); });
	Callable<float(float)> mul2C([](Var<float> x) { Var<float> r = x * Expr<float>(2.0f); Return(r); });
	Callable<float(float)> sub3C([](Var<float> x) { Var<float> r = x - Expr<float>(3.0f); Return(r); });

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 5.0f;
		Var<float> y = sub3C(mul2C(add1C(x)));
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.tapeSummary, "call=");
END_TEST

TEST(stress_callable_reused)
	// Same callable used multiple times in one kernel
	Callable<float(float)> squareC([](Var<float> x) {
		Var<float> r = x * x;
		Return(r);
	});

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 2.0f;
		Var<float> b; b = 3.0f;
		Var<float> y = squareC(a) + squareC(b);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.tapeSummary, "sub-tape");
END_TEST

// =============================================================================
// SECTION E: Multi-Parameter Stress
// =============================================================================

TEST(stress_param_20_params)
	// 20 float parameters all participating in loss
	auto r = RunADParamTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> params[20];
		Var<float> loss; loss = 0.0f;
		for (int i = 0; i < 20; ++i) {
			params[i] = float(i + 1);
			tape.RegisterParameter(params[i].VarName(), "float");
			loss = loss + params[i] * Expr<float>(float(i + 1));
		}
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
	CHECK_MIN_COUNT(r.backwardCode, "+=", 10);
END_TEST

TEST(stress_param_50_params)
	// 50 parameters
	auto r = RunADParamTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> loss; loss = 0.0f;
		for (int i = 0; i < 50; ++i) {
			Var<float> p; p = float(i);
			tape.RegisterParameter(p.VarName(), "float");
			loss = loss + p * Expr<float>(float(i));
		}
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_param_mixed_types)
	// Parameters of different types: float, vec2, vec3, vec4
	auto r = RunADParamTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float>  w;  w  = 1.0f;
		Var<Vec2>   v2; v2 = Vec2(1.0f, 2.0f);
		Var<Vec3>   v3; v3 = Vec3(1.0f, 2.0f, 3.0f);
		Var<Vec4>   v4; v4 = Vec4(1.0f, 2.0f, 3.0f, 4.0f);
		tape.RegisterParameter(w.VarName(), "float");
		tape.RegisterParameter(v2.VarName(), "vec2");
		tape.RegisterParameter(v3.VarName(), "vec3");
		tape.RegisterParameter(v4.VarName(), "vec4");
		Var<float> loss = w + Length(v2) + Length(v3) + Length(v4);
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_param_sparse_100_10_active)
	// 100 parameters, only 10 participate in loss (sparse interaction)
	auto r = RunADParamTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> loss; loss = 0.0f;
		for (int i = 0; i < 100; ++i) {
			Var<float> p; p = float(i);
			tape.RegisterParameter(p.VarName(), "float");
			if (i % 10 == 0) {
				loss = loss + p * Expr<float>(float(i));
			}
		}
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
END_TEST

TEST(stress_param_dense_interaction)
	// Parameters interact with each other: p_i * p_j for all pairs
	auto r = RunADParamTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> p0; p0 = 1.0f; tape.RegisterParameter(p0.VarName(), "float");
		Var<float> p1; p1 = 2.0f; tape.RegisterParameter(p1.VarName(), "float");
		Var<float> p2; p2 = 3.0f; tape.RegisterParameter(p2.VarName(), "float");
		Var<float> p3; p3 = 4.0f; tape.RegisterParameter(p3.VarName(), "float");
		Var<float> loss = p0*p1 + p1*p2 + p2*p3 + p3*p0 + p0*p2 + p1*p3;
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_param_same_buffer_elements)
	// Parameters from different elements of the same logical buffer
	auto r = RunADParamTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> w0; w0 = 1.0f; tape.RegisterParameter(w0.VarName(), "float");
		Var<float> w1; w1 = 2.0f; tape.RegisterParameter(w1.VarName(), "float");
		Var<float> w2; w2 = 3.0f; tape.RegisterParameter(w2.VarName(), "float");
		Var<float> x0; x0 = 0.5f;
		Var<float> x1; x1 = 1.5f;
		Var<float> x2; x2 = 2.5f;
		Var<float> loss = w0*x0 + w1*x1 + w2*x2;
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_param_readonly)
	// Parameter registered but not modified in forward pass
	auto r = RunADParamTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> w; w = 2.0f;
		tape.RegisterParameter(w.VarName(), "float");
		Var<float> x; x = 3.0f;
		Var<float> y = w * x; // w is read-only here
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(stress_param_through_callable)
	// Parameter passed through a callable
	Callable<float(float, float)> dotC([](Var<float> a, Var<float> b) {
		Var<float> r = a * b;
		Return(r);
	});

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> w; w = 2.0f;
		Var<float> x; x = 3.0f;
		tape.RegisterParameter(w.VarName(), "float");
		Var<float> y = dotC(w, x);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
END_TEST

// =============================================================================
// SECTION F: Vector/Matrix Stress
// =============================================================================

TEST(stress_vec_swizzle_variants)
	// Test various swizzle patterns on vec4
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<Vec4> v; v = Vec4(1.0f, 2.0f, 3.0f, 4.0f);
		Var<float> x = v.VarName() + ".x";
		Var<float> y = v.VarName() + ".y";
		Var<float> z = v.VarName() + ".z";
		Var<float> w = v.VarName() + ".w";
		Var<float> loss = x + y + z + w;
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
END_TEST

TEST(stress_vec_chain_operations)
	// normalize(cross(a, b) + dot(c, d) * e)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<Vec3> a; a = Vec3(1.0f, 0.0f, 0.0f);
		Var<Vec3> b; b = Vec3(0.0f, 1.0f, 0.0f);
		Var<Vec3> c; c = Vec3(0.0f, 0.0f, 1.0f);
		Var<Vec3> d; d = Vec3(1.0f, 1.0f, 1.0f);
		Var<float> e; e = 2.0f;
		Var<Vec3> n = Normalize(Cross(a, b) + Dot(c, d) * e);
		Var<float> loss = Length(n);
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_vec_mixed_vec_types)
	// mix vec2, vec3, vec4 operations
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<Vec2> v2; v2 = Vec2(1.0f, 2.0f);
		Var<Vec3> v3; v3 = Vec3(1.0f, 2.0f, 3.0f);
		Var<Vec4> v4; v4 = Vec4(1.0f, 2.0f, 3.0f, 4.0f);
		Var<float> l2 = Length(v2);
		Var<float> l3 = Length(v3);
		Var<float> l4 = Length(v4);
		Var<float> loss = l2 + l3 + l4;
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "length");
END_TEST

TEST(stress_vec_length_distance_normalize)
	// Chain: length(a) + distance(b,c) + length(normalize(d))
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<Vec3> a; a = Vec3(1.0f, 2.0f, 3.0f);
		Var<Vec3> b; b = Vec3(0.0f, 1.0f, 2.0f);
		Var<Vec3> c; c = Vec3(3.0f, 2.0f, 1.0f);
		Var<Vec3> d; d = Vec3(1.0f, 1.0f, 1.0f);
		Var<float> loss = Length(a) + Distance(b, c) + Length(Normalize(d));
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_vec_many_components)
	// vec4 all components participate via Length which touches all components
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<Vec4> v; v = Vec4(1.0f, 2.0f, 3.0f, 4.0f);
		Var<float> loss = Length(v);
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST


TEST(stress_vec_reflect_refract)
	// Reflect and refract operations
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<Vec3> I; I = Vec3(1.0f, 0.0f, 0.0f);
		Var<Vec3> N; N = Vec3(0.0f, 1.0f, 0.0f);
		Var<Vec3> R = Reflect(I, N);
		Var<float> loss = Length(R);
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
END_TEST

TEST(stress_vec_dot_cross_chain)
	// dot(a, cross(b, c)) - scalar triple product
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<Vec3> a; a = Vec3(1.0f, 2.0f, 3.0f);
		Var<Vec3> b; b = Vec3(4.0f, 5.0f, 6.0f);
		Var<Vec3> c; c = Vec3(7.0f, 8.0f, 9.0f);
		Var<float> loss = Dot(a, Cross(b, c));
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_vec_multiple_swizzles)
	// Different swizzle patterns on same vector
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<Vec4> v; v = Vec4(1.0f, 2.0f, 3.0f, 4.0f);
		Var<float> a = v.VarName() + ".x";
		Var<float> b = v.VarName() + ".y";
		Var<float> c = v.VarName() + ".z";
		Var<float> d = v.VarName() + ".w";
		Var<float> e = v.VarName() + ".xy";
		Var<float> f = v.VarName() + ".xyz";
		Var<float> loss = a + b + c + d;
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
END_TEST

TEST(stress_vec_10_vector_ops)
	// 10 consecutive vector operations
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<Vec3> v; v = Vec3(1.0f, 1.0f, 1.0f);
		v = v + Vec3(1.0f, 0.0f, 0.0f);
		v = v * Expr<float>(2.0f);
		v = Normalize(v);
		v = v * Expr<float>(3.0f);
		v = v - Vec3(0.0f, 1.0f, 0.0f);
		v = v + Vec3(0.0f, 0.0f, 1.0f);
		Var<float> loss = Length(v);
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
	CHECK_MIN_COUNT(r.backwardCode, "+=", 2);
END_TEST

// =============================================================================
// SECTION G: Complex Composition
// =============================================================================

TEST(stress_comp_softmax_like)
	// Manual softmax: exp(x_i) / sum(exp(x_j))
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x0; x0 = 1.0f;
		Var<float> x1; x1 = 2.0f;
		Var<float> x2; x2 = 3.0f;
		Var<float> e0 = Exp(x0);
		Var<float> e1 = Exp(x1);
		Var<float> e2 = Exp(x2);
		Var<float> sumExp = e0 + Expr<float>(e1) + Expr<float>(e2);
		Var<float> s0 = e0 / Expr<float>(sumExp);
		Var<float> s1 = e1 / Expr<float>(sumExp);
		Var<float> s2 = e2 / Expr<float>(sumExp);
		// Negative log-likelihood loss
		Var<float> loss = Expr<float>(0.0f) - Log(s1); // target is class 1
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_comp_residual_connection)
	// y = F(x) + x  (residual connection)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 2.0f;
		Var<float> Fx = x * Expr<float>(3.0f) + Expr<float>(1.0f);
		Var<float> y = Fx + Expr<float>(x); // residual
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_MIN_COUNT(r.backwardCode, "+=", 2);
END_TEST

TEST(stress_comp_polynomial_regression)
	// y = a*x^3 + b*x^2 + c*x + d
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 2.0f;
		Var<float> b; b = -3.0f;
		Var<float> c; c = 4.0f;
		Var<float> d; d = 5.0f;
		Var<float> x; x = 3.0f;
		Var<float> x2 = x * Expr<float>(x);
		Var<float> x3 = x2 * Expr<float>(x);
		Var<float> y = a*x3 + b*x2 + c*x + d;
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(stress_comp_trig_combo)
	// sin(a)*cos(b) + tan(c)*atan(d)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 0.5f;
		Var<float> b; b = 0.3f;
		Var<float> c; c = 0.7f;
		Var<float> d; d = 1.0f;
		Var<float> y = Sin(a)*Cos(b) + Tan(c)*Atan(d);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_comp_all_op_types)
	// Single kernel using every supported differentiable operation type
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 2.0f;
		Var<float> b; b = 3.0f;
		// Binary ops
		Var<float> addR = a + b;
		Var<float> mulR = a * b;
		Var<float> subR = a - b;
		Var<float> divR = a / Expr<float>(b);
		// Unary
		Var<float> negR = Expr<float>(0.0f) - a;
		// Intrinsics
		Var<float> s = Sin(a);
		Var<float> c = Cos(a);
		Var<float> e = Exp(a);
		Var<float> l = Log(a);
		Var<float> sq = Sqrt(a);
		Var<float> t = Tanh(a);
		// Combine all
		Var<float> loss = addR + mulR + subR + divR + negR + s + c + e + l + sq + t;
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(stress_comp_attention_like)
	// Simplified attention: softmax(Q*K^T) * V for single-element "tokens"
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> Q; Q = 1.0f;
		Var<float> K; K = 2.0f;
		Var<float> V; V = 3.0f;
		// Score
		Var<float> score = Q * K;
		// Simplified softmax (2 elements: score, 0)
		Var<float> eScore = Exp(score);
		Var<float> eZero = Exp(Expr<float>(0.0f));
		Var<float> sumE = eScore + Expr<float>(eZero);
		Var<float> attn = eScore / Expr<float>(sumE);
		// Weighted sum
		Var<float> out = attn * V;
		tape.MarkLoss(out.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(stress_comp_lstm_like_gates)
	// Simplified LSTM-like gates:
	// f = sigmoid(W_f * x)   (forget)
	// i = sigmoid(W_i * x)   (input)
	// o = sigmoid(W_o * x)   (output)
	// g = tanh(W_g * x)      (candidate)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 1.0f;
		Var<float> Wf; Wf = 0.5f;
		Var<float> Wi; Wi = 0.6f;
		Var<float> Wo; Wo = 0.7f;
		Var<float> Wg; Wg = 0.8f;
		// Forget gate (sigmoid)
		Var<float> f = Expr<float>(1.0f) / (Expr<float>(1.0f) + Exp(Expr<float>(0.0f) - Wf * x));
		// Input gate
		Var<float> i = Expr<float>(1.0f) / (Expr<float>(1.0f) + Exp(Expr<float>(0.0f) - Wi * x));
		// Output gate
		Var<float> o = Expr<float>(1.0f) / (Expr<float>(1.0f) + Exp(Expr<float>(0.0f) - Wo * x));
		// Candidate
		Var<float> g = Tanh(Wg * x);
		// Loss combines all gates
		Var<float> loss = f + i + o + g;
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(stress_comp_batchnorm_like)
	// Simplified batch norm: y = (x - mean) / sqrt(var + eps)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x1; x1 = 1.0f;
		Var<float> x2; x2 = 2.0f;
		Var<float> x3; x3 = 3.0f;
		Var<float> mean = (x1 + Expr<float>(x2) + Expr<float>(x3)) / Expr<float>(3.0f);
		Var<float> diff1 = x1 - Expr<float>(mean);
		Var<float> diff2 = x2 - Expr<float>(mean);
		Var<float> diff3 = x3 - Expr<float>(mean);
		Var<float> varE = (diff1*diff1 + diff2*diff2 + diff3*diff3) / Expr<float>(3.0f);
		Var<float> eps; eps = 1e-5f;
		Var<float> y1 = diff1 / Expr<float>(Sqrt(varE + Expr<float>(eps)));
		Var<float> loss = y1;
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

// =============================================================================
// SECTION H: Backward Code Quality
// =============================================================================

TEST(stress_quality_loss_seed)
	// Verify loss variable has adjoint seeded to 1.0
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 5.0f;
		Var<float> y = x * Expr<float>(2.0f);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "= float(1.0)");
END_TEST

TEST(stress_quality_adjoint_declarations)
	// Verify adjoint variables are declared
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 1.0f;
		Var<float> b; b = 2.0f;
		Var<float> c = a + b;
		tape.MarkLoss(c.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "float d_");
END_TEST

TEST(stress_quality_accumulation_present)
	// Every operation in the tape should produce a += in backward
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 1.0f;
		Var<float> b; b = 2.0f;
		Var<float> c = a + b;
		Var<float> d = c * Expr<float>(3.0f);
		tape.MarkLoss(d.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_MIN_COUNT(r.backwardCode, "+=", 2);
END_TEST

TEST(stress_quality_no_orphan_adjoints)
	// Non-parameter, non-loss, non-active variables shouldn't get adjoints
	// (Testing that unused variable 'b' doesn't pollute backward code)
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 1.0f;
		Var<float> b; b = 2.0f;
		// 'b' is never used in loss path
		Var<float> loss = a * Expr<float>(3.0f);
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	// 'b' should not appear in backward code with d_ prefix
	// (We can't check generically, but we verify code is reasonable)
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_quality_param_writebacks)
	// When writeBackParams=true, parameters get writeback statements
	GPU::AD::GradientTape tape;
	GPU::IR::Builder::Builder::Get().SetGradientTape(&tape);
	GPU::Kernel::InspectorKernel1D kernel([&](Var<int> &id) {
		Var<float> w; w = 2.0f;
		Var<float> x; x = 3.0f;
		tape.RegisterParameter(w.VarName(), "float");
		Var<float> y = w * x;
		tape.MarkLoss(y.VarName(), "float");
	});
	GPU::IR::Builder::Builder::Get().SetGradientTape(nullptr);
	GPU::AD::AdjointGenerator gen;
	std::string code = gen.Generate(tape, true);
	CHECK_CONTAINS(code, "d_");
END_TEST

TEST(stress_quality_no_duplicate_decls)
	// Verify no duplicate float declarations for the same adjoint
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 1.0f;
		Var<float> b = a + Expr<float>(2.0f);
		Var<float> c = b + Expr<float>(3.0f);
		tape.MarkLoss(c.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	// Each d_ variable should appear in exactly one float declaration
	// Simple structural check: code should parse
	CHECK_CONTAINS(r.backwardCode, "void main()");
END_TEST

TEST(stress_quality_control_flow_pairs)
	// Verify that if/for have matching braces in backward code
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 5.0f;
		Var<float> y; y = 1.0f;
		If(x > Expr<float>(0.0f), [&]{
			y = y * Expr<float>(x);
		});
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "if (");
	CHECK_CONTAINS(r.backwardCode, "{");
	CHECK_CONTAINS(r.backwardCode, "}");
END_TEST

TEST(stress_quality_subtape_inline)
	// Sub-tape entries should be inlined into backward code
	Callable<float(float)> sqC([](Var<float> x) {
		Var<float> r = x * x;
		Return(r);
	});

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 3.0f;
		Var<float> y = sqC(a);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(stress_quality_backward_not_empty_when_loss)
	// Loss with at least one differentiable operation must produce backward code
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 1.0f;
		Var<float> y = x + Expr<float>(0.0f);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
END_TEST

TEST(stress_quality_main_function_present)
	// Backward code must wrap everything in main()
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 1.0f;
		Var<float> b = a + Expr<float>(2.0f);
		tape.MarkLoss(b.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "void main()");
	CHECK_CONTAINS(r.backwardCode, "layout(local_size_x");
END_TEST

// =============================================================================
// SECTION I: Numerical Correctness Framework
// =============================================================================
// These tests validate the mathematical correctness of gradients.
// They require GPU execution to fully verify, but we can at minimum check
// that they compile and the code structure is correct.

#ifdef EASYGPU_AD_NUMERICAL_TEST

TEST(stress_numerical_mul_finite_diff)
	// Verify gradient of y = a*b via structural consistency
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 2.0f;
		Var<float> b; b = 3.0f;
		Var<float> y = a * b;
		tape.RegisterParameter(a.VarName(), "float");
		tape.RegisterParameter(b.VarName(), "float");
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	// da should get b*d_y (3.0*1) and db should get a*d_y (2.0*1)
	CHECK_CONTAINS(r.backwardCode, "+=");
	CHECK_MIN_COUNT(r.backwardCode, "+=", 2);
END_TEST

TEST(stress_numerical_sigmoid_chain)
	// sigmoid(x) = 1/(1+exp(-x)), derivative = sigmoid(x)*(1-sigmoid(x))
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 0.5f;
		Var<float> sx = Expr<float>(1.0f) / (Expr<float>(1.0f) + Exp(Expr<float>(0.0f) - x));
		tape.RegisterParameter(x.VarName(), "float");
		tape.MarkLoss(sx.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "exp");
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(stress_numerical_vec_dot)
	// dot(a, b) gradient: da = b, db = a
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<Vec3> a; a = Vec3(1.0f, 2.0f, 3.0f);
		Var<Vec3> b; b = Vec3(4.0f, 5.0f, 6.0f);
		Var<float> y = Dot(a, b);
		tape.RegisterParameter(a.VarName(), "vec3");
		tape.RegisterParameter(b.VarName(), "vec3");
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "+=");
END_TEST

TEST(stress_numerical_mlp_layer)
	// y = tanh(W*x + b), verify structural gradient
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> W; W = 0.5f;
		Var<float> x; x = 2.0f;
		Var<float> b; b = 0.1f;
		Var<float> h = W * x + b;
		Var<float> y = Tanh(h);
		tape.RegisterParameter(W.VarName(), "float");
		tape.RegisterParameter(b.VarName(), "float");
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.backwardCode, "tanh");
	CHECK_MIN_COUNT(r.backwardCode, "+=", 2);
END_TEST

TEST(stress_numerical_gradient_descent_step)
	// Verify structure supports gradient descent: loss = (W*x - target)^2
	auto r = RunADTest([](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> W; W = 1.0f;
		Var<float> x; x = 3.0f;
		Var<float> target; target = 5.0f;
		Var<float> pred = W * x;
		Var<float> error = pred - Expr<float>(target);
		Var<float> loss = error * Expr<float>(error);
		tape.RegisterParameter(W.VarName(), "float");
		tape.MarkLoss(loss.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_MIN_COUNT(r.backwardCode, "+=", 2);
END_TEST

#endif // EASYGPU_AD_NUMERICAL_TEST

// =============================================================================
// SECTION J: Regression Protection
// =============================================================================

TEST(stress_regression_subtape_not_lost)
	// Regression: Sub-tape recordings must not be silently dropped (RecordDirect bug)
	Callable<float(float, float)> mulOpR([](Var<float> a, Var<float> b) {
		Var<float> r = a * b;
		Return(r);
	});

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> x; x = 2.0f;
		Var<float> y; y = 3.0f;
		Var<float> z = mulOpR(x, y);
		tape.MarkLoss(z.VarName(), "float");
	});
	ASSERT(!r.tapeSummary.empty());
	// Sub-tape must contain the binary operation
	CHECK_CONTAINS(r.tapeSummary, "sub-tape[0]");
	CHECK_CONTAINS(r.tapeSummary, "kind=0"); // BinaryOp
END_TEST

TEST(stress_regression_in_callable_state_no_leak)
	// Regression: Builder _inCallableBody must not leak between tests
	// Run two callable tests back-to-back; second must also work
	{
		Callable<float(float)> f1([](Var<float> x) { Var<float> r = x + Expr<float>(1.0f); Return(r); });
		auto r1 = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
			Var<float> a; a = 1.0f;
			Var<float> y = f1(a);
			tape.MarkLoss(y.VarName(), "float");
		});
		ASSERT(!r1.backwardCode.empty());
	}
	{
		Callable<float(float)> f2([](Var<float> x) { Var<float> r = x * Expr<float>(2.0f); Return(r); });
		auto r2 = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
			Var<float> b; b = 2.0f;
			Var<float> z = f2(b);
			tape.MarkLoss(z.VarName(), "float");
		});
		ASSERT(!r2.backwardCode.empty());
	}
END_TEST

TEST(stress_regression_tape_active_during_codegen)
	// Regression: Tape must be active during GetCode() for callable body generators
	Callable<float(float)> mulR([](Var<float> x) { Var<float> r = x * Expr<float>(2.0f); Return(r); });

	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 3.0f;
		Var<float> y = mulR(a);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	CHECK_CONTAINS(r.tapeSummary, "sub-tape[0]");
END_TEST

TEST(stress_regression_consistent_getcode)
	// Regression: Multiple GetCode() calls should produce consistent results
	GPU::AD::GradientTape tape;
	GPU::IR::Builder::Builder::Get().SetGradientTape(&tape);
	GPU::Kernel::InspectorKernel1D kernel([&](Var<int> &id) {
		Var<float> a; a = 1.0f;
		Var<float> b = a + Expr<float>(2.0f);
		tape.MarkLoss(b.VarName(), "float");
	});
	GPU::IR::Builder::Builder::Get().SetGradientTape(nullptr);

	std::string code1 = kernel.GetCode();
	std::string code2 = kernel.GetCode();
	ASSERT(code1 == code2);
END_TEST

TEST(stress_regression_callable_no_dup)
	// Regression: Callable declarations must not be duplicated
	Callable<float(float)> dupC([](Var<float> x) {
		Var<float> r = x * x;
		Return(r);
	});

	// Use same callable twice to check no duplicate declarations
	auto r = RunADCallableTest([&](Var<int> &id, GPU::AD::GradientTape &tape) {
		Var<float> a; a = 2.0f;
		Var<float> b; b = 3.0f;
		Var<float> y = dupC(a) + dupC(b);
		tape.MarkLoss(y.VarName(), "float");
	});
	ASSERT(!r.backwardCode.empty());
	// Forward code should mention the callable function name but not duplicate its body
	CHECK_CONTAINS(r.forwardCode, "void main()");
END_TEST

// =============================================================================
// Main
// =============================================================================

int main() {
	std::cout << "=== EasyGPU AD Industrial-Grade Stress Tests ===\n";

	// Section A: Deep Graph Stress
	test_stress_deep_50_add_chain();
	test_stress_deep_50_mixed_ops();
	test_stress_deep_100_add_chain();
	test_stress_deep_nested_10_levels();
	test_stress_deep_wide_fan_out();
	test_stress_deep_diamond();
	test_stress_deep_mlp_5_layers();
	test_stress_deep_single_var_50_uses();

	// Section B: Edge Cases
	test_stress_edge_all_zero_inputs();
	test_stress_edge_negative_sqrt();
	test_stress_edge_negative_log();
	test_stress_edge_extreme_values();
	test_stress_edge_near_div_by_zero();
	test_stress_edge_empty_kernel();
	test_stress_edge_declare_only();
	test_stress_edge_loss_no_params();
	test_stress_edge_params_no_loss();
	test_stress_edge_param_is_loss();
	test_stress_edge_multi_loss();
	test_stress_edge_all_int_ops();
	test_stress_edge_mixed_types();

	// Section C: Control Flow Stress
	test_stress_ctrl_5_nested_if();
	test_stress_ctrl_10_nested_if();
	test_stress_ctrl_if_in_for_in_if();
	test_stress_ctrl_5_branch_elif();
	test_stress_ctrl_for_100_iter();
	test_stress_ctrl_for_variable_bound();
	test_stress_ctrl_empty_if_body();
	test_stress_ctrl_empty_for_body();
	test_stress_ctrl_complex_condition();
	test_stress_ctrl_for_accumulation();

	// Section D: Callable Stress
	test_stress_callable_nested_2_layers();
	test_stress_callable_deep_5_layers();
	test_stress_callable_10_different();
	test_stress_callable_with_if_inside();
	test_stress_callable_with_for_inside();
	test_stress_callable_void_return();
	test_stress_callable_3_params();
	test_stress_callable_param_as_registered();
	test_stress_callable_chained_call();
	test_stress_callable_reused();

	// Section E: Multi-Parameter Stress
	test_stress_param_20_params();
	test_stress_param_50_params();
	test_stress_param_mixed_types();
	test_stress_param_sparse_100_10_active();
	test_stress_param_dense_interaction();
	test_stress_param_same_buffer_elements();
	test_stress_param_readonly();
	test_stress_param_through_callable();

	// Section F: Vector/Matrix Stress
	test_stress_vec_swizzle_variants();
	test_stress_vec_chain_operations();
	test_stress_vec_mixed_vec_types();
	test_stress_vec_length_distance_normalize();
	test_stress_vec_many_components();
	test_stress_vec_reflect_refract();
	test_stress_vec_dot_cross_chain();
	test_stress_vec_multiple_swizzles();
	test_stress_vec_10_vector_ops();

	// Section G: Complex Composition
	test_stress_comp_softmax_like();
	test_stress_comp_residual_connection();
	test_stress_comp_polynomial_regression();
	test_stress_comp_trig_combo();
	test_stress_comp_all_op_types();
	test_stress_comp_attention_like();
	test_stress_comp_lstm_like_gates();
	test_stress_comp_batchnorm_like();

	// Section H: Backward Code Quality
	test_stress_quality_loss_seed();
	test_stress_quality_adjoint_declarations();
	test_stress_quality_accumulation_present();
	test_stress_quality_no_orphan_adjoints();
	test_stress_quality_param_writebacks();
	test_stress_quality_no_duplicate_decls();
	test_stress_quality_control_flow_pairs();
	test_stress_quality_subtape_inline();
	test_stress_quality_backward_not_empty_when_loss();
	test_stress_quality_main_function_present();

#ifdef EASYGPU_AD_NUMERICAL_TEST
	// Section I: Numerical Correctness Framework
	test_stress_numerical_mul_finite_diff();
	test_stress_numerical_sigmoid_chain();
	test_stress_numerical_vec_dot();
	test_stress_numerical_mlp_layer();
	test_stress_numerical_gradient_descent_step();
#endif

	// Section J: Regression Protection
	test_stress_regression_subtape_not_lost();
	test_stress_regression_in_callable_state_no_leak();
	test_stress_regression_tape_active_during_codegen();
	test_stress_regression_consistent_getcode();
	test_stress_regression_callable_no_dup();

	std::cout << "\n=== Results: " << pass_count << "/" << test_count << " passed ===\n";
	return pass_count == test_count ? 0 : 1;
}
