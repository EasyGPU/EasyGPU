/**
 * @file TestNestedControlFlow.cpp
 * @brief Tests deeply nested and mixed control flow constructs.
 */

#include <GPU.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;

#define TEST(name)                                                                                                     \
	void test_##name() {                                                                                               \
		std::cout << "\n[TEST] " #name " ... ";                                                                        \
		try {

#define END_TEST                                                                                                       \
	std::cout << "PASSED\n";                                                                                           \
	}                                                                                                                  \
	catch (const std::exception &e) {                                                                                  \
		std::cout << "FAILED: " << e.what() << "\n";                                                                   \
		throw;                                                                                                         \
	}                                                                                                                  \
	}

#define ASSERT(cond)                                                                                                   \
	if (!(cond)) {                                                                                                     \
		throw std::runtime_error("Assertion failed: " #cond);                                                          \
	}

#define ASSERT_EQ(a, b)                                                                                                \
	if ((a) != (b)) {                                                                                                  \
		throw std::runtime_error("Assertion failed: " #a " != " #b);                                                   \
	}

// =============================================================================
// Code Generation Tests (InspectorKernel)
// =============================================================================

TEST(nested_if_inside_if)
InspectorKernel1D inspector([&](Int i) {
	If(i > 0, [&]() {
		If(i > 10, [&]() { Var<float> a = MakeFloat(1.0f); }).Else([&]() { Var<float> b = MakeFloat(2.0f); });
	});
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("if (") != std::string::npos);
// Must have at least two if statements
int	   count = 0;
size_t pos	 = 0;
while ((pos = code.find("if (", pos)) != std::string::npos) {
	++count;
	++pos;
}
ASSERT(count >= 2);
END_TEST

TEST(if_inside_for)
InspectorKernel1D inspector([&](Int i) {
	For(0, 10, [&](Int &j) { If(j > 5, [&]() { Var<float> a = MakeFloat(1.0f); }); });
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("for (") != std::string::npos);
ASSERT(code.find("if (") != std::string::npos);
END_TEST

TEST(for_inside_if)
InspectorKernel1D inspector([&](Int i) {
	If(i > 0, [&]() { For(0, 5, [&](Int &j) { Var<float> a = MakeFloat(1.0f); }); });
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("for (") != std::string::npos);
ASSERT(code.find("if (") != std::string::npos);
END_TEST

TEST(nested_for_loops)
InspectorKernel1D inspector([&](Int i) {
	For(0, 4, [&](Int &j) { For(0, 4, [&](Int &k) { Var<float> a = MakeFloat(1.0f); }); });
});
std::string		  code	= inspector.GetCode();
int				  count = 0;
size_t			  pos	= 0;
while ((pos = code.find("for (", pos)) != std::string::npos) {
	++count;
	++pos;
}
ASSERT(count >= 2);
END_TEST

TEST(while_inside_if)
InspectorKernel1D inspector([&](Int i) {
	If(i > 0, [&]() {
		Var<int> counter = MakeInt(0);
		While(counter < 5, [&]() { counter = counter + MakeInt(1); });
	});
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("while (") != std::string::npos);
ASSERT(code.find("if (") != std::string::npos);
END_TEST

TEST(break_inside_nested_for)
InspectorKernel1D inspector([&](Int i) { For(0, 10, [&](Int &j) { If(j > 5, [&]() { Break(); }); }); });
std::string		  code = inspector.GetCode();
ASSERT(code.find("break;") != std::string::npos);
END_TEST

TEST(continue_inside_nested_if)
InspectorKernel1D inspector([&](Int i) {
	For(0, 10, [&](Int &j) {
		If(j % 2 == 0, [&]() { Continue(); });
		Var<float> a = MakeFloat(1.0f);
	});
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("continue;") != std::string::npos);
END_TEST

TEST(complex_nested_control_flow)
InspectorKernel1D inspector([&](Int i) {
	If(i > 0, [&]() {
		For(0, 4, [&](Int &j) {
			If(j > 2, [&]() {
				Var<int> c = MakeInt(0);
				While(c < 3, [&]() {
					If(c == 1, [&]() { Break(); });
					c = c + MakeInt(1);
				});
			}).Elif(j == 1, [&]() { Continue(); });
		});
	}).Else([&]() { Var<float> d = MakeFloat(99.0f); });
});
std::string		  code = inspector.GetCode();
ASSERT(code.find("if (") != std::string::npos);
ASSERT(code.find("for (") != std::string::npos);
ASSERT(code.find("while (") != std::string::npos);
ASSERT(code.find("break;") != std::string::npos);
ASSERT(code.find("continue;") != std::string::npos);
ASSERT(code.find("} else {") != std::string::npos);
END_TEST

// =============================================================================
// Runtime Execution Tests
// =============================================================================

TEST(runtime_nested_if_correctness)
constexpr int	 N = 64;
std::vector<int> input(N);
for (int i = 0; i < N; ++i) {
	input[i] = i;
}
Runtime::Buffer<int> bufIn(input);
Runtime::Buffer<int> bufOut(N);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto	 in		= bufIn.Bind();
		auto	 out	= bufOut.Bind();
		Var<int> v		= in[id];
		Var<int> result = MakeInt(0);
		If(v > 30, [&]() {
			If(v > 50, [&]() { result = MakeInt(3); }).Else([&]() { result = MakeInt(2); });
		}).Else([&]() { result = MakeInt(1); });
		out[id] = result;
	},
	256);

kernel.Dispatch(1, true);

std::vector<int> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	int expected = (i > 30) ? ((i > 50) ? 3 : 2) : 1;
	ASSERT_EQ(output[i], expected);
}
END_TEST

TEST(runtime_for_loop_accumulation)
constexpr int		 N = 64;
Runtime::Buffer<int> bufOut(N);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto	 out = bufOut.Bind();
		Var<int> sum = MakeInt(0);
		For(0, id + 1, [&](Int &j) { sum = sum + j; });
		out[id] = sum;
	},
	256);

kernel.Dispatch(1, true);

std::vector<int> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	int expected = i * (i + 1) / 2; // sum(0..i)
	ASSERT_EQ(output[i], expected);
}
END_TEST

TEST(runtime_for_with_step)
constexpr int		 N = 64;
Runtime::Buffer<int> bufOut(N);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto	 out = bufOut.Bind();
		Var<int> sum = MakeInt(0);
		// Sum even numbers up to id*2
		For(0, id * 2 + 1, 2, [&](Int &j) { sum = sum + j; });
		out[id] = sum;
	},
	256);

kernel.Dispatch(1, true);

std::vector<int> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	// sum of 0, 2, 4, ..., 2i = 2 * (0+1+...+i) = i*(i+1)
	int expected = i * (i + 1);
	ASSERT_EQ(output[i], expected);
}
END_TEST

TEST(runtime_break_in_loop)
constexpr int		 N = 64;
Runtime::Buffer<int> bufOut(N);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto	 out   = bufOut.Bind();
		Var<int> count = MakeInt(0);
		For(0, 100, [&](Int &j) {
			If(j >= id, [&]() { Break(); });
			count = count + MakeInt(1);
		});
		out[id] = count;
	},
	256);

kernel.Dispatch(1, true);

std::vector<int> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	// count = min(id, 100) but since id < 64, count = id
	ASSERT_EQ(output[i], i);
}
END_TEST

TEST(runtime_continue_in_loop)
constexpr int		 N = 64;
Runtime::Buffer<int> bufOut(N);

Kernel1D			 kernel(
	[&, N](Var<int> &id) {
		auto	 out = bufOut.Bind();
		Var<int> sum = MakeInt(0);
		For(0, id + 1, [&](Int &j) {
			If(j % 2 == 0, [&]() { Continue(); });
			sum = sum + j;
		});
		out[id] = sum;
	},
	256);

kernel.Dispatch(1, true);

std::vector<int> output(N);
bufOut.Download(output.data(), N);
for (int i = 0; i < N; ++i) {
	// Sum of odd numbers up to i
	int expected = 0;
	for (int j = 0; j <= i; ++j) {
		if (j % 2 != 0)
			expected += j;
	}
	ASSERT_EQ(output[i], expected);
}
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "  EasyGPU Nested Control Flow Tests     " << std::endl;
	std::cout << "========================================" << std::endl;

	try {
		test_nested_if_inside_if();
		test_if_inside_for();
		test_for_inside_if();
		test_nested_for_loops();
		test_while_inside_if();
		test_break_inside_nested_for();
		test_continue_inside_nested_if();
		test_complex_nested_control_flow();
		test_runtime_nested_if_correctness();
		test_runtime_for_loop_accumulation();
		test_runtime_for_with_step();
		test_runtime_break_in_loop();
		test_runtime_continue_in_loop();

		std::cout << "\n========================================" << std::endl;
		std::cout << "  All nested control flow tests passed! " << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << std::endl;
		return 1;
	}
}
