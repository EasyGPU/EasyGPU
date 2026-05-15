/**
 * @file TestInspectorValidate.cpp
 * @brief GLSL validation (no GPU required). Tests for.
 */

#include <Flow/IfFlow.h>
#include <Kernel/Kernel.h>
#include <Runtime/Buffer.h>

#include <iostream>
#include <string>

using namespace GPU;
using namespace GPU::Flow;
using namespace GPU::IR::Value;
using namespace GPU::Runtime;
using namespace GPU::Kernel;

static int testsPassed = 0;
static int testsTotal  = 0;

#define RUN_TEST(name, body)                                                                                           \
	do {                                                                                                               \
		std::cout << "[Test " << #name << "] ... " << std::flush;                                                      \
		testsTotal++;                                                                                                  \
		try {                                                                                                          \
			body                                                                                                       \
		} catch (const std::exception &e) {                                                                            \
			std::cout << "FAIL: " << e.what() << std::endl;                                                            \
			continue;                                                                                                  \
		}                                                                                                              \
		std::cout << "PASS" << std::endl;                                                                              \
		testsPassed++;                                                                                                 \
	} while (0)

int main() {
	std::cout << "=== InspectorKernel Validate() Tests ===" << std::endl;

	// ==================================================================
	// Test 1: Validate valid 1D kernel
	// ==================================================================
	RUN_TEST(Validate1D_Valid, {
		InspectorKernel1D kernel([](Var<int> &id) { Var<float> f = Expr<float>(id) * 2.0f; });

		std::string		  error;
		if (!kernel.Validate(error)) {
			throw std::runtime_error("Valid 1D kernel failed validation: " + error);
		}
		if (!kernel.Validate()) {
			throw std::runtime_error("Valid 1D kernel failed validation (no msg overload)");
		}
	});

	// ==================================================================
	// Test 2: Validate valid 2D kernel
	// ==================================================================
	RUN_TEST(Validate2D_Valid, {
		InspectorKernel2D kernel([](Var<int> &x, Var<int> &y) { Var<int> idx = y * 100 + x; });

		std::string		  error;
		if (!kernel.Validate(error)) {
			throw std::runtime_error("Valid 2D kernel failed validation: " + error);
		}
	});

	// ==================================================================
	// Test 3: Validate valid 3D kernel
	// ==================================================================
	RUN_TEST(Validate3D_Valid, {
		InspectorKernel3D kernel([](Var<int> &x, Var<int> &y, Var<int> &z) { Var<int> idx = (z * 100 + y) * 100 + x; });

		std::string		  error;
		if (!kernel.Validate(error)) {
			throw std::runtime_error("Valid 3D kernel failed validation: " + error);
		}
	});

	// ==================================================================
	// Test 4: Validate with valid generated code
	// ==================================================================
	RUN_TEST(Validate_GeneratedCode, {
		InspectorKernel1D kernel([](Var<int> &id) { Var<int> x = id; });

		std::string		  error;
		if (!kernel.Validate(error)) {
			throw std::runtime_error("Kernel with code should validate: " + error);
		}
	});

	// ==================================================================
	// Test 5: Compile() with graceful degradation
	// ==================================================================
	RUN_TEST(Compile_GracefulDegradation, {
		InspectorKernel1D kernel([](Var<int> &id) { Var<int> x = id * 2; });

		std::string		  error;
		bool			  result = kernel.Compile(error);
		if (!result) {
			std::cout << " (compile note: " << error << ")" << std::flush;
		}
		// Validate should always work for valid code
		if (!kernel.Validate(error)) {
			throw std::runtime_error("Validate failed after Compile: " + error);
		}
	});

	// ==================================================================
	// Test 6: Validate with buffer binding
	// ==================================================================
	RUN_TEST(Validate_WithBuffer, {
		Buffer<float>	  buffer(256, BufferMode::ReadWrite);

		InspectorKernel1D kernel([&](Var<int> &id) {
			auto buf = buffer.Bind();
			buf[id]	 = Expr<float>(id) * 2.0f;
		});

		std::string		  error;
		if (!kernel.Validate(error)) {
			throw std::runtime_error("Validate with buffer failed: " + error);
		}
	});

	// ==================================================================
	// Test 7: Validate with control flow
	// ==================================================================
	RUN_TEST(Validate_WithControlFlow, {
		Buffer<float>	  buffer(256, BufferMode::ReadWrite);

		InspectorKernel1D kernel([&](Var<int> &id) {
			auto buf = buffer.Bind();
			If(id < 128, [&]() { buf[id] = 1.0f; }).Else([&]() { buf[id] = 2.0f; });
		});

		std::string		  error;
		if (!kernel.Validate(error)) {
			throw std::runtime_error("Validate with control flow failed: " + error);
		}
	});

	std::cout << "\n========================================" << std::endl;
	std::cout << "Test Results: " << testsPassed << "/" << testsTotal << " passed" << std::endl;
	std::cout << "========================================" << std::endl;

	return (testsPassed == testsTotal) ? 0 : 1;
}
