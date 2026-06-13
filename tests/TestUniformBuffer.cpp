/**
 * @file TestUniformBuffer.cpp
 * @brief Tests for High #8: UniformBuffer UBO support.
 */

#include <GPU.h>
#include <iostream>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
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

// Define a struct for UBO testing using raw math types
EASYGPU_STRUCT(MyUBOConfig, (GPU::Math::Vec3, color), (float, exposure));

int main() {
	std::cout << "=== UniformBuffer (UBO) Tests ===" << std::endl;

	// ==================================================================
	// Test 1: Basic UniformBuffer creation and value access
	// ==================================================================
	RUN_TEST(BasicCreateAndAccess, {
		MyUBOConfig cfgValue;
		cfgValue.color	  = Vec3(1.0f, 0.5f, 0.25f);
		cfgValue.exposure = 2.0f;

		UniformBuffer<MyUBOConfig> ubo(cfgValue);

		const auto				  &val = ubo.GetValue();
		if (val.color.x != 1.0f || val.color.y != 0.5f || val.color.z != 0.25f || val.exposure != 2.0f) {
			throw std::runtime_error("Value mismatch");
		}
	});

	// ==================================================================
	// Test 2: SetValue updates the value
	// ==================================================================
	RUN_TEST(SetValue, {
		UniformBuffer<MyUBOConfig> ubo;

		MyUBOConfig				   cfg;
		cfg.color	 = Vec3(0.1f, 0.2f, 0.3f);
		cfg.exposure = 1.5f;
		ubo.SetValue(cfg);

		if (ubo.GetValue().exposure != 1.5f) {
			throw std::runtime_error("SetValue failed");
		}
	});

	// ==================================================================
	// Test 3: Assignment operator
	// ==================================================================
	RUN_TEST(AssignmentOperator, {
		UniformBuffer<MyUBOConfig> ubo;
		MyUBOConfig				   cfg;
		cfg.color	 = Vec3(0.5f, 0.5f, 0.5f);
		cfg.exposure = 3.0f;
		ubo			 = cfg;

		if (ubo.GetValue().exposure != 3.0f) {
			throw std::runtime_error("Assignment failed");
		}
	});

	// ==================================================================
	// Test 4: UniformBuffer works in a kernel (GPU round-trip)
	// ==================================================================
	RUN_TEST(KernelGPUUsage, {
		UniformBuffer<MyUBOConfig> ubo;
		MyUBOConfig				   cfg;
		cfg.color	 = Vec3(2.0f, 3.0f, 4.0f);
		cfg.exposure = 10.0f;
		ubo			 = cfg;

		std::vector<float> resultData(4, 0.0f);
		Buffer<float>	   result(resultData, BufferMode::Write);

		Kernel1D		   kernel(
			[&](Int i) {
				auto buf = result.Bind();
				auto c	 = ubo.Load();

				buf[0]	 = c.color().x();
				buf[1]	 = c.color().y();
				buf[2]	 = c.color().z();
				buf[3]	 = c.exposure();
			},
			1);

		kernel.Dispatch(1, true);
		result.Download(resultData);

		std::cout << " (got: " << resultData[0] << ", " << resultData[1] << ", " << resultData[2] << ", "
				  << resultData[3] << ")" << std::flush;

		if (std::abs(resultData[0] - 2.0f) > 0.01f || std::abs(resultData[1] - 3.0f) > 0.01f ||
			std::abs(resultData[2] - 4.0f) > 0.01f || std::abs(resultData[3] - 10.0f) > 0.01f) {
			throw std::runtime_error("GPU result mismatch");
		}
	});

	// ==================================================================
	// Test 5: UniformBuffer with multiple dispatches (value update)
	// ==================================================================
	RUN_TEST(MultipleDispatches, {
		UniformBuffer<MyUBOConfig> ubo;
		std::vector<float>		   resultData(4, 0.0f);
		Buffer<float>			   result(resultData, BufferMode::Write);

		Kernel1D				   kernel(
			[&](Int i) {
				auto buf = result.Bind();
				auto c	 = ubo.Load();
				buf[0]	 = c.color().x();
				buf[1]	 = c.color().y();
				buf[2]	 = c.color().z();
				buf[3]	 = c.exposure();
			},
			1);

		// First dispatch
		MyUBOConfig cfg1;
		cfg1.color	  = Vec3(1.0f, 1.0f, 1.0f);
		cfg1.exposure = 1.0f;
		ubo			  = cfg1;

		kernel.Dispatch(1, true);
		result.Download(resultData);

		if (std::abs(resultData[0] - 1.0f) > 0.01f || std::abs(resultData[3] - 1.0f) > 0.01f) {
			throw std::runtime_error("First dispatch failed");
		}

		// Second dispatch with different values
		MyUBOConfig cfg2;
		cfg2.color	  = Vec3(5.0f, 6.0f, 7.0f);
		cfg2.exposure = 99.0f;
		ubo			  = cfg2;

		kernel.Dispatch(1, true);
		result.Download(resultData);

		if (std::abs(resultData[0] - 5.0f) > 0.01f || std::abs(resultData[1] - 6.0f) > 0.01f ||
			std::abs(resultData[2] - 7.0f) > 0.01f || std::abs(resultData[3] - 99.0f) > 0.01f) {
			throw std::runtime_error("Second dispatch failed: got " + std::to_string(resultData[0]) + ", " +
									 std::to_string(resultData[3]));
		}
	});

	// ==================================================================
	// Test 6: InspectorKernel with UniformBuffer (verify GLSL code)
	// ==================================================================
	RUN_TEST(InspectorGLSL, {
		UniformBuffer<MyUBOConfig> ubo;
		MyUBOConfig				   cfg;
		cfg.color	 = Vec3(1.0f, 0.0f, 0.0f);
		cfg.exposure = 1.0f;
		ubo			 = cfg;

		InspectorKernel1D kernel([&](Var<int> &id) {
			auto c = ubo.Load();
			(void)id;
			(void)c;
		});

		std::string		  code = kernel.GetCode();

		if (code.find("uniform EasyGPU_UBO_") == std::string::npos) {
			throw std::runtime_error("UBO declaration not found");
		}
		if (code.find("layout(std140") == std::string::npos) {
			throw std::runtime_error("std140 layout not found");
		}
		if (code.find("MyUBOConfig") == std::string::npos) {
			throw std::runtime_error("Struct type name not found in:\n" + code);
		}
	});

	std::cout << "\n========================================" << std::endl;
	std::cout << "Test Results: " << testsPassed << "/" << testsTotal << " passed" << std::endl;
	std::cout << "========================================" << std::endl;

	return (testsPassed == testsTotal) ? 0 : 1;
}
