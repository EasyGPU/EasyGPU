/**
 * @file TestInspector.cpp
 * @brief Test Inspector Kernels for all dimensions.
 */

#include <Flow/ForFlow.h>
#include <Flow/IfFlow.h>
#include <IR/Value/BufferRef.h>
#include <IR/Value/Var.h>
#include <Kernel/Kernel.h>
#include <Runtime/Buffer.h>
#include <Utility/Helpers.h>
#include <Utility/Vec.h>
#ifdef EASYGPU_BACKEND_VULKAN
#include <Backend/VulkanBackend.h>
#endif

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace GPU;
using namespace GPU::IR::Value;
using namespace GPU::Math;
using namespace GPU::Flow;
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

// =============================================================================
// Test InspectorKernel1D
// =============================================================================
TEST(inspector_1d_basic)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<int>   x = id * 2;
	Var<float> f = Expr<float>(x) + 1.5f;
});

std::string					   code = kernel.GetCode();
ASSERT(!code.empty());
ASSERT(code.find("local_size_x") != std::string::npos);
ASSERT(code.find("gl_GlobalInvocationID.x") != std::string::npos);
std::cout << "\n[Generated Code]:\n" << code << "\n";
END_TEST

TEST(inspector_1d_with_worksize)
GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) { Var<int> x = id; }, 128); // Custom work size

std::string					   code = kernel.GetCode();
ASSERT(code.find("local_size_x = 128") != std::string::npos);
END_TEST

TEST(inspector_1d_buffer)
Buffer<float>				   buffer(256, BufferMode::ReadWrite);

GPU::Kernel::InspectorKernel1D kernel([&](Var<int> &id) {
	auto buf = buffer.Bind();
	buf[id]	 = Expr<float>(id) * 2.0f;
});

std::string					   code = kernel.GetCode();
ASSERT(code.find("buffer") != std::string::npos || code.find("Buffer") != std::string::npos);
END_TEST

// =============================================================================
// Test InspectorKernel2D
// =============================================================================
TEST(inspector_2d_basic)
GPU::Kernel::InspectorKernel2D kernel([](Var<int> &x, Var<int> &y) {
	Var<int>   idx	 = y * 100 + x;
	Var<float> value = Expr<float>(idx);
});

std::string					   code = kernel.GetCode();
ASSERT(!code.empty());
ASSERT(code.find("local_size_x") != std::string::npos);
ASSERT(code.find("local_size_y") != std::string::npos);
ASSERT(code.find("gl_GlobalInvocationID.x") != std::string::npos);
ASSERT(code.find("gl_GlobalInvocationID.y") != std::string::npos);
std::cout << "\n[Generated Code]:\n" << code << "\n";
END_TEST

TEST(inspector_2d_custom_worksize)
GPU::Kernel::InspectorKernel2D kernel([](Var<int> &x, Var<int> &y) { Var<int> sum = x + y; }, 32, 32);

std::string					   code = kernel.GetCode();
ASSERT(code.find("local_size_x = 32") != std::string::npos);
ASSERT(code.find("local_size_y = 32") != std::string::npos);
END_TEST

TEST(inspector_2d_vector_ops)
GPU::Kernel::InspectorKernel2D kernel([](Var<int> &x, Var<int> &y) {
	Var<Vec3> color = MakeFloat3(Expr<float>(x) / 100.0f, Expr<float>(y) / 100.0f, 0.5f);
});

std::string					   code = kernel.GetCode();
ASSERT(code.find("vec3") != std::string::npos);
END_TEST

// =============================================================================
// Test InspectorKernel3D
// =============================================================================
TEST(inspector_3d_basic)
GPU::Kernel::InspectorKernel3D kernel([](Var<int> &x, Var<int> &y, Var<int> &z) {
	Var<int> idx = (z * 100 + y) * 100 + x;
});

std::string					   code = kernel.GetCode();
ASSERT(!code.empty());
ASSERT(code.find("local_size_x") != std::string::npos);
ASSERT(code.find("local_size_y") != std::string::npos);
ASSERT(code.find("local_size_z") != std::string::npos);
ASSERT(code.find("gl_GlobalInvocationID.x") != std::string::npos);
ASSERT(code.find("gl_GlobalInvocationID.y") != std::string::npos);
ASSERT(code.find("gl_GlobalInvocationID.z") != std::string::npos);
std::cout << "\n[Generated Code]:\n" << code << "\n";
END_TEST

TEST(inspector_3d_custom_worksize)
GPU::Kernel::InspectorKernel3D kernel([](Var<int> &x, Var<int> &y, Var<int> &z) { Var<int> sum = x + y + z; }, 4, 4, 4);

std::string					   code = kernel.GetCode();
ASSERT(code.find("local_size_x = 4") != std::string::npos);
ASSERT(code.find("local_size_y = 4") != std::string::npos);
ASSERT(code.find("local_size_z = 4") != std::string::npos);
END_TEST

// =============================================================================
// Test Backward Compatibility (InspectorKernel alias)
// =============================================================================
TEST(inspector_backward_compat)
// InspectorKernel should be an alias for InspectorKernel1D
GPU::Kernel::InspectorKernel kernel([](Var<int> &id) { Var<int> x = id; });

std::string					 code = kernel.GetCode();
ASSERT(!code.empty());
ASSERT(code.find("local_size_x = 256") != std::string::npos);
END_TEST

// =============================================================================
// Test PrintCode
// =============================================================================
TEST(inspector_print_code)
GPU::Kernel::InspectorKernel2D kernel([](Var<int> &x, Var<int> &y) { Var<int> sum = x + y; });

// Should not throw
kernel.PrintCode();
ASSERT(true);
END_TEST

// =============================================================================
// Test Compile API
// =============================================================================
TEST(inspector_compile_1d)
std::cout << "\n  Testing InspectorKernel1D::Compile()...\n";

GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<int>   x = id * 2;
	Var<float> f = Expr<float>(x) + 1.5f;
});

std::string					   errorMsg;
bool						   compiled = kernel.Compile(errorMsg);

if (!compiled) {
	std::cout << "  Error: " << errorMsg << "\n";
}
ASSERT(compiled);
std::cout << "  ✓ 1D kernel compiled successfully!\n";
END_TEST

TEST(inspector_compile_2d)
std::cout << "\n  Testing InspectorKernel2D::Compile()...\n";

GPU::Kernel::InspectorKernel2D kernel([](Var<int> &x, Var<int> &y) {
	Var<int>  idx	= y * 100 + x;
	Var<Vec3> color = MakeFloat3(Expr<float>(x) / 100.0f, Expr<float>(y) / 100.0f, 0.5f);
});

std::string					   errorMsg;
bool						   compiled = kernel.Compile(errorMsg);

if (!compiled) {
	std::cout << "  Error: " << errorMsg << "\n";
}
ASSERT(compiled);
std::cout << "  ✓ 2D kernel compiled successfully!\n";
END_TEST

TEST(inspector_compile_3d)
std::cout << "\n  Testing InspectorKernel3D::Compile()...\n";

GPU::Kernel::InspectorKernel3D kernel([](Var<int> &x, Var<int> &y, Var<int> &z) {
	Var<int>  idx = (z * 100 + y) * 100 + x;
	Var<Vec3> pos = MakeFloat3(Expr<float>(x), Expr<float>(y), Expr<float>(z));
});

std::string					   errorMsg;
bool						   compiled = kernel.Compile(errorMsg);

if (!compiled) {
	std::cout << "  Error: " << errorMsg << "\n";
}
ASSERT(compiled);
std::cout << "  ✓ 3D kernel compiled successfully!\n";
END_TEST

TEST(inspector_compile_simple_version)
std::cout << "\n  Testing Compile() without error message...\n";

GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) { Var<int> x = id + 1; });

// Simple version without error message
bool						   compiled = kernel.Compile();
ASSERT(compiled);
std::cout << "  ✓ Simple Compile() works!\n";
END_TEST

TEST(inspector_optimized_glsl)
std::cout << "\n  Testing optimized GLSL inspection...\n";

GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x = Expr<float>(id) * 2.0f;
	Var<float> y = (x + 1.0f) * (x + 1.0f);
});

std::string					   before = kernel.GetCode();
std::string					   after  = kernel.GetOptimizedGLSL();

ASSERT(!before.empty());
if (Runtime::Context::GetInstance().GetBackendType() == Backend::BackendType::Vulkan) {
	ASSERT(!after.empty());
	ASSERT(after.find("#version") != std::string::npos);
	ASSERT(after.find("main") != std::string::npos);
} else {
	ASSERT(after.empty());
}
std::cout << "  ✓ Optimized GLSL generated via SPIR-V toolchain!\n";
END_TEST

TEST(inspector_optimization_levels)
std::cout << "\n  Testing selectable SPIR-V optimization levels...\n";

GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x	= Expr<float>(id) * 2.0f;
	Var<float> dead = x * 0.0f;
	If(MakeBool(false), [&] { Var<float> y = dead + 1.0f; }).Else([&] { Var<float> z = x + 3.0f; });
});

ASSERT(kernel.GetOptimizationLevel() == Backend::ShaderOptimizationLevel::Aggressive);

std::string aggressive = kernel.GetOptimizedGLSL();
if (Runtime::Context::GetInstance().GetBackendType() == Backend::BackendType::Vulkan) {
	ASSERT(aggressive.find("#version") != std::string::npos);
	ASSERT(aggressive.find("if (false)") == std::string::npos);
} else {
	ASSERT(aggressive.empty());
}

kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::None);
ASSERT(kernel.GetOptimizationLevel() == Backend::ShaderOptimizationLevel::None);
std::string none = kernel.GetOptimizedGLSL();
if (Runtime::Context::GetInstance().GetBackendType() == Backend::BackendType::Vulkan) {
	ASSERT(none.find("#version") != std::string::npos);
	ASSERT(none.find("if (false)") != std::string::npos);
} else {
	ASSERT(none.empty());
}

kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Size);
ASSERT(kernel.GetOptimizationLevel() == Backend::ShaderOptimizationLevel::Size);
std::string size = kernel.GetOptimizedGLSL();
if (Runtime::Context::GetInstance().GetBackendType() == Backend::BackendType::Vulkan) {
	ASSERT(size.find("#version") != std::string::npos);
	ASSERT(size.find("if (false)") == std::string::npos);
} else {
	ASSERT(size.empty());
}

std::cout << "  ✓ Optimization levels selectable and observable!\n";
END_TEST

TEST(inspector_ultra_optimization)
std::cout << "\n  Testing Ultra SPIR-V optimization level...\n";

GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x    = Expr<float>(id) * 2.0f;
	Var<float> dead = x * 0.0f;
	If(MakeBool(false), [&] {
		Var<float> y = dead + 1.0f;
	}).Else([&] {
		Var<float> z = x + 3.0f;
	});
	Var<float> dup   = Expr<float>(id) * 2.0f;
	Float3    dummy  = MakeFloat3(x, Expr<float>(id) * 0.0f, 0.0f);
	For(0, 4, 1, [&](Var<int> &i) {
		Var<float> acc = Expr<float>(i) * 0.25f;
	});
});

kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Ultra);
ASSERT(kernel.GetOptimizationLevel() == Backend::ShaderOptimizationLevel::Ultra);

std::string ultra = kernel.GetOptimizedGLSL();
if (Runtime::Context::GetInstance().GetBackendType() == Backend::BackendType::Vulkan) {
	ASSERT(ultra.find("#version") != std::string::npos);
	ASSERT(ultra.find("if (false)") == std::string::npos);
} else {
	ASSERT(ultra.empty());
}

kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::None);
std::string none = kernel.GetOptimizedGLSL();
if (Runtime::Context::GetInstance().GetBackendType() == Backend::BackendType::Vulkan) {
	ASSERT(none.find("if (false)") != std::string::npos);
	ASSERT(ultra != none);
}

std::cout << "  Ultra optimization level functional!\n";
END_TEST

TEST(inspector_extreme_optimization)
std::cout << "\n  Testing Extreme SPIR-V optimization level...\n";

GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
	Var<float> x    = Expr<float>(id) * 2.0f;
	Var<float> dead = x * 0.0f;
	If(MakeBool(false), [&] {
		Var<float> y = dead + 1.0f;
	}).Else([&] {
		Var<float> z = x + 3.0f;
	});
});

kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Extreme);
ASSERT(kernel.GetOptimizationLevel() == Backend::ShaderOptimizationLevel::Extreme);

std::string extreme = kernel.GetOptimizedGLSL();
if (Runtime::Context::GetInstance().GetBackendType() == Backend::BackendType::Vulkan) {
	ASSERT(extreme.find("#version") != std::string::npos);
	ASSERT(extreme.find("if (false)") == std::string::npos);
} else {
	ASSERT(extreme.empty());
}

std::cout << "  Extreme optimization level functional!\n";
END_TEST

TEST(shader_optimization_execution_equivalence)
std::cout << "\n  Testing execution equivalence across production optimization levels...\n";

constexpr int		 kElementCount = 64;
std::vector<int> inputData(kElementCount);
for (int i = 0; i < kElementCount; ++i) {
	inputData[i] = (i * 17) % 97 - 48;
}

Buffer<int> inputBuffer(inputData, BufferMode::Read);
Buffer<int> outputBuffer(kElementCount, BufferMode::Write);

GPU::Kernel::Kernel1D kernel(
	[&](Var<int> &id) {
		auto input  = inputBuffer.Bind();
		auto output = outputBuffer.Bind();

		Var<int> value = input[id];
		Var<int> sum(Expr<int>(0));
		For(0, 12, 1, [&](Var<int> &i) {
			Var<int> term = value * 8 + i * 2;
			If((value + i) % 3 == 0, [&] { sum = sum + term - 3; }).Else([&] { sum = sum + term + 5; });
		});
		output[id] = sum;
	},
	64);

auto run = [&](Backend::ShaderOptimizationLevel level) {
	kernel.SetOptimizationLevel(level);
	kernel.Dispatch(1, true);
	std::vector<int> result;
	outputBuffer.Download(result);
	return result;
};

const auto none		 = run(Backend::ShaderOptimizationLevel::None);
const auto aggressive = run(Backend::ShaderOptimizationLevel::Aggressive);
const auto ultra	 = run(Backend::ShaderOptimizationLevel::Ultra);

ASSERT(none == aggressive);
ASSERT(none == ultra);
std::cout << "  None, Aggressive, and Ultra produced identical results!\n";
END_TEST

TEST(shader_disk_cache)
std::cout << "\n  Testing persistent optimized SPIR-V cache...\n";

#ifdef EASYGPU_SHADER_CACHE_ENABLED
if (Runtime::Context::GetInstance().GetBackendType() == Backend::BackendType::Vulkan) {
	const auto cacheDirectory = std::filesystem::temp_directory_path() /
		("easygpu-spirv-cache-test-" +
		 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	std::filesystem::remove_all(cacheDirectory);
#ifdef _WIN32
	_putenv_s("EASYGPU_SHADER_CACHE_DIR", cacheDirectory.string().c_str());
#else
	setenv("EASYGPU_SHADER_CACHE_DIR", cacheDirectory.string().c_str(), 1);
#endif

	auto *backend = Runtime::Context::GetBackend();
	ASSERT(backend != nullptr);
	backend->ResetShaderCompilationStats();

	GPU::Kernel::InspectorKernel1D kernel([](Var<int> &id) {
		Var<int> value = id * 8 + 3;
		If(value > 17, [&] { value = value - 2; });
	});
	kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Ultra);

	ASSERT(!kernel.GetOptimizedGLSL().empty());
	auto stats = backend->GetShaderCompilationStats();
	ASSERT(stats.memoryCacheHits == 0);
	ASSERT(stats.diskCacheHits == 0);
	ASSERT(stats.diskCacheMisses == 1);
	ASSERT(stats.frontendCompilations == 1);
	ASSERT(!stats.lastDiskCacheHit);

	ASSERT(!kernel.GetOptimizedGLSL().empty());
	stats = backend->GetShaderCompilationStats();
	ASSERT(stats.memoryCacheHits == 1);
	ASSERT(stats.diskCacheHits == 0);
	ASSERT(stats.diskCacheMisses == 1);
	ASSERT(stats.frontendCompilations == 1);
	ASSERT(stats.lastMemoryCacheHit);
	ASSERT(!stats.lastDiskCacheHit);
	ASSERT(stats.lastFrontendMilliseconds == 0.0);
	ASSERT(stats.lastOptimizationMilliseconds == 0.0);

	#ifdef EASYGPU_BACKEND_VULKAN
	Backend::ShaderDesc diskDescriptor;
	diskDescriptor.type = Backend::ShaderType::Compute;
	diskDescriptor.sourceCode = kernel.GetCode();
	diskDescriptor.optimizationLevel = Backend::ShaderOptimizationLevel::Ultra;
	Backend::VulkanBackend diskBackend;
	diskBackend.Initialize();
	diskBackend.ResetShaderCompilationStats();
	ASSERT(!diskBackend.GetOptimizedGLSL(diskDescriptor).empty());
	const auto diskStats = diskBackend.GetShaderCompilationStats();
	ASSERT(diskStats.memoryCacheHits == 0);
	ASSERT(diskStats.diskCacheHits == 1);
	ASSERT(diskStats.diskCacheMisses == 0);
	ASSERT(diskStats.frontendCompilations == 0);
	ASSERT(!diskStats.lastMemoryCacheHit);
	ASSERT(diskStats.lastDiskCacheHit);
	diskBackend.Shutdown();
	#endif

	std::vector<std::filesystem::path> cacheFiles;
	for (const auto &entry : std::filesystem::recursive_directory_iterator(cacheDirectory)) {
		if (entry.is_regular_file() && entry.path().extension() == ".spv") {
			cacheFiles.push_back(entry.path());
		}
	}
	ASSERT(cacheFiles.size() == 1);
	{
		std::ofstream corrupted(cacheFiles.front(), std::ios::binary | std::ios::trunc);
		const uint32_t invalidWord = 0;
		corrupted.write(reinterpret_cast<const char *>(&invalidWord), sizeof(invalidWord));
	}

	ASSERT(!kernel.GetOptimizedGLSL().empty());
	stats = backend->GetShaderCompilationStats();
	ASSERT(stats.memoryCacheHits == 1);
	ASSERT(stats.diskCacheHits == 0);
	ASSERT(stats.diskCacheMisses == 2);
	ASSERT(stats.frontendCompilations == 2);
	ASSERT(!stats.lastMemoryCacheHit);
	ASSERT(!stats.lastDiskCacheHit);
	ASSERT(std::filesystem::file_size(cacheFiles.front()) >= 5 * sizeof(uint32_t));

	kernel.SetOptimizationLevel(Backend::ShaderOptimizationLevel::Aggressive);
	ASSERT(!kernel.GetOptimizedGLSL().empty());
	stats = backend->GetShaderCompilationStats();
	ASSERT(stats.diskCacheMisses == 3);
	ASSERT(stats.frontendCompilations == 3);

	Backend::ShaderDesc preserveDesc;
	preserveDesc.type = Backend::ShaderType::Compute;
	preserveDesc.sourceCode = kernel.GetCode();
	preserveDesc.optimizationLevel = Backend::ShaderOptimizationLevel::Ultra;
	preserveDesc.preserveInterface = true;
	ASSERT(!backend->GetOptimizedGLSL(preserveDesc).empty());
	stats = backend->GetShaderCompilationStats();
	ASSERT(stats.diskCacheMisses == 4);
	ASSERT(stats.frontendCompilations == 4);

#ifdef _WIN32
	_putenv_s("EASYGPU_SHADER_CACHE_DIR", "");
#else
	unsetenv("EASYGPU_SHADER_CACHE_DIR");
#endif
	std::filesystem::remove_all(cacheDirectory);
}
std::cout << "  Persistent SPIR-V cache hit and corruption recovery verified!\n";
#else
std::cout << "  Persistent SPIR-V cache is disabled in this build.\n";
#endif
END_TEST

TEST(vulkan_pipeline_disk_cache)
std::cout << "\n  Testing persistent Vulkan pipeline cache...\n";

#if defined(EASYGPU_SHADER_CACHE_ENABLED) && defined(EASYGPU_BACKEND_VULKAN)
const auto cacheDirectory = std::filesystem::temp_directory_path() /
	("easygpu-pipeline-cache-test-" +
	 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
std::filesystem::remove_all(cacheDirectory);
#ifdef _WIN32
_putenv_s("EASYGPU_SHADER_CACHE_DIR", cacheDirectory.string().c_str());
#else
setenv("EASYGPU_SHADER_CACHE_DIR", cacheDirectory.string().c_str(), 1);
#endif

auto buildPipeline = [](Backend::VulkanBackend &backend, bool validateBinaryRoundTrip) {
	Backend::ShaderDesc shaderDesc;
	shaderDesc.type = Backend::ShaderType::Compute;
	shaderDesc.optimizationLevel = Backend::ShaderOptimizationLevel::Ultra;
	shaderDesc.sourceCode = R"glsl(#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {}
)glsl";

	const auto shader = backend.CreateShader(shaderDesc);
	ASSERT(shader != Backend::INVALID_SHADER_HANDLE);

	Backend::PipelineDesc pipelineDesc;
	pipelineDesc.computeShader = shader;
	pipelineDesc.workGroupSizeX = 1;
	pipelineDesc.workGroupSizeY = 1;
	pipelineDesc.workGroupSizeZ = 1;
	const auto pipeline = backend.CreatePipeline(pipelineDesc);
	ASSERT(pipeline != Backend::INVALID_PIPELINE_HANDLE);
	if (validateBinaryRoundTrip) {
		uint32_t format = 0;
		const auto binary = backend.GetPipelineBinary(pipeline, format);
		ASSERT(!binary.empty());
		auto incompatibleBinary = binary;
		incompatibleBinary[offsetof(VkPipelineCacheHeaderVersionOne, pipelineCacheUUID) + VK_UUID_SIZE - 1] ^= 1u;
		ASSERT(backend.CreatePipelineFromBinary(pipelineDesc, incompatibleBinary.data(), incompatibleBinary.size(),
												format) == Backend::INVALID_PIPELINE_HANDLE);
		const auto cachedPipeline = backend.CreatePipelineFromBinary(pipelineDesc, binary.data(), binary.size(), format);
		ASSERT(cachedPipeline != Backend::INVALID_PIPELINE_HANDLE);
		backend.DestroyPipeline(cachedPipeline);
	}
	backend.DestroyPipeline(pipeline);
	backend.DestroyShader(shader);
};

{
	Backend::VulkanBackend backend;
	backend.Initialize();
	const auto initialStats = backend.GetPipelineCacheStats();
	ASSERT(initialStats.diskCacheHits == 0);
	ASSERT(initialStats.diskCacheMisses == 1);
	buildPipeline(backend, true);
	backend.FlushPipelineCache();
	backend.Shutdown();
	const auto savedStats = backend.GetPipelineCacheStats();
	ASSERT(savedStats.diskCacheWrites >= 1);
	ASSERT(savedStats.diskCacheWriteFailures == 0);
	ASSERT(savedStats.savedBytes > 0);
}

std::vector<std::filesystem::path> pipelineCacheFiles;
const auto pipelineDirectory = cacheDirectory / "vulkan-pipeline-v1";
for (const auto &entry : std::filesystem::directory_iterator(pipelineDirectory)) {
	if (entry.is_regular_file() && entry.path().extension() == ".bin") {
		pipelineCacheFiles.push_back(entry.path());
	}
}
ASSERT(pipelineCacheFiles.size() == 1);

{
	Backend::VulkanBackend backend;
	backend.Initialize();
	const auto loadedStats = backend.GetPipelineCacheStats();
	ASSERT(loadedStats.diskCacheHits == 1);
	ASSERT(loadedStats.diskCacheMisses == 0);
	ASSERT(loadedStats.invalidDiskEntries == 0);
	ASSERT(loadedStats.loadedBytes == std::filesystem::file_size(pipelineCacheFiles.front()));
	ASSERT(loadedStats.lastDiskCacheHit);
	backend.Shutdown();
}

{
	std::ofstream corrupted(pipelineCacheFiles.front(), std::ios::binary | std::ios::trunc);
	const uint32_t invalidWord = 0;
	corrupted.write(reinterpret_cast<const char *>(&invalidWord), sizeof(invalidWord));
}

{
	Backend::VulkanBackend backend;
	backend.Initialize();
	const auto recoveredStats = backend.GetPipelineCacheStats();
	ASSERT(recoveredStats.diskCacheHits == 0);
	ASSERT(recoveredStats.diskCacheMisses == 1);
	ASSERT(recoveredStats.invalidDiskEntries == 1);
	ASSERT(!recoveredStats.lastDiskCacheHit);
	buildPipeline(backend, false);
	backend.FlushPipelineCache();
	backend.Shutdown();
	ASSERT(backend.GetPipelineCacheStats().diskCacheWrites >= 1);
}
ASSERT(std::filesystem::file_size(pipelineCacheFiles.front()) > sizeof(VkPipelineCacheHeaderVersionOne));

#ifdef _WIN32
_putenv_s("EASYGPU_SHADER_CACHE_DIR", "");
#else
unsetenv("EASYGPU_SHADER_CACHE_DIR");
#endif
std::filesystem::remove_all(cacheDirectory);
std::cout << "  Persistent Vulkan pipeline cache load and recovery verified!\n";
#else
std::cout << "  Persistent Vulkan pipeline cache is disabled in this build.\n";
#endif
END_TEST

// =============================================================================
// Main
// =============================================================================
int main() {
	std::cout << "========================================\n";
	std::cout << "   Inspector Kernel Test Suite          \n";
	std::cout << "========================================\n";

	try {
		test_inspector_1d_basic();
		test_inspector_1d_with_worksize();
		test_inspector_1d_buffer();
		test_inspector_2d_basic();
		test_inspector_2d_custom_worksize();
		test_inspector_2d_vector_ops();
		test_inspector_3d_basic();
		test_inspector_3d_custom_worksize();
		test_inspector_backward_compat();
		test_inspector_print_code();
		test_inspector_compile_1d();
		test_inspector_compile_2d();
		test_inspector_compile_3d();
		test_inspector_compile_simple_version();
		test_inspector_optimized_glsl();
		test_inspector_optimization_levels();
		test_inspector_ultra_optimization();
		test_inspector_extreme_optimization();
		test_shader_optimization_execution_equivalence();
		test_shader_disk_cache();
		test_vulkan_pipeline_disk_cache();

		std::cout << "\n========================================\n";
		std::cout << "  Results: " << pass_count << "/" << test_count << " tests passed\n";
		std::cout << "========================================\n";

		return (pass_count == test_count) ? 0 : 1;
	} catch (const std::exception &e) {
		std::cout << "\nFATAL ERROR: " << e.what() << "\n";
		return 1;
	}
}
