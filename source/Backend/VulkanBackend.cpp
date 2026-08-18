/**
 * @file VulkanBackend.cpp
 * @brief Vulkan backend implementation.
 */

#include <Backend/VulkanBackend.h>
#include <Utility/SHA256.h>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_set>

#ifdef EASYGPU_SHADER_CACHE_ENABLED
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

#include <process.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif
#endif

// glslang includes for GLSL to SPIR-V compilation
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#if defined(EASYGPU_SPIRV_OPT_ENABLED) || defined(EASYGPU_SHADER_CACHE_ENABLED)
#include <spirv-tools/libspirv.hpp>
#endif

#ifdef EASYGPU_SPIRV_OPT_ENABLED
#include <spirv-tools/optimizer.hpp>
#endif

#ifdef EASYGPU_SPIRV_CROSS_GLSL_ENABLED
#include <spirv_cross/spirv_glsl.hpp>
#endif

namespace GPU::Backend {

namespace {

constexpr uintmax_t kMaximumCachedPipelineBytes = 256u * 1024u * 1024u;

bool ValidatePipelineCacheData(const void *data, size_t byteCount, const VkPhysicalDeviceProperties &properties) {
	if (data == nullptr || byteCount < sizeof(VkPipelineCacheHeaderVersionOne) ||
		byteCount > kMaximumCachedPipelineBytes) {
		return false;
	}

	VkPipelineCacheHeaderVersionOne header{};
	std::memcpy(&header, data, sizeof(header));
	return header.headerSize >= sizeof(VkPipelineCacheHeaderVersionOne) && header.headerSize <= byteCount &&
		   header.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE && header.vendorID == properties.vendorID &&
		   header.deviceID == properties.deviceID &&
		   std::memcmp(header.pipelineCacheUUID, properties.pipelineCacheUUID, VK_UUID_SIZE) == 0;
}

std::optional<std::vector<uint8_t>> GetPipelineCacheBytes(VkDevice device, VkPipelineCache cache) {
	for (int attempt = 0; attempt < 3; ++attempt) {
		size_t byteCount = 0;
		if (vkGetPipelineCacheData(device, cache, &byteCount, nullptr) != VK_SUCCESS ||
			byteCount < sizeof(VkPipelineCacheHeaderVersionOne) || byteCount > kMaximumCachedPipelineBytes) {
			return std::nullopt;
		}

		std::vector<uint8_t> data(byteCount);
		const VkResult result = vkGetPipelineCacheData(device, cache, &byteCount, data.data());
		if (result == VK_SUCCESS) {
			data.resize(byteCount);
			return data;
		}
		if (result != VK_INCOMPLETE) {
			return std::nullopt;
		}
	}
	return std::nullopt;
}

#ifdef EASYGPU_SHADER_CACHE_ENABLED

constexpr uint32_t kSpirvMagicNumber = 0x07230203u;
constexpr uintmax_t kMaximumCachedSpirvBytes = 64u * 1024u * 1024u;
constexpr std::string_view kSpirvCacheSchema = "easygpu-spirv-cache-v2";
constexpr std::string_view kPipelineCacheSchema = "easygpu-vulkan-pipeline-cache-v1";

static std::atomic<uint64_t> g_cacheTemporaryFileCounter{0};

uint64_t GetProcessIdForCacheFile() {
#ifdef _WIN32
	return static_cast<uint64_t>(_getpid());
#else
	return static_cast<uint64_t>(getpid());
#endif
}

std::optional<std::filesystem::path> GetShaderCacheRootDirectory() {
	if (const char *runtimeDirectory = std::getenv("EASYGPU_SHADER_CACHE_DIR");
		runtimeDirectory != nullptr && runtimeDirectory[0] != '\0') {
		return std::filesystem::path(runtimeDirectory);
	}

#ifdef EASYGPU_SHADER_CACHE_DIR
	return std::filesystem::path(EASYGPU_SHADER_CACHE_DIR);
#elif defined(_WIN32)
	if (const char *localAppData = std::getenv("LOCALAPPDATA"); localAppData != nullptr && localAppData[0] != '\0') {
		return std::filesystem::path(localAppData) / "EasyGPU" / "shader-cache";
	}
#elif defined(__APPLE__)
	if (const char *home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
		return std::filesystem::path(home) / "Library" / "Caches" / "EasyGPU";
	}
#else
	if (const char *xdgCache = std::getenv("XDG_CACHE_HOME"); xdgCache != nullptr && xdgCache[0] != '\0') {
		return std::filesystem::path(xdgCache) / "easygpu";
	}
	if (const char *home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
		return std::filesystem::path(home) / ".cache" / "easygpu";
	}
#endif

	return std::nullopt;
}

std::optional<std::filesystem::path> GetSpirvCacheDirectory() {
	if (const auto root = GetShaderCacheRootDirectory()) {
		return *root / "spirv-v2";
	}
	return std::nullopt;
}

std::filesystem::path MakeCacheTemporaryPath(const std::filesystem::path &path) {
	auto temporaryPath = path;
	temporaryPath += ".tmp-" + std::to_string(GetProcessIdForCacheFile()) + "-" +
					 std::to_string(g_cacheTemporaryFileCounter.fetch_add(1, std::memory_order_relaxed));
	return temporaryPath;
}

bool ReplaceCacheFileAtomically(const std::filesystem::path &path, const void *data, size_t byteCount) {
	std::error_code error;
	std::filesystem::create_directories(path.parent_path(), error);
	if (error) {
		return false;
	}

	const auto temporaryPath = MakeCacheTemporaryPath(path);
	std::ofstream stream(temporaryPath, std::ios::binary | std::ios::trunc);
	if (!stream) {
		return false;
	}
	if (byteCount != 0) {
		stream.write(static_cast<const char *>(data), static_cast<std::streamsize>(byteCount));
	}
	stream.close();
	if (!stream) {
		std::filesystem::remove(temporaryPath, error);
		return false;
	}

#ifdef _WIN32
	if (MoveFileExW(temporaryPath.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0) {
		return true;
	}
#else
	std::filesystem::rename(temporaryPath, path, error);
	if (!error) {
		return true;
	}
#endif

	error.clear();
	std::filesystem::remove(temporaryPath, error);
	return false;
}

class CacheFileLock {
public:
	explicit CacheFileLock(const std::filesystem::path &path) {
#ifdef _WIN32
		_handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
							  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS,
							  FILE_ATTRIBUTE_NORMAL, nullptr);
		if (_handle != INVALID_HANDLE_VALUE &&
			LockFileEx(_handle, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &_overlapped) == 0) {
			CloseHandle(_handle);
			_handle = INVALID_HANDLE_VALUE;
		}
#else
		_descriptor = open(path.c_str(), O_CREAT | O_RDWR, 0600);
		if (_descriptor >= 0 && flock(_descriptor, LOCK_EX) != 0) {
			close(_descriptor);
			_descriptor = -1;
		}
#endif
	}

	~CacheFileLock() {
#ifdef _WIN32
		if (_handle != INVALID_HANDLE_VALUE) {
			UnlockFileEx(_handle, 0, MAXDWORD, MAXDWORD, &_overlapped);
			CloseHandle(_handle);
		}
#else
		if (_descriptor >= 0) {
			flock(_descriptor, LOCK_UN);
			close(_descriptor);
		}
#endif
	}

	CacheFileLock(const CacheFileLock &) = delete;
	CacheFileLock &operator=(const CacheFileLock &) = delete;

	bool IsLocked() const {
#ifdef _WIN32
		return _handle != INVALID_HANDLE_VALUE;
#else
		return _descriptor >= 0;
#endif
	}

private:
#ifdef _WIN32
	HANDLE		 _handle = INVALID_HANDLE_VALUE;
	OVERLAPPED _overlapped{};
#else
	int _descriptor = -1;
#endif
};

std::filesystem::path BuildPipelineCachePath(const std::filesystem::path &root,
											 const VkPhysicalDeviceProperties &properties) {
	std::ostringstream identity;
	identity << kPipelineCacheSchema << '\n';
	identity << "vendor=" << properties.vendorID << '\n';
	identity << "device=" << properties.deviceID << '\n';
	identity << "driver=" << properties.driverVersion << '\n';
	identity << "api=" << properties.apiVersion << '\n';
	identity.write(reinterpret_cast<const char *>(properties.pipelineCacheUUID), VK_UUID_SIZE);
	return root / "vulkan-pipeline-v1" / (Utility::ComputeSHA256(identity.str()) + ".bin");
}

struct PipelineCacheLoadResult {
	std::vector<uint8_t> data;
	bool				 found = false;
	bool				 invalid = false;
};

PipelineCacheLoadResult LoadPipelineCacheData(const std::filesystem::path &path,
											  const VkPhysicalDeviceProperties &properties) {
	PipelineCacheLoadResult result;
	std::error_code		 error;
	result.found = std::filesystem::exists(path, error) && !error;
	if (!result.found) {
		return result;
	}

	const auto byteCount = std::filesystem::file_size(path, error);
	if (error || byteCount < sizeof(VkPipelineCacheHeaderVersionOne) || byteCount > kMaximumCachedPipelineBytes) {
		result.invalid = true;
	} else {
		std::ifstream stream(path, std::ios::binary);
		if (stream) {
			result.data.resize(static_cast<size_t>(byteCount));
			stream.read(reinterpret_cast<char *>(result.data.data()), static_cast<std::streamsize>(byteCount));
			if (!stream || !ValidatePipelineCacheData(result.data.data(), result.data.size(), properties)) {
				result.invalid = true;
				result.data.clear();
			}
		} else {
			result.invalid = true;
		}
	}

	if (result.invalid) {
		error.clear();
		std::filesystem::remove(path, error);
	}
	return result;
}

std::string BuildSpirvCacheKey(const std::string &glslSource, ShaderType type,
							   ShaderOptimizationLevel optimizationLevel, bool preserveInterface) {
	const auto glslangVersion = glslang::GetVersion();
	std::ostringstream key;
	key << kSpirvCacheSchema << '\n';
	key << "glslang=" << glslangVersion.major << '.' << glslangVersion.minor << '.' << glslangVersion.patch << '-'
		<< (glslangVersion.flavor == nullptr ? "" : glslangVersion.flavor) << '\n';
	key << "spirv-tools=" << spvSoftwareVersionDetailsString() << '\n';
	key << "target=vulkan1.1;spirv1.3\n";
	key << "stage=" << static_cast<uint32_t>(type) << '\n';
	key << "optimization=" << static_cast<uint32_t>(optimizationLevel) << '\n';
	key << "preserve-interface=" << (preserveInterface ? 1 : 0) << '\n';
#ifdef EASYGPU_SPIRV_OPT_ENABLED
	key << "optimizer-enabled=1\n";
#else
	key << "optimizer-enabled=0\n";
#endif
	key << "source-bytes=" << glslSource.size() << '\n';
	key << glslSource;
	return Utility::ComputeSHA256(key.str());
}

bool ValidateCachedSpirv(const std::vector<uint32_t> &spirv) {
	if (spirv.size() < 5 || spirv.front() != kSpirvMagicNumber) {
		return false;
	}
	spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_1);
	return validator.IsValid() && validator.Validate(spirv);
}

std::optional<std::vector<uint32_t>> LoadCachedSpirv(const std::filesystem::path &path) {
	std::error_code error;
	const auto byteCount = std::filesystem::file_size(path, error);
	if (error || byteCount < 5 * sizeof(uint32_t) || byteCount > kMaximumCachedSpirvBytes ||
		byteCount % sizeof(uint32_t) != 0) {
		if (!error) {
			std::filesystem::remove(path, error);
		}
		return std::nullopt;
	}

	std::ifstream stream(path, std::ios::binary);
	if (!stream) {
		return std::nullopt;
	}

	std::vector<uint32_t> spirv(static_cast<size_t>(byteCount / sizeof(uint32_t)));
	stream.read(reinterpret_cast<char *>(spirv.data()), static_cast<std::streamsize>(byteCount));
	if (!stream || !ValidateCachedSpirv(spirv)) {
		stream.close();
		std::filesystem::remove(path, error);
		return std::nullopt;
	}
	return spirv;
}

bool StoreCachedSpirv(const std::filesystem::path &path, const std::vector<uint32_t> &spirv) {
	std::error_code error;
	std::filesystem::create_directories(path.parent_path(), error);
	if (error) {
		return false;
	}
	if (std::filesystem::exists(path, error) && !error) {
		if (LoadCachedSpirv(path)) {
			return true;
		}
		error.clear();
		if (std::filesystem::exists(path, error) || error) {
			return false;
		}
	}

	return ReplaceCacheFileAtomically(path, spirv.data(), spirv.size() * sizeof(uint32_t));
}

#endif

VulkanBackend::InstanceExtensionProvider &GetInstanceExtensionProvider() {
	static VulkanBackend::InstanceExtensionProvider provider;
	return provider;
}

bool HasInstanceExtensionProvider() {
	return static_cast<bool>(GetInstanceExtensionProvider());
}

VkFilter ToVkFilter(SamplerFilter filter) {
	switch (filter) {
	case SamplerFilter::Nearest:
		return VK_FILTER_NEAREST;
	case SamplerFilter::Linear:
		return VK_FILTER_LINEAR;
	default:
		throw std::runtime_error("Unsupported sampler filter");
	}
}

VkSamplerAddressMode ToVkAddressMode(SamplerAddressMode mode) {
	switch (mode) {
	case SamplerAddressMode::ClampToEdge:
		return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	case SamplerAddressMode::Repeat:
		return VK_SAMPLER_ADDRESS_MODE_REPEAT;
	case SamplerAddressMode::MirroredRepeat:
		return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
	case SamplerAddressMode::ClampToBorder:
		return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	default:
		throw std::runtime_error("Unsupported sampler address mode");
	}
}

VkSamplerMipmapMode ToVkSamplerMipmapMode(SamplerMipmapMode mode) {
	switch (mode) {
	case SamplerMipmapMode::Nearest:
		return VK_SAMPLER_MIPMAP_MODE_NEAREST;
	case SamplerMipmapMode::Linear:
		return VK_SAMPLER_MIPMAP_MODE_LINEAR;
	default:
		throw std::runtime_error("Unsupported sampler mipmap mode");
	}
}

VkBorderColor ToVkBorderColor(SamplerBorderColor color) {
	switch (color) {
	case SamplerBorderColor::FloatTransparentBlack:
		return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
	case SamplerBorderColor::IntTransparentBlack:
		return VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;
	case SamplerBorderColor::FloatOpaqueBlack:
		return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
	case SamplerBorderColor::IntOpaqueBlack:
		return VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	case SamplerBorderColor::FloatOpaqueWhite:
		return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	case SamplerBorderColor::IntOpaqueWhite:
		return VK_BORDER_COLOR_INT_OPAQUE_WHITE;
	default:
		throw std::runtime_error("Unsupported sampler border color");
	}
}

VkBlendFactor ToVkBlendFactor(BlendFactor factor) {
	switch (factor) {
	case BlendFactor::Zero:
		return VK_BLEND_FACTOR_ZERO;
	case BlendFactor::One:
		return VK_BLEND_FACTOR_ONE;
	case BlendFactor::SrcColor:
		return VK_BLEND_FACTOR_SRC_COLOR;
	case BlendFactor::OneMinusSrcColor:
		return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
	case BlendFactor::DstColor:
		return VK_BLEND_FACTOR_DST_COLOR;
	case BlendFactor::OneMinusDstColor:
		return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
	case BlendFactor::SrcAlpha:
		return VK_BLEND_FACTOR_SRC_ALPHA;
	case BlendFactor::OneMinusSrcAlpha:
		return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	case BlendFactor::DstAlpha:
		return VK_BLEND_FACTOR_DST_ALPHA;
	case BlendFactor::OneMinusDstAlpha:
		return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
	default:
		throw std::runtime_error("Unsupported blend factor");
	}
}

VkBlendOp ToVkBlendOp(BlendOp op) {
	switch (op) {
	case BlendOp::Add:
		return VK_BLEND_OP_ADD;
	case BlendOp::Subtract:
		return VK_BLEND_OP_SUBTRACT;
	case BlendOp::ReverseSubtract:
		return VK_BLEND_OP_REVERSE_SUBTRACT;
	case BlendOp::Min:
		return VK_BLEND_OP_MIN;
	case BlendOp::Max:
		return VK_BLEND_OP_MAX;
	default:
		throw std::runtime_error("Unsupported blend op");
	}
}

VkCompareOp ToVkCompareOp(CompareOp op) {
	switch (op) {
	case CompareOp::Never:
		return VK_COMPARE_OP_NEVER;
	case CompareOp::Less:
		return VK_COMPARE_OP_LESS;
	case CompareOp::Equal:
		return VK_COMPARE_OP_EQUAL;
	case CompareOp::LessOrEqual:
		return VK_COMPARE_OP_LESS_OR_EQUAL;
	case CompareOp::Greater:
		return VK_COMPARE_OP_GREATER;
	case CompareOp::NotEqual:
		return VK_COMPARE_OP_NOT_EQUAL;
	case CompareOp::GreaterOrEqual:
		return VK_COMPARE_OP_GREATER_OR_EQUAL;
	case CompareOp::Always:
		return VK_COMPARE_OP_ALWAYS;
	default:
		throw std::runtime_error("Unsupported compare op");
	}
}

VkStencilOp ToVkStencilOp(StencilOp op) {
	switch (op) {
	case StencilOp::Keep:
		return VK_STENCIL_OP_KEEP;
	case StencilOp::Zero:
		return VK_STENCIL_OP_ZERO;
	case StencilOp::Replace:
		return VK_STENCIL_OP_REPLACE;
	case StencilOp::IncrementAndClamp:
		return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
	case StencilOp::DecrementAndClamp:
		return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
	case StencilOp::Invert:
		return VK_STENCIL_OP_INVERT;
	case StencilOp::IncrementAndWrap:
		return VK_STENCIL_OP_INCREMENT_AND_WRAP;
	case StencilOp::DecrementAndWrap:
		return VK_STENCIL_OP_DECREMENT_AND_WRAP;
	default:
		throw std::runtime_error("Unsupported stencil op");
	}
}

VkStencilOpState ToVkStencilOpState(const StencilFaceState &state, uint32_t readMask, uint32_t writeMask, uint32_t reference) {
	VkStencilOpState vkState = {};
	vkState.failOp			  = ToVkStencilOp(state.failOp);
	vkState.passOp			  = ToVkStencilOp(state.passOp);
	vkState.depthFailOp		  = ToVkStencilOp(state.depthFailOp);
	vkState.compareOp		  = ToVkCompareOp(state.compareOp);
	vkState.compareMask		  = readMask;
	vkState.writeMask		  = writeMask;
	vkState.reference		  = reference;
	return vkState;
}

VkCullModeFlags ToVkCullMode(CullMode mode) {
	switch (mode) {
	case CullMode::None:
		return VK_CULL_MODE_NONE;
	case CullMode::Front:
		return VK_CULL_MODE_FRONT_BIT;
	case CullMode::Back:
		return VK_CULL_MODE_BACK_BIT;
	case CullMode::FrontAndBack:
		return VK_CULL_MODE_FRONT_AND_BACK;
	default:
		throw std::runtime_error("Unsupported cull mode");
	}
}

VkFrontFace ToVkFrontFace(FrontFace face) {
	switch (face) {
	case FrontFace::CounterClockwise:
		return VK_FRONT_FACE_COUNTER_CLOCKWISE;
	case FrontFace::Clockwise:
		return VK_FRONT_FACE_CLOCKWISE;
	default:
		throw std::runtime_error("Unsupported front face");
	}
}

VkPolygonMode ToVkPolygonMode(PolygonMode mode) {
	switch (mode) {
	case PolygonMode::Fill:
		return VK_POLYGON_MODE_FILL;
	case PolygonMode::Line:
		return VK_POLYGON_MODE_LINE;
	case PolygonMode::Point:
		return VK_POLYGON_MODE_POINT;
	default:
		throw std::runtime_error("Unsupported polygon mode");
	}
}

VkColorComponentFlags ToVkColorWriteMask(uint32_t mask) {
	VkColorComponentFlags flags = 0;
	if ((mask & ColorWriteRed) != 0)
		flags |= VK_COLOR_COMPONENT_R_BIT;
	if ((mask & ColorWriteGreen) != 0)
		flags |= VK_COLOR_COMPONENT_G_BIT;
	if ((mask & ColorWriteBlue) != 0)
		flags |= VK_COLOR_COMPONENT_B_BIT;
	if ((mask & ColorWriteAlpha) != 0)
		flags |= VK_COLOR_COMPONENT_A_BIT;
	return flags;
}

bool TraceVulkan() {
	const char *value = std::getenv("EASYGPU_VULKAN_TRACE");
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool EnableVulkanValidation() {
	const char *value = std::getenv("EASYGPU_ENABLE_VALIDATION");
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
												   VkDebugUtilsMessageTypeFlagsEXT messageTypes,
												   const VkDebugUtilsMessengerCallbackDataEXT *callbackData,
												   void *userData) {
	(void)messageSeverity;
	(void)messageTypes;
	(void)userData;
	std::cerr << "[easygpu vulkan validation] "
			  << (callbackData && callbackData->pMessage ? callbackData->pMessage : "<no message>") << "\n";
	return VK_FALSE;
}

const char *BindingTypeName(BindingType type) {
	switch (type) {
	case BindingType::Buffer:
		return "Buffer";
	case BindingType::Texture:
		return "Texture";
	case BindingType::Sampler:
		return "Sampler";
	default:
		return "Unknown";
	}
}

} // namespace

void VulkanBackend::RegisterInstanceExtensionProvider(InstanceExtensionProvider provider) {
	GetInstanceExtensionProvider() = std::move(provider);
}

// =============================================================================
// Helper Functions
// =============================================================================

static VkResult CreateShaderModule(VkDevice device, const std::vector<uint32_t> &code, VkShaderModule *shaderModule) {
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType					= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize					= code.size() * sizeof(uint32_t);
	createInfo.pCode					= code.data();

	return vkCreateShaderModule(device, &createInfo, nullptr, shaderModule);
}

static const char *VkResultToString(VkResult result) {
	switch (result) {
	case VK_SUCCESS:
		return "VK_SUCCESS";
	case VK_NOT_READY:
		return "VK_NOT_READY";
	case VK_TIMEOUT:
		return "VK_TIMEOUT";
	case VK_EVENT_SET:
		return "VK_EVENT_SET";
	case VK_EVENT_RESET:
		return "VK_EVENT_RESET";
	case VK_INCOMPLETE:
		return "VK_INCOMPLETE";
	case VK_ERROR_OUT_OF_HOST_MEMORY:
		return "VK_ERROR_OUT_OF_HOST_MEMORY";
	case VK_ERROR_OUT_OF_DEVICE_MEMORY:
		return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
	case VK_ERROR_INITIALIZATION_FAILED:
		return "VK_ERROR_INITIALIZATION_FAILED";
	case VK_ERROR_DEVICE_LOST:
		return "VK_ERROR_DEVICE_LOST";
	case VK_ERROR_MEMORY_MAP_FAILED:
		return "VK_ERROR_MEMORY_MAP_FAILED";
	case VK_ERROR_LAYER_NOT_PRESENT:
		return "VK_ERROR_LAYER_NOT_PRESENT";
	case VK_ERROR_EXTENSION_NOT_PRESENT:
		return "VK_ERROR_EXTENSION_NOT_PRESENT";
	case VK_ERROR_FEATURE_NOT_PRESENT:
		return "VK_ERROR_FEATURE_NOT_PRESENT";
	case VK_ERROR_INCOMPATIBLE_DRIVER:
		return "VK_ERROR_INCOMPATIBLE_DRIVER";
	case VK_ERROR_TOO_MANY_OBJECTS:
		return "VK_ERROR_TOO_MANY_OBJECTS";
	case VK_ERROR_FORMAT_NOT_SUPPORTED:
		return "VK_ERROR_FORMAT_NOT_SUPPORTED";
	case VK_ERROR_FRAGMENTED_POOL:
		return "VK_ERROR_FRAGMENTED_POOL";
	case VK_ERROR_UNKNOWN:
		return "VK_ERROR_UNKNOWN";
	default:
		return "Unknown VkResult";
	}
}

static void CheckVkResult(VkResult result, const char *operation) {
	if (result != VK_SUCCESS) {
		throw std::runtime_error(std::string("Vulkan operation failed: ") + operation + " - " +
								 VkResultToString(result));
	}
}

static size_t PixelFormatByteSize(PixelFormat format) {
	switch (format) {
	case PixelFormat::R8:
		return 1;
	case PixelFormat::RG8:
		return 2;
	case PixelFormat::RGBA8:
	case PixelFormat::R32F:
	case PixelFormat::R32I:
	case PixelFormat::R32UI:
	case PixelFormat::RG16F:
	case PixelFormat::D32F:
	case PixelFormat::D24S8:
		return 4;
	case PixelFormat::RG32F:
	case PixelFormat::RGBA16F:
	case PixelFormat::RG32I:
	case PixelFormat::RG32UI:
		return 8;
	case PixelFormat::RGB32F:
	case PixelFormat::RGB32I:
	case PixelFormat::RGB32UI:
		return 12;
	case PixelFormat::RGBA32F:
	case PixelFormat::RGBA32I:
	case PixelFormat::RGBA32UI:
		return 16;
	case PixelFormat::R16F:
		return 2;
	}
	return 4;
}

struct ScopedVulkanBuffer {
	VkDevice	   device = nullptr;
	VkBuffer	   buffer = nullptr;
	VkDeviceMemory memory = nullptr;

	ScopedVulkanBuffer()  = default;
	ScopedVulkanBuffer(VkDevice device_) : device(device_) {
	}
	~ScopedVulkanBuffer() {
		if (buffer) {
			vkDestroyBuffer(device, buffer, nullptr);
		}
		if (memory) {
			vkFreeMemory(device, memory, nullptr);
		}
	}

	ScopedVulkanBuffer(const ScopedVulkanBuffer &)			  = delete;
	ScopedVulkanBuffer &operator=(const ScopedVulkanBuffer &) = delete;
};

// =============================================================================
// glslang Initialization Helper
// =============================================================================

static std::mutex g_glslangMutex;
static int		  g_glslangRefCount = 0;

static void		  InitializeGlslang() {
	std::lock_guard<std::mutex> lock(g_glslangMutex);
	if (g_glslangRefCount == 0) {
		glslang::InitializeProcess();
	}
	++g_glslangRefCount;
}

static void ShutdownGlslang() {
	std::lock_guard<std::mutex> lock(g_glslangMutex);
	--g_glslangRefCount;
	if (g_glslangRefCount == 0) {
		glslang::FinalizeProcess();
	}
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

VulkanBackend::VulkanBackend() = default;

VulkanBackend::~VulkanBackend() {
	if (_initialized) {
		try {
			Shutdown();
		} catch (...) {
			// Ignore exceptions during destruction to prevent std::terminate
			// Destruction should be noexcept
		}
	}
}

// =============================================================================
// Initialization and Shutdown
// =============================================================================

void VulkanBackend::Initialize() {
	std::lock_guard<std::mutex> lock(_mutex);

	if (_initialized) {
		return;
	}
	_spirvMemoryCache.clear();
	_spirvMemoryCacheBytes = 0;
	_spirvMemoryCacheAccess = 0;

	try {
		InitializeGlslang();
		CreateInstance();
		SelectPhysicalDevice();
		CreateDevice();
		CreateCommandPool();
		CreateDescriptorPool();
		CreateDefaultSampler();
		CreateQueryPool();
		InitializePipelineCache();

		_initialized = true;
	} catch (const std::exception &e) {
		CleanupVulkan();
		ShutdownGlslang();
		throw std::runtime_error(std::string("Failed to initialize Vulkan backend: ") + e.what());
	}
}

void VulkanBackend::Shutdown() {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		return;
	}

	CleanupVulkan();
	ShutdownGlslang();
	_spirvMemoryCache.clear();
	_spirvMemoryCacheBytes = 0;
	_spirvMemoryCacheAccess = 0;

	_initialized = false;
}

bool VulkanBackend::IsInitialized() const {
	return _initialized;
}

void VulkanBackend::CleanupVulkan() {
	if (_device) {
		vkDeviceWaitIdle(_device);
		PersistPipelineCache();
		_descriptorSets.clear();
		_inFlightDescriptorSets.clear();
		for (auto &[key, sampler] : _samplerCache) {
			(void)key;
			if (sampler)
				vkDestroySampler(_device, sampler, nullptr);
		}
		_samplerCache.clear();

		for (auto &attachment : _msaaAttachments) {
			if (attachment.view)
				vkDestroyImageView(_device, attachment.view, nullptr);
			if (attachment.image)
				vkDestroyImage(_device, attachment.image, nullptr);
			if (attachment.memory)
				vkFreeMemory(_device, attachment.memory, nullptr);
		}
		_msaaAttachments.clear();

		// Destroy pipelines
		for (auto &[handle, info] : _pipelines) {
			if (info.pipeline)
				vkDestroyPipeline(_device, info.pipeline, nullptr);
			if (info.layout)
				vkDestroyPipelineLayout(_device, info.layout, nullptr);
			if (info.descriptorSetLayout)
				vkDestroyDescriptorSetLayout(_device, info.descriptorSetLayout, nullptr);
		}
		_pipelines.clear();

		// Destroy shaders
		for (auto &[handle, info] : _shaders) {
			if (info.module)
				vkDestroyShaderModule(_device, info.module, nullptr);
		}
		_shaders.clear();

		// Destroy textures
		for (auto &[handle, info] : _textures) {
			if (info.sampledView)
				vkDestroyImageView(_device, info.sampledView, nullptr);
			if (info.view)
				vkDestroyImageView(_device, info.view, nullptr);
			if (info.image) {
				vkDestroyImage(_device, info.image, nullptr);
			}
			if (info.memory)
				vkFreeMemory(_device, info.memory, nullptr);
		}
		_textures.clear();

		// Destroy buffers
		for (auto &[handle, info] : _buffers) {
			if (info.isMapped && info.stagingMemory) {
				vkUnmapMemory(_device, info.stagingMemory);
			}
			if (info.buffer)
				vkDestroyBuffer(_device, info.buffer, nullptr);
			if (info.stagingBuffer)
				vkDestroyBuffer(_device, info.stagingBuffer, nullptr);
			if (info.memory)
				vkFreeMemory(_device, info.memory, nullptr);
			if (info.stagingMemory)
				vkFreeMemory(_device, info.stagingMemory, nullptr);
		}
		_buffers.clear();

		// Destroy descriptor pools
		if (_defaultSampler)
			vkDestroySampler(_device, _defaultSampler, nullptr);
		if (_mipmapSampler)
			vkDestroySampler(_device, _mipmapSampler, nullptr);
		for (auto pool : _descriptorPools) {
			if (pool)
				vkDestroyDescriptorPool(_device, pool, nullptr);
		}
		_descriptorPools.clear();
		_descriptorPool = nullptr;

		// Destroy pipeline cache
		if (_pipelineCache) {
			vkDestroyPipelineCache(_device, _pipelineCache, nullptr);
			_pipelineCache = nullptr;
		}

		// Destroy query pool
		if (_queryPool)
			vkDestroyQueryPool(_device, _queryPool, nullptr);

		// Destroy command resources
		for (auto &[handle, submission] : _submissions) {
			(void)handle;
			DestroySubmissionResources(submission);
		}
		_submissions.clear();
		for (auto &submission : _availableSubmissionResources) {
			DestroySubmissionResources(submission);
		}
		_availableSubmissionResources.clear();
		if (_commandFence)
			vkDestroyFence(_device, _commandFence, nullptr);
		if (_commandPool)
			vkDestroyCommandPool(_device, _commandPool, nullptr);
		_commandFence = nullptr;
		_commandPool = nullptr;
		_commandBuffer = nullptr;
		_commandBufferRecording = false;

		// Destroy device
		vkDestroyDevice(_device, nullptr);
		_device = nullptr;
	}

	// Destroy instance
	if (_instance) {
		if (_debugMessenger) {
			auto destroyDebugMessenger =
				reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
					vkGetInstanceProcAddr(_instance, "vkDestroyDebugUtilsMessengerEXT"));
			if (destroyDebugMessenger) {
				destroyDebugMessenger(_instance, _debugMessenger, nullptr);
			}
			_debugMessenger = nullptr;
		}
		vkDestroyInstance(_instance, nullptr);
		_instance = nullptr;
	}
}

void VulkanBackend::InitializePipelineCache() {
	_pipelineCacheStats = {};
	_pipelineCachePath.clear();
	_pipelineCacheDirty = false;

	std::vector<uint8_t> initialData;
#ifdef EASYGPU_SHADER_CACHE_ENABLED
	try {
		if (const auto root = GetShaderCacheRootDirectory()) {
			VkPhysicalDeviceProperties properties{};
			vkGetPhysicalDeviceProperties(_physicalDevice, &properties);
			const auto path = BuildPipelineCachePath(*root, properties);
			_pipelineCachePath = path;

			std::error_code error;
			std::filesystem::create_directories(path.parent_path(), error);
			if (!error) {
				auto lockPath = path;
				lockPath += ".lock";
				CacheFileLock cacheLock(lockPath);
				if (cacheLock.IsLocked()) {
					auto cached = LoadPipelineCacheData(path, properties);
					if (!cached.data.empty()) {
						initialData = std::move(cached.data);
					} else {
						++_pipelineCacheStats.diskCacheMisses;
						if (cached.invalid) {
							++_pipelineCacheStats.invalidDiskEntries;
						}
					}
				}
			}
		}
	} catch (...) {
		_pipelineCachePath.clear();
		initialData.clear();
	}
#endif

	VkPipelineCacheCreateInfo cacheCreateInfo{};
	cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	cacheCreateInfo.initialDataSize = initialData.size();
	cacheCreateInfo.pInitialData = initialData.empty() ? nullptr : initialData.data();

	VkResult result = vkCreatePipelineCache(_device, &cacheCreateInfo, nullptr, &_pipelineCache);
	if (result == VK_SUCCESS && !initialData.empty()) {
		++_pipelineCacheStats.diskCacheHits;
		_pipelineCacheStats.loadedBytes = initialData.size();
		_pipelineCacheStats.lastDiskCacheHit = true;
		return;
	}

	if (!initialData.empty()) {
		++_pipelineCacheStats.diskCacheMisses;
		++_pipelineCacheStats.invalidDiskEntries;
		std::error_code error;
		std::filesystem::remove(_pipelineCachePath, error);
	}

	if (result != VK_SUCCESS) {
		cacheCreateInfo.initialDataSize = 0;
		cacheCreateInfo.pInitialData = nullptr;
		result = vkCreatePipelineCache(_device, &cacheCreateInfo, nullptr, &_pipelineCache);
	}
	if (result != VK_SUCCESS) {
		_pipelineCache = nullptr;
	}
}

void VulkanBackend::PersistPipelineCache() {
#ifdef EASYGPU_SHADER_CACHE_ENABLED
	if (!_pipelineCacheDirty || _pipelineCache == nullptr || _pipelineCachePath.empty()) {
		return;
	}

	try {
		const std::filesystem::path path = _pipelineCachePath;
		std::error_code error;
		std::filesystem::create_directories(path.parent_path(), error);
		if (error) {
			++_pipelineCacheStats.diskCacheWriteFailures;
			return;
		}

		auto lockPath = path;
		lockPath += ".lock";
		CacheFileLock cacheLock(lockPath);
		if (!cacheLock.IsLocked()) {
			++_pipelineCacheStats.diskCacheWriteFailures;
			return;
		}

		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(_physicalDevice, &properties);
		auto latest = LoadPipelineCacheData(path, properties);
		if (latest.invalid) {
			++_pipelineCacheStats.invalidDiskEntries;
		}
		if (!latest.data.empty()) {
			VkPipelineCacheCreateInfo mergeCreateInfo{};
			mergeCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
			mergeCreateInfo.initialDataSize = latest.data.size();
			mergeCreateInfo.pInitialData = latest.data.data();

			VkPipelineCache mergeCache = nullptr;
			if (vkCreatePipelineCache(_device, &mergeCreateInfo, nullptr, &mergeCache) == VK_SUCCESS) {
				const VkResult mergeResult = vkMergePipelineCaches(_device, _pipelineCache, 1, &mergeCache);
				vkDestroyPipelineCache(_device, mergeCache, nullptr);
				if (mergeResult != VK_SUCCESS) {
					++_pipelineCacheStats.diskCacheWriteFailures;
					return;
				}
			} else {
				++_pipelineCacheStats.invalidDiskEntries;
				error.clear();
				std::filesystem::remove(path, error);
			}
		}

		auto data = GetPipelineCacheBytes(_device, _pipelineCache);
		if (!data || data->size() <= sizeof(VkPipelineCacheHeaderVersionOne) ||
			!ValidatePipelineCacheData(data->data(), data->size(), properties)) {
			_pipelineCacheDirty = false;
			return;
		}
		if (!latest.data.empty() && latest.data == *data) {
			_pipelineCacheDirty = false;
			return;
		}

		if (ReplaceCacheFileAtomically(path, data->data(), data->size())) {
			++_pipelineCacheStats.diskCacheWrites;
			_pipelineCacheStats.savedBytes = data->size();
			_pipelineCacheDirty = false;
		} else {
			++_pipelineCacheStats.diskCacheWriteFailures;
		}
	} catch (...) {
		++_pipelineCacheStats.diskCacheWriteFailures;
	}
#else
	_pipelineCacheDirty = false;
#endif
}

void VulkanBackend::CreateInstance() {
	VkApplicationInfo appInfo		= {};
	appInfo.sType					= VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName		= "EasyGPU";
	appInfo.applicationVersion		= VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName				= "EasyGPU";
	appInfo.engineVersion			= VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion				= VK_API_VERSION_1_1;

	VkInstanceCreateInfo createInfo = {};
	createInfo.sType				= VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo		= &appInfo;

	const char *validationLayer = "VK_LAYER_KHRONOS_validation";
	bool		enableValidation = false;
	// Enable validation layers only when explicitly requested and available.
	if (EnableVulkanValidation()) {
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		if (layerCount != 0) {
			vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
		}

		const bool hasValidationLayer = std::any_of(
			availableLayers.begin(), availableLayers.end(),
			[&](const VkLayerProperties &layer) { return std::strcmp(layer.layerName, validationLayer) == 0; });
		if (hasValidationLayer) {
			createInfo.enabledLayerCount	 = 1;
			createInfo.ppEnabledLayerNames = &validationLayer;
			enableValidation = true;
		}
	}

	std::vector<const char *> instanceExtensions;
	if (auto &provider = GetInstanceExtensionProvider()) {
		for (const char *extension : provider()) {
			if (extension) {
				instanceExtensions.push_back(extension);
			}
		}
	}

	if (enableValidation) {
		instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

#ifdef __APPLE__
	instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

	createInfo.enabledExtensionCount   = static_cast<uint32_t>(instanceExtensions.size());
	createInfo.ppEnabledExtensionNames = instanceExtensions.empty() ? nullptr : instanceExtensions.data();

	VkResult result					   = vkCreateInstance(&createInfo, nullptr, &_instance);
	CheckVkResult(result, "vkCreateInstance");

	if (enableValidation) {
		auto createDebugMessenger =
			reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
				vkGetInstanceProcAddr(_instance, "vkCreateDebugUtilsMessengerEXT"));
		if (createDebugMessenger) {
			VkDebugUtilsMessengerCreateInfoEXT debugInfo = {};
			debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
										VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
									VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
									VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			debugInfo.pfnUserCallback = VulkanDebugCallback;
			result = createDebugMessenger(_instance, &debugInfo, nullptr, &_debugMessenger);
			if (result != VK_SUCCESS) {
				_debugMessenger = nullptr;
			}
		}
	}
}

void VulkanBackend::SelectPhysicalDevice() {
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr);

	if (deviceCount == 0) {
		throw std::runtime_error("No Vulkan-capable physical devices found");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data());

	// Select the first device that supports both graphics and compute (preferred)
	for (auto device : devices) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(device, &props);

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		// First pass: look for a queue family with both graphics and compute
		for (uint32_t i = 0; i < queueFamilyCount; ++i) {
			if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
				(queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
				_physicalDevice				   = device;
				_computeQueueFamilyIndex	   = i;

				// Store capabilities
				_caps.versionString			   = std::to_string(VK_VERSION_MAJOR(props.apiVersion)) + "." +
												 std::to_string(VK_VERSION_MINOR(props.apiVersion));

				VkPhysicalDeviceLimits limits  = props.limits;
				_caps.maxWorkGroupSizeX		   = limits.maxComputeWorkGroupSize[0];
				_caps.maxWorkGroupSizeY		   = limits.maxComputeWorkGroupSize[1];
				_caps.maxWorkGroupSizeZ		   = limits.maxComputeWorkGroupSize[2];
				_caps.maxBufferBindings		   = limits.maxPerStageDescriptorStorageBuffers;
				_caps.maxTextureBindings	   = limits.maxPerStageDescriptorStorageImages;
				_caps.supportsComputeShaders   = true;
				_caps.supportsGraphics		   = true;
				_caps.supportsAsyncTransfer	   = false;
				_caps.supportsMultiQueue	   = false;
				_caps.supportsTimestampQueries = queueFamilies[i].timestampValidBits != 0;
#ifdef __APPLE__
				// MoltenVK timestamp queries are not reliable across all supported
				// macOS/GPU combinations. Use the profiler's synchronized CPU fallback.
				_caps.supportsTimestampQueries = false;
#endif
				_timestampPeriod	 = limits.timestampPeriod;
				_maxPushConstantSize = limits.maxPushConstantsSize;
				VkPhysicalDeviceFeatures features{};
				vkGetPhysicalDeviceFeatures(device, &features);
				_samplerAnisotropySupported = features.samplerAnisotropy == VK_TRUE;
				_depthClampSupported = features.depthClamp == VK_TRUE;
				_fillModeNonSolidSupported = features.fillModeNonSolid == VK_TRUE;
				_caps.supportsDepthClamp = _depthClampSupported;
				_caps.supportsNonFillPolygonMode = _fillModeNonSolidSupported;
				_maxSamplerAnisotropy = std::max(1.0f, limits.maxSamplerAnisotropy);
				return;
			}
		}

		// Fallback: find any device with compute support (compute-only)
		for (uint32_t i = 0; i < queueFamilyCount; ++i) {
			if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				_physicalDevice				   = device;
				_computeQueueFamilyIndex	   = i;

				_caps.versionString			   = std::to_string(VK_VERSION_MAJOR(props.apiVersion)) + "." +
												 std::to_string(VK_VERSION_MINOR(props.apiVersion));

				VkPhysicalDeviceLimits limits  = props.limits;
				_caps.maxWorkGroupSizeX		   = limits.maxComputeWorkGroupSize[0];
				_caps.maxWorkGroupSizeY		   = limits.maxComputeWorkGroupSize[1];
				_caps.maxWorkGroupSizeZ		   = limits.maxComputeWorkGroupSize[2];
				_caps.maxBufferBindings		   = limits.maxPerStageDescriptorStorageBuffers;
				_caps.maxTextureBindings	   = limits.maxPerStageDescriptorStorageImages;
				_caps.supportsComputeShaders   = true;
				_caps.supportsGraphics		   = false;
				_caps.supportsAsyncTransfer	   = false;
				_caps.supportsMultiQueue	   = false;
				_caps.supportsTimestampQueries = queueFamilies[i].timestampValidBits != 0;
#ifdef __APPLE__
				_caps.supportsTimestampQueries = false;
#endif
				_timestampPeriod	 = limits.timestampPeriod;
				_maxPushConstantSize = limits.maxPushConstantsSize;
				VkPhysicalDeviceFeatures features{};
				vkGetPhysicalDeviceFeatures(device, &features);
				_samplerAnisotropySupported = features.samplerAnisotropy == VK_TRUE;
				_depthClampSupported = features.depthClamp == VK_TRUE;
				_fillModeNonSolidSupported = features.fillModeNonSolid == VK_TRUE;
				_caps.supportsDepthClamp = _depthClampSupported;
				_caps.supportsNonFillPolygonMode = _fillModeNonSolidSupported;
				_maxSamplerAnisotropy = std::max(1.0f, limits.maxSamplerAnisotropy);
				return;
			}
		}
	}

	throw std::runtime_error("No Vulkan device with compute support found");
}

void VulkanBackend::CreateDevice() {
	float					queuePriority	   = 1.0f;

	VkDeviceQueueCreateInfo queueCreateInfo	   = {};
	queueCreateInfo.sType					   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex		   = _computeQueueFamilyIndex;
	queueCreateInfo.queueCount				   = 1;
	queueCreateInfo.pQueuePriorities		   = &queuePriority;

	// Enable only features actually supported by the device.
	VkPhysicalDeviceFeatures supportedFeatures = {};
	vkGetPhysicalDeviceFeatures(_physicalDevice, &supportedFeatures);

	VkPhysicalDeviceFeatures deviceFeatures				= {};
	deviceFeatures.shaderStorageImageReadWithoutFormat	= supportedFeatures.shaderStorageImageReadWithoutFormat;
	deviceFeatures.shaderStorageImageWriteWithoutFormat = supportedFeatures.shaderStorageImageWriteWithoutFormat;
	deviceFeatures.samplerAnisotropy					= supportedFeatures.samplerAnisotropy;
	deviceFeatures.depthClamp							= supportedFeatures.depthClamp;
	deviceFeatures.fillModeNonSolid						= supportedFeatures.fillModeNonSolid;
	_depthClampSupported								= supportedFeatures.depthClamp == VK_TRUE;
	_fillModeNonSolidSupported							= supportedFeatures.fillModeNonSolid == VK_TRUE;
	_caps.supportsDepthClamp							= _depthClampSupported;
	_caps.supportsNonFillPolygonMode					= _fillModeNonSolidSupported;

	// Enable dynamic rendering feature (VK_KHR_dynamic_rendering)
	VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures = {};
	dynamicRenderingFeatures.sType			  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
	dynamicRenderingFeatures.dynamicRendering = VK_TRUE;

	uint32_t extensionCount					  = 0;
	vkEnumerateDeviceExtensionProperties(_physicalDevice, nullptr, &extensionCount, nullptr);
	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	if (extensionCount != 0) {
		vkEnumerateDeviceExtensionProperties(_physicalDevice, nullptr, &extensionCount, availableExtensions.data());
	}

	std::set<std::string> availableExtensionNames;
	for (const auto &extension : availableExtensions) {
		availableExtensionNames.insert(extension.extensionName);
	}

	std::vector<const char *> deviceExtensions;
	if (_caps.supportsGraphics) {
		if (availableExtensionNames.count(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) == 0) {
			_caps.supportsGraphics = false;
		} else {
			deviceExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
			if (availableExtensionNames.count(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME) != 0) {
				deviceExtensions.push_back(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
			}
		}
	}
	if (HasInstanceExtensionProvider()) {
		if (availableExtensionNames.count(VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
			throw std::runtime_error("Vulkan device does not support VK_KHR_swapchain");
		}
		deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#ifdef __APPLE__
		if (availableExtensionNames.count("VK_KHR_portability_subset") != 0) {
			deviceExtensions.push_back("VK_KHR_portability_subset");
		}
#endif
	}

	VkDeviceCreateInfo createInfo	   = {};
	createInfo.sType				   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.pNext				   = _caps.supportsGraphics ? &dynamicRenderingFeatures : nullptr;
	createInfo.queueCreateInfoCount	   = 1;
	createInfo.pQueueCreateInfos	   = &queueCreateInfo;
	createInfo.pEnabledFeatures		   = &deviceFeatures;
	createInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
	createInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();
	createInfo.enabledLayerCount	   = 0;

	VkResult result					   = vkCreateDevice(_physicalDevice, &createInfo, nullptr, &_device);
	CheckVkResult(result, "vkCreateDevice");

	// Get the queue (works for both graphics and compute)
	vkGetDeviceQueue(_device, _computeQueueFamilyIndex, 0, &_computeQueue);

	// Load dynamic rendering function pointers (may fail gracefully if not supported)
	if (_caps.supportsGraphics) {
		_vkCmdBeginRenderingKHR = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(_device, "vkCmdBeginRenderingKHR");
		_vkCmdEndRenderingKHR	= (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(_device, "vkCmdEndRenderingKHR");
		if (!_vkCmdBeginRenderingKHR || !_vkCmdEndRenderingKHR) {
			// Dynamic rendering not available, disable graphics
			_caps.supportsGraphics = false;
		}
	}
}
void VulkanBackend::CreateCommandPool() {
	if (_commandPool != nullptr) {
		return;
	}
	if (!_availableSubmissionResources.empty()) {
		auto resources = _availableSubmissionResources.back();
		_availableSubmissionResources.pop_back();
		_commandPool = resources.pool;
		_commandBuffer = resources.commandBuffer;
		_commandFence = resources.fence;
		CheckVkResult(vkResetCommandPool(_device, _commandPool, 0), "vkResetCommandPool");
		CheckVkResult(vkResetFences(_device, 1, &_commandFence), "vkResetFences");
		return;
	}

	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType					 = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex		 = _computeQueueFamilyIndex;
	poolInfo.flags					 = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	VkResult result					 = vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool);
	CheckVkResult(result, "vkCreateCommandPool");

	// Create command buffer
	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType						  = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool				  = _commandPool;
	allocInfo.level						  = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount		  = 1;

	result								  = vkAllocateCommandBuffers(_device, &allocInfo, &_commandBuffer);
	CheckVkResult(result, "vkAllocateCommandBuffers");

	// Create fence for synchronization
	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType				= VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	result						= vkCreateFence(_device, &fenceInfo, nullptr, &_commandFence);
	CheckVkResult(result, "vkCreateFence");
}

void VulkanBackend::CreateDescriptorPool() {
	VkDescriptorPoolSize poolSizes[3]		   = {};

	// Storage buffers
	poolSizes[0].type						   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[0].descriptorCount			   = MAX_BUFFER_BINDINGS * MAX_DESCRIPTOR_SETS;

	// Storage images
	poolSizes[1].type						   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[1].descriptorCount			   = MAX_TEXTURE_BINDINGS * MAX_DESCRIPTOR_SETS;

	// Sampled textures use combined image samplers.
	poolSizes[2].type						   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[2].descriptorCount			   = MAX_TEXTURE_BINDINGS * MAX_DESCRIPTOR_SETS;
	// Uniform buffers (for UBO support)
	VkDescriptorPoolSize poolSizeUBO		   = {};
	poolSizeUBO.type						   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizeUBO.descriptorCount				   = MAX_BUFFER_BINDINGS * MAX_DESCRIPTOR_SETS;

	VkDescriptorPoolSize	   allPoolSizes[4] = {poolSizes[0], poolSizes[1], poolSizes[2], poolSizeUBO};

	VkDescriptorPoolCreateInfo poolInfo		   = {};
	poolInfo.sType							   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount					   = 4;
	poolInfo.pPoolSizes						   = allPoolSizes;
	poolInfo.maxSets						   = MAX_DESCRIPTOR_SETS;
	poolInfo.flags							   = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

	VkDescriptorPool pool					   = nullptr;
	VkResult result							   = vkCreateDescriptorPool(_device, &poolInfo, nullptr, &pool);
	CheckVkResult(result, "vkCreateDescriptorPool");
	_descriptorPools.push_back(pool);
	_descriptorPool = pool;
}

void VulkanBackend::CreateDefaultSampler() {
	VkSamplerCreateInfo samplerInfo		= {};
	samplerInfo.sType					= VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter				= VK_FILTER_NEAREST;
	samplerInfo.minFilter				= VK_FILTER_NEAREST;
	samplerInfo.mipmapMode				= VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.addressModeU			= VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV			= VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW			= VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.mipLodBias				= 0.0f;
	samplerInfo.anisotropyEnable		= VK_FALSE;
	samplerInfo.maxAnisotropy			= 1.0f;
	samplerInfo.compareEnable			= VK_FALSE;
	samplerInfo.compareOp				= VK_COMPARE_OP_ALWAYS;
	samplerInfo.minLod					= 0.0f;
	samplerInfo.maxLod					= 0.0f;
	samplerInfo.borderColor				= VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;

	CheckVkResult(vkCreateSampler(_device, &samplerInfo, nullptr, &_defaultSampler), "vkCreateSampler");

	samplerInfo.magFilter  = VK_FILTER_LINEAR;
	samplerInfo.minFilter  = VK_FILTER_LINEAR;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.anisotropyEnable = _samplerAnisotropySupported ? VK_TRUE : VK_FALSE;
	samplerInfo.maxAnisotropy = _samplerAnisotropySupported ? _maxSamplerAnisotropy : 1.0f;
	samplerInfo.maxLod	   = VK_LOD_CLAMP_NONE;
	CheckVkResult(vkCreateSampler(_device, &samplerInfo, nullptr, &_mipmapSampler), "vkCreateSampler (mipmap)");
}

void VulkanBackend::CreateQueryPool() {
	if (!_caps.supportsTimestampQueries) {
		return;
	}

	VkQueryPoolCreateInfo queryPoolInfo = {};
	queryPoolInfo.sType					= VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	queryPoolInfo.queryType				= VK_QUERY_TYPE_TIMESTAMP;
	queryPoolInfo.queryCount			= MAX_QUERIES * 2; // Start and end timestamps

	VkResult result						= vkCreateQueryPool(_device, &queryPoolInfo, nullptr, &_queryPool);
	CheckVkResult(result, "vkCreateQueryPool");
}

// =============================================================================
// Context Management
// =============================================================================

void VulkanBackend::MakeCurrent() {
	std::lock_guard<std::mutex> lock(_mutex);
	_isCurrent = true;
	// Vulkan does not have an OpenGL-style current context. Command buffers are
	// opened lazily by operations that actually record GPU work; doing it here
	// can leak an empty recording scope across API calls.
}

void VulkanBackend::MakeNoneCurrent() {
	std::lock_guard<std::mutex> lock(_mutex);
	if (_commandBufferRecording) {
		EndCommandBuffer();
		SubmitCommandBuffer(false);
	}
	_isCurrent = false;
}

BackendCaps VulkanBackend::GetCaps() const {
	return _caps;
}

// =============================================================================
// Command Buffer Management
// =============================================================================

void VulkanBackend::EnsureCommandBuffer() {
	if (!_commandBufferRecording) {
		BeginCommandBuffer();
	}
}

void VulkanBackend::BeginCommandBuffer() {
	ReapReleasedSubmissions();
	CreateCommandPool();

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType					   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags					   = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VkResult result					   = vkBeginCommandBuffer(_commandBuffer, &beginInfo);
	CheckVkResult(result, "vkBeginCommandBuffer");

	_commandBufferRecording = true;
}

void VulkanBackend::EndCommandBuffer() {
	if (!_commandBufferRecording) {
		return;
	}

	VkResult result = vkEndCommandBuffer(_commandBuffer);
	CheckVkResult(result, "vkEndCommandBuffer");

	_commandBufferRecording = false;
}

SubmissionHandle VulkanBackend::SubmitCommandBuffer(bool wait, bool externallyVisible) {
	if (_commandPool == nullptr || _commandBuffer == nullptr || _commandFence == nullptr) {
		throw std::runtime_error("No Vulkan command buffer is available for submission");
	}
	if (_nextSubmissionHandle == INVALID_SUBMISSION_HANDLE) {
		throw std::runtime_error("Vulkan submission handle space exhausted");
	}

	VkSubmitInfo submitInfo		  = {};
	submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers	  = &_commandBuffer;

	VkResult result = vkQueueSubmit(_computeQueue, 1, &submitInfo, _commandFence);
	if (result != VK_SUCCESS) {
		throw std::runtime_error(std::string("vkQueueSubmit failed: ") + VkResultToString(result));
	}

	const SubmissionHandle submission = _nextSubmissionHandle++;
	_submissions.emplace(submission, SubmissionInfo{
		_commandPool,
		_commandBuffer,
		_commandFence,
		false,
		!externallyVisible
	});
	_commandPool = nullptr;
	_commandBuffer = nullptr;
	_commandFence = nullptr;

	if (wait) {
		(void)UpdateSubmissionStatus(submission, UINT64_MAX, true);
		ReapReleasedSubmissions();
	}
	return submission;
}

void VulkanBackend::WaitForSubmittedWork() {
	std::vector<SubmissionHandle> submissions;
	submissions.reserve(_submissions.size());
	for (const auto &[handle, info] : _submissions) {
		(void)info;
		submissions.push_back(handle);
	}
	for (const auto submission : submissions) {
		(void)UpdateSubmissionStatus(submission, UINT64_MAX, true);
	}
	ReapReleasedSubmissions();
}

bool VulkanBackend::UpdateSubmissionStatus(SubmissionHandle submission, uint64_t timeoutNanoseconds, bool wait) {
	auto it = _submissions.find(submission);
	if (it == _submissions.end()) {
		throw std::runtime_error("Invalid submission handle");
	}
	auto &info = it->second;
	if (info.completed) {
		return true;
	}
	if (submission <= _completedSubmissionWatermark) {
		info.completed = true;
		RecycleSubmissionResources(info);
		return true;
	}

	const VkResult result = wait
		? vkWaitForFences(_device, 1, &info.fence, VK_TRUE, timeoutNanoseconds)
		: vkGetFenceStatus(_device, info.fence);
	if (result == VK_TIMEOUT || result == VK_NOT_READY) {
		return false;
	}
	if (result != VK_SUCCESS) {
		throw std::runtime_error(std::string(wait ? "vkWaitForFences failed: " : "vkGetFenceStatus failed: ") +
								 VkResultToString(result));
	}

	_completedSubmissionWatermark = std::max(_completedSubmissionWatermark, submission);
	info.completed = true;
	RecycleSubmissionResources(info);
	return true;
}

void VulkanBackend::ReapReleasedSubmissions() {
	for (auto it = _submissions.begin(); it != _submissions.end();) {
		if (!it->second.released) {
			++it;
			continue;
		}
		if (!it->second.completed && it->first > _completedSubmissionWatermark) {
			const VkResult result = vkGetFenceStatus(_device, it->second.fence);
			if (result == VK_NOT_READY) {
				++it;
				continue;
			}
			if (result != VK_SUCCESS) {
				throw std::runtime_error(std::string("vkGetFenceStatus failed: ") + VkResultToString(result));
			}
			_completedSubmissionWatermark = std::max(_completedSubmissionWatermark, it->first);
		}
		it->second.completed = true;
		RecycleSubmissionResources(it->second);
		it = _submissions.erase(it);
	}
}

void VulkanBackend::RecycleSubmissionResources(SubmissionInfo &submission) {
	if (submission.pool == nullptr || submission.commandBuffer == nullptr || submission.fence == nullptr) {
		DestroySubmissionResources(submission);
		return;
	}
	if (_availableSubmissionResources.size() >= MAX_CACHED_SUBMISSION_RESOURCES) {
		DestroySubmissionResources(submission);
		return;
	}

	_availableSubmissionResources.push_back(SubmissionInfo{
		submission.pool,
		submission.commandBuffer,
		submission.fence,
		false,
		false
	});
	submission.pool = nullptr;
	submission.commandBuffer = nullptr;
	submission.fence = nullptr;
}

void VulkanBackend::DestroySubmissionResources(SubmissionInfo &submission) {
	if (submission.fence != nullptr) {
		vkDestroyFence(_device, submission.fence, nullptr);
	}
	if (submission.pool != nullptr) {
		vkDestroyCommandPool(_device, submission.pool, nullptr);
	}
	submission.pool = nullptr;
	submission.commandBuffer = nullptr;
	submission.fence = nullptr;
}

void VulkanBackend::EnsureNoPendingGpuWork() {
	if (_commandBufferRecording) {
		EndCommandBuffer();
		(void)SubmitCommandBuffer(false);
	}
	if (!_submissions.empty()) {
		WaitForSubmittedWork();
	}
}

void VulkanBackend::InvalidateAllDescriptorCaches() {
	_descriptorSets.clear();
}

void VulkanBackend::InvalidateDescriptorCachesForPipeline(PipelineHandle pipeline) {
	auto						 eraseBegin =
		std::remove_if(_descriptorSets.begin(), _descriptorSets.end(), [&](const DescriptorSetCache &cache) {
			if (cache.pipeline == pipeline) {
				return true;
			}
			return false;
		});
	_descriptorSets.erase(eraseBegin, _descriptorSets.end());
}

void VulkanBackend::InvalidateDescriptorCachesForBuffer(BufferHandle buffer) {
	auto						 eraseBegin =
		std::remove_if(_descriptorSets.begin(), _descriptorSets.end(), [&](const DescriptorSetCache &cache) {
			for (uint32_t i = 0; i < MAX_BUFFER_BINDINGS; ++i) {
				if ((cache.bufferMask & (1ull << i)) != 0 && cache.boundBuffers[i] == buffer) {
					return true;
				}
			}
			return false;
		});
	_descriptorSets.erase(eraseBegin, _descriptorSets.end());
}

void VulkanBackend::InvalidateDescriptorCachesForTexture(TextureHandle texture) {
	auto						 eraseBegin =
		std::remove_if(_descriptorSets.begin(), _descriptorSets.end(), [&](const DescriptorSetCache &cache) {
			for (uint32_t i = 0; i < MAX_TEXTURE_BINDINGS; ++i) {
				if ((cache.textureMask & (1ull << i)) != 0 && cache.boundTextures[i] == texture) {
					return true;
				}
			}
			return false;
		});
	_descriptorSets.erase(eraseBegin, _descriptorSets.end());
}

// =============================================================================
// Buffer Management
// =============================================================================

BufferHandle VulkanBackend::CreateBuffer(const BufferDesc &desc) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
	}

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType			  = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size				  = desc.sizeInBytes;
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
					   VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
					   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer buffer		   = nullptr;
	VkResult result		   = vkCreateBuffer(_device, &bufferInfo, nullptr, &buffer);
	CheckVkResult(result, "vkCreateBuffer");

	VkMemoryPropertyFlags memProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	VkDeviceMemory		  memory		= nullptr;
	try {
		AllocateBufferMemory(buffer, memory, memProperties, desc.sizeInBytes);
	} catch (const std::exception &) {
		memProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		AllocateBufferMemory(buffer, memory, memProperties, desc.sizeInBytes);
	}

	result = vkBindBufferMemory(_device, buffer, memory, 0);
	CheckVkResult(result, "vkBindBufferMemory");

	VkBufferCreateInfo stagingInfo = {};
	stagingInfo.sType			   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingInfo.size			   = desc.sizeInBytes;
	stagingInfo.usage			   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	stagingInfo.sharingMode		   = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer stagingBuffer		   = nullptr;
	result						   = vkCreateBuffer(_device, &stagingInfo, nullptr, &stagingBuffer);
	CheckVkResult(result, "vkCreateBuffer (persistent staging)");

	constexpr VkMemoryPropertyFlags stagingProperties =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VkDeviceMemory stagingMemory = nullptr;
	AllocateBufferMemory(stagingBuffer, stagingMemory, stagingProperties, desc.sizeInBytes);

	result = vkBindBufferMemory(_device, stagingBuffer, stagingMemory, 0);
	CheckVkResult(result, "vkBindBufferMemory (persistent staging)");

	// Upload initial data if provided
	if (desc.initialData) {
		UploadBufferInternal(buffer, desc.sizeInBytes, desc.initialData);
	}

	BufferHandle handle = _nextBufferHandle++;
	BufferInfo	 info;
	info.buffer				= buffer;
	info.memory				= memory;
	info.stagingBuffer		= stagingBuffer;
	info.stagingMemory		= stagingMemory;
	info.size				= desc.sizeInBytes;
	info.mappedPtr			= nullptr;
	info.mode				= desc.mode;
	info.isMapped			= false;
	info.mappedForRead		= false;
	info.mappedForWrite		= false;
	info.memoryFlags		= memProperties;
	info.stagingMemoryFlags = stagingProperties;

	_buffers[handle]		= info;

	return handle;
}

void VulkanBackend::UploadBufferInternal(VkBuffer buffer, size_t size, const void *data) {
	// Create staging buffer
	VkBufferCreateInfo stagingInfo = {};
	stagingInfo.sType			   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingInfo.size			   = size;
	stagingInfo.usage			   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	stagingInfo.sharingMode		   = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer stagingBuffer		   = nullptr;
	VkResult result				   = vkCreateBuffer(_device, &stagingInfo, nullptr, &stagingBuffer);
	CheckVkResult(result, "vkCreateBuffer (staging)");

	// Allocate staging memory (host visible)
	VkDeviceMemory stagingMemory = nullptr;
	AllocateBufferMemory(stagingBuffer, stagingMemory,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, size);

	result = vkBindBufferMemory(_device, stagingBuffer, stagingMemory, 0);
	CheckVkResult(result, "vkBindBufferMemory (staging)");

	// Map and copy data
	void *mapped = nullptr;
	result		 = vkMapMemory(_device, stagingMemory, 0, size, 0, &mapped);
	CheckVkResult(result, "vkMapMemory");
	std::memcpy(mapped, data, size);
	vkUnmapMemory(_device, stagingMemory);

	// Ensure command buffer is recording
	EnsureCommandBuffer();

	// Copy from staging to device buffer
	VkBufferCopy copyRegion = {};
	copyRegion.srcOffset	= 0;
	copyRegion.dstOffset	= 0;
	copyRegion.size			= size;

	vkCmdCopyBuffer(_commandBuffer, stagingBuffer, buffer, 1, &copyRegion);

	VkBufferMemoryBarrier barrier = {};
	barrier.sType				   = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	barrier.srcAccessMask		   = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask		   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
									 VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
	barrier.srcQueueFamilyIndex	   = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex	   = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer				   = buffer;
	barrier.offset				   = 0;
	barrier.size				   = size;

	vkCmdPipelineBarrier(_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
						 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
							 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
						 0, 0, nullptr, 1, &barrier, 0, nullptr);

	// Submit and wait for upload to complete
	EndCommandBuffer();
	SubmitCommandBuffer(true);

	// Cleanup staging resources
	vkDestroyBuffer(_device, stagingBuffer, nullptr);
	vkFreeMemory(_device, stagingMemory, nullptr);
}

void VulkanBackend::DestroyBuffer(BufferHandle buffer) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		return;
	}

	EnsureNoPendingGpuWork();
	InvalidateDescriptorCachesForBuffer(buffer);

	if (it->second.isMapped && it->second.memory) {
		vkUnmapMemory(_device, it->second.stagingMemory);
	}
	if (it->second.buffer) {
		vkDestroyBuffer(_device, it->second.buffer, nullptr);
	}
	if (it->second.stagingBuffer) {
		vkDestroyBuffer(_device, it->second.stagingBuffer, nullptr);
	}
	if (it->second.memory) {
		vkFreeMemory(_device, it->second.memory, nullptr);
	}
	if (it->second.stagingMemory) {
		vkFreeMemory(_device, it->second.stagingMemory, nullptr);
	}

	_buffers.erase(it);
}

void VulkanBackend::UploadBuffer(BufferHandle buffer, size_t offset, size_t size, const void *data) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		throw std::runtime_error("Invalid buffer handle");
	}
	if (data == nullptr && size != 0) {
		throw std::runtime_error("UploadBuffer received null data");
	}
	if (offset > it->second.size || size > it->second.size - offset) {
		throw std::runtime_error("UploadBuffer range exceeds buffer size");
	}
	EnsureNoPendingGpuWork();

	VkResult result = VK_SUCCESS;
	void	*mapped = nullptr;
	result			= vkMapMemory(_device, it->second.stagingMemory, offset, size, 0, &mapped);
	CheckVkResult(result, "vkMapMemory");
	std::memcpy(mapped, data, size);
	vkUnmapMemory(_device, it->second.stagingMemory);

	EnsureCommandBuffer();

	VkBufferCopy copyRegion = {};
	copyRegion.srcOffset	= offset;
	copyRegion.dstOffset	= offset;
	copyRegion.size			= size;

	vkCmdCopyBuffer(_commandBuffer, it->second.stagingBuffer, it->second.buffer, 1, &copyRegion);

	// Insert barrier to ensure upload is complete before use
	VkBufferMemoryBarrier barrier = {};
	barrier.sType				  = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	barrier.srcAccessMask		  = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask		  = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	barrier.srcQueueFamilyIndex	  = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex	  = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer				  = it->second.buffer;
	barrier.offset				  = offset;
	barrier.size				  = size;

	vkCmdPipelineBarrier(_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
						 nullptr, 1, &barrier, 0, nullptr);

	EndCommandBuffer();
	SubmitCommandBuffer(true);
}

void VulkanBackend::DownloadBuffer(BufferHandle buffer, size_t offset, size_t size, void *outData) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		throw std::runtime_error("Invalid buffer handle");
	}
	if (outData == nullptr && size != 0) {
		throw std::runtime_error("DownloadBuffer received null output pointer");
	}
	if (offset > it->second.size || size > it->second.size - offset) {
		throw std::runtime_error("DownloadBuffer range exceeds buffer size");
	}

	EnsureNoPendingGpuWork();
	EnsureCommandBuffer();

	VkBufferCopy copyRegion = {};
	copyRegion.srcOffset	= offset;
	copyRegion.dstOffset	= offset;
	copyRegion.size			= size;

	vkCmdCopyBuffer(_commandBuffer, it->second.buffer, it->second.stagingBuffer, 1, &copyRegion);

	EndCommandBuffer();
	SubmitCommandBuffer(true);

	VkResult result = VK_SUCCESS;
	void	*mapped = nullptr;
	result			= vkMapMemory(_device, it->second.stagingMemory, offset, size, 0, &mapped);
	CheckVkResult(result, "vkMapMemory");
	std::memcpy(outData, mapped, size);
	vkUnmapMemory(_device, it->second.stagingMemory);
}

void VulkanBackend::CopyBuffer(BufferHandle source, size_t sourceOffset, BufferHandle destination,
							   size_t destinationOffset, size_t size) {
	std::lock_guard<std::mutex> lock(_mutex);
	auto sourceIt = _buffers.find(source);
	auto destinationIt = _buffers.find(destination);
	if (sourceIt == _buffers.end() || destinationIt == _buffers.end()) {
		throw std::runtime_error("Invalid buffer handle");
	}
	if (sourceOffset > sourceIt->second.size || size > sourceIt->second.size - sourceOffset ||
		destinationOffset > destinationIt->second.size || size > destinationIt->second.size - destinationOffset) {
		throw std::runtime_error("CopyBuffer range exceeds buffer size");
	}
	if (source == destination && sourceOffset < destinationOffset + size && destinationOffset < sourceOffset + size) {
		throw std::runtime_error("CopyBuffer does not support overlapping ranges in the same buffer");
	}
	if (size == 0) {
		return;
	}

	EnsureCommandBuffer();
	VkBufferCopy copyRegion = {};
	copyRegion.srcOffset = sourceOffset;
	copyRegion.dstOffset = destinationOffset;
	copyRegion.size = size;
	vkCmdCopyBuffer(_commandBuffer, sourceIt->second.buffer, destinationIt->second.buffer, 1, &copyRegion);

	VkBufferMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer = destinationIt->second.buffer;
	barrier.offset = destinationOffset;
	barrier.size = size;
	vkCmdPipelineBarrier(_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
		0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void *VulkanBackend::MapBuffer(BufferHandle buffer, bool read, bool write) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		return nullptr;
	}
	if (!read && !write) {
		throw std::runtime_error("MapBuffer requires read and/or write access");
	}

	if (it->second.isMapped) {
		return it->second.mappedPtr;
	}

	if ((it->second.stagingMemoryFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
		throw std::runtime_error("MapBuffer requires HOST_VISIBLE staging memory in Vulkan backend");
	}
	EnsureNoPendingGpuWork();

	if (read) {
		EnsureCommandBuffer();

		VkBufferCopy copyRegion = {};
		copyRegion.srcOffset	= 0;
		copyRegion.dstOffset	= 0;
		copyRegion.size			= it->second.size;

		vkCmdCopyBuffer(_commandBuffer, it->second.buffer, it->second.stagingBuffer, 1, &copyRegion);
		EndCommandBuffer();
		SubmitCommandBuffer(true);
	}

	VkResult result = vkMapMemory(_device, it->second.stagingMemory, 0, it->second.size, 0, &it->second.mappedPtr);
	if (result != VK_SUCCESS) {
		return nullptr;
	}

	it->second.isMapped		  = true;
	it->second.mappedForRead  = read;
	it->second.mappedForWrite = write;
	return it->second.mappedPtr;
}

void VulkanBackend::UnmapBuffer(BufferHandle buffer) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		return;
	}

	if (it->second.isMapped) {
		vkUnmapMemory(_device, it->second.stagingMemory);

		if (it->second.mappedForWrite) {
			EnsureCommandBuffer();

			VkBufferCopy copyRegion = {};
			copyRegion.srcOffset	= 0;
			copyRegion.dstOffset	= 0;
			copyRegion.size			= it->second.size;

			vkCmdCopyBuffer(_commandBuffer, it->second.stagingBuffer, it->second.buffer, 1, &copyRegion);

			VkBufferMemoryBarrier barrier = {};
			barrier.sType				  = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			barrier.srcAccessMask		  = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask		  = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			barrier.srcQueueFamilyIndex	  = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex	  = VK_QUEUE_FAMILY_IGNORED;
			barrier.buffer				  = it->second.buffer;
			barrier.offset				  = 0;
			barrier.size				  = it->second.size;

			vkCmdPipelineBarrier(_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								 0, 0, nullptr, 1, &barrier, 0, nullptr);

			EndCommandBuffer();
			SubmitCommandBuffer(true);
		}

		it->second.isMapped		  = false;
		it->second.mappedPtr	  = nullptr;
		it->second.mappedForRead  = false;
		it->second.mappedForWrite = false;
	}
}

// =============================================================================
// Memory Management
// =============================================================================

uint32_t VulkanBackend::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("Failed to find suitable memory type");
}

void VulkanBackend::AllocateBufferMemory(VkBuffer buffer, VkDeviceMemory &memory, VkMemoryPropertyFlags properties,
										 size_t size) {
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(_device, buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType				   = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize	   = memRequirements.size;
	allocInfo.memoryTypeIndex	   = FindMemoryType(memRequirements.memoryTypeBits, properties);

	VkResult result				   = vkAllocateMemory(_device, &allocInfo, nullptr, &memory);
	CheckVkResult(result, "vkAllocateMemory");
}

void VulkanBackend::AllocateImageMemory(VkImage image, VkDeviceMemory &memory, VkMemoryPropertyFlags properties) {
	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(_device, image, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType				   = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize	   = memRequirements.size;
	allocInfo.memoryTypeIndex	   = FindMemoryType(memRequirements.memoryTypeBits, properties);

	VkResult result				   = vkAllocateMemory(_device, &allocInfo, nullptr, &memory);
	CheckVkResult(result, "vkAllocateMemory (image)");
}

// =============================================================================
// Texture Management
// =============================================================================

TextureHandle VulkanBackend::CreateTexture(const TextureDesc &desc) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
	}

	VkFormat format		  = GetVkFormat(desc.format);
	bool	 is3D		  = desc.depth > 1;
	uint32_t mipLevels	  = std::max(1u, desc.mipLevels);
	uint32_t maxMipLevels = 1;
	for (uint32_t size = std::max(desc.width, desc.height); size > 1; size /= 2)
		++maxMipLevels;
	if (mipLevels > maxMipLevels)
		throw std::invalid_argument("Texture mip level count exceeds its dimensions");
	if (is3D && mipLevels > 1)
		throw std::runtime_error("VulkanBackend mipmap generation currently supports 2D textures only");

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType				= VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType			= is3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
	imageInfo.extent.width		= desc.width;
	imageInfo.extent.height		= desc.height;
	imageInfo.extent.depth		= desc.depth;
	imageInfo.mipLevels			= mipLevels;
	imageInfo.arrayLayers		= 1;
	imageInfo.format			= format;
	imageInfo.tiling			= VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout		= VK_IMAGE_LAYOUT_UNDEFINED;
	VkImageUsageFlags usage = 0;
	if (desc.usage == TextureUsageNone) {
		usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
				VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		if (!is3D) {
			usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		}
	} else {
		if ((desc.usage & TextureUsageStorage) != 0) {
			usage |= VK_IMAGE_USAGE_STORAGE_BIT;
		}
		if ((desc.usage & TextureUsageSampled) != 0) {
			usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
		}
		if ((desc.usage & TextureUsageTransferSrc) != 0) {
			usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		}
		if ((desc.usage & TextureUsageTransferDst) != 0) {
			usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		}
		if ((desc.usage & TextureUsageColorAttachment) != 0) {
			if (is3D) {
				throw std::runtime_error("3D textures cannot be created as color attachments");
			}
			usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		}
		if ((desc.usage & TextureUsageDepthStencilAttachment) != 0) {
			if (is3D) {
				throw std::runtime_error("3D textures cannot be created as depth/stencil attachments");
			}
			if (format != VK_FORMAT_D32_SFLOAT && format != VK_FORMAT_D24_UNORM_S8_UINT) {
				throw std::runtime_error("Depth/stencil attachment usage requires D32F or D24S8 format");
			}
			usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		}
	}
	if (usage == 0) {
		throw std::runtime_error("TextureDesc must request at least one usage flag");
	}
	imageInfo.usage = usage;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples	  = VK_SAMPLE_COUNT_1_BIT;

	VkImage	 image		  = nullptr;
	VkResult result		  = vkCreateImage(_device, &imageInfo, nullptr, &image);
	CheckVkResult(result, "vkCreateImage");

	VkDeviceMemory memory = nullptr;
	AllocateImageMemory(image, memory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	result = vkBindImageMemory(_device, image, memory, 0);
	CheckVkResult(result, "vkBindImageMemory");

	// Create image view
	VkImageViewCreateInfo viewInfo			 = {};
	viewInfo.sType							 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image							 = image;
	viewInfo.viewType						 = is3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format							 = format;
	viewInfo.subresourceRange.aspectMask	 =
		format == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT :
		format == VK_FORMAT_D24_UNORM_S8_UINT ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) :
		VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel	 = 0;
	viewInfo.subresourceRange.levelCount	 = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount	 = 1;

	VkImageView view						 = nullptr;
	result									 = vkCreateImageView(_device, &viewInfo, nullptr, &view);
	CheckVkResult(result, "vkCreateImageView");

	VkImageView sampledView = nullptr;
	if (mipLevels > 1) {
		viewInfo.subresourceRange.levelCount = mipLevels;
		result								 = vkCreateImageView(_device, &viewInfo, nullptr, &sampledView);
		CheckVkResult(result, "vkCreateImageView (sampled mip chain)");
	}

	TextureHandle handle = _nextTextureHandle++;
	TextureInfo	  info;
	info.image		   = image;
	info.memory		   = memory;
	info.view		   = view;
	info.sampledView   = sampledView;
	info.width		   = desc.width;
	info.height		   = desc.height;
	info.depth		   = desc.depth;
	info.mipLevels	   = mipLevels;
	info.format		   = desc.format;
	info.vkFormat	   = format;
	info.samples	   = VK_SAMPLE_COUNT_1_BIT;
	info.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	_textures[handle]  = info;

	// Upload initial data if provided
	if (desc.initialData) {
		UploadTextureInternal(_textures[handle], 0, 0, 0, desc.width, desc.height, desc.depth, desc.initialData);
	}

	return handle;
}

VulkanBackend::NativeTextureInfo VulkanBackend::GetNativeTextureInfo(TextureHandle texture) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}

	const auto &info = it->second;
	return NativeTextureInfo{.image	 = info.image,
							 .format = info.vkFormat,
							 .layout = info.currentLayout,
							 .width	 = info.width,
							 .height = info.height,
							 .depth	 = info.depth};
}

void VulkanBackend::SetNativeTextureLayout(TextureHandle texture, VkImageLayout layout, VkPipelineStageFlags stage,
										   VkAccessFlags access) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}
	it->second.currentLayout = layout;
	it->second.lastStage	 = stage;
	it->second.lastAccess	 = access;
}

void VulkanBackend::DestroyTexture(TextureHandle texture) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _textures.find(texture);
	if (it == _textures.end()) {
		return;
	}

	EnsureNoPendingGpuWork();
	InvalidateDescriptorCachesForTexture(texture);

	for (auto attachmentIt = _msaaAttachments.begin(); attachmentIt != _msaaAttachments.end();) {
		if (attachmentIt->resolveTarget == texture) {
			if (attachmentIt->view)
				vkDestroyImageView(_device, attachmentIt->view, nullptr);
			if (attachmentIt->image)
				vkDestroyImage(_device, attachmentIt->image, nullptr);
			if (attachmentIt->memory)
				vkFreeMemory(_device, attachmentIt->memory, nullptr);
			attachmentIt = _msaaAttachments.erase(attachmentIt);
		} else {
			++attachmentIt;
		}
	}

	if (it->second.sampledView)
		vkDestroyImageView(_device, it->second.sampledView, nullptr);
	if (it->second.view)
		vkDestroyImageView(_device, it->second.view, nullptr);
	if (it->second.image)
		vkDestroyImage(_device, it->second.image, nullptr);
	if (it->second.memory)
		vkFreeMemory(_device, it->second.memory, nullptr);

	_textures.erase(it);
}

void VulkanBackend::UploadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
								  const void *data) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}

	UploadTextureInternal(it->second, x, y, 0, width, height, 1, data);
}

void VulkanBackend::UploadTextureFromBuffer(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width,
											uint32_t height, BufferHandle source, size_t sourceOffset) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						texIt = _textures.find(texture);
	if (texIt == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}
	auto bufIt = _buffers.find(source);
	if (bufIt == _buffers.end()) {
		throw std::runtime_error("Invalid source buffer handle");
	}
	size_t dataSize = static_cast<size_t>(width) * height * PixelFormatByteSize(texIt->second.format);
	if (sourceOffset + dataSize > bufIt->second.size) {
		throw std::runtime_error("UploadTextureFromBuffer range exceeds source buffer size");
	}

	CopyBufferToTexture(texIt->second, bufIt->second.buffer, sourceOffset, x, y, 0, width, height, 1);
}

void VulkanBackend::GenerateMipmaps(TextureHandle texture) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _textures.find(texture);
	if (it == _textures.end())
		throw std::runtime_error("Invalid texture handle");
	TextureInfo &info = it->second;
	if (info.mipLevels <= 1)
		return;
	if (info.depth > 1)
		throw std::runtime_error("VulkanBackend mipmap generation currently supports 2D textures only");

	VkFormatProperties properties{};
	vkGetPhysicalDeviceFormatProperties(_physicalDevice, info.vkFormat, &properties);
	if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0)
		throw std::runtime_error("Texture format does not support linear blit mipmap generation");

	EnsureCommandBuffer();
	TransitionTexture(info, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
					  VK_ACCESS_TRANSFER_WRITE_BIT);

	int32_t mipWidth  = static_cast<int32_t>(info.width);
	int32_t mipHeight = static_cast<int32_t>(info.height);
	for (uint32_t level = 1; level < info.mipLevels; ++level) {
		VkImageMemoryBarrier barrier{};
		barrier.sType							= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image							= info.image;
		barrier.srcQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask		= VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel	= level - 1;
		barrier.subresourceRange.levelCount		= 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount		= 1;
		barrier.oldLayout						= VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout						= VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcAccessMask					= VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask					= VK_ACCESS_TRANSFER_READ_BIT;
		vkCmdPipelineBarrier(_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
							 nullptr, 0, nullptr, 1, &barrier);

		int32_t		nextWidth  = std::max(1, mipWidth / 2);
		int32_t		nextHeight = std::max(1, mipHeight / 2);
		VkImageBlit blit{};
		blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
		blit.srcOffsets[1]	= {mipWidth, mipHeight, 1};
		blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
		blit.dstOffsets[1]	= {nextWidth, nextHeight, 1};
		vkCmdBlitImage(_commandBuffer, info.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, info.image,
					   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

		barrier.oldLayout	  = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout	  = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
							 nullptr, 0, nullptr, 1, &barrier);
		mipWidth  = nextWidth;
		mipHeight = nextHeight;
	}

	info.currentLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	info.lastStage	   = VK_PIPELINE_STAGE_TRANSFER_BIT;
	info.lastAccess	   = VK_ACCESS_TRANSFER_WRITE_BIT;
	TransitionTexture(info, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					  VK_ACCESS_SHADER_READ_BIT);
	EndCommandBuffer();
	SubmitCommandBuffer(true);
}

void VulkanBackend::UploadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
									uint32_t height, uint32_t depth, const void *data) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}

	UploadTextureInternal(it->second, x, y, z, width, height, depth, data);
}

void VulkanBackend::UploadTexture3DFromBuffer(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
											  uint32_t height, uint32_t depth, BufferHandle source,
											  size_t sourceOffset) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						texIt = _textures.find(texture);
	if (texIt == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}
	auto bufIt = _buffers.find(source);
	if (bufIt == _buffers.end()) {
		throw std::runtime_error("Invalid source buffer handle");
	}
	size_t dataSize = static_cast<size_t>(width) * height * depth * PixelFormatByteSize(texIt->second.format);
	if (sourceOffset + dataSize > bufIt->second.size) {
		throw std::runtime_error("UploadTexture3DFromBuffer range exceeds source buffer size");
	}

	CopyBufferToTexture(texIt->second, bufIt->second.buffer, sourceOffset, x, y, z, width, height, depth);
}

void VulkanBackend::UploadTextureInternal(TextureInfo &info, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
										  uint32_t height, uint32_t depth, const void *data) {
	if (data == nullptr && (width != 0 || height != 0 || depth != 0)) {
		throw std::runtime_error("UploadTexture received null data");
	}
	if (x + width > info.width || y + height > info.height || z + depth > info.depth) {
		throw std::runtime_error("UploadTexture region exceeds texture bounds");
	}
	if (width == 0 || height == 0 || depth == 0) {
		return;
	}

	size_t			   dataSize	   = static_cast<size_t>(width) * height * depth * PixelFormatByteSize(info.format);

	// Create staging buffer
	VkBufferCreateInfo stagingInfo = {};
	stagingInfo.sType			   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingInfo.size			   = dataSize;
	stagingInfo.usage			   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	stagingInfo.sharingMode		   = VK_SHARING_MODE_EXCLUSIVE;

	ScopedVulkanBuffer staging(_device);
	VkResult		   result = vkCreateBuffer(_device, &stagingInfo, nullptr, &staging.buffer);
	CheckVkResult(result, "vkCreateBuffer (staging)");

	AllocateBufferMemory(staging.buffer, staging.memory,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, dataSize);

	result = vkBindBufferMemory(_device, staging.buffer, staging.memory, 0);
	CheckVkResult(result, "vkBindBufferMemory (staging)");

	// Copy data to staging buffer
	void *mapped = nullptr;
	result		 = vkMapMemory(_device, staging.memory, 0, dataSize, 0, &mapped);
	CheckVkResult(result, "vkMapMemory");
	std::memcpy(mapped, data, dataSize);
	vkUnmapMemory(_device, staging.memory);

	CopyBufferToTexture(info, staging.buffer, 0, x, y, z, width, height, depth);
}

void VulkanBackend::CopyBufferToTexture(TextureInfo &info, VkBuffer sourceBuffer, size_t sourceOffset, uint32_t x,
										uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth) {
	if (x + width > info.width || y + height > info.height || z + depth > info.depth) {
		throw std::runtime_error("UploadTexture region exceeds texture bounds");
	}
	if (width == 0 || height == 0 || depth == 0) {
		return;
	}
	EnsureCommandBuffer();

	TransitionTexture(info, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
					  VK_ACCESS_TRANSFER_WRITE_BIT);

	// Copy buffer to image
	VkBufferImageCopy region			   = {};
	region.bufferOffset					   = sourceOffset;
	region.bufferRowLength				   = 0;
	region.bufferImageHeight			   = 0;
	region.imageSubresource.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel	   = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount	   = 1;
	region.imageOffset = {static_cast<int32_t>(x), static_cast<int32_t>(y), static_cast<int32_t>(z)};
	region.imageExtent = {width, height, depth};

	vkCmdCopyBufferToImage(_commandBuffer, sourceBuffer, info.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	TransitionTexture(info, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

	EndCommandBuffer();
	SubmitCommandBuffer(true);
}

void VulkanBackend::DownloadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
									void *outData) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}

	DownloadTextureInternal(it->second, x, y, 0, width, height, 1, outData);
}

void VulkanBackend::DownloadTextureToBuffer(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width,
											uint32_t height, BufferHandle destination, size_t destinationOffset) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						texIt = _textures.find(texture);
	if (texIt == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}
	auto bufIt = _buffers.find(destination);
	if (bufIt == _buffers.end()) {
		throw std::runtime_error("Invalid destination buffer handle");
	}
	size_t dataSize = static_cast<size_t>(width) * height * PixelFormatByteSize(texIt->second.format);
	if (destinationOffset + dataSize > bufIt->second.size) {
		throw std::runtime_error("DownloadTextureToBuffer range exceeds destination buffer size");
	}

	CopyTextureToBuffer(texIt->second, bufIt->second.buffer, destinationOffset, x, y, 0, width, height, 1);
}

void VulkanBackend::DownloadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
									  uint32_t height, uint32_t depth, void *outData) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}

	DownloadTextureInternal(it->second, x, y, z, width, height, depth, outData);
}

void VulkanBackend::DownloadTexture3DToBuffer(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
											  uint32_t height, uint32_t depth, BufferHandle destination,
											  size_t destinationOffset) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						texIt = _textures.find(texture);
	if (texIt == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}
	auto bufIt = _buffers.find(destination);
	if (bufIt == _buffers.end()) {
		throw std::runtime_error("Invalid destination buffer handle");
	}
	size_t dataSize = static_cast<size_t>(width) * height * depth * PixelFormatByteSize(texIt->second.format);
	if (destinationOffset + dataSize > bufIt->second.size) {
		throw std::runtime_error("DownloadTexture3DToBuffer range exceeds destination buffer size");
	}

	CopyTextureToBuffer(texIt->second, bufIt->second.buffer, destinationOffset, x, y, z, width, height, depth);
}

void VulkanBackend::DownloadTextureInternal(TextureInfo &info, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
											uint32_t height, uint32_t depth, void *outData) {
	if (outData == nullptr && (width != 0 || height != 0 || depth != 0)) {
		throw std::runtime_error("DownloadTexture received null output pointer");
	}
	if (x + width > info.width || y + height > info.height || z + depth > info.depth) {
		throw std::runtime_error("DownloadTexture region exceeds texture bounds");
	}
	if (width == 0 || height == 0 || depth == 0) {
		return;
	}

	size_t			   dataSize	   = static_cast<size_t>(width) * height * depth * PixelFormatByteSize(info.format);

	// Create staging buffer
	VkBufferCreateInfo stagingInfo = {};
	stagingInfo.sType			   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingInfo.size			   = dataSize;
	stagingInfo.usage			   = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	stagingInfo.sharingMode		   = VK_SHARING_MODE_EXCLUSIVE;

	ScopedVulkanBuffer staging(_device);
	VkResult		   result = vkCreateBuffer(_device, &stagingInfo, nullptr, &staging.buffer);
	CheckVkResult(result, "vkCreateBuffer (staging)");

	AllocateBufferMemory(staging.buffer, staging.memory,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, dataSize);

	result = vkBindBufferMemory(_device, staging.buffer, staging.memory, 0);
	CheckVkResult(result, "vkBindBufferMemory (staging)");

	CopyTextureToBuffer(info, staging.buffer, 0, x, y, z, width, height, depth);

	// Map and copy data
	void *mapped = nullptr;
	result		 = vkMapMemory(_device, staging.memory, 0, dataSize, 0, &mapped);
	CheckVkResult(result, "vkMapMemory");
	std::memcpy(outData, mapped, dataSize);
	vkUnmapMemory(_device, staging.memory);
}

void VulkanBackend::CopyTextureToBuffer(TextureInfo &info, VkBuffer destinationBuffer, size_t destinationOffset,
										uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height,
										uint32_t depth) {
	if (x + width > info.width || y + height > info.height || z + depth > info.depth) {
		throw std::runtime_error("DownloadTexture region exceeds texture bounds");
	}
	if (width == 0 || height == 0 || depth == 0) {
		return;
	}

	EnsureCommandBuffer();

	TransitionTexture(info, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
					  VK_ACCESS_TRANSFER_READ_BIT);

	// Copy image to buffer
	VkBufferImageCopy region			   = {};
	region.bufferOffset					   = destinationOffset;
	region.bufferRowLength				   = 0;
	region.bufferImageHeight			   = 0;
	region.imageSubresource.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel	   = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount	   = 1;
	region.imageOffset = {static_cast<int32_t>(x), static_cast<int32_t>(y), static_cast<int32_t>(z)};
	region.imageExtent = {width, height, depth};

	vkCmdCopyImageToBuffer(_commandBuffer, info.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destinationBuffer, 1,
						   &region);

	TransitionTexture(info, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

	EndCommandBuffer();
	SubmitCommandBuffer(true);
}

// =============================================================================
// Format Conversions
// =============================================================================

VkFormat VulkanBackend::GetVkFormat(PixelFormat format) {
	switch (format) {
	case PixelFormat::R8:
		return VK_FORMAT_R8_UNORM;
	case PixelFormat::RG8:
		return VK_FORMAT_R8G8_UNORM;
	case PixelFormat::RGBA8:
		return VK_FORMAT_R8G8B8A8_UNORM;
	case PixelFormat::R32F:
		return VK_FORMAT_R32_SFLOAT;
	case PixelFormat::RG32F:
		return VK_FORMAT_R32G32_SFLOAT;
	case PixelFormat::RGB32F:
		return VK_FORMAT_R32G32B32_SFLOAT;
	case PixelFormat::RGBA32F:
		return VK_FORMAT_R32G32B32A32_SFLOAT;
	case PixelFormat::R16F:
		return VK_FORMAT_R16_SFLOAT;
	case PixelFormat::RG16F:
		return VK_FORMAT_R16G16_SFLOAT;
	case PixelFormat::RGBA16F:
		return VK_FORMAT_R16G16B16A16_SFLOAT;
	case PixelFormat::R32I:
		return VK_FORMAT_R32_SINT;
	case PixelFormat::RG32I:
		return VK_FORMAT_R32G32_SINT;
	case PixelFormat::RGB32I:
		return VK_FORMAT_R32G32B32_SINT;
	case PixelFormat::RGBA32I:
		return VK_FORMAT_R32G32B32A32_SINT;
	case PixelFormat::R32UI:
		return VK_FORMAT_R32_UINT;
	case PixelFormat::RG32UI:
		return VK_FORMAT_R32G32_UINT;
	case PixelFormat::RGB32UI:
		return VK_FORMAT_R32G32B32_UINT;
	case PixelFormat::RGBA32UI:
		return VK_FORMAT_R32G32B32A32_UINT;
	case PixelFormat::D32F:
		return VK_FORMAT_D32_SFLOAT;
	case PixelFormat::D24S8:
		return VK_FORMAT_D24_UNORM_S8_UINT;
	default:
		return VK_FORMAT_R8G8B8A8_UNORM;
	}
}

VkSampleCountFlagBits VulkanBackend::GetVkSampleCount(SampleCount sampleCount) {
	switch (sampleCount) {
	case SampleCount::X1:
		return VK_SAMPLE_COUNT_1_BIT;
	case SampleCount::X2:
		return VK_SAMPLE_COUNT_2_BIT;
	case SampleCount::X4:
		return VK_SAMPLE_COUNT_4_BIT;
	case SampleCount::X8:
		return VK_SAMPLE_COUNT_8_BIT;
	case SampleCount::X16:
		return VK_SAMPLE_COUNT_16_BIT;
	default:
		return VK_SAMPLE_COUNT_1_BIT;
	}
}

VkDescriptorType VulkanBackend::GetVkDescriptorType(BindingType type) {
	switch (type) {
	case BindingType::Buffer:
		return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	case BindingType::Texture:
		return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	case BindingType::Sampler:
		return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	default:
		return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	}
}

VkImageLayout VulkanBackend::GetVkImageLayout(PixelFormat format, bool readOnly) {
	if (readOnly) {
		return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}
	return VK_IMAGE_LAYOUT_GENERAL;
}

VkShaderStageFlags VulkanBackend::GetVkShaderStage(ShaderType type) {
	switch (type) {
	case ShaderType::Compute:
		return VK_SHADER_STAGE_COMPUTE_BIT;
	case ShaderType::Vertex:
		return VK_SHADER_STAGE_VERTEX_BIT;
	case ShaderType::Fragment:
		return VK_SHADER_STAGE_FRAGMENT_BIT;
	default:
		return VK_SHADER_STAGE_COMPUTE_BIT;
	}
}

VkShaderStageFlags VulkanBackend::GetVkResourceStages(uint32_t stageFlags, bool graphicsPipeline) {
	if (stageFlags == ResourceStageNone) {
		return graphicsPipeline ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
								: VK_SHADER_STAGE_COMPUTE_BIT;
	}

	VkShaderStageFlags stages = 0;
	if ((stageFlags & ResourceStageCompute) != 0) {
		stages |= VK_SHADER_STAGE_COMPUTE_BIT;
	}
	if ((stageFlags & ResourceStageVertex) != 0) {
		stages |= VK_SHADER_STAGE_VERTEX_BIT;
	}
	if ((stageFlags & ResourceStageFragment) != 0) {
		stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
	}
	if (stages == 0) {
		return graphicsPipeline ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
								: VK_SHADER_STAGE_COMPUTE_BIT;
	}
	return stages;
}

// =============================================================================
// SPIR-V Compilation using glslang
// =============================================================================

std::vector<uint32_t> VulkanBackend::OptimizeSPIRV(const std::vector<uint32_t> &spirv,
												   ShaderOptimizationLevel optimizationLevel,
												   bool					 preserveInterface) {
	if (optimizationLevel == ShaderOptimizationLevel::None) {
		return spirv;
	}

#ifdef EASYGPU_SPIRV_OPT_ENABLED
	spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_1);
	switch (optimizationLevel) {
	case ShaderOptimizationLevel::Aggressive:
		optimizer.RegisterPerformancePasses(preserveInterface);
		break;
	case ShaderOptimizationLevel::Size:
		optimizer.RegisterSizePasses(preserveInterface);
		break;
	case ShaderOptimizationLevel::Ultra:
	case ShaderOptimizationLevel::Extreme: {
		// Track SPIRV-Tools' maintained -O recipe first, then add a small
		// target-independent tail. This avoids copying a recipe that changes
		// between SPIRV-Tools releases.
		optimizer.RegisterPerformancePasses(preserveInterface);
		optimizer.RegisterPass(spvtools::CreateLoopInvariantCodeMotionPass());
		optimizer.RegisterPass(spvtools::CreateStrengthReductionPass());
		optimizer.RegisterPass(spvtools::CreateLocalRedundancyEliminationPass());
		optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
		optimizer.RegisterPass(spvtools::CreateCodeSinkingPass());

		if (optimizationLevel == ShaderOptimizationLevel::Extreme) {
			// These transformations can increase code size/register pressure or
			// reduce precision. Keep them out of the production Ultra preset.
			optimizer.RegisterPass(spvtools::CreateLoopUnswitchPass());
			optimizer.RegisterPass(spvtools::CreateLoopPeelingPass());
			optimizer.RegisterPass(spvtools::CreateLoopFissionPass(64));
			optimizer.RegisterPass(spvtools::CreateLoopFusionPass(64));
			optimizer.RegisterPass(spvtools::CreateConvertRelaxedToHalfPass());
			optimizer.RegisterPass(spvtools::CreateFlattenDecorationPass());
			optimizer.RegisterPass(spvtools::CreateAmdExtToKhrPass());
		}

		optimizer.RegisterPass(spvtools::CreateCombineAccessChainsPass());
		optimizer.RegisterPass(spvtools::CreateSimplificationPass());
		optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
		optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass(preserveInterface));
		optimizer.RegisterPass(spvtools::CreateCFGCleanupPass());
		optimizer.RegisterPass(spvtools::CreateCompactIdsPass());
		break;
	}
	case ShaderOptimizationLevel::None:
		break;
	}

	std::vector<uint32_t> optimized;
	optimizer.SetMessageConsumer([](spv_message_level_t, const char *, const spv_position_t &position,
									const char *message) {
		std::cerr << "SPIRV-Tools optimizer: " << position.line << ":" << position.column << ": " << message << '\n';
	});

	if (!optimizer.Run(spirv.data(), spirv.size(), &optimized)) {
		throw std::runtime_error("SPIR-V optimization failed");
	}

	const auto &result = optimized.empty() ? spirv : optimized;
	spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_1);
	validator.SetMessageConsumer([](spv_message_level_t, const char *, const spv_position_t &position,
									const char *message) {
		std::cerr << "SPIRV-Tools validator: " << position.line << ":" << position.column << ": " << message << '\n';
	});
	if (!validator.Validate(result)) {
		throw std::runtime_error("SPIR-V validation failed after optimization");
	}

	return result;
#else
	(void)preserveInterface;
	return spirv;
#endif
}

std::string VulkanBackend::DecompileSPIRVToGLSL(const std::vector<uint32_t> &spirv, ShaderType type) {
#ifdef EASYGPU_SPIRV_CROSS_GLSL_ENABLED
	spirv_cross::CompilerGLSL		   compiler(spirv);
	spirv_cross::CompilerGLSL::Options options;
	options.version			 = type == ShaderType::Compute ? 430 : 450;
	options.es				 = false;
	options.vulkan_semantics = true;
	compiler.set_common_options(options);
	return compiler.compile();
#else
	(void)spirv;
	(void)type;
	throw std::runtime_error("SPIRV-Cross GLSL inspection is disabled");
#endif
}

std::optional<std::vector<uint32_t>> VulkanBackend::LoadMemoryCachedSpirv(const std::filesystem::path &path) {
	const auto key = path.string();
	auto it = _spirvMemoryCache.find(key);
	if (it == _spirvMemoryCache.end()) {
		return std::nullopt;
	}

	std::error_code error;
	const auto fileSize = std::filesystem::file_size(path, error);
	if (error) {
		_spirvMemoryCacheBytes -= it->second.spirv.size() * sizeof(uint32_t);
		_spirvMemoryCache.erase(it);
		return std::nullopt;
	}
	const auto lastWriteTime = std::filesystem::last_write_time(path, error);
	if (error || fileSize != it->second.fileSize || lastWriteTime != it->second.lastWriteTime) {
		_spirvMemoryCacheBytes -= it->second.spirv.size() * sizeof(uint32_t);
		_spirvMemoryCache.erase(it);
		return std::nullopt;
	}

	it->second.lastAccess = ++_spirvMemoryCacheAccess;
	return it->second.spirv;
}

void VulkanBackend::StoreMemoryCachedSpirv(const std::filesystem::path &path, const std::vector<uint32_t> &spirv) {
	const size_t byteCount = spirv.size() * sizeof(uint32_t);
	if (byteCount > MAX_CACHED_SPIRV_MEMORY_BYTES) {
		return;
	}

	std::error_code error;
	const auto fileSize = std::filesystem::file_size(path, error);
	if (error) {
		return;
	}
	const auto lastWriteTime = std::filesystem::last_write_time(path, error);
	if (error) {
		return;
	}

	const auto key = path.string();
	auto existing = _spirvMemoryCache.find(key);
	if (existing != _spirvMemoryCache.end()) {
		_spirvMemoryCacheBytes -= existing->second.spirv.size() * sizeof(uint32_t);
		_spirvMemoryCache.erase(existing);
	}
	while (!_spirvMemoryCache.empty() &&
		   (_spirvMemoryCache.size() >= MAX_CACHED_SPIRV_MODULES ||
			_spirvMemoryCacheBytes + byteCount > MAX_CACHED_SPIRV_MEMORY_BYTES)) {
		const auto oldest = std::min_element(
			_spirvMemoryCache.begin(), _spirvMemoryCache.end(),
			[](const auto &left, const auto &right) { return left.second.lastAccess < right.second.lastAccess; });
		if (oldest != _spirvMemoryCache.end()) {
			_spirvMemoryCacheBytes -= oldest->second.spirv.size() * sizeof(uint32_t);
			_spirvMemoryCache.erase(oldest);
		}
	}

	_spirvMemoryCache[key] = {spirv, fileSize, lastWriteTime, ++_spirvMemoryCacheAccess};
	_spirvMemoryCacheBytes += byteCount;
}

std::vector<uint32_t> VulkanBackend::CompileGLSLToSPIRV(const std::string &glslSource, ShaderType type,
												ShaderOptimizationLevel optimizationLevel,
												bool					preserveInterface) {
	_shaderCompilationStats.lastMemoryCacheHit = false;
	_shaderCompilationStats.lastDiskCacheHit = false;
	_shaderCompilationStats.lastFrontendMilliseconds = 0.0;
	_shaderCompilationStats.lastOptimizationMilliseconds = 0.0;

#ifdef EASYGPU_SHADER_CACHE_ENABLED
	std::optional<std::filesystem::path> cachePath;
	try {
		if (const auto cacheDirectory = GetSpirvCacheDirectory()) {
			cachePath = *cacheDirectory / (BuildSpirvCacheKey(glslSource, type, optimizationLevel, preserveInterface) + ".spv");
			if (auto cachedSpirv = LoadMemoryCachedSpirv(*cachePath)) {
				++_shaderCompilationStats.memoryCacheHits;
				_shaderCompilationStats.lastMemoryCacheHit = true;
				return std::move(*cachedSpirv);
			}
			if (auto cachedSpirv = LoadCachedSpirv(*cachePath)) {
				StoreMemoryCachedSpirv(*cachePath, *cachedSpirv);
				++_shaderCompilationStats.diskCacheHits;
				_shaderCompilationStats.lastDiskCacheHit = true;
				return std::move(*cachedSpirv);
			}
			++_shaderCompilationStats.diskCacheMisses;
		}
	} catch (...) {
		cachePath.reset();
	}
#endif

	++_shaderCompilationStats.frontendCompilations;
	const auto frontendStart = std::chrono::steady_clock::now();
	EShLanguage stage;
	switch (type) {
	case ShaderType::Compute:
		stage = EShLangCompute;
		break;
	case ShaderType::Vertex:
		stage = EShLangVertex;
		break;
	case ShaderType::Fragment:
		stage = EShLangFragment;
		break;
	default:
		stage = EShLangCompute;
	}

	glslang::TShader shader(stage);
	const char		*sourceCStr = glslSource.c_str();
	shader.setStrings(&sourceCStr, 1);

	// Set up shader options
	shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 430);
	shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
	shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);

	// Parse and compile
	TBuiltInResource resources = *GetDefaultResources();
	EShMessages		 messages  = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules | EShMsgDebugInfo);

	if (!shader.parse(&resources, 430, false, messages)) {
		std::string errorMsg  = "GLSL parsing failed:\n";
		errorMsg			 += shader.getInfoLog();
		errorMsg			 += "\n" + std::string(shader.getInfoDebugLog());
		throw std::runtime_error(errorMsg);
	}

	// Link into program
	glslang::TProgram program;
	program.addShader(&shader);

	if (!program.link(messages)) {
		std::string errorMsg  = "GLSL linking failed:\n";
		errorMsg			 += program.getInfoLog();
		errorMsg			 += "\n" + std::string(program.getInfoDebugLog());
		throw std::runtime_error(errorMsg);
	}

	// Generate SPIR-V
	std::vector<uint32_t> spirv;

	glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);

	if (spirv.empty()) {
		throw std::runtime_error("SPIR-V generation failed: empty output");
	}

	const auto frontendEnd = std::chrono::steady_clock::now();
	_shaderCompilationStats.lastFrontendMilliseconds =
		std::chrono::duration<double, std::milli>(frontendEnd - frontendStart).count();

	const auto optimizationStart = std::chrono::steady_clock::now();
	auto optimized = OptimizeSPIRV(spirv, optimizationLevel, preserveInterface);
	const auto optimizationEnd = std::chrono::steady_clock::now();
	_shaderCompilationStats.lastOptimizationMilliseconds =
		std::chrono::duration<double, std::milli>(optimizationEnd - optimizationStart).count();

#ifdef EASYGPU_SHADER_CACHE_ENABLED
	if (cachePath) {
		try {
			if (StoreCachedSpirv(*cachePath, optimized)) {
				StoreMemoryCachedSpirv(*cachePath, optimized);
			} else {
				++_shaderCompilationStats.diskCacheWriteFailures;
			}
		} catch (...) {
			++_shaderCompilationStats.diskCacheWriteFailures;
		}
	}
#endif

	return optimized;
}

// =============================================================================
// Shader Management
// =============================================================================

std::string VulkanBackend::GetOptimizedGLSL(const ShaderDesc &desc) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
	}

	return DecompileSPIRVToGLSL(
		CompileGLSLToSPIRV(desc.sourceCode, desc.type, desc.optimizationLevel, desc.preserveInterface), desc.type);
}

ShaderCompilationStats VulkanBackend::GetShaderCompilationStats() const {
	std::lock_guard<std::mutex> lock(_mutex);
	return _shaderCompilationStats;
}

void VulkanBackend::ResetShaderCompilationStats() {
	std::lock_guard<std::mutex> lock(_mutex);
	_shaderCompilationStats = {};
}

ShaderHandle VulkanBackend::CreateShader(const ShaderDesc &desc) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
	}

	// Compile GLSL to SPIR-V
	std::vector<uint32_t> spirv =
		CompileGLSLToSPIRV(desc.sourceCode, desc.type, desc.optimizationLevel, desc.preserveInterface);

	// Create shader module
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType					= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize					= spirv.size() * sizeof(uint32_t);
	createInfo.pCode					= spirv.data();

	VkShaderModule shaderModule			= nullptr;
	VkResult	   result				= vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule);
	if (result != VK_SUCCESS) {
		throw std::runtime_error(std::string("vkCreateShaderModule failed: ") + VkResultToString(result));
	}

	ShaderHandle handle = _nextShaderHandle++;
	ShaderInfo	 info;
	info.module		 = shaderModule;
	info.type		 = desc.type;
	info.spirvCode	 = std::move(spirv);

	_shaders[handle] = std::move(info);

	return handle;
}

void VulkanBackend::DestroyShader(ShaderHandle shader) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _shaders.find(shader);
	if (it == _shaders.end()) {
		return;
	}

	EnsureNoPendingGpuWork();

	if (it->second.module) {
		vkDestroyShaderModule(_device, it->second.module, nullptr);
	}

	_shaders.erase(it);
}

// =============================================================================
// Pipeline Management
// =============================================================================

PipelineHandle VulkanBackend::CreatePipeline(const PipelineDesc &desc) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
	}

	auto shaderIt = _shaders.find(desc.computeShader);
	if (shaderIt == _shaders.end()) {
		throw std::runtime_error("Invalid shader handle");
	}
	if (desc.pushConstantSize > _maxPushConstantSize) {
		throw std::runtime_error("Pipeline push constant size exceeds device limit");
	}

	std::vector<VkDescriptorSetLayoutBinding> bindings;

	std::vector<ResourceLayoutEntry>		  sortedResources = desc.resources;
	std::sort(sortedResources.begin(), sortedResources.end());
	for (size_t i = 1; i < sortedResources.size(); ++i) {
		if (sortedResources[i - 1].binding == sortedResources[i].binding &&
			!(sortedResources[i - 1] == sortedResources[i])) {
			throw std::runtime_error("Pipeline resource layout contains conflicting declarations for the same binding");
		}
	}
	sortedResources.erase(std::unique(sortedResources.begin(), sortedResources.end()), sortedResources.end());

	for (const auto &entry : sortedResources) {
		if (entry.type == BindingType::Buffer && entry.binding >= MAX_BUFFER_BINDINGS) {
			throw std::runtime_error("Vulkan pipeline buffer binding exceeds Vulkan backend cache limits");
		}
		if ((entry.type == BindingType::Texture || entry.type == BindingType::Sampler) &&
			entry.binding >= MAX_TEXTURE_BINDINGS) {
			throw std::runtime_error("Vulkan pipeline texture binding exceeds Vulkan backend cache limits");
		}
		VkDescriptorSetLayoutBinding binding = {};
		binding.binding						 = entry.binding;
		binding.descriptorType				 = GetVkDescriptorType(entry);
		binding.descriptorCount				 = 1;
		binding.stageFlags					 = GetVkResourceStages(entry.stageFlags, false);
		binding.pImmutableSamplers			 = nullptr;
		bindings.push_back(binding);
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType						   = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount					   = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings					   = bindings.data();

	VkDescriptorSetLayout descriptorSetLayout  = nullptr;
	VkResult			  result = vkCreateDescriptorSetLayout(_device, &layoutInfo, nullptr, &descriptorSetLayout);
	CheckVkResult(result, "vkCreateDescriptorSetLayout");

	// Create pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType					  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount			  = bindings.empty() ? 0u : 1u;
	pipelineLayoutInfo.pSetLayouts				  = bindings.empty() ? nullptr : &descriptorSetLayout;

	VkPushConstantRange pushConstantRange		  = {};
	if (desc.pushConstantSize != 0) {
		pushConstantRange.stageFlags			  = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset				  = 0;
		pushConstantRange.size					  = desc.pushConstantSize;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		pipelineLayoutInfo.pPushConstantRanges	  = &pushConstantRange;
	} else {
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges	  = nullptr;
	}

	VkPipelineLayout pipelineLayout = nullptr;
	result							= vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
	CheckVkResult(result, "vkCreatePipelineLayout");

	// Create compute pipeline
	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType							= VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage							= VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStage.module							= shaderIt->second.module;
	shaderStage.pName							= "main";

	VkComputePipelineCreateInfo pipelineInfo	= {};
	pipelineInfo.sType							= VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage							= shaderStage;
	pipelineInfo.layout							= pipelineLayout;

	VkPipeline pipeline							= nullptr;
	result = vkCreateComputePipelines(_device, _pipelineCache, 1, &pipelineInfo, nullptr, &pipeline);
	if (result != VK_SUCCESS) {
		vkDestroyPipelineLayout(_device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(_device, descriptorSetLayout, nullptr);
		throw std::runtime_error(std::string("vkCreateComputePipelines failed: ") + VkResultToString(result));
	}

	PipelineHandle handle = _nextPipelineHandle++;
	PipelineInfo   info;
	info.pipeline			 = pipeline;
	info.layout				 = pipelineLayout;
	info.descriptorSetLayout = descriptorSetLayout;
	info.workGroupSizeX		 = desc.workGroupSizeX;
	info.workGroupSizeY		 = desc.workGroupSizeY;
	info.workGroupSizeZ		 = desc.workGroupSizeZ;
	info.pushConstantSize	 = desc.pushConstantSize;
	info.resources			 = std::move(sortedResources);

	_pipelines[handle]		 = std::move(info);
	_pipelineCacheDirty	 = true;

	return handle;
}

void VulkanBackend::DestroyPipeline(PipelineHandle pipeline) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _pipelines.find(pipeline);
	if (it == _pipelines.end()) {
		return;
	}

	EnsureNoPendingGpuWork();
	InvalidateDescriptorCachesForPipeline(pipeline);

	if (it->second.pipeline)
		vkDestroyPipeline(_device, it->second.pipeline, nullptr);
	if (it->second.layout)
		vkDestroyPipelineLayout(_device, it->second.layout, nullptr);
	if (it->second.descriptorSetLayout)
		vkDestroyDescriptorSetLayout(_device, it->second.descriptorSetLayout, nullptr);
	if (_currentPipeline == pipeline) {
		_currentPipeline = INVALID_PIPELINE_HANDLE;
	}

	_pipelines.erase(it);
}

VkDescriptorType VulkanBackend::GetVkDescriptorType(const ResourceLayoutEntry &entry) {
	switch (entry.type) {
	case BindingType::Buffer:
		return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	case BindingType::Texture:
		return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	case BindingType::Sampler:
		return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	default:
		return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	}
}

VulkanBackend::SamplerKey VulkanBackend::MakeSamplerKey(const SamplerDesc &desc, bool hasMipmaps) const {
	SamplerKey key;
	key.minFilter = desc.minFilter;
	key.magFilter = desc.magFilter;
	key.mipmapMode = desc.mipmapMode;
	key.addressU = desc.addressU;
	key.addressV = desc.addressV;
	key.addressW = desc.addressW;
	key.mipLodBias = desc.mipLodBias;
	key.minLod = desc.minLod;
	key.maxLod = hasMipmaps ? desc.maxLod : 0.0f;
	key.anisotropyEnable = desc.anisotropyEnable && _samplerAnisotropySupported;
	key.maxAnisotropy = std::max(1.0f, desc.maxAnisotropy);
	if (_samplerAnisotropySupported) {
		key.maxAnisotropy = std::min(key.maxAnisotropy, _maxSamplerAnisotropy);
	}
	key.compareEnable = desc.compareEnable;
	key.compareOp = desc.compareOp;
	key.borderColor = desc.borderColor;
	return key;
}

VkSampler VulkanBackend::GetOrCreateSampler(const SamplerKey &key) {
	for (const auto &[cachedKey, sampler] : _samplerCache) {
		if (cachedKey == key) {
			return sampler;
		}
	}

	VkSamplerCreateInfo samplerInfo		= {};
	samplerInfo.sType					= VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter				= ToVkFilter(key.magFilter);
	samplerInfo.minFilter				= ToVkFilter(key.minFilter);
	samplerInfo.mipmapMode				= ToVkSamplerMipmapMode(key.mipmapMode);
	samplerInfo.addressModeU			= ToVkAddressMode(key.addressU);
	samplerInfo.addressModeV			= ToVkAddressMode(key.addressV);
	samplerInfo.addressModeW			= ToVkAddressMode(key.addressW);
	samplerInfo.mipLodBias				= key.mipLodBias;
	samplerInfo.anisotropyEnable		= key.anisotropyEnable ? VK_TRUE : VK_FALSE;
	samplerInfo.maxAnisotropy			= samplerInfo.anisotropyEnable == VK_TRUE
											  ? key.maxAnisotropy
											  : 1.0f;
	samplerInfo.compareEnable			= key.compareEnable ? VK_TRUE : VK_FALSE;
	samplerInfo.compareOp				= ToVkCompareOp(key.compareOp);
	samplerInfo.minLod					= key.minLod;
	samplerInfo.maxLod					= key.maxLod;
	samplerInfo.borderColor				= ToVkBorderColor(key.borderColor);
	samplerInfo.unnormalizedCoordinates = VK_FALSE;

	VkSampler sampler = nullptr;
	CheckVkResult(vkCreateSampler(_device, &samplerInfo, nullptr, &sampler), "vkCreateSampler (descriptor)");
	_samplerCache.emplace_back(key, sampler);
	return sampler;
}

void VulkanBackend::TransitionTexture(TextureInfo &info, VkImageLayout newLayout, VkPipelineStageFlags dstStage,
									  VkAccessFlags dstAccess) {
	if (info.currentLayout == newLayout && info.lastStage == dstStage && info.lastAccess == dstAccess) {
		return;
	}

	VkImageMemoryBarrier barrier = {};
	barrier.sType				 = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout			 = info.currentLayout;
	barrier.newLayout			 = newLayout;
	barrier.srcQueueFamilyIndex	 = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex	 = VK_QUEUE_FAMILY_IGNORED;
	barrier.image				 = info.image;
	barrier.subresourceRange.aspectMask =
		info.vkFormat == VK_FORMAT_D24_UNORM_S8_UINT ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) :
		(info.vkFormat == VK_FORMAT_D16_UNORM || info.vkFormat == VK_FORMAT_D32_SFLOAT) ? VK_IMAGE_ASPECT_DEPTH_BIT
																						: VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel	= 0;
	barrier.subresourceRange.levelCount		= info.mipLevels;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount		= 1;
	barrier.srcAccessMask					= info.lastAccess;
	barrier.dstAccessMask					= dstAccess;

	vkCmdPipelineBarrier(_commandBuffer, info.lastStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	info.currentLayout = newLayout;
	info.lastStage	   = dstStage;
	info.lastAccess	   = dstAccess;
}

void VulkanBackend::TransitionMsaaAttachment(MsaaAttachment &info, VkImageLayout newLayout,
											 VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
	if (info.currentLayout == newLayout && info.lastStage == dstStage && info.lastAccess == dstAccess) {
		return;
	}

	VkImageMemoryBarrier barrier			= {};
	barrier.sType							= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout						= info.currentLayout;
	barrier.newLayout						= newLayout;
	barrier.srcQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex				= VK_QUEUE_FAMILY_IGNORED;
	barrier.image							= info.image;
	barrier.subresourceRange.aspectMask		= info.aspectMask;
	barrier.subresourceRange.baseMipLevel	= 0;
	barrier.subresourceRange.levelCount		= 1;
	barrier.subresourceRange.baseArrayLayer	= 0;
	barrier.subresourceRange.layerCount		= 1;
	barrier.srcAccessMask					= info.lastAccess;
	barrier.dstAccessMask					= dstAccess;

	vkCmdPipelineBarrier(_commandBuffer, info.lastStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	info.currentLayout = newLayout;
	info.lastStage	   = dstStage;
	info.lastAccess	   = dstAccess;
}

VulkanBackend::MsaaAttachment &VulkanBackend::GetOrCreateMsaaAttachment(uint32_t width, uint32_t height, uint32_t slot,
																		VkFormat format,
																		VkSampleCountFlagBits samples,
																		TextureHandle resolveTarget,
																		VkImageUsageFlags usage,
																		VkImageAspectFlags aspectMask) {
	for (auto &attachment : _msaaAttachments) {
		if (attachment.width == width && attachment.height == height && attachment.format == format &&
			attachment.samples == samples && attachment.aspectMask == aspectMask && attachment.slot == slot &&
			attachment.resolveTarget == resolveTarget) {
			return attachment;
		}
	}

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType				= VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType			= VK_IMAGE_TYPE_2D;
	imageInfo.extent.width		= width;
	imageInfo.extent.height		= height;
	imageInfo.extent.depth		= 1;
	imageInfo.mipLevels			= 1;
	imageInfo.arrayLayers		= 1;
	imageInfo.format			= format;
	imageInfo.tiling			= VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout		= VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage				= usage;
	imageInfo.sharingMode		= VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples			= samples;

	VkImage image				= nullptr;
	VkResult result				= vkCreateImage(_device, &imageInfo, nullptr, &image);
	CheckVkResult(result, "vkCreateImage (MSAA attachment)");

	VkDeviceMemory memory		= nullptr;
	AllocateImageMemory(image, memory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	result = vkBindImageMemory(_device, image, memory, 0);
	CheckVkResult(result, "vkBindImageMemory (MSAA attachment)");

	VkImageViewCreateInfo viewInfo			 = {};
	viewInfo.sType							 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image							 = image;
	viewInfo.viewType						 = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format							 = format;
	viewInfo.subresourceRange.aspectMask	 = aspectMask;
	viewInfo.subresourceRange.baseMipLevel	 = 0;
	viewInfo.subresourceRange.levelCount	 = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount	 = 1;

	VkImageView view						 = nullptr;
	result									 = vkCreateImageView(_device, &viewInfo, nullptr, &view);
	CheckVkResult(result, "vkCreateImageView (MSAA attachment)");

	MsaaAttachment attachment;
	attachment.image		 = image;
	attachment.memory		 = memory;
	attachment.view			 = view;
	attachment.width		 = width;
	attachment.height		 = height;
	attachment.slot			 = slot;
	attachment.resolveTarget = resolveTarget;
	attachment.format		 = format;
	attachment.samples		 = samples;
	attachment.aspectMask	 = aspectMask;
	_msaaAttachments.push_back(attachment);
	return _msaaAttachments.back();
}

void VulkanBackend::BindPipeline(PipelineHandle pipeline) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _pipelines.find(pipeline);
	if (it == _pipelines.end()) {
		throw std::runtime_error("Invalid pipeline handle");
	}

	EnsureCommandBuffer();

	if (it->second.isGraphics) {
		vkCmdBindPipeline(_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, it->second.pipeline);
	} else {
		vkCmdBindPipeline(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, it->second.pipeline);
	}

	_currentPipeline = pipeline;
}

void VulkanBackend::SetUniform(PipelineHandle pipeline, const std::string &name, const std::string &type,
							   const void *data) {
	(void)pipeline;
	(void)name;
	(void)type;
	(void)data;
	throw std::runtime_error(
		"Vulkan backend does not implement SetUniform yet; use buffers or add a real uniform path");
}

void VulkanBackend::SetUniformData(PipelineHandle pipeline, const void *data, size_t size) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _pipelines.find(pipeline);
	if (it == _pipelines.end()) {
		throw std::runtime_error("Invalid pipeline handle for SetUniformData");
	}
	if (size == 0) {
		return;
	}
	if (it->second.pushConstantSize == 0) {
		throw std::runtime_error("Pipeline does not declare push constants");
	}
	if (size > it->second.pushConstantSize) {
		throw std::runtime_error("Push constant upload exceeds pipeline declaration");
	}

	EnsureCommandBuffer();
	VkShaderStageFlags pushStages = it->second.isGraphics ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
														  : VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdPushConstants(_commandBuffer, it->second.layout, pushStages, 0, static_cast<uint32_t>(size), data);
}

// =============================================================================
// Resource Binding
// =============================================================================

void VulkanBackend::UpdateDescriptorSet(const DescriptorSetCache &cache) {
	auto pipelineIt = _pipelines.find(cache.pipeline);
	if (pipelineIt == _pipelines.end()) {
		throw std::runtime_error("Descriptor cache references an invalid pipeline");
	}

	const auto						   &pipelineInfo = pipelineIt->second;
	std::vector<VkWriteDescriptorSet>	descriptorWrites;
	std::vector<VkDescriptorBufferInfo> bufferInfos;
	std::vector<VkDescriptorImageInfo>	imageInfos;
	descriptorWrites.reserve(pipelineInfo.resources.size());
	bufferInfos.reserve(pipelineInfo.resources.size());
	imageInfos.reserve(pipelineInfo.resources.size());
	size_t bufferInfoIndex = 0;
	size_t imageInfoIndex  = 0;

	for (const auto &resource : pipelineInfo.resources) {
		if (resource.type == BindingType::Buffer) {
			const BufferHandle handle	= cache.boundBuffers[resource.binding];
			auto			   bufferIt = _buffers.find(handle);
			if (bufferIt == _buffers.end()) {
				throw std::runtime_error("Descriptor cache references an invalid buffer handle");
			}

			bufferInfos.push_back({});
			VkDescriptorBufferInfo &bufferInfo = bufferInfos.back();
			bufferInfo.buffer				   = bufferIt->second.buffer;
			bufferInfo.offset				   = 0;
			bufferInfo.range				   = bufferIt->second.size;
			if (TraceVulkan()) {
				std::cerr << "[easygpu vulkan] descriptor buffer binding=" << resource.binding
						  << " handle=" << handle << " size=" << bufferInfo.range << "\n";
			}

			VkWriteDescriptorSet write = {};
			write.sType				   = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet			   = cache.set;
			write.dstBinding		   = resource.binding;
			write.descriptorType	   = GetVkDescriptorType(resource);
			write.descriptorCount	   = 1;
			write.pBufferInfo		   = &bufferInfos[bufferInfoIndex++];
			descriptorWrites.push_back(write);
		} else if (resource.type == BindingType::Texture) {
			const TextureHandle handle	  = cache.boundTextures[resource.binding];
			auto				textureIt = _textures.find(handle);
			if (textureIt == _textures.end()) {
				throw std::runtime_error("Descriptor cache references an invalid texture handle");
			}

			imageInfos.push_back({});
			VkDescriptorImageInfo &imageInfo = imageInfos.back();
			imageInfo.imageView				 = textureIt->second.view;
			imageInfo.imageLayout			 = VK_IMAGE_LAYOUT_GENERAL;
			if (TraceVulkan()) {
				std::cerr << "[easygpu vulkan] descriptor texture binding=" << resource.binding
						  << " handle=" << handle << " format=" << static_cast<int>(textureIt->second.format)
						  << " vkFormat=" << textureIt->second.vkFormat << " layout=" << imageInfo.imageLayout
						  << " view=" << imageInfo.imageView << "\n";
			}

			VkWriteDescriptorSet write = {};
			write.sType				   = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet			   = cache.set;
			write.dstBinding		   = resource.binding;
			write.descriptorType	   = GetVkDescriptorType(resource);
			write.descriptorCount	   = 1;
			write.pImageInfo		   = &imageInfos[imageInfoIndex++];
			descriptorWrites.push_back(write);
		} else if (resource.type == BindingType::Sampler) {
			const TextureHandle handle	  = cache.boundTextures[resource.binding];
			auto				textureIt = _textures.find(handle);
			if (textureIt == _textures.end()) {
				throw std::runtime_error("Descriptor cache references an invalid sampled texture handle");
			}

			const auto samplerKey = cache.boundSamplers[resource.binding];
			imageInfos.push_back({});
			VkDescriptorImageInfo &imageInfo = imageInfos.back();
			imageInfo.sampler				 = GetOrCreateSampler(samplerKey);
			imageInfo.imageView =
				textureIt->second.sampledView ? textureIt->second.sampledView : textureIt->second.view;
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			if (TraceVulkan()) {
				std::cerr << "[easygpu vulkan] descriptor sampler binding=" << resource.binding
						  << " handle=" << handle << " format=" << static_cast<int>(textureIt->second.format)
						  << " vkFormat=" << textureIt->second.vkFormat << " layout=" << imageInfo.imageLayout
						  << " view=" << imageInfo.imageView << " sampler=" << imageInfo.sampler
						  << " mipLevels=" << textureIt->second.mipLevels << "\n";
			}

			VkWriteDescriptorSet write = {};
			write.sType				   = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet			   = cache.set;
			write.dstBinding		   = resource.binding;
			write.descriptorType	   = GetVkDescriptorType(resource);
			write.descriptorCount	   = 1;
			write.pImageInfo		   = &imageInfos[imageInfoIndex++];
			descriptorWrites.push_back(write);
		}
	}

	if (!descriptorWrites.empty()) {
		vkUpdateDescriptorSets(_device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0,
							   nullptr);
	}
}

VulkanBackend::DescriptorSetCache *VulkanBackend::FindOrCreateDescriptorSet(const ResourceBinding *bindings,
																			uint32_t			   count) {
	auto pipelineIt = _pipelines.find(_currentPipeline);
	if (pipelineIt == _pipelines.end()) {
		throw std::runtime_error("Current pipeline handle is stale");
	}

	DescriptorSetCache requested = {};
	requested.pipeline			 = _currentPipeline;

	for (uint32_t i = 0; i < count; ++i) {
		const auto &binding = bindings[i];
		if (binding.type == BindingType::Buffer) {
			requested.bufferMask					|= (1ull << binding.binding);
			requested.boundBuffers[binding.binding]	 = binding.buffer;
		} else if (binding.type == BindingType::Texture || binding.type == BindingType::Sampler) {
			requested.textureMask						 |= (1ull << binding.binding);
			requested.boundTextures[binding.binding]	  = binding.texture;
			requested.boundTextureTypes[binding.binding]  = binding.type;
			requested.boundFormats[binding.binding]		  = binding.format;
			requested.boundReadOnly[binding.binding]	  = binding.readOnly;
			if (binding.type == BindingType::Sampler) {
				auto textureIt = _textures.find(binding.texture);
				if (textureIt == _textures.end()) {
					throw std::runtime_error("Invalid sampled texture handle in BindResources");
				}
				requested.boundSamplers[binding.binding] = MakeSamplerKey(binding.sampler, textureIt->second.mipLevels > 1);
			}
		}
	}

	for (auto &cache : _descriptorSets) {
		if (cache.pipeline != requested.pipeline) {
			continue;
		}
		if (cache.bufferMask != requested.bufferMask || cache.textureMask != requested.textureMask) {
			continue;
		}

		bool matches = true;
		for (uint32_t i = 0; i < MAX_BUFFER_BINDINGS; ++i) {
			if ((requested.bufferMask & (1ull << i)) != 0 && cache.boundBuffers[i] != requested.boundBuffers[i]) {
				matches = false;
				break;
			}
		}
		if (!matches) {
			continue;
		}
		for (uint32_t i = 0; i < MAX_TEXTURE_BINDINGS; ++i) {
			if ((requested.textureMask & (1ull << i)) == 0) {
				continue;
			}
			if (cache.boundTextures[i] != requested.boundTextures[i] ||
				cache.boundTextureTypes[i] != requested.boundTextureTypes[i] ||
				cache.boundFormats[i] != requested.boundFormats[i] ||
				cache.boundReadOnly[i] != requested.boundReadOnly[i] ||
				(cache.boundTextureTypes[i] == BindingType::Sampler &&
				 !(cache.boundSamplers[i] == requested.boundSamplers[i]))) {
				matches = false;
				break;
			}
		}
		if (matches) {
			return &cache;
		}
	}

	const auto				   &pipelineInfo = pipelineIt->second;
	VkDescriptorSetAllocateInfo allocInfo	 = {};
	allocInfo.sType							 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool				 = _descriptorPool;
	allocInfo.descriptorSetCount			 = 1;
	allocInfo.pSetLayouts					 = &pipelineInfo.descriptorSetLayout;

	VkResult result							 = vkAllocateDescriptorSets(_device, &allocInfo, &requested.set);
	if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
		CreateDescriptorPool();
		allocInfo.descriptorPool = _descriptorPool;
		result = vkAllocateDescriptorSets(_device, &allocInfo, &requested.set);
	}
	CheckVkResult(result, "vkAllocateDescriptorSets");

	_descriptorSets.push_back(std::move(requested));
	UpdateDescriptorSet(_descriptorSets.back());
	return &_descriptorSets.back();
}

void VulkanBackend::BindResources(const ResourceBinding *bindings, uint32_t count) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (_currentPipeline == INVALID_PIPELINE_HANDLE) {
		throw std::runtime_error("No pipeline bound");
	}

	EnsureCommandBuffer();

	auto pipelineIt = _pipelines.find(_currentPipeline);
	if (pipelineIt == _pipelines.end()) {
		throw std::runtime_error("Current pipeline handle is stale");
	}
	auto &pipelineInfo = pipelineIt->second;
	if (pipelineInfo.resources.empty()) {
		if (count != 0) {
			throw std::runtime_error("Pipeline does not declare descriptor-backed resources");
		}
		return;
	}

	std::vector<bool>			 seenResources(pipelineInfo.resources.size(), false);
	std::unordered_set<uint32_t> seenBindings;

	for (uint32_t i = 0; i < count; ++i) {
		const auto &binding = bindings[i];
		if (!seenBindings.insert(binding.binding).second) {
			throw std::runtime_error("BindResources received duplicate binding indices");
		}
		auto layoutIt =
			std::find_if(pipelineInfo.resources.begin(), pipelineInfo.resources.end(),
						 [&](const ResourceLayoutEntry &entry) { return entry.binding == binding.binding; });
		if (layoutIt == pipelineInfo.resources.end()) {
			throw std::runtime_error("BindingResources encountered a binding not declared by the pipeline");
		}
		seenResources[static_cast<size_t>(layoutIt - pipelineInfo.resources.begin())] = true;
		if (layoutIt->type != binding.type) {
			throw std::runtime_error("BindingResources encountered a binding type mismatch");
		}

		if (binding.type == BindingType::Buffer) {
			if (binding.binding >= MAX_BUFFER_BINDINGS) {
				throw std::runtime_error("BindResources buffer binding exceeds Vulkan backend cache limits");
			}
			auto it = _buffers.find(binding.buffer);
			if (it == _buffers.end()) {
				throw std::runtime_error("Invalid buffer handle in BindResources");
			}
		} else if (binding.type == BindingType::Texture || binding.type == BindingType::Sampler) {
			if (binding.binding >= MAX_TEXTURE_BINDINGS) {
				throw std::runtime_error("BindResources texture binding exceeds Vulkan backend cache limits");
			}
			auto it = _textures.find(binding.texture);
			if (it == _textures.end()) {
				throw std::runtime_error("Invalid texture handle in BindResources");
			}
			if (binding.format != layoutIt->format) {
				throw std::runtime_error("BindResources texture format does not match pipeline layout");
			}
			if (binding.readOnly != layoutIt->readOnly) {
				throw std::runtime_error("BindResources texture access mode does not match pipeline layout");
			}

			if (binding.type == BindingType::Sampler) {
				if (_insideRenderPass) {
					if (it->second.currentLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
						throw std::runtime_error(
							"BindResources cannot transition sampled textures inside an active render pass");
					}
				} else {
					TransitionTexture(it->second, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
									  pipelineInfo.isGraphics ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
															  : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
									  VK_ACCESS_SHADER_READ_BIT);
				}
			} else {
				const VkAccessFlags shaderAccess = layoutIt->readOnly
													   ? VK_ACCESS_SHADER_READ_BIT
													   : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
				if (_insideRenderPass) {
					if (it->second.currentLayout != VK_IMAGE_LAYOUT_GENERAL) {
						throw std::runtime_error(
							"BindResources cannot transition storage textures inside an active render pass");
					}
				} else {
					TransitionTexture(it->second, VK_IMAGE_LAYOUT_GENERAL,
									  pipelineInfo.isGraphics ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
															  : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
									  shaderAccess);
				}
			}
		}
	}

	for (size_t i = 0; i < seenResources.size(); ++i) {
		if (!seenResources[i]) {
			throw std::runtime_error("Not all pipeline resources were provided to BindResources");
		}
	}

	DescriptorSetCache *cache = FindOrCreateDescriptorSet(bindings, count);
	if (TraceVulkan()) {
		std::cerr << "[easygpu vulkan] bind descriptor set=" << cache->set << " pipeline=" << _currentPipeline
				  << " count=" << count << " inRenderPass=" << (_insideRenderPass ? 1 : 0) << "\n";
	}

	// Bind descriptor set
	VkPipelineBindPoint bindPoint =
		pipelineInfo.isGraphics ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE;
	vkCmdBindDescriptorSets(_commandBuffer, bindPoint, pipelineInfo.layout, 0, 1, &cache->set, 0, nullptr);
}

// =============================================================================
// Dispatch
// =============================================================================

void VulkanBackend::Dispatch(uint32_t groupX, uint32_t groupY, uint32_t groupZ) {
	std::lock_guard<std::mutex> lock(_mutex);

	EnsureCommandBuffer();

	vkCmdDispatch(_commandBuffer, groupX, groupY, groupZ);
}

// =============================================================================
// Memory Barriers
// =============================================================================

void VulkanBackend::MemoryBarrier(BarrierType barrierType) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_commandBufferRecording) {
		return;
	}

	VkPipelineStageFlags srcStage =
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkPipelineStageFlags dstStage =
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	VkMemoryBarrier memoryBarrier = {};
	memoryBarrier.sType			  = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memoryBarrier.srcAccessMask	  = 0;
	memoryBarrier.dstAccessMask	  = 0;

	if (HasFlag(barrierType, BarrierType::Buffer)) {
		memoryBarrier.srcAccessMask |= VK_ACCESS_SHADER_WRITE_BIT;
		memoryBarrier.dstAccessMask |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	}

	if (HasFlag(barrierType, BarrierType::Texture)) {
		memoryBarrier.srcAccessMask |= VK_ACCESS_SHADER_WRITE_BIT;
		memoryBarrier.dstAccessMask |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	}

	if (memoryBarrier.srcAccessMask != 0 || memoryBarrier.dstAccessMask != 0) {
		vkCmdPipelineBarrier(_commandBuffer, srcStage, dstStage, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
	}
}

void VulkanBackend::Finish() {
	std::lock_guard<std::mutex> lock(_mutex);

	if (_commandBufferRecording) {
		EndCommandBuffer();
		(void)SubmitCommandBuffer(false);
	}
	if (!_submissions.empty()) {
		WaitForSubmittedWork();
	}
}

SubmissionHandle VulkanBackend::Submit() {
	std::lock_guard<std::mutex> lock(_mutex);
	if (_insideRenderPass) {
		throw std::runtime_error("Cannot submit while a render pass is active");
	}
	EnsureCommandBuffer();
	EndCommandBuffer();
	return SubmitCommandBuffer(false, true);
}

bool VulkanBackend::IsSubmissionComplete(SubmissionHandle submission) {
	std::lock_guard<std::mutex> lock(_mutex);
	auto it = _submissions.find(submission);
	if (it == _submissions.end() || it->second.released) {
		throw std::runtime_error("Invalid submission handle");
	}
	return UpdateSubmissionStatus(submission, 0, false);
}

bool VulkanBackend::WaitForSubmission(SubmissionHandle submission, uint64_t timeoutNanoseconds) {
	std::lock_guard<std::mutex> lock(_mutex);
	auto it = _submissions.find(submission);
	if (it == _submissions.end() || it->second.released) {
		throw std::runtime_error("Invalid submission handle");
	}
	return UpdateSubmissionStatus(submission, timeoutNanoseconds, true);
}

void VulkanBackend::ReleaseSubmission(SubmissionHandle submission) {
	std::lock_guard<std::mutex> lock(_mutex);
	auto it = _submissions.find(submission);
	if (it == _submissions.end() || it->second.released) {
		throw std::runtime_error("Invalid submission handle");
	}
	if (it->second.completed) {
		_submissions.erase(it);
		return;
	}
	it->second.released = true;
	ReapReleasedSubmissions();
}

// =============================================================================
// Query / Timing
// =============================================================================

uint32_t VulkanBackend::BeginQuery() {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_queryPool) {
		return 0;
	}

	EnsureCommandBuffer();

	uint32_t queryIndex = _nextQueryIndex;
	if (queryIndex >= MAX_QUERIES * 2 - 1) {
		// Query pool exhausted, reset and start over
		// In a real implementation, you'd want to flush and get results first
		_nextQueryIndex = 0;
		queryIndex		= 0;
	}

	vkCmdResetQueryPool(_commandBuffer, _queryPool, queryIndex, 2);
	// TOP/BOTTOM_OF_PIPE are supported for timestamp queries even on devices
	// that do not expose timestamps at every graphics/compute pipeline stage.
	vkCmdWriteTimestamp(_commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, _queryPool, queryIndex);

	_nextQueryIndex += 2;

	// Find or create query info
	for (uint32_t i = 0; i < _queries.size(); ++i) {
		if (!_queries[i].active) {
			_queries[i].queryIndex = queryIndex;
			_queries[i].active	   = true;
			_queries[i].result	   = 0;
			return i + 1;
		}
	}

	QueryInfo info;
	info.queryIndex = queryIndex;
	info.active		= true;
	info.result		= 0;
	_queries.push_back(info);

	return static_cast<uint32_t>(_queries.size());
}

uint64_t VulkanBackend::EndQuery(uint32_t query) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (query == 0 || !_queryPool) {
		return 0;
	}
	const uint32_t querySlot = query - 1;
	if (querySlot >= _queries.size() || !_queries[querySlot].active) {
		return 0;
	}

	EnsureCommandBuffer();

	uint32_t queryIndex = _queries[querySlot].queryIndex;
	vkCmdWriteTimestamp(_commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, _queryPool, queryIndex + 1);

	// Flush commands and get results
	EndCommandBuffer();
	SubmitCommandBuffer(true);

	// Get query results
	uint64_t timestamps[2] = {0, 0};
	VkResult result		   = vkGetQueryPoolResults(_device, _queryPool, queryIndex, 2, sizeof(timestamps), timestamps,
												   sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

	_queries[querySlot].active = false;

	if (result != VK_SUCCESS) {
		return 0;
	}

	double elapsedNanoseconds =
		static_cast<double>(timestamps[1] - timestamps[0]) * static_cast<double>(_timestampPeriod);
	return static_cast<uint64_t>(elapsedNanoseconds);
}

// =============================================================================
// Binary Cache Support
// =============================================================================

PipelineHandle VulkanBackend::CreatePipelineFromBinary(const PipelineDesc &desc, const void *binaryData,
													   size_t binarySize, uint32_t format) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
	}

	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(_physicalDevice, &properties);
	uint32_t expectedFormat = 0;
	std::memcpy(&expectedFormat, properties.pipelineCacheUUID, sizeof(expectedFormat));
	if (format != expectedFormat || !ValidatePipelineCacheData(binaryData, binarySize, properties)) {
		return INVALID_PIPELINE_HANDLE;
	}

	// Create a temporary pipeline cache from the binary data
	VkPipelineCacheCreateInfo cacheCreateInfo = {};
	cacheCreateInfo.sType					  = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	cacheCreateInfo.initialDataSize			  = binarySize;
	cacheCreateInfo.pInitialData			  = binaryData;

	VkPipelineCache tempCache				  = nullptr;
	VkResult		result					  = vkCreatePipelineCache(_device, &cacheCreateInfo, nullptr, &tempCache);
	if (result != VK_SUCCESS) {
		return INVALID_PIPELINE_HANDLE;
	}

	auto shaderIt = _shaders.find(desc.computeShader);
	if (shaderIt == _shaders.end()) {
		vkDestroyPipelineCache(_device, tempCache, nullptr);
		throw std::runtime_error("Invalid shader handle");
	}

	// Create descriptor set layout (same as CreatePipeline)
	std::vector<ResourceLayoutEntry> sortedResources = desc.resources;
	std::sort(sortedResources.begin(), sortedResources.end());
	sortedResources.erase(std::unique(sortedResources.begin(), sortedResources.end()), sortedResources.end());

	std::vector<VkDescriptorSetLayoutBinding> bindings;
	for (const auto &entry : sortedResources) {
		VkDescriptorSetLayoutBinding binding = {};
		binding.binding						 = entry.binding;
		binding.descriptorType				 = GetVkDescriptorType(entry);
		binding.descriptorCount				 = 1;
		binding.stageFlags					 = GetVkResourceStages(entry.stageFlags, false);
		bindings.push_back(binding);
	}

	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	if (!bindings.empty()) {
		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType						   = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount					   = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings					   = bindings.data();
		result = vkCreateDescriptorSetLayout(_device, &layoutInfo, nullptr, &descriptorSetLayout);
		if (result != VK_SUCCESS) {
			vkDestroyPipelineCache(_device, tempCache, nullptr);
			CheckVkResult(result, "vkCreateDescriptorSetLayout");
		}
	}

	// Create pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType					  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount			  = bindings.empty() ? 0u : 1u;
	pipelineLayoutInfo.pSetLayouts				  = bindings.empty() ? nullptr : &descriptorSetLayout;

	VkPushConstantRange pushConstantRange		  = {};
	if (desc.pushConstantSize != 0) {
		pushConstantRange.stageFlags			  = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset				  = 0;
		pushConstantRange.size					  = desc.pushConstantSize;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		pipelineLayoutInfo.pPushConstantRanges	  = &pushConstantRange;
	}

	VkPipelineLayout pipelineLayout = nullptr;
	result							= vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
	if (result != VK_SUCCESS) {
		vkDestroyPipelineCache(_device, tempCache, nullptr);
		if (descriptorSetLayout)
			vkDestroyDescriptorSetLayout(_device, descriptorSetLayout, nullptr);
		CheckVkResult(result, "vkCreatePipelineLayout");
	}

	// Create compute pipeline using temporary cache
	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType							= VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage							= VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStage.module							= shaderIt->second.module;
	shaderStage.pName							= "main";

	VkComputePipelineCreateInfo pipelineInfo	= {};
	pipelineInfo.sType							= VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage							= shaderStage;
	pipelineInfo.layout							= pipelineLayout;

	VkPipeline pipeline							= nullptr;
	result = vkCreateComputePipelines(_device, tempCache, 1, &pipelineInfo, nullptr, &pipeline);

	if (result == VK_SUCCESS && _pipelineCache != nullptr &&
		vkMergePipelineCaches(_device, _pipelineCache, 1, &tempCache) == VK_SUCCESS) {
		_pipelineCacheDirty = true;
	}

	// Destroy temporary cache
	vkDestroyPipelineCache(_device, tempCache, nullptr);

	if (result != VK_SUCCESS) {
		vkDestroyPipelineLayout(_device, pipelineLayout, nullptr);
		if (descriptorSetLayout)
			vkDestroyDescriptorSetLayout(_device, descriptorSetLayout, nullptr);
		return INVALID_PIPELINE_HANDLE;
	}

	PipelineHandle handle = _nextPipelineHandle++;
	PipelineInfo   info;
	info.pipeline			 = pipeline;
	info.layout				 = pipelineLayout;
	info.descriptorSetLayout = descriptorSetLayout;
	info.workGroupSizeX		 = desc.workGroupSizeX;
	info.workGroupSizeY		 = desc.workGroupSizeY;
	info.workGroupSizeZ		 = desc.workGroupSizeZ;
	info.pushConstantSize	 = desc.pushConstantSize;
	info.resources			 = std::move(sortedResources);
	_pipelines[handle]		 = std::move(info);

	return handle;
}

std::vector<uint8_t> VulkanBackend::GetPipelineBinary(PipelineHandle pipeline, uint32_t &format) {
	format = 0;

	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _pipelines.find(pipeline);
	if (it == _pipelines.end()) {
		return {};
	}

	if (_pipelineCache == nullptr) {
		return {};
	}

	auto data = GetPipelineCacheBytes(_device, _pipelineCache);
	if (!data) {
		return {};
	}

	VkPipelineCacheHeaderVersionOne header{};
	std::memcpy(&header, data->data(), sizeof(header));
	std::memcpy(&format, header.pipelineCacheUUID, sizeof(format));

	return std::move(*data);
}

bool VulkanBackend::SupportsPipelineCache() const {
	return _initialized && _pipelineCache != nullptr;
}

uint32_t VulkanBackend::GetPipelineCacheFormat() const {
	if (!_initialized || !_physicalDevice) {
		return 0;
	}

	// Use device properties pipelineCacheUUID as format identifier
	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(_physicalDevice, &props);
	uint32_t format = 0;
	std::memcpy(&format, props.pipelineCacheUUID, sizeof(format));
	return format;
}

PipelineCacheStats VulkanBackend::GetPipelineCacheStats() const {
	std::lock_guard<std::mutex> lock(_mutex);
	return _pipelineCacheStats;
}

void VulkanBackend::FlushPipelineCache() {
	std::lock_guard<std::mutex> lock(_mutex);
	if (_initialized) {
		PersistPipelineCache();
	}
}

// =============================================================================
// Backend Factory
// =============================================================================

Backend *CreateVulkanBackend() {
	return new VulkanBackend();
}

// =============================================================================
// Graphics Pipeline Support
// =============================================================================

PipelineHandle VulkanBackend::CreateGraphicsPipeline(const GraphicsPipelineDesc &desc) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
	}
	if (!_caps.supportsGraphics) {
		throw std::runtime_error("Graphics pipeline not supported on this device");
	}

	// Validate shaders
	auto vsIt = _shaders.find(desc.vertexShader);
	if (vsIt == _shaders.end())
		throw std::runtime_error("Invalid vertex shader handle");
	auto fsIt = _shaders.find(desc.fragmentShader);
	if (fsIt == _shaders.end())
		throw std::runtime_error("Invalid fragment shader handle");

	if (desc.pushConstantSize > _maxPushConstantSize) {
		throw std::runtime_error("Pipeline push constant size exceeds device limit");
	}
	if (desc.depthClampEnable && !_depthClampSupported) {
		throw std::runtime_error("Graphics pipeline depth clamp requires Vulkan depthClamp feature support");
	}
	if (desc.polygonMode != PolygonMode::Fill && !_fillModeNonSolidSupported) {
		throw std::runtime_error("Graphics pipeline non-fill polygon mode requires Vulkan fillModeNonSolid feature support");
	}

	std::vector<PixelFormat> colorAttachmentFormats = desc.colorAttachmentFormats;
	if (colorAttachmentFormats.empty()) {
		colorAttachmentFormats.push_back(desc.colorAttachmentFormat);
	}
	if (colorAttachmentFormats.empty() || colorAttachmentFormats.size() > MAX_COLOR_ATTACHMENTS) {
		throw std::runtime_error("Graphics pipeline color attachment count must be between 1 and MAX_COLOR_ATTACHMENTS");
	}

	VkSampleCountFlagBits sampleCount = GetVkSampleCount(desc.sampleCount);
	if (sampleCount != VK_SAMPLE_COUNT_1_BIT) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(_physicalDevice, &props);
		VkSampleCountFlags supported = props.limits.framebufferColorSampleCounts;
		if (desc.depthTestEnable) {
			supported &= props.limits.framebufferDepthSampleCounts;
		}
		if ((supported & sampleCount) == 0) {
			throw std::runtime_error("Requested graphics pipeline MSAA sample count is not supported by this device");
		}
	}

	// Create descriptor set layout for resources
	std::vector<ResourceLayoutEntry> sortedResources = desc.resources;
	std::sort(sortedResources.begin(), sortedResources.end());
	sortedResources.erase(std::unique(sortedResources.begin(), sortedResources.end()), sortedResources.end());

	std::vector<VkDescriptorSetLayoutBinding> bindings;
	for (const auto &entry : sortedResources) {
		if (entry.type == BindingType::Buffer && entry.binding >= MAX_BUFFER_BINDINGS) {
			throw std::runtime_error("Buffer binding exceeds Vulkan backend cache limits");
		}
		if ((entry.type == BindingType::Texture || entry.type == BindingType::Sampler) &&
			entry.binding >= MAX_TEXTURE_BINDINGS) {
			throw std::runtime_error("Texture binding exceeds Vulkan backend cache limits");
		}
		VkDescriptorSetLayoutBinding binding = {};
		binding.binding						 = entry.binding;
		binding.descriptorType				 = GetVkDescriptorType(entry);
		binding.descriptorCount				 = 1;
		binding.stageFlags					 = GetVkResourceStages(entry.stageFlags, true);
		binding.pImmutableSamplers			 = nullptr;
		bindings.push_back(binding);
	}
	if (TraceVulkan()) {
		std::cerr << "[easygpu vulkan] create graphics pipeline resources=" << sortedResources.size()
				  << " colorAttachments=" << colorAttachmentFormats.size() << " sampleCount="
				  << static_cast<int>(sampleCount) << " depth=" << (desc.depthTestEnable ? 1 : 0) << "\n";
		for (const auto &entry : sortedResources) {
			std::cerr << "[easygpu vulkan]   layout binding=" << entry.binding << " type="
					  << BindingTypeName(entry.type) << " format=" << static_cast<int>(entry.format)
					  << " readOnly=" << (entry.readOnly ? 1 : 0) << " stages=" << entry.stageFlags << "\n";
		}
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType						   = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount					   = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings					   = bindings.data();

	VkDescriptorSetLayout descriptorSetLayout  = nullptr;
	VkResult			  result = vkCreateDescriptorSetLayout(_device, &layoutInfo, nullptr, &descriptorSetLayout);
	CheckVkResult(result, "vkCreateDescriptorSetLayout");

	// Create pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType					  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount			  = bindings.empty() ? 0u : 1u;
	pipelineLayoutInfo.pSetLayouts				  = bindings.empty() ? nullptr : &descriptorSetLayout;

	VkPushConstantRange pushConstantRange		  = {};
	if (desc.pushConstantSize != 0) {
		pushConstantRange.stageFlags			  = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset				  = 0;
		pushConstantRange.size					  = desc.pushConstantSize;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		pipelineLayoutInfo.pPushConstantRanges	  = &pushConstantRange;
	}

	VkPipelineLayout pipelineLayout = nullptr;
	result							= vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
	if (result != VK_SUCCESS) {
		vkDestroyDescriptorSetLayout(_device, descriptorSetLayout, nullptr);
		throw std::runtime_error(std::string("vkCreatePipelineLayout failed: ") + VkResultToString(result));
	}

	// Shader stages
	VkPipelineShaderStageCreateInfo vsStage			 = {};
	vsStage.sType									 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vsStage.stage									 = VK_SHADER_STAGE_VERTEX_BIT;
	vsStage.module									 = vsIt->second.module;
	vsStage.pName									 = "main";

	VkPipelineShaderStageCreateInfo fsStage			 = {};
	fsStage.sType									 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fsStage.stage									 = VK_SHADER_STAGE_FRAGMENT_BIT;
	fsStage.module									 = fsIt->second.module;
	fsStage.pName									 = "main";

	VkPipelineShaderStageCreateInfo		 stages[]	 = {vsStage, fsStage};

	// Vertex input state
	VkPipelineVertexInputStateCreateInfo vertexInput = {};
	vertexInput.sType								 = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	std::vector<VkVertexInputBindingDescription>   vertexBindings;
	std::vector<VkVertexInputAttributeDescription> vertexAttribs;

	if (!desc.vertexLayout.empty()) {
		VkVertexInputBindingDescription bindingDesc = {};
		bindingDesc.binding							= 0;
		bindingDesc.inputRate						= VK_VERTEX_INPUT_RATE_VERTEX;

		// Calculate stride from last entry's offset + size
		uint32_t stride								= 0;
		for (const auto &entry : desc.vertexLayout) {
			uint32_t size = 4; // Default float size
			switch (entry.format) {
			case PixelFormat::R32F:
			case PixelFormat::R32I:
			case PixelFormat::R32UI:
				size = 4;
				break;
			case PixelFormat::RG32F:
			case PixelFormat::RG32I:
			case PixelFormat::RG32UI:
				size = 8;
				break;
			case PixelFormat::RGB32F:
			case PixelFormat::RGB32I:
			case PixelFormat::RGB32UI:
				size = 12;
				break;
			case PixelFormat::RGBA8:
				size = 4;
				break;
			case PixelFormat::RGBA32F:
			case PixelFormat::RGBA32I:
			case PixelFormat::RGBA32UI:
				size = 16;
				break;
			default:
				size = 16;
				break;
			}
			stride = std::max(stride, entry.offset + size);
		}
		bindingDesc.stride = stride;
		vertexBindings.push_back(bindingDesc);

		for (const auto &entry : desc.vertexLayout) {
			VkVertexInputAttributeDescription attr = {};
			attr.location						   = entry.location;
			attr.binding						   = 0;
			attr.format							   = GetVkFormat(entry.format);
			attr.offset							   = entry.offset;
			vertexAttribs.push_back(attr);
		}
	}

	vertexInput.vertexBindingDescriptionCount			 = static_cast<uint32_t>(vertexBindings.size());
	vertexInput.pVertexBindingDescriptions				 = vertexBindings.data();
	vertexInput.vertexAttributeDescriptionCount			 = static_cast<uint32_t>(vertexAttribs.size());
	vertexInput.pVertexAttributeDescriptions			 = vertexAttribs.data();

	// Input assembly
	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType									 = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	switch (desc.topology) {
	case PrimitiveTopology::PointList:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		break;
	case PrimitiveTopology::LineList:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		break;
	case PrimitiveTopology::LineStrip:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
		break;
	case PrimitiveTopology::TriangleList:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		break;
	case PrimitiveTopology::TriangleStrip:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
		break;
	case PrimitiveTopology::TriangleFan:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
		break;
	}
	inputAssembly.primitiveRestartEnable			   = VK_FALSE;

	// Viewport state (dynamic)
	VkPipelineViewportStateCreateInfo viewportState	   = {};
	viewportState.sType								   = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount						   = 1;
	viewportState.scissorCount						   = 1;

	// Rasterization
	VkPipelineRasterizationStateCreateInfo rasterizer  = {};
	rasterizer.sType								   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable						   = desc.depthClampEnable ? VK_TRUE : VK_FALSE;
	rasterizer.rasterizerDiscardEnable				   = VK_FALSE;
	rasterizer.polygonMode							   = ToVkPolygonMode(desc.polygonMode);
	rasterizer.cullMode								   = ToVkCullMode(desc.cullMode);
	rasterizer.frontFace							   = ToVkFrontFace(desc.frontFace);
	rasterizer.depthBiasEnable						   = VK_FALSE;
	rasterizer.lineWidth							   = 1.0f;

	// Multisampling
	VkPipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sType								   = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.rasterizationSamples				   = sampleCount;
	multisampling.minSampleShading					   = 1.0f;

	// Depth/stencil
	VkPipelineDepthStencilStateCreateInfo depthStencil = {};
	depthStencil.sType								   = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable					   = desc.depthTestEnable ? VK_TRUE : VK_FALSE;
	depthStencil.depthWriteEnable					   = desc.depthWriteEnable ? VK_TRUE : VK_FALSE;
	depthStencil.depthCompareOp						   = ToVkCompareOp(desc.depthCompareOp);
	depthStencil.depthBoundsTestEnable				   = VK_FALSE;
	depthStencil.stencilTestEnable					   = desc.stencilTestEnable ? VK_TRUE : VK_FALSE;
	depthStencil.front = ToVkStencilOpState(desc.stencilFront, desc.stencilReadMask, desc.stencilWriteMask,
											desc.stencilReference);
	depthStencil.back = ToVkStencilOpState(desc.stencilBack, desc.stencilReadMask, desc.stencilWriteMask,
										   desc.stencilReference);

	// Color blend
	std::vector<ColorAttachmentBlendState> blendStates = desc.colorBlendAttachments;
	if (blendStates.empty()) {
		ColorAttachmentBlendState state;
		state.blendEnable = desc.blendEnable;
		state.srcColorBlendFactor = desc.blendSrcColor;
		state.dstColorBlendFactor = desc.blendDstColor;
		state.colorBlendOp = desc.blendColorOp;
		state.srcAlphaBlendFactor = desc.blendSrcAlpha;
		state.dstAlphaBlendFactor = desc.blendDstAlpha;
		state.alphaBlendOp = desc.blendAlphaOp;
		state.colorWriteMask = desc.colorWriteMask;
		blendStates.assign(colorAttachmentFormats.size(), state);
	}
	if (blendStates.size() != colorAttachmentFormats.size()) {
		throw std::runtime_error("Graphics pipeline color blend attachment count must match color attachment formats");
	}
	std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments;
	colorBlendAttachments.reserve(blendStates.size());
	for (const auto &state : blendStates) {
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.blendEnable			  = state.blendEnable ? VK_TRUE : VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor	  = ToVkBlendFactor(state.srcColorBlendFactor);
		colorBlendAttachment.dstColorBlendFactor	  = ToVkBlendFactor(state.dstColorBlendFactor);
		colorBlendAttachment.colorBlendOp			  = ToVkBlendOp(state.colorBlendOp);
		colorBlendAttachment.srcAlphaBlendFactor	  = ToVkBlendFactor(state.srcAlphaBlendFactor);
		colorBlendAttachment.dstAlphaBlendFactor	  = ToVkBlendFactor(state.dstAlphaBlendFactor);
		colorBlendAttachment.alphaBlendOp			  = ToVkBlendOp(state.alphaBlendOp);
		colorBlendAttachment.colorWriteMask		  = ToVkColorWriteMask(state.colorWriteMask);
		colorBlendAttachments.push_back(colorBlendAttachment);
	}

	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType								  = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable						  = VK_FALSE;
	colorBlending.attachmentCount					  = static_cast<uint32_t>(colorBlendAttachments.size());
	colorBlending.pAttachments						  = colorBlendAttachments.data();

	// Dynamic state
	VkDynamicState					 dynamicStates[]  = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamicState	  = {};
	dynamicState.sType								  = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount					  = 2;
	dynamicState.pDynamicStates						  = dynamicStates;

	// Dynamic rendering
	std::vector<VkFormat>			 colorFormats;
	colorFormats.reserve(colorAttachmentFormats.size());
	for (PixelFormat format : colorAttachmentFormats) {
		colorFormats.push_back(GetVkFormat(format));
	}
	VkPipelineRenderingCreateInfoKHR renderingInfo	  = {};
	renderingInfo.sType								  = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
	renderingInfo.colorAttachmentCount				  = static_cast<uint32_t>(colorFormats.size());
	renderingInfo.pColorAttachmentFormats			  = colorFormats.data();
	if (desc.depthTestEnable || desc.depthWriteEnable || desc.stencilTestEnable) {
		const auto depthFormat = GetVkFormat(desc.depthAttachmentFormat);
		if (depthFormat != VK_FORMAT_D32_SFLOAT && depthFormat != VK_FORMAT_D24_UNORM_S8_UINT) {
			throw std::runtime_error("Graphics pipeline depth attachment format must be D32F or D24S8");
		}
		renderingInfo.depthAttachmentFormat = depthFormat;
		if (depthFormat == VK_FORMAT_D24_UNORM_S8_UINT) {
			renderingInfo.stencilAttachmentFormat = VK_FORMAT_D24_UNORM_S8_UINT;
		}
	}

	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType						  = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext						  = &renderingInfo;
	pipelineInfo.stageCount					  = 2;
	pipelineInfo.pStages					  = stages;
	pipelineInfo.pVertexInputState			  = &vertexInput;
	pipelineInfo.pInputAssemblyState		  = &inputAssembly;
	pipelineInfo.pViewportState				  = &viewportState;
	pipelineInfo.pRasterizationState		  = &rasterizer;
	pipelineInfo.pMultisampleState			  = &multisampling;
	pipelineInfo.pDepthStencilState			  = &depthStencil;
	pipelineInfo.pColorBlendState			  = &colorBlending;
	pipelineInfo.pDynamicState				  = &dynamicState;
	pipelineInfo.layout						  = pipelineLayout;
	pipelineInfo.renderPass					  = nullptr;
	pipelineInfo.subpass					  = 0;

	VkPipeline pipeline						  = nullptr;
	result = vkCreateGraphicsPipelines(_device, _pipelineCache, 1, &pipelineInfo, nullptr, &pipeline);
	if (result != VK_SUCCESS) {
		vkDestroyPipelineLayout(_device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(_device, descriptorSetLayout, nullptr);
		throw std::runtime_error(std::string("vkCreateGraphicsPipelines failed: ") + VkResultToString(result));
	}

	PipelineHandle handle = _nextPipelineHandle++;
	PipelineInfo   info;
	info.pipeline			 = pipeline;
	info.layout				 = pipelineLayout;
	info.descriptorSetLayout = descriptorSetLayout;
	info.pushConstantSize	 = desc.pushConstantSize;
	info.resources			 = std::move(sortedResources);
	info.isGraphics			 = true;
	info.vertexShader		 = desc.vertexShader;
	info.fragmentShader		 = desc.fragmentShader;
	info.topology			 = desc.topology;
	info.colorFormat		 = colorAttachmentFormats.front();
	info.colorFormats		 = std::move(colorAttachmentFormats);
	info.samples			 = sampleCount;
	info.depthEnable		 = desc.depthTestEnable;
	info.vertexLayout		 = desc.vertexLayout;

	_pipelines[handle]		 = std::move(info);
	_pipelineCacheDirty	 = true;
	return handle;
}

void VulkanBackend::BeginRendering(const RenderPassBeginDesc &desc) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_caps.supportsGraphics) {
		throw std::runtime_error("Graphics not supported on this device");
	}
	if (_insideRenderPass) {
		throw std::runtime_error("BeginRendering called while already inside a render pass");
	}
	if (!_vkCmdBeginRenderingKHR) {
		throw std::runtime_error("Dynamic rendering function not available");
	}

	EnsureCommandBuffer();

	const VkSampleCountFlagBits sampleCount = GetVkSampleCount(desc.sampleCount);

	AttachmentLoadOp effectiveColorLoadOp = desc.colorLoadOp;
	if (effectiveColorLoadOp == AttachmentLoadOp::Default) {
		effectiveColorLoadOp = desc.clearColorFlag ? AttachmentLoadOp::Clear : AttachmentLoadOp::Load;
	}

	std::vector<TextureHandle> colorHandles = desc.colorAttachments;
	if (colorHandles.empty() && desc.colorAttachment != INVALID_TEXTURE_HANDLE) {
		colorHandles.push_back(desc.colorAttachment);
	}
	if (colorHandles.empty() || colorHandles.size() > MAX_COLOR_ATTACHMENTS) {
		throw std::runtime_error("BeginRendering color attachment count must be between 1 and MAX_COLOR_ATTACHMENTS");
	}

	std::vector<std::unordered_map<TextureHandle, TextureInfo>::iterator> colorTextureIters;
	colorTextureIters.reserve(colorHandles.size());
	for (TextureHandle handle : colorHandles) {
		auto colorIt = _textures.find(handle);
		if (colorIt == _textures.end()) {
			throw std::runtime_error("Invalid color attachment texture handle");
		}
		colorTextureIters.push_back(colorIt);
	}

	const uint32_t renderWidth	= colorTextureIters.front()->second.width;
	const uint32_t renderHeight = colorTextureIters.front()->second.height;
	for (const auto &colorIt : colorTextureIters) {
		if (colorIt->second.width != renderWidth || colorIt->second.height != renderHeight) {
			throw std::runtime_error("BeginRendering MRT color attachments must have identical dimensions");
		}
	}

	std::vector<VkRenderingAttachmentInfoKHR> colorAttachments;
	colorAttachments.reserve(colorTextureIters.size());
	for (size_t colorIndex = 0; colorIndex < colorTextureIters.size(); ++colorIndex) {
		auto &colorIt = colorTextureIters[colorIndex];
		TransitionTexture(colorIt->second, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
						  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
						  VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

		VkRenderingAttachmentInfoKHR colorAttachment = {};
		colorAttachment.sType						= VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
		colorAttachment.imageLayout					= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		switch (effectiveColorLoadOp) {
		case AttachmentLoadOp::Load:
			colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
			break;
		case AttachmentLoadOp::Clear:
			colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			break;
		case AttachmentLoadOp::DontCare:
			colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			break;
		case AttachmentLoadOp::Default:
		default:
			colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
			break;
		}
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.clearValue.color.float32[0] = desc.clearColor[0];
		colorAttachment.clearValue.color.float32[1] = desc.clearColor[1];
		colorAttachment.clearValue.color.float32[2] = desc.clearColor[2];
		colorAttachment.clearValue.color.float32[3] = desc.clearColor[3];

		if (sampleCount == VK_SAMPLE_COUNT_1_BIT) {
			colorAttachment.imageView = colorIt->second.view;
		} else {
			auto &msaaColor = GetOrCreateMsaaAttachment(
				renderWidth, renderHeight, static_cast<uint32_t>(colorIndex), colorIt->second.vkFormat, sampleCount,
				colorHandles[colorIndex], VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				VK_IMAGE_ASPECT_COLOR_BIT);
			const VkAccessFlags msaaAccess = colorAttachment.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD
												? (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
												   VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
												: VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			TransitionMsaaAttachment(msaaColor, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
									 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, msaaAccess);

			colorAttachment.imageView	   = msaaColor.view;
			colorAttachment.storeOp		   = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachment.resolveMode	   = VK_RESOLVE_MODE_AVERAGE_BIT;
			colorAttachment.resolveImageView = colorIt->second.view;
			colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		}
		if (TraceVulkan()) {
			std::cerr << "[easygpu vulkan] color attachment slot=" << colorIndex
					  << " handle=" << colorHandles[colorIndex] << " format=" << static_cast<int>(colorIt->second.format)
					  << " vkFormat=" << colorIt->second.vkFormat << " size=" << colorIt->second.width << "x"
					  << colorIt->second.height << " view=" << colorAttachment.imageView
					  << " loadOp=" << colorAttachment.loadOp << "\n";
		}
		colorAttachments.push_back(colorAttachment);
	}

	VkRenderingAttachmentInfoKHR depthAttachment = {};
	bool						 hasDepth		 = false;
	bool						 hasStencil		 = false;

	if (desc.depthAttachment != INVALID_TEXTURE_HANDLE) {
		auto depthIt = _textures.find(desc.depthAttachment);
		if (depthIt != _textures.end()) {
			hasStencil = depthIt->second.vkFormat == VK_FORMAT_D24_UNORM_S8_UINT;
			depthAttachment.sType		= VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
			depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			depthAttachment.loadOp	= desc.clearDepthFlag ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
			depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			depthAttachment.clearValue.depthStencil.depth	= desc.clearDepth;
			depthAttachment.clearValue.depthStencil.stencil = 0;

			if (sampleCount == VK_SAMPLE_COUNT_1_BIT) {
				depthAttachment.imageView = depthIt->second.view;
				const VkAccessFlags depthAccess = desc.clearDepthFlag
											 ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
											 : (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
												VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
				TransitionTexture(depthIt->second, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
								  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
									  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
								  depthAccess);
			} else {
				auto &msaaDepth = GetOrCreateMsaaAttachment(
					renderWidth, renderHeight, MAX_COLOR_ATTACHMENTS, depthIt->second.vkFormat, sampleCount,
					desc.depthAttachment,
					VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
					hasStencil ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_DEPTH_BIT);
				const VkAccessFlags depthAccess = desc.clearDepthFlag
											 ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
											 : (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
												VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
				TransitionMsaaAttachment(msaaDepth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
										 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
											 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
										 depthAccess);
				depthAttachment.imageView = msaaDepth.view;
			}
			hasDepth = true;
		}
	}

	VkRenderingInfoKHR renderingInfo	   = {};
	renderingInfo.sType					   = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
	renderingInfo.renderArea.extent.width  = renderWidth;
	renderingInfo.renderArea.extent.height = renderHeight;
	renderingInfo.layerCount			   = 1;
	renderingInfo.colorAttachmentCount	   = static_cast<uint32_t>(colorAttachments.size());
	renderingInfo.pColorAttachments		   = colorAttachments.data();
	if (hasDepth) {
		renderingInfo.pDepthAttachment = &depthAttachment;
		if (hasStencil) {
			renderingInfo.pStencilAttachment = &depthAttachment;
		}
	}

	_vkCmdBeginRenderingKHR(_commandBuffer, &renderingInfo);
	_insideRenderPass = true;
}

void VulkanBackend::EndRendering() {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_insideRenderPass) {
		throw std::runtime_error("EndRendering called without matching BeginRendering");
	}

	_vkCmdEndRenderingKHR(_commandBuffer);
	_insideRenderPass = false;
}

void VulkanBackend::SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
	std::lock_guard<std::mutex> lock(_mutex);

	VkViewport					viewport = {};
	viewport.x							 = static_cast<float>(x);
	viewport.y							 = static_cast<float>(y);
	viewport.width						 = static_cast<float>(width);
	viewport.height						 = static_cast<float>(height);
	viewport.minDepth					 = 0.0f;
	viewport.maxDepth					 = 1.0f;

	EnsureCommandBuffer();
	vkCmdSetViewport(_commandBuffer, 0, 1, &viewport);
}

void VulkanBackend::SetScissor(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
	std::lock_guard<std::mutex> lock(_mutex);

	VkRect2D					scissor = {};
	scissor.offset						= {static_cast<int32_t>(x), static_cast<int32_t>(y)};
	scissor.extent						= {width, height};

	EnsureCommandBuffer();
	vkCmdSetScissor(_commandBuffer, 0, 1, &scissor);
}

void VulkanBackend::BindVertexBuffer(BufferHandle buffer, uint32_t stride) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (_currentPipeline != INVALID_PIPELINE_HANDLE) {
		auto pipelineIt = _pipelines.find(_currentPipeline);
		if (pipelineIt != _pipelines.end() && pipelineIt->second.isGraphics && pipelineIt->second.vertexLayout.empty()) {
			_currentVertexBuffer = INVALID_BUFFER_HANDLE;
			return;
		}
	}

	auto						it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		throw std::runtime_error("Invalid vertex buffer handle");
	}

	EnsureCommandBuffer();
	VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(_commandBuffer, 0, 1, &it->second.buffer, &offset);
	_currentVertexBuffer = buffer;
}

void VulkanBackend::BindIndexBuffer(BufferHandle buffer) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _buffers.find(buffer);
	if (it == _buffers.end()) {
		throw std::runtime_error("Invalid index buffer handle");
	}

	EnsureCommandBuffer();
	vkCmdBindIndexBuffer(_commandBuffer, it->second.buffer, 0, VK_INDEX_TYPE_UINT32);
	_currentIndexBuffer = buffer;
}

void VulkanBackend::Draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_insideRenderPass) {
		throw std::runtime_error("Draw called outside a render pass");
	}

	EnsureCommandBuffer();
	vkCmdDraw(_commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
}

void VulkanBackend::DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
								uint32_t firstInstance) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_insideRenderPass) {
		throw std::runtime_error("DrawIndexed called outside a render pass");
	}

	EnsureCommandBuffer();
	vkCmdDrawIndexed(_commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

// =============================================================================
// Depth Buffer
// =============================================================================

TextureHandle VulkanBackend::CreateDepthBuffer(uint32_t width, uint32_t height) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
	}

	VkFormat		  depthFormat = VK_FORMAT_D32_SFLOAT;

	VkImageCreateInfo imageInfo	  = {};
	imageInfo.sType				  = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType			  = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width		  = width;
	imageInfo.extent.height		  = height;
	imageInfo.extent.depth		  = 1;
	imageInfo.mipLevels			  = 1;
	imageInfo.arrayLayers		  = 1;
	imageInfo.format			  = depthFormat;
	imageInfo.tiling			  = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout		  = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage				  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	imageInfo.sharingMode		  = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples			  = VK_SAMPLE_COUNT_1_BIT;

	VkImage	 image				  = nullptr;
	VkResult result				  = vkCreateImage(_device, &imageInfo, nullptr, &image);
	CheckVkResult(result, "vkCreateImage (depth)");

	VkDeviceMemory memory = nullptr;
	AllocateImageMemory(image, memory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	result = vkBindImageMemory(_device, image, memory, 0);
	CheckVkResult(result, "vkBindImageMemory (depth)");

	VkImageViewCreateInfo viewInfo			 = {};
	viewInfo.sType							 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image							 = image;
	viewInfo.viewType						 = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format							 = depthFormat;
	viewInfo.subresourceRange.aspectMask	 = VK_IMAGE_ASPECT_DEPTH_BIT;
	viewInfo.subresourceRange.baseMipLevel	 = 0;
	viewInfo.subresourceRange.levelCount	 = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount	 = 1;

	VkImageView view						 = nullptr;
	result									 = vkCreateImageView(_device, &viewInfo, nullptr, &view);
	CheckVkResult(result, "vkCreateImageView (depth)");

	TextureHandle handle = _nextTextureHandle++;
	TextureInfo	  info;
	info.image		   = image;
	info.memory		   = memory;
	info.view		   = view;
	info.width		   = width;
	info.height		   = height;
	info.depth		   = 1;
	info.mipLevels	   = 1;
	info.format		   = PixelFormat::R32F;
	info.vkFormat	   = depthFormat;
	info.samples	   = VK_SAMPLE_COUNT_1_BIT;
	info.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	_textures[handle]  = std::move(info);
	return handle;
}

void VulkanBackend::DestroyDepthBuffer(TextureHandle texture) {
	DestroyTexture(texture);
}

// =============================================================================
// Uniform Buffer Support
// =============================================================================

BufferHandle VulkanBackend::CreateUniformBuffer(size_t size, const void *data) {
	std::lock_guard<std::mutex> lock(_mutex);

	BufferDesc					desc;
	desc.sizeInBytes = size;
	desc.mode		 = BufferMode::Read;
	desc.initialData = data;

	// Create a buffer with UNIFORM_BUFFER usage
	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
	}

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType			  = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size				  = size;
	bufferInfo.usage			  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode		  = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer buffer				  = nullptr;
	VkResult result				  = vkCreateBuffer(_device, &bufferInfo, nullptr, &buffer);
	CheckVkResult(result, "vkCreateBuffer (UBO)");

	VkDeviceMemory memory = nullptr;
	AllocateBufferMemory(buffer, memory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, size);

	result = vkBindBufferMemory(_device, buffer, memory, 0);
	CheckVkResult(result, "vkBindBufferMemory (UBO)");

	// Upload initial data if provided
	if (data != nullptr) {
		UploadBufferInternal(buffer, size, data);
	}

	BufferHandle handle = _nextBufferHandle++;
	BufferInfo	 info;
	info.buffer		 = buffer;
	info.memory		 = memory;
	info.size		 = size;
	info.mode		 = BufferMode::Read;
	info.memoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	_buffers[handle] = std::move(info);
	return handle;
}

void VulkanBackend::UploadUniformBuffer(BufferHandle handle, const void *data, size_t size) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _buffers.find(handle);
	if (it == _buffers.end()) {
		throw std::runtime_error("Invalid uniform buffer handle");
	}

	UploadBufferInternal(it->second.buffer, size, data);
}

} // namespace GPU::Backend
