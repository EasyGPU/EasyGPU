/**
 * @file VulkanBackend.cpp
 * @brief Vulkan backend implementation
 */

#include <Backend/VulkanBackend.h>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <unordered_set>

// glslang includes for GLSL to SPIR-V compilation
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

namespace GPU::Backend {

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
	if (_initialized) {
		return;
	}

	std::lock_guard<std::mutex> lock(_mutex);

	try {
		InitializeGlslang();
		CreateInstance();
		SelectPhysicalDevice();
		CreateDevice();
		CreateCommandPool();
		CreateDescriptorPool();
		CreateDefaultSampler();
		CreateQueryPool();

		// Create persistent pipeline cache for binary caching
		VkPipelineCacheCreateInfo cacheCreateInfo = {};
		cacheCreateInfo.sType					  = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		auto result = vkCreatePipelineCache(_device, &cacheCreateInfo, nullptr, &_pipelineCache);
		if (result != VK_SUCCESS) {
			_pipelineCache = nullptr; // Pipeline cache is optional
		}

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

	_initialized = false;
}

bool VulkanBackend::IsInitialized() const {
	return _initialized;
}

void VulkanBackend::CleanupVulkan() {
	if (_device) {
		vkDeviceWaitIdle(_device);
		_descriptorSets.clear();
		_inFlightDescriptorSets.clear();

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
		if (_descriptorPool)
			vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);

		// Destroy pipeline cache
		if (_pipelineCache)
			vkDestroyPipelineCache(_device, _pipelineCache, nullptr);

		// Destroy query pool
		if (_queryPool)
			vkDestroyQueryPool(_device, _queryPool, nullptr);

		// Destroy command resources
		if (_commandFence)
			vkDestroyFence(_device, _commandFence, nullptr);
		if (_commandPool)
			vkDestroyCommandPool(_device, _commandPool, nullptr);

		// Destroy device
		vkDestroyDevice(_device, nullptr);
		_device = nullptr;
	}

	// Destroy instance
	if (_instance) {
		vkDestroyInstance(_instance, nullptr);
		_instance = nullptr;
	}
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

	// Enable validation layers only when explicitly requested and available.
#ifdef EASYGPU_ENABLE_VALIDATION
	const char *validationLayer = "VK_LAYER_KHRONOS_validation";
	uint32_t	layerCount		= 0;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
	std::vector<VkLayerProperties> availableLayers(layerCount);
	if (layerCount != 0) {
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
	}

	const bool hasValidationLayer =
		std::any_of(availableLayers.begin(), availableLayers.end(),
					[&](const VkLayerProperties &layer) { return std::strcmp(layer.layerName, validationLayer) == 0; });
	if (hasValidationLayer) {
		createInfo.enabledLayerCount   = 1;
		createInfo.ppEnabledLayerNames = &validationLayer;
	}
#endif

	// No instance extensions required for headless compute
	createInfo.enabledExtensionCount = 0;

	VkResult result					 = vkCreateInstance(&createInfo, nullptr, &_instance);
	CheckVkResult(result, "vkCreateInstance");
}

void VulkanBackend::SelectPhysicalDevice() {
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr);

	if (deviceCount == 0) {
		throw std::runtime_error("No Vulkan-capable physical devices found");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data());

	// Select the first device that supports compute
	for (auto device : devices) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(device, &props);

		// Check for compute queue family
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		for (uint32_t i = 0; i < queueFamilyCount; ++i) {
			if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				_physicalDevice			 = device;
				_computeQueueFamilyIndex = i;

				// Store capabilities
				_caps.versionString		 = std::to_string(VK_VERSION_MAJOR(props.apiVersion)) + "." +
									  std::to_string(VK_VERSION_MINOR(props.apiVersion));

				VkPhysicalDeviceLimits limits = props.limits;
				_caps.maxWorkGroupSizeX		  = limits.maxComputeWorkGroupSize[0];
				_caps.maxWorkGroupSizeY		  = limits.maxComputeWorkGroupSize[1];
				_caps.maxWorkGroupSizeZ		  = limits.maxComputeWorkGroupSize[2];
				_caps.maxBufferBindings		  = limits.maxPerStageDescriptorStorageBuffers;
				_caps.maxTextureBindings	  = limits.maxPerStageDescriptorStorageImages;
				_caps.supportsComputeShaders  = true;
				_caps.supportsAsyncTransfer	  = false;
				_caps.supportsMultiQueue	  = false;
				_timestampPeriod			  = limits.timestampPeriod;
				_maxPushConstantSize		  = limits.maxPushConstantsSize;
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

	VkDeviceCreateInfo createInfo						= {};
	createInfo.sType									= VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.queueCreateInfoCount						= 1;
	createInfo.pQueueCreateInfos						= &queueCreateInfo;
	createInfo.pEnabledFeatures							= &deviceFeatures;
	createInfo.enabledExtensionCount					= 0;
	createInfo.enabledLayerCount						= 0;

	VkResult result = vkCreateDevice(_physicalDevice, &createInfo, nullptr, &_device);
	CheckVkResult(result, "vkCreateDevice");

	// Get the compute queue
	vkGetDeviceQueue(_device, _computeQueueFamilyIndex, 0, &_computeQueue);
}

void VulkanBackend::CreateCommandPool() {
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
	VkDescriptorPoolSize poolSizes[3]	= {};

	// Storage buffers
	poolSizes[0].type					= VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[0].descriptorCount		= MAX_BUFFER_BINDINGS * MAX_DESCRIPTOR_SETS;

	// Storage images
	poolSizes[1].type					= VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[1].descriptorCount		= MAX_TEXTURE_BINDINGS * MAX_DESCRIPTOR_SETS;

	// Sampled textures use combined image samplers.
	poolSizes[2].type					= VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[2].descriptorCount		= MAX_TEXTURE_BINDINGS * MAX_DESCRIPTOR_SETS;

	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType						= VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount				= 3;
	poolInfo.pPoolSizes					= poolSizes;
	poolInfo.maxSets					= MAX_DESCRIPTOR_SETS;
	poolInfo.flags						= VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

	VkResult result						= vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_descriptorPool);
	CheckVkResult(result, "vkCreateDescriptorPool");
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
}

void VulkanBackend::CreateQueryPool() {
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
	// Vulkan doesn't require explicit context making like OpenGL
	// But we ensure the command buffer is ready
	EnsureCommandBuffer();
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
	if (_submissionPending) {
		WaitForSubmittedWork();
	}

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType					   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags					   = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VkResult result					   = vkBeginCommandBuffer(_commandBuffer, &beginInfo);
	CheckVkResult(result, "vkBeginCommandBuffer");

	_commandBufferRecording = true;

	// Reset query pool indices
	_nextQueryIndex			= 0;
}

void VulkanBackend::EndCommandBuffer() {
	if (!_commandBufferRecording) {
		return;
	}

	VkResult result = vkEndCommandBuffer(_commandBuffer);
	CheckVkResult(result, "vkEndCommandBuffer");

	_commandBufferRecording = false;
}

void VulkanBackend::SubmitCommandBuffer(bool wait) {
	if (_submissionPending) {
		WaitForSubmittedWork();
	}

	VkResult result = vkResetFences(_device, 1, &_commandFence);
	CheckVkResult(result, "vkResetFences");

	VkSubmitInfo submitInfo		  = {};
	submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers	  = &_commandBuffer;

	result						  = vkQueueSubmit(_computeQueue, 1, &submitInfo, _commandFence);
	if (result != VK_SUCCESS) {
		throw std::runtime_error(std::string("vkQueueSubmit failed: ") + VkResultToString(result));
	}

	_submissionPending = true;

	if (wait) {
		WaitForSubmittedWork();
	}
}

void VulkanBackend::WaitForSubmittedWork() {
	if (!_submissionPending) {
		return;
	}

	VkResult result = vkWaitForFences(_device, 1, &_commandFence, VK_TRUE, UINT64_MAX);
	if (result != VK_SUCCESS) {
		throw std::runtime_error(std::string("vkWaitForFences failed: ") + VkResultToString(result));
	}
	result = vkResetCommandPool(_device, _commandPool, 0);
	CheckVkResult(result, "vkResetCommandPool");

	_submissionPending = false;
}

void VulkanBackend::EnsureNoPendingGpuWork() {
	if (_commandBufferRecording) {
		EndCommandBuffer();
		SubmitCommandBuffer(true);
		return;
	}

	if (_submissionPending) {
		WaitForSubmittedWork();
	}
}

void VulkanBackend::InvalidateAllDescriptorCaches() {
	std::vector<VkDescriptorSet> setsToFree;
	setsToFree.reserve(_descriptorSets.size());
	for (const auto &cache : _descriptorSets) {
		if (cache.set != nullptr) {
			setsToFree.push_back(cache.set);
		}
	}
	if (!setsToFree.empty()) {
		CheckVkResult(
			vkFreeDescriptorSets(_device, _descriptorPool, static_cast<uint32_t>(setsToFree.size()), setsToFree.data()),
			"vkFreeDescriptorSets");
	}
	_descriptorSets.clear();
}

void VulkanBackend::InvalidateDescriptorCachesForPipeline(PipelineHandle pipeline) {
	std::vector<VkDescriptorSet> setsToFree;
	auto						 eraseBegin =
		std::remove_if(_descriptorSets.begin(), _descriptorSets.end(), [&](const DescriptorSetCache &cache) {
			if (cache.pipeline == pipeline) {
				if (cache.set != nullptr) {
					setsToFree.push_back(cache.set);
				}
				return true;
			}
			return false;
		});
	if (!setsToFree.empty()) {
		CheckVkResult(
			vkFreeDescriptorSets(_device, _descriptorPool, static_cast<uint32_t>(setsToFree.size()), setsToFree.data()),
			"vkFreeDescriptorSets");
	}
	_descriptorSets.erase(eraseBegin, _descriptorSets.end());
}

void VulkanBackend::InvalidateDescriptorCachesForBuffer(BufferHandle buffer) {
	std::vector<VkDescriptorSet> setsToFree;
	auto						 eraseBegin =
		std::remove_if(_descriptorSets.begin(), _descriptorSets.end(), [&](const DescriptorSetCache &cache) {
			for (uint32_t i = 0; i < MAX_BUFFER_BINDINGS; ++i) {
				if ((cache.bufferMask & (1ull << i)) != 0 && cache.boundBuffers[i] == buffer) {
					if (cache.set != nullptr) {
						setsToFree.push_back(cache.set);
					}
					return true;
				}
			}
			return false;
		});
	if (!setsToFree.empty()) {
		CheckVkResult(
			vkFreeDescriptorSets(_device, _descriptorPool, static_cast<uint32_t>(setsToFree.size()), setsToFree.data()),
			"vkFreeDescriptorSets");
	}
	_descriptorSets.erase(eraseBegin, _descriptorSets.end());
}

void VulkanBackend::InvalidateDescriptorCachesForTexture(TextureHandle texture) {
	std::vector<VkDescriptorSet> setsToFree;
	auto						 eraseBegin =
		std::remove_if(_descriptorSets.begin(), _descriptorSets.end(), [&](const DescriptorSetCache &cache) {
			for (uint32_t i = 0; i < MAX_TEXTURE_BINDINGS; ++i) {
				if ((cache.textureMask & (1ull << i)) != 0 && cache.boundTextures[i] == texture) {
					if (cache.set != nullptr) {
						setsToFree.push_back(cache.set);
					}
					return true;
				}
			}
			return false;
		});
	if (!setsToFree.empty()) {
		CheckVkResult(
			vkFreeDescriptorSets(_device, _descriptorPool, static_cast<uint32_t>(setsToFree.size()), setsToFree.data()),
			"vkFreeDescriptorSets");
	}
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
	bufferInfo.usage =
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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
	if (offset + size > it->second.size) {
		throw std::runtime_error("UploadBuffer range exceeds buffer size");
	}

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
	if (offset + size > it->second.size) {
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

	if (read) {
		EnsureNoPendingGpuWork();
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
	if (desc.depth != 1) {
		throw std::runtime_error("Vulkan backend currently supports only 2D textures (depth must be 1)");
	}

	VkFormat		  format	= GetVkFormat(desc.format);

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType				= VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType			= VK_IMAGE_TYPE_2D;
	imageInfo.extent.width		= desc.width;
	imageInfo.extent.height		= desc.height;
	imageInfo.extent.depth		= 1;
	imageInfo.mipLevels			= 1;
	imageInfo.arrayLayers		= 1;
	imageInfo.format			= format;
	imageInfo.tiling			= VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout		= VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
					  VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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
	viewInfo.viewType						 = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format							 = format;
	viewInfo.subresourceRange.aspectMask	 = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel	 = 0;
	viewInfo.subresourceRange.levelCount	 = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount	 = 1;

	VkImageView view						 = nullptr;
	result									 = vkCreateImageView(_device, &viewInfo, nullptr, &view);
	CheckVkResult(result, "vkCreateImageView");

	TextureHandle handle = _nextTextureHandle++;
	TextureInfo	  info;
	info.image		   = image;
	info.memory		   = memory;
	info.view		   = view;
	info.width		   = desc.width;
	info.height		   = desc.height;
	info.depth		   = 1;
	info.format		   = desc.format;
	info.vkFormat	   = format;
	info.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	_textures[handle]  = info;

	// Upload initial data if provided
	if (desc.initialData) {
		UploadTextureInternal(_textures[handle], 0, 0, desc.width, desc.height, desc.initialData);
	}

	return handle;
}

void VulkanBackend::DestroyTexture(TextureHandle texture) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _textures.find(texture);
	if (it == _textures.end()) {
		return;
	}

	EnsureNoPendingGpuWork();
	InvalidateDescriptorCachesForTexture(texture);

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

	UploadTextureInternal(it->second, x, y, width, height, data);
}

void VulkanBackend::UploadTextureInternal(TextureInfo &info, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
										  const void *data) {
	if (data == nullptr && (width != 0 || height != 0)) {
		throw std::runtime_error("UploadTexture received null data");
	}
	if (x + width > info.width || y + height > info.height) {
		throw std::runtime_error("UploadTexture region exceeds texture bounds");
	}
	if (width == 0 || height == 0) {
		return;
	}

	// Calculate data size based on format
	uint32_t pixelSize = 4; // Default to 4 bytes
	switch (info.format) {
	case PixelFormat::R8:
		pixelSize = 1;
		break;
	case PixelFormat::RG8:
		pixelSize = 2;
		break;
	case PixelFormat::RGBA8:
		pixelSize = 4;
		break;
	case PixelFormat::R32F:
		pixelSize = 4;
		break;
	case PixelFormat::RG32F:
		pixelSize = 8;
		break;
	case PixelFormat::RGBA32F:
		pixelSize = 16;
		break;
	case PixelFormat::R16F:
		pixelSize = 2;
		break;
	case PixelFormat::RG16F:
		pixelSize = 4;
		break;
	case PixelFormat::RGBA16F:
		pixelSize = 8;
		break;
	case PixelFormat::R32I:
		pixelSize = 4;
		break;
	case PixelFormat::RG32I:
		pixelSize = 8;
		break;
	case PixelFormat::RGBA32I:
		pixelSize = 16;
		break;
	case PixelFormat::R32UI:
		pixelSize = 4;
		break;
	case PixelFormat::RG32UI:
		pixelSize = 8;
		break;
	case PixelFormat::RGBA32UI:
		pixelSize = 16;
		break;
	}

	size_t			   dataSize	   = width * height * pixelSize;

	// Create staging buffer
	VkBufferCreateInfo stagingInfo = {};
	stagingInfo.sType			   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingInfo.size			   = dataSize;
	stagingInfo.usage			   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	stagingInfo.sharingMode		   = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer stagingBuffer		   = nullptr;
	VkResult result				   = vkCreateBuffer(_device, &stagingInfo, nullptr, &stagingBuffer);
	CheckVkResult(result, "vkCreateBuffer (staging)");

	VkDeviceMemory stagingMemory = nullptr;
	AllocateBufferMemory(stagingBuffer, stagingMemory,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, dataSize);

	result = vkBindBufferMemory(_device, stagingBuffer, stagingMemory, 0);
	CheckVkResult(result, "vkBindBufferMemory (staging)");

	// Copy data to staging buffer
	void *mapped = nullptr;
	result		 = vkMapMemory(_device, stagingMemory, 0, dataSize, 0, &mapped);
	CheckVkResult(result, "vkMapMemory");
	std::memcpy(mapped, data, dataSize);
	vkUnmapMemory(_device, stagingMemory);

	EnsureCommandBuffer();

	TransitionTexture(info, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
					  VK_ACCESS_TRANSFER_WRITE_BIT);

	// Copy buffer to image
	VkBufferImageCopy region			   = {};
	region.bufferOffset					   = 0;
	region.bufferRowLength				   = 0;
	region.bufferImageHeight			   = 0;
	region.imageSubresource.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel	   = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount	   = 1;
	region.imageOffset					   = {static_cast<int32_t>(x), static_cast<int32_t>(y), 0};
	region.imageExtent					   = {width, height, 1};

	vkCmdCopyBufferToImage(_commandBuffer, stagingBuffer, info.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	TransitionTexture(info, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

	EndCommandBuffer();
	SubmitCommandBuffer(true);

	// Cleanup staging
	vkDestroyBuffer(_device, stagingBuffer, nullptr);
	vkFreeMemory(_device, stagingMemory, nullptr);
}

void VulkanBackend::DownloadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
									void *outData) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _textures.find(texture);
	if (it == _textures.end()) {
		throw std::runtime_error("Invalid texture handle");
	}
	if (outData == nullptr && (width != 0 || height != 0)) {
		throw std::runtime_error("DownloadTexture received null output pointer");
	}
	if (x + width > it->second.width || y + height > it->second.height) {
		throw std::runtime_error("DownloadTexture region exceeds texture bounds");
	}
	if (width == 0 || height == 0) {
		return;
	}

	// Flush pending commands
	if (_commandBufferRecording) {
		EndCommandBuffer();
		SubmitCommandBuffer(true);
	}

	// Calculate data size
	uint32_t pixelSize = 4;
	switch (it->second.format) {
	case PixelFormat::R8:
		pixelSize = 1;
		break;
	case PixelFormat::RG8:
		pixelSize = 2;
		break;
	case PixelFormat::RGBA8:
		pixelSize = 4;
		break;
	case PixelFormat::R32F:
		pixelSize = 4;
		break;
	case PixelFormat::RG32F:
		pixelSize = 8;
		break;
	case PixelFormat::RGBA32F:
		pixelSize = 16;
		break;
	case PixelFormat::R16F:
		pixelSize = 2;
		break;
	case PixelFormat::RG16F:
		pixelSize = 4;
		break;
	case PixelFormat::RGBA16F:
		pixelSize = 8;
		break;
	case PixelFormat::R32I:
		pixelSize = 4;
		break;
	case PixelFormat::RG32I:
		pixelSize = 8;
		break;
	case PixelFormat::RGBA32I:
		pixelSize = 16;
		break;
	case PixelFormat::R32UI:
		pixelSize = 4;
		break;
	case PixelFormat::RG32UI:
		pixelSize = 8;
		break;
	case PixelFormat::RGBA32UI:
		pixelSize = 16;
		break;
	}

	size_t			   dataSize	   = width * height * pixelSize;

	// Create staging buffer
	VkBufferCreateInfo stagingInfo = {};
	stagingInfo.sType			   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingInfo.size			   = dataSize;
	stagingInfo.usage			   = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	stagingInfo.sharingMode		   = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer stagingBuffer		   = nullptr;
	VkResult result				   = vkCreateBuffer(_device, &stagingInfo, nullptr, &stagingBuffer);
	CheckVkResult(result, "vkCreateBuffer (staging)");

	VkDeviceMemory stagingMemory = nullptr;
	AllocateBufferMemory(stagingBuffer, stagingMemory,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, dataSize);

	result = vkBindBufferMemory(_device, stagingBuffer, stagingMemory, 0);
	CheckVkResult(result, "vkBindBufferMemory (staging)");

	EnsureCommandBuffer();

	TransitionTexture(it->second, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
					  VK_ACCESS_TRANSFER_READ_BIT);

	// Copy image to buffer
	VkBufferImageCopy region			   = {};
	region.bufferOffset					   = 0;
	region.bufferRowLength				   = 0;
	region.bufferImageHeight			   = 0;
	region.imageSubresource.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel	   = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount	   = 1;
	region.imageOffset					   = {static_cast<int32_t>(x), static_cast<int32_t>(y), 0};
	region.imageExtent					   = {width, height, 1};

	vkCmdCopyImageToBuffer(_commandBuffer, it->second.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, 1,
						   &region);

	TransitionTexture(it->second, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

	EndCommandBuffer();
	SubmitCommandBuffer(true);

	// Map and copy data
	void *mapped = nullptr;
	result		 = vkMapMemory(_device, stagingMemory, 0, dataSize, 0, &mapped);
	CheckVkResult(result, "vkMapMemory");
	std::memcpy(outData, mapped, dataSize);
	vkUnmapMemory(_device, stagingMemory);

	// Cleanup
	vkDestroyBuffer(_device, stagingBuffer, nullptr);
	vkFreeMemory(_device, stagingMemory, nullptr);
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
	case PixelFormat::RGBA32I:
		return VK_FORMAT_R32G32B32A32_SINT;
	case PixelFormat::R32UI:
		return VK_FORMAT_R32_UINT;
	case PixelFormat::RG32UI:
		return VK_FORMAT_R32G32_UINT;
	case PixelFormat::RGBA32UI:
		return VK_FORMAT_R32G32B32A32_UINT;
	default:
		return VK_FORMAT_R8G8B8A8_UNORM;
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

// =============================================================================
// SPIR-V Compilation using glslang
// =============================================================================

std::vector<uint32_t> VulkanBackend::CompileGLSLToSPIRV(const std::string &glslSource, ShaderType type) {
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

	return spirv;
}

// =============================================================================
// Shader Management
// =============================================================================

ShaderHandle VulkanBackend::CreateShader(const ShaderDesc &desc) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
	}

	// Compile GLSL to SPIR-V
	std::vector<uint32_t>	 spirv		= CompileGLSLToSPIRV(desc.sourceCode, desc.type);

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
		binding.stageFlags					 = VK_SHADER_STAGE_COMPUTE_BIT;
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

void VulkanBackend::TransitionTexture(TextureInfo &info, VkImageLayout newLayout, VkPipelineStageFlags dstStage,
									  VkAccessFlags dstAccess) {
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
	barrier.subresourceRange.aspectMask		= VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel	= 0;
	barrier.subresourceRange.levelCount		= 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount		= 1;
	barrier.srcAccessMask					= info.lastAccess;
	barrier.dstAccessMask					= dstAccess;

	vkCmdPipelineBarrier(_commandBuffer, info.lastStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	info.currentLayout = newLayout;
	info.lastStage	   = dstStage;
	info.lastAccess	   = dstAccess;
}

void VulkanBackend::BindPipeline(PipelineHandle pipeline) {
	std::lock_guard<std::mutex> lock(_mutex);

	auto						it = _pipelines.find(pipeline);
	if (it == _pipelines.end()) {
		throw std::runtime_error("Invalid pipeline handle");
	}

	EnsureCommandBuffer();

	vkCmdBindPipeline(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, it->second.pipeline);

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
	vkCmdPushConstants(_commandBuffer, it->second.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(size),
					   data);
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

	for (const auto &resource : pipelineInfo.resources) {
		if (resource.type == BindingType::Buffer) {
			const BufferHandle handle	= cache.boundBuffers[resource.binding];
			auto			   bufferIt = _buffers.find(handle);
			if (bufferIt == _buffers.end()) {
				throw std::runtime_error("Descriptor cache references an invalid buffer handle");
			}

			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer				  = bufferIt->second.buffer;
			bufferInfo.offset				  = 0;
			bufferInfo.range				  = bufferIt->second.size;
			bufferInfos.push_back(bufferInfo);

			VkWriteDescriptorSet write = {};
			write.sType				   = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet			   = cache.set;
			write.dstBinding		   = resource.binding;
			write.descriptorType	   = GetVkDescriptorType(resource);
			write.descriptorCount	   = 1;
			write.pBufferInfo		   = &bufferInfos.back();
			descriptorWrites.push_back(write);
		} else if (resource.type == BindingType::Texture) {
			const TextureHandle handle	  = cache.boundTextures[resource.binding];
			auto				textureIt = _textures.find(handle);
			if (textureIt == _textures.end()) {
				throw std::runtime_error("Descriptor cache references an invalid texture handle");
			}

			VkDescriptorImageInfo imageInfo = {};
			imageInfo.imageView				= textureIt->second.view;
			imageInfo.imageLayout			= VK_IMAGE_LAYOUT_GENERAL;
			imageInfos.push_back(imageInfo);

			VkWriteDescriptorSet write = {};
			write.sType				   = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet			   = cache.set;
			write.dstBinding		   = resource.binding;
			write.descriptorType	   = GetVkDescriptorType(resource);
			write.descriptorCount	   = 1;
			write.pImageInfo		   = &imageInfos.back();
			descriptorWrites.push_back(write);
		} else if (resource.type == BindingType::Sampler) {
			const TextureHandle handle	  = cache.boundTextures[resource.binding];
			auto				textureIt = _textures.find(handle);
			if (textureIt == _textures.end()) {
				throw std::runtime_error("Descriptor cache references an invalid sampled texture handle");
			}

			VkDescriptorImageInfo imageInfo = {};
			imageInfo.sampler				= _defaultSampler;
			imageInfo.imageView				= textureIt->second.view;
			imageInfo.imageLayout			= VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfos.push_back(imageInfo);

			VkWriteDescriptorSet write = {};
			write.sType				   = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet			   = cache.set;
			write.dstBinding		   = resource.binding;
			write.descriptorType	   = GetVkDescriptorType(resource);
			write.descriptorCount	   = 1;
			write.pImageInfo		   = &imageInfos.back();
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
				cache.boundReadOnly[i] != requested.boundReadOnly[i]) {
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
		EnsureNoPendingGpuWork();
		InvalidateAllDescriptorCaches();
		CheckVkResult(vkResetDescriptorPool(_device, _descriptorPool, 0), "vkResetDescriptorPool");
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
				TransitionTexture(it->second, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
								  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
			} else {
				const VkAccessFlags shaderAccess = layoutIt->readOnly
													   ? VK_ACCESS_SHADER_READ_BIT
													   : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
				TransitionTexture(it->second, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								  shaderAccess);
			}
		}
	}

	for (size_t i = 0; i < seenResources.size(); ++i) {
		if (!seenResources[i]) {
			throw std::runtime_error("Not all pipeline resources were provided to BindResources");
		}
	}

	DescriptorSetCache *cache = FindOrCreateDescriptorSet(bindings, count);

	// Bind descriptor set
	vkCmdBindDescriptorSets(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineInfo.layout, 0, 1, &cache->set, 0,
							nullptr);
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

	VkPipelineStageFlags srcStage	   = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	VkPipelineStageFlags dstStage	   = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	VkMemoryBarrier		 memoryBarrier = {};
	memoryBarrier.sType				   = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memoryBarrier.srcAccessMask		   = 0;
	memoryBarrier.dstAccessMask		   = 0;

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
		SubmitCommandBuffer(true);
	} else if (_submissionPending) {
		WaitForSubmittedWork();
	} else {
		vkDeviceWaitIdle(_device);
	}
}

// =============================================================================
// Query / Timing
// =============================================================================

uint32_t VulkanBackend::BeginQuery() {
	std::lock_guard<std::mutex> lock(_mutex);

	EnsureCommandBuffer();

	uint32_t queryIndex = _nextQueryIndex;
	if (queryIndex >= MAX_QUERIES * 2 - 1) {
		// Query pool exhausted, reset and start over
		// In a real implementation, you'd want to flush and get results first
		_nextQueryIndex = 0;
		queryIndex		= 0;
	}

	vkCmdResetQueryPool(_commandBuffer, _queryPool, queryIndex, 2);
	vkCmdWriteTimestamp(_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, _queryPool, queryIndex);

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

	if (query == 0) {
		return 0;
	}
	const uint32_t querySlot = query - 1;
	if (querySlot >= _queries.size() || !_queries[querySlot].active) {
		return 0;
	}

	EnsureCommandBuffer();

	uint32_t queryIndex = _queries[querySlot].queryIndex;
	vkCmdWriteTimestamp(_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, _queryPool, queryIndex + 1);

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
	(void)format; // Vulkan doesn't use format parameter

	std::lock_guard<std::mutex> lock(_mutex);

	if (!_initialized) {
		throw std::runtime_error("Vulkan backend not initialized");
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
		binding.stageFlags					 = VK_SHADER_STAGE_COMPUTE_BIT;
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

	// Get pipeline cache data from our persistent cache
	// Note: In Vulkan, pipeline binaries are obtained from the pipeline cache,
	// not individual pipelines. We merge pipeline data into our persistent cache
	// during creation.
	size_t	 cacheSize = 0;
	VkResult result	   = vkGetPipelineCacheData(_device, _pipelineCache, &cacheSize, nullptr);
	if (result != VK_SUCCESS || cacheSize == 0) {
		return {};
	}

	std::vector<uint8_t> data(cacheSize);
	result = vkGetPipelineCacheData(_device, _pipelineCache, &cacheSize, data.data());
	if (result != VK_SUCCESS) {
		return {};
	}

	data.resize(cacheSize);

	// Use Vulkan pipeline cache header UUID as format identifier
	if (cacheSize >= sizeof(VkPipelineCacheHeaderVersionOne)) {
		auto *header = reinterpret_cast<const VkPipelineCacheHeaderVersionOne *>(data.data());
		// Create a simple hash from the first few bytes of UUID
		format		 = *reinterpret_cast<const uint32_t *>(header->pipelineCacheUUID);
	}

	return data;
}

bool VulkanBackend::SupportsPipelineCache() const {
	return _initialized && _pipelineCache != nullptr;
}

uint32_t VulkanBackend::GetPipelineCacheFormat() const {
	if (!_initialized || !_physicalDevice) {
		return 0;
	}

	// Use device properties pipelineCacheUUID as format identifier
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(_physicalDevice, &props);
	return *reinterpret_cast<const uint32_t *>(props.pipelineCacheUUID);
}

// =============================================================================
// Backend Factory
// =============================================================================

Backend *CreateVulkanBackend() {
	return new VulkanBackend();
}

} // namespace GPU::Backend
