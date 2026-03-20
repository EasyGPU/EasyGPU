#pragma once

/**
 * @file VulkanBackend.h
 * @brief Vulkan implementation of the Backend interface
 */

#ifndef EASYGPU_VULKANBACKEND_H
#define EASYGPU_VULKANBACKEND_H

#include <Backend/Backend.h>

#include <array>
#include <mutex>
#include <unordered_map>
#include <vector>

// Include Vulkan header
#include <vulkan/vulkan.h>

namespace GPU::Backend {

constexpr uint32_t MAX_BUFFER_BINDINGS	= 32;
constexpr uint32_t MAX_TEXTURE_BINDINGS = 32;
constexpr uint32_t MAX_DESCRIPTOR_SETS	= 1024;
constexpr uint32_t MAX_QUERIES			= 256;

/**
 * @brief Vulkan backend implementation
 *
 * This backend uses Vulkan 1.1+ compute shaders for GPU compute operations.
 * GLSL shaders are compiled to SPIR-V using glslang at runtime.
 */
class VulkanBackend : public Backend {
public:
	VulkanBackend();
	~VulkanBackend() override;

	VulkanBackend(const VulkanBackend &)			= delete;
	VulkanBackend &operator=(const VulkanBackend &) = delete;
	VulkanBackend(VulkanBackend &&)					= delete;
	VulkanBackend &operator=(VulkanBackend &&)		= delete;

	void		   Initialize() override;
	void		   Shutdown() override;
	bool		   IsInitialized() const override;
	void		   MakeCurrent() override;
	void		   MakeNoneCurrent() override;
	BackendCaps	   GetCaps() const override;

	BufferHandle   CreateBuffer(const BufferDesc &desc) override;
	void		   DestroyBuffer(BufferHandle buffer) override;
	void		   UploadBuffer(BufferHandle buffer, size_t offset, size_t size, const void *data) override;
	void		   DownloadBuffer(BufferHandle buffer, size_t offset, size_t size, void *outData) override;
	void		  *MapBuffer(BufferHandle buffer, bool read, bool write) override;
	void		   UnmapBuffer(BufferHandle buffer) override;

	TextureHandle  CreateTexture(const TextureDesc &desc) override;
	void		   DestroyTexture(TextureHandle texture) override;
	void		   UploadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
								 const void *data) override;
	void		   DownloadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
								   void *outData) override;

	ShaderHandle   CreateShader(const ShaderDesc &desc) override;
	void		   DestroyShader(ShaderHandle shader) override;

	PipelineHandle CreatePipeline(const PipelineDesc &desc) override;
	void		   DestroyPipeline(PipelineHandle pipeline) override;

	void		   BindPipeline(PipelineHandle pipeline) override;
	void		   BindResources(const ResourceBinding *bindings, uint32_t count) override;
	void		   SetUniform(PipelineHandle pipeline, const std::string &name, const std::string &type,
							  const void *data) override;
	void		   SetUniformData(PipelineHandle pipeline, const void *data, size_t size) override;
	void		   Dispatch(uint32_t groupX, uint32_t groupY, uint32_t groupZ) override;
	void		   MemoryBarrier(BarrierType barrierType) override;
	void		   Finish() override;

	uint32_t	   BeginQuery() override;
	uint64_t	   EndQuery(uint32_t query) override;

private:
	// Vulkan resource info structures
	struct BufferInfo {
		VkBuffer			  buffer			 = nullptr;
		VkDeviceMemory		  memory			 = nullptr;
		VkBuffer			  stagingBuffer		 = nullptr;
		VkDeviceMemory		  stagingMemory		 = nullptr;
		size_t				  size				 = 0;
		void				 *mappedPtr			 = nullptr;
		BufferMode			  mode				 = BufferMode::ReadWrite;
		bool				  isMapped			 = false;
		bool				  mappedForRead		 = false;
		bool				  mappedForWrite	 = false;
		VkMemoryPropertyFlags memoryFlags		 = 0;
		VkMemoryPropertyFlags stagingMemoryFlags = 0;
	};

	struct TextureInfo {
		VkImage				 image		   = nullptr;
		VkDeviceMemory		 memory		   = nullptr;
		VkImageView			 view		   = nullptr;
		uint32_t			 width		   = 0;
		uint32_t			 height		   = 0;
		uint32_t			 depth		   = 1;
		PixelFormat			 format		   = PixelFormat::RGBA8;
		VkFormat			 vkFormat	   = VK_FORMAT_UNDEFINED;
		VkImageLayout		 currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VkPipelineStageFlags lastStage	   = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkAccessFlags		 lastAccess	   = 0;
	};

	struct ShaderInfo {
		VkShaderModule		  module = nullptr;
		ShaderType			  type	 = ShaderType::Compute;
		std::vector<uint32_t> spirvCode; // Cached SPIR-V code
	};

	struct PipelineInfo {
		VkPipeline						 pipeline			 = nullptr;
		VkPipelineLayout				 layout				 = nullptr;
		VkDescriptorSetLayout			 descriptorSetLayout = nullptr;
		uint32_t						 workGroupSizeX		 = 1;
		uint32_t						 workGroupSizeY		 = 1;
		uint32_t						 workGroupSizeZ		 = 1;
		uint32_t						 pushConstantSize	 = 0;
		std::vector<ResourceLayoutEntry> resources;
	};

	struct QueryInfo {
		uint32_t queryIndex = 0;
		bool	 active		= false;
		uint64_t result		= 0;
	};

	struct DescriptorSetCache {
		PipelineHandle									pipeline = INVALID_PIPELINE_HANDLE;
		VkDescriptorSet									set		 = nullptr;
		std::array<BufferHandle, MAX_BUFFER_BINDINGS>	boundBuffers{};
		std::array<TextureHandle, MAX_TEXTURE_BINDINGS> boundTextures{};
		std::array<BindingType, MAX_TEXTURE_BINDINGS>	boundTextureTypes{};
		std::array<PixelFormat, MAX_TEXTURE_BINDINGS>	boundFormats{};
		std::array<bool, MAX_TEXTURE_BINDINGS>			boundReadOnly{};
		uint64_t										bufferMask	= 0;
		uint64_t										textureMask = 0;
	};

	// Internal helpers
	void	 UploadBufferInternal(VkBuffer buffer, size_t size, const void *data);
	void	 UploadTextureInternal(TextureInfo &info, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
								   const void *data);
	void	 EnsureNoPendingGpuWork();
	void	 TransitionTexture(TextureInfo &info, VkImageLayout newLayout, VkPipelineStageFlags dstStage,
							   VkAccessFlags dstAccess);
	void	 InvalidateAllDescriptorCaches();
	void	 InvalidateDescriptorCachesForPipeline(PipelineHandle pipeline);
	void	 InvalidateDescriptorCachesForBuffer(BufferHandle buffer);
	void	 InvalidateDescriptorCachesForTexture(TextureHandle texture);

	// Initialization helpers
	void	 CreateInstance();
	void	 SelectPhysicalDevice();
	void	 CreateDevice();
	void	 CreateCommandPool();
	void	 CreateDescriptorPool();
	void	 CreateQueryPool();
	void	 CreateDefaultSampler();

	// Cleanup helpers
	void	 CleanupVulkan();

	// Command buffer management
	void	 BeginCommandBuffer();
	void	 EndCommandBuffer();
	void	 SubmitCommandBuffer(bool wait = false);
	void	 EnsureCommandBuffer();
	void	 WaitForSubmittedWork();

	// Memory management
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	void AllocateBufferMemory(VkBuffer buffer, VkDeviceMemory &memory, VkMemoryPropertyFlags properties, size_t size);
	void AllocateImageMemory(VkImage image, VkDeviceMemory &memory, VkMemoryPropertyFlags properties);

	// Format conversions
	static VkFormat			  GetVkFormat(PixelFormat format);
	static VkDescriptorType	  GetVkDescriptorType(BindingType type);
	static VkImageLayout	  GetVkImageLayout(PixelFormat format, bool readOnly);
	static VkShaderStageFlags GetVkShaderStage(ShaderType type);

	// SPIR-V compilation
	std::vector<uint32_t>	  CompileGLSLToSPIRV(const std::string &glslSource, ShaderType type);

	// Barrier helpers
	void InsertBufferBarrier(VkBuffer buffer, size_t offset, size_t size, VkPipelineStageFlags srcStage,
							 VkPipelineStageFlags dstStage, VkAccessFlags srcAccess, VkAccessFlags dstAccess);
	void InsertImageBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
							VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkAccessFlags srcAccess,
							VkAccessFlags dstAccess);
	static VkDescriptorType GetVkDescriptorType(const ResourceLayoutEntry &entry);

	// Descriptor set management
	void					UpdateDescriptorSet(const DescriptorSetCache &cache);
	DescriptorSetCache	   *FindOrCreateDescriptorSet(const ResourceBinding *bindings, uint32_t count);

private:
	// Vulkan handles
	VkInstance										 _instance				  = nullptr;
	VkPhysicalDevice								 _physicalDevice		  = nullptr;
	VkDevice										 _device				  = nullptr;
	VkQueue											 _computeQueue			  = nullptr;
	uint32_t										 _computeQueueFamilyIndex = 0;

	// Command resources
	VkCommandPool									 _commandPool			  = nullptr;
	VkCommandBuffer									 _commandBuffer			  = nullptr;
	VkFence											 _commandFence			  = nullptr;
	bool											 _commandBufferRecording  = false;
	bool											 _submissionPending		  = false;

	// Descriptor resources
	VkDescriptorPool								 _descriptorPool		  = nullptr;
	VkSampler										 _defaultSampler		  = nullptr;
	std::vector<VkDescriptorSet>					 _inFlightDescriptorSets;

	// Query pool for timing
	VkQueryPool										 _queryPool		 = nullptr;
	uint32_t										 _nextQueryIndex = 0;

	// Resource maps
	std::unordered_map<BufferHandle, BufferInfo>	 _buffers;
	std::unordered_map<TextureHandle, TextureInfo>	 _textures;
	std::unordered_map<ShaderHandle, ShaderInfo>	 _shaders;
	std::unordered_map<PipelineHandle, PipelineInfo> _pipelines;
	std::vector<QueryInfo>							 _queries;
	std::vector<DescriptorSetCache>					 _descriptorSets;

	// Handle counters
	BufferHandle									 _nextBufferHandle	 = 1;
	TextureHandle									 _nextTextureHandle	 = 1;
	ShaderHandle									 _nextShaderHandle	 = 1;
	PipelineHandle									 _nextPipelineHandle = 1;

	// State
	bool											 _initialized		 = false;
	PipelineHandle									 _currentPipeline	 = INVALID_PIPELINE_HANDLE;

	// Capabilities
	BackendCaps										 _caps;
	float											 _timestampPeriod	  = 1.0f;
	uint32_t										 _maxPushConstantSize = 0;

	// Thread safety
	std::mutex										 _mutex;
	bool											 _isCurrent = false;
};

Backend *CreateVulkanBackend();

} // namespace GPU::Backend

#endif // EASYGPU_VULKANBACKEND_H
