#pragma once

/**
 * @file VulkanBackend.h
 * @brief Vulkan implementation of the Backend interface.
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

constexpr uint32_t MAX_DESCRIPTOR_SETS = 1024;
constexpr uint32_t MAX_QUERIES		   = 256;

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

	/** @copydoc Backend::Initialize */
	void		   Initialize() override;
	/** @copydoc Backend::Shutdown */
	void		   Shutdown() override;
	/** @copydoc Backend::IsInitialized */
	bool		   IsInitialized() const override;
	/** @copydoc Backend::MakeCurrent */
	void		   MakeCurrent() override;
	/** @copydoc Backend::MakeNoneCurrent */
	void		   MakeNoneCurrent() override;
	/** @copydoc Backend::GetCaps */
	BackendCaps	   GetCaps() const override;

	/** @copydoc Backend::CreateBuffer */
	BufferHandle   CreateBuffer(const BufferDesc &desc) override;
	/** @copydoc Backend::DestroyBuffer */
	void		   DestroyBuffer(BufferHandle buffer) override;
	/** @copydoc Backend::UploadBuffer */
	void		   UploadBuffer(BufferHandle buffer, size_t offset, size_t size, const void *data) override;
	/** @copydoc Backend::DownloadBuffer */
	void		   DownloadBuffer(BufferHandle buffer, size_t offset, size_t size, void *outData) override;
	/** @copydoc Backend::MapBuffer */
	void		  *MapBuffer(BufferHandle buffer, bool read, bool write) override;
	/** @copydoc Backend::UnmapBuffer */
	void		   UnmapBuffer(BufferHandle buffer) override;

	/** @copydoc Backend::CreateTexture */
	TextureHandle  CreateTexture(const TextureDesc &desc) override;
	/** @copydoc Backend::DestroyTexture */
	void		   DestroyTexture(TextureHandle texture) override;
	/** @copydoc Backend::UploadTexture */
	void		   UploadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
								 const void *data) override;
	/** @copydoc Backend::UploadTexture3D */
	void UploadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height,
						 uint32_t depth, const void *data) override;
	/** @copydoc Backend::DownloadTexture */
	void DownloadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
						 void *outData) override;
	/** @copydoc Backend::DownloadTexture3D */
	void DownloadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height,
						   uint32_t depth, void *outData) override;

	/** @copydoc Backend::CreateShader */
	ShaderHandle		 CreateShader(const ShaderDesc &desc) override;
	/** @copydoc Backend::DestroyShader */
	void				 DestroyShader(ShaderHandle shader) override;

	/** @copydoc Backend::CreatePipeline */
	PipelineHandle		 CreatePipeline(const PipelineDesc &desc) override;
	/** @copydoc Backend::DestroyPipeline */
	void				 DestroyPipeline(PipelineHandle pipeline) override;

	/** @copydoc Backend::BindPipeline */
	void				 BindPipeline(PipelineHandle pipeline) override;
	/** @copydoc Backend::BindResources */
	void				 BindResources(const ResourceBinding *bindings, uint32_t count) override;
	/** @copydoc Backend::SetUniform */
	void				 SetUniform(PipelineHandle pipeline, const std::string &name, const std::string &type,
									const void *data) override;
	/** @copydoc Backend::SetUniformData */
	void				 SetUniformData(PipelineHandle pipeline, const void *data, size_t size) override;
	/** @copydoc Backend::Dispatch */
	void				 Dispatch(uint32_t groupX, uint32_t groupY, uint32_t groupZ) override;
	/** @copydoc Backend::MemoryBarrier */
	void				 MemoryBarrier(BarrierType barrierType) override;
	/** @copydoc Backend::Finish */
	void				 Finish() override;

	/** @copydoc Backend::BeginQuery */
	uint32_t			 BeginQuery() override;
	/** @copydoc Backend::EndQuery */
	uint64_t			 EndQuery(uint32_t query) override;

	/** @copydoc Backend::CreatePipelineFromBinary */
	PipelineHandle		 CreatePipelineFromBinary(const PipelineDesc &desc, const void *binaryData, size_t binarySize,
												  uint32_t format) override;
	/** @copydoc Backend::GetPipelineBinary */
	std::vector<uint8_t> GetPipelineBinary(PipelineHandle pipeline, uint32_t &format) override;
	/** @copydoc Backend::SupportsPipelineCache */
	bool				 SupportsPipelineCache() const override;
	/** @copydoc Backend::GetPipelineCacheFormat */
	uint32_t			 GetPipelineCacheFormat() const override;

	/** @copydoc Backend::GetType */
	BackendType			 GetType() const override {
		 return BackendType::Vulkan;
	}

private:
	/** @brief Internal Vulkan buffer resource information. */
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

	/** @brief Internal Vulkan texture resource information. */
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

	/** @brief Internal Vulkan shader resource information. */
	struct ShaderInfo {
		VkShaderModule		  module = nullptr;
		ShaderType			  type	 = ShaderType::Compute;
		std::vector<uint32_t> spirvCode; // Cached SPIR-V code
	};

	/** @brief Internal Vulkan pipeline resource information. */
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

	/** @brief Internal Vulkan query resource information. */
	struct QueryInfo {
		uint32_t queryIndex = 0;
		bool	 active		= false;
		uint64_t result		= 0;
	};

	/** @brief Cached Vulkan descriptor set with bound resource state. */
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

	/**
	 * @brief Upload data to a Vulkan buffer using a staging buffer.
	 * @param buffer Destination Vulkan buffer.
	 * @param size Size in bytes.
	 * @param data Source data pointer.
	 */
	void UploadBufferInternal(VkBuffer buffer, size_t size, const void *data);
	/**
	 * @brief Upload voxel data to a texture using a staging buffer.
	 * @param info Texture info structure.
	 * @param x Destination x offset.
	 * @param y Destination y offset.
	 * @param z Destination z offset.
	 * @param width Region width.
	 * @param height Region height.
	 * @param depth Region depth.
	 * @param data Source voxel data.
	 */
	void UploadTextureInternal(TextureInfo &info, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height,
							   uint32_t depth, const void *data);
	/**
	 * @brief Download voxel data from a texture using a staging buffer.
	 * @param info Texture info structure.
	 * @param x Source x offset.
	 * @param y Source y offset.
	 * @param z Source z offset.
	 * @param width Region width.
	 * @param height Region height.
	 * @param depth Region depth.
	 * @param outData Destination voxel buffer.
	 */
	void DownloadTextureInternal(TextureInfo &info, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height,
								 uint32_t depth, void *outData);
	/** @brief Wait for all pending GPU work to finish. */
	void EnsureNoPendingGpuWork();
	/**
	 * @brief Transition a texture to a new image layout.
	 * @param info Texture info structure.
	 * @param newLayout Target image layout.
	 * @param dstStage Destination pipeline stage.
	 * @param dstAccess Destination access flags.
	 */
	void TransitionTexture(TextureInfo &info, VkImageLayout newLayout, VkPipelineStageFlags dstStage,
						   VkAccessFlags dstAccess);
	/** @brief Invalidate all cached descriptor sets. */
	void InvalidateAllDescriptorCaches();
	/**
	 * @brief Invalidate descriptor caches for a specific pipeline.
	 * @param pipeline Pipeline handle.
	 */
	void InvalidateDescriptorCachesForPipeline(PipelineHandle pipeline);
	/**
	 * @brief Invalidate descriptor caches referencing a specific buffer.
	 * @param buffer Buffer handle.
	 */
	void InvalidateDescriptorCachesForBuffer(BufferHandle buffer);
	/**
	 * @brief Invalidate descriptor caches referencing a specific texture.
	 * @param texture Texture handle.
	 */
	void InvalidateDescriptorCachesForTexture(TextureHandle texture);

	/** @brief Create the Vulkan instance. */
	void CreateInstance();
	/** @brief Select a suitable physical device (GPU). */
	void SelectPhysicalDevice();
	/** @brief Create the logical device and compute queue. */
	void CreateDevice();
	/** @brief Create the command pool for compute operations. */
	void CreateCommandPool();
	/** @brief Create the descriptor pool for resource bindings. */
	void CreateDescriptorPool();
	/** @brief Create the query pool for timestamp queries. */
	void CreateQueryPool();
	/** @brief Create the default texture sampler. */
	void CreateDefaultSampler();

	/** @brief Release all Vulkan resources. */
	void CleanupVulkan();

	/** @brief Begin recording a command buffer. */
	void BeginCommandBuffer();
	/** @brief End recording the current command buffer. */
	void EndCommandBuffer();
	/**
	 * @brief Submit the recorded command buffer to the queue.
	 * @param wait If true, wait for the submission to complete.
	 */
	void SubmitCommandBuffer(bool wait = false);
	/** @brief Ensure a command buffer is available and recording. */
	void EnsureCommandBuffer();
	/** @brief Wait for all submitted GPU work to complete. */
	void WaitForSubmittedWork();

	/**
	 * @brief Find a suitable memory type index for allocation.
	 * @param typeFilter Bitmask of allowed memory types.
	 * @param properties Required memory property flags.
	 * @return Memory type index.
	 */
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	/**
	 * @brief Allocate and bind memory for a buffer.
	 * @param buffer Vulkan buffer handle.
	 * @param[out] memory Allocated device memory.
	 * @param properties Required memory property flags.
	 * @param size Allocation size in bytes.
	 */
	void AllocateBufferMemory(VkBuffer buffer, VkDeviceMemory &memory, VkMemoryPropertyFlags properties, size_t size);
	/**
	 * @brief Allocate and bind memory for an image.
	 * @param image Vulkan image handle.
	 * @param[out] memory Allocated device memory.
	 * @param properties Required memory property flags.
	 */
	void AllocateImageMemory(VkImage image, VkDeviceMemory &memory, VkMemoryPropertyFlags properties);

	/**
	 * @brief Convert PixelFormat to Vulkan format.
	 * @param format Pixel format.
	 * @return Corresponding VkFormat.
	 */
	static VkFormat			  GetVkFormat(PixelFormat format);
	/**
	 * @brief Convert BindingType to Vulkan descriptor type.
	 * @param type Binding type.
	 * @return Corresponding VkDescriptorType.
	 */
	static VkDescriptorType	  GetVkDescriptorType(BindingType type);
	/**
	 * @brief Get optimal image layout for a pixel format and access pattern.
	 * @param format Pixel format.
	 * @param readOnly Whether read-only access is desired.
	 * @return Corresponding VkImageLayout.
	 */
	static VkImageLayout	  GetVkImageLayout(PixelFormat format, bool readOnly);
	/**
	 * @brief Convert ShaderType to Vulkan shader stage flags.
	 * @param type Shader type.
	 * @return Corresponding VkShaderStageFlags.
	 */
	static VkShaderStageFlags GetVkShaderStage(ShaderType type);

	/**
	 * @brief Compile GLSL source code to SPIR-V bytecode.
	 * @param glslSource GLSL source string.
	 * @param type Shader type.
	 * @return SPIR-V binary as uint32_t vector.
	 */
	std::vector<uint32_t>	  CompileGLSLToSPIRV(const std::string &glslSource, ShaderType type);

	/**
	 * @brief Insert a pipeline barrier for a buffer range.
	 * @param buffer Vulkan buffer handle.
	 * @param offset Byte offset into the buffer.
	 * @param size Byte range size.
	 * @param srcStage Source pipeline stage.
	 * @param dstStage Destination pipeline stage.
	 * @param srcAccess Source access flags.
	 * @param dstAccess Destination access flags.
	 */
	void InsertBufferBarrier(VkBuffer buffer, size_t offset, size_t size, VkPipelineStageFlags srcStage,
							 VkPipelineStageFlags dstStage, VkAccessFlags srcAccess, VkAccessFlags dstAccess);
	/**
	 * @brief Insert a pipeline barrier for an image layout transition.
	 * @param image Vulkan image handle.
	 * @param oldLayout Current image layout.
	 * @param newLayout Target image layout.
	 * @param srcStage Source pipeline stage.
	 * @param dstStage Destination pipeline stage.
	 * @param srcAccess Source access flags.
	 * @param dstAccess Destination access flags.
	 */
	void InsertImageBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
							VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkAccessFlags srcAccess,
							VkAccessFlags dstAccess);
	/**
	 * @brief Get Vulkan descriptor type from a resource layout entry.
	 * @param entry Resource layout entry.
	 * @return Corresponding VkDescriptorType.
	 */
	static VkDescriptorType GetVkDescriptorType(const ResourceLayoutEntry &entry);

	/**
	 * @brief Update a Vulkan descriptor set with cached bindings.
	 * @param cache Descriptor set cache to write to the GPU.
	 */
	void					UpdateDescriptorSet(const DescriptorSetCache &cache);
	/**
	 * @brief Find an existing descriptor set or create a new one for the given bindings.
	 * @param bindings Array of resource bindings.
	 * @param count Number of bindings.
	 * @return Pointer to the matching descriptor set cache entry.
	 */
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
	VkPipelineCache									 _pipelineCache			  = nullptr;
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
