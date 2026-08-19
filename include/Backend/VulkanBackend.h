#pragma once

/**
 * @file VulkanBackend.h
 * @brief Vulkan implementation of the Backend interface.
 */

#ifndef EASYGPU_VULKANBACKEND_H
#define EASYGPU_VULKANBACKEND_H

#include <Backend/Backend.h>

#include <array>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
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
	using InstanceExtensionProvider = std::function<std::vector<const char *>()>;

	static void RegisterInstanceExtensionProvider(InstanceExtensionProvider provider);

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
	/** @copydoc Backend::CopyBuffer */
	void		   CopyBuffer(BufferHandle source, size_t sourceOffset, BufferHandle destination,
						  size_t destinationOffset, size_t size) override;
	/** @copydoc Backend::MapBuffer */
	void		  *MapBuffer(BufferHandle buffer, bool read, bool write) override;
	/** @copydoc Backend::UnmapBuffer */
	void		   UnmapBuffer(BufferHandle buffer) override;
	/** @copydoc Backend::BeginTextureReadback */
	SubmissionHandle BeginTextureReadback(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width,
										  uint32_t height, BufferHandle stagingBuffer, size_t stagingOffset) override;
	/** @copydoc Backend::MapTextureReadback */
	TextureReadbackMapping MapTextureReadback(SubmissionHandle submission) override;
	/** @copydoc Backend::UnmapTextureReadback */
	void				   UnmapTextureReadback(SubmissionHandle submission) override;

	/** @copydoc Backend::CreateTexture */
	TextureHandle  CreateTexture(const TextureDesc &desc) override;
	/** @copydoc Backend::DestroyTexture */
	void		   DestroyTexture(TextureHandle texture) override;
	/** @copydoc Backend::UploadTexture */
	void		   UploadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
								 const void *data) override;
	void		   UploadTextureFromBuffer(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width,
										   uint32_t height, BufferHandle source, size_t sourceOffset) override;
	void		   GenerateMipmaps(TextureHandle texture) override;
	/** @copydoc Backend::UploadTexture3D */
	void UploadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height,
						 uint32_t depth, const void *data) override;
	void UploadTexture3DFromBuffer(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
								   uint32_t height, uint32_t depth, BufferHandle source, size_t sourceOffset) override;
	/** @copydoc Backend::DownloadTexture */
	void DownloadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
						 void *outData) override;
	void DownloadTextureToBuffer(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
								 BufferHandle destination, size_t destinationOffset) override;
	/** @copydoc Backend::DownloadTexture3D */
	void DownloadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height,
						   uint32_t depth, void *outData) override;
	void DownloadTexture3DToBuffer(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
								   uint32_t height, uint32_t depth, BufferHandle destination,
								   size_t destinationOffset) override;

	/** @copydoc Backend::CreateShader */
	ShaderHandle		 CreateShader(const ShaderDesc &desc) override;
	/** @copydoc Backend::DestroyShader */
	void				 DestroyShader(ShaderHandle shader) override;
	/** @copydoc Backend::GetOptimizedGLSL */
	std::string			 GetOptimizedGLSL(const ShaderDesc &desc) override;
	/** @copydoc Backend::GetShaderCompilationStats */
	ShaderCompilationStats GetShaderCompilationStats() const override;
	/** @copydoc Backend::ResetShaderCompilationStats */
	void				 ResetShaderCompilationStats() override;

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
	/** @copydoc Backend::Submit */
	SubmissionHandle	 Submit() override;
	/** @copydoc Backend::IsSubmissionComplete */
	bool				 IsSubmissionComplete(SubmissionHandle submission) override;
	/** @copydoc Backend::WaitForSubmission */
	bool				 WaitForSubmission(SubmissionHandle submission, uint64_t timeoutNanoseconds) override;
	/** @copydoc Backend::ReleaseSubmission */
	void				 ReleaseSubmission(SubmissionHandle submission) override;
	/** @copydoc Backend::GetOperationCounters */
	BackendOperationCounters GetOperationCounters() const override;

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
	/** @copydoc Backend::GetPipelineCacheStats */
	PipelineCacheStats GetPipelineCacheStats() const override;
	/** @copydoc Backend::FlushPipelineCache */
	void				 FlushPipelineCache() override;
	/** @copydoc Backend::CreateGraphicsPipeline */
	PipelineHandle		 CreateGraphicsPipeline(const GraphicsPipelineDesc &desc) override;
	/** @copydoc Backend::BeginRendering */
	void				 BeginRendering(const RenderPassBeginDesc &desc) override;
	/** @copydoc Backend::EndRendering */
	void				 EndRendering() override;
	/** @copydoc Backend::SetViewport */
	void				 SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override;
	/** @copydoc Backend::SetScissor */
	void				 SetScissor(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override;
	/** @copydoc Backend::BindVertexBuffer */
	void				 BindVertexBuffer(BufferHandle buffer, uint32_t stride) override;
	/** @copydoc Backend::BindIndexBuffer */
	void				 BindIndexBuffer(BufferHandle buffer) override;
	/** @copydoc Backend::Draw */
	void Draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) override;
	/** @copydoc Backend::DrawIndexed */
	void DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
					 uint32_t firstInstance) override;
	/** @copydoc Backend::CreateDepthBuffer */
	TextureHandle CreateDepthBuffer(uint32_t width, uint32_t height) override;
	/** @copydoc Backend::DestroyDepthBuffer */
	void		  DestroyDepthBuffer(TextureHandle texture) override;
	/** @copydoc Backend::CreateUniformBuffer */
	BufferHandle  CreateUniformBuffer(size_t size, const void *data) override;
	/** @copydoc Backend::UploadUniformBuffer */
	void		  UploadUniformBuffer(BufferHandle handle, const void *data, size_t size) override;

	/** @copydoc Backend::GetType */
	BackendType	  GetType() const override {
		return BackendType::Vulkan;
	}

	struct NativeTextureInfo {
		VkImage		  image	 = VK_NULL_HANDLE;
		VkFormat	  format = VK_FORMAT_UNDEFINED;
		VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
		uint32_t	  width	 = 0;
		uint32_t	  height = 0;
		uint32_t	  depth	 = 1;
	};

	[[nodiscard]] VkInstance Instance() const {
		return _instance;
	}
	[[nodiscard]] VkPhysicalDevice PhysicalDevice() const {
		return _physicalDevice;
	}
	[[nodiscard]] VkDevice Device() const {
		return _device;
	}
	[[nodiscard]] VkQueue Queue() const {
		return _computeQueue;
	}
	[[nodiscard]] uint32_t QueueFamilyIndex() const {
		return _computeQueueFamilyIndex;
	}
	[[nodiscard]] NativeTextureInfo GetNativeTextureInfo(TextureHandle texture);
	void SetNativeTextureLayout(TextureHandle texture, VkImageLayout layout, VkPipelineStageFlags stage,
								VkAccessFlags access);

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
		SubmissionHandle	  mappedReadback	 = INVALID_SUBMISSION_HANDLE;
		uint32_t			  gpuUseLeases		 = 0;
		uint32_t			  readbackLeases	 = 0;
		bool				  destroyRequested	 = false;
		VkMemoryPropertyFlags memoryFlags		 = 0;
		VkMemoryPropertyFlags stagingMemoryFlags = 0;
	};

	/** @brief Internal Vulkan texture resource information. */
	struct TextureInfo {
		VkImage				 image		   = nullptr;
		VkDeviceMemory		 memory		   = nullptr;
		VkImageView			 view		   = nullptr;
		VkImageView			 sampledView   = nullptr;
		uint32_t			 width		   = 0;
		uint32_t			 height		   = 0;
		uint32_t			 depth		   = 1;
		uint32_t			 mipLevels	   = 1;
		PixelFormat			 format		   = PixelFormat::RGBA8;
		VkFormat			 vkFormat	   = VK_FORMAT_UNDEFINED;
		VkImageUsageFlags	  usage			   = 0;
		VkSampleCountFlagBits samples	   = VK_SAMPLE_COUNT_1_BIT;
		VkImageLayout		 currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VkPipelineStageFlags lastStage	   = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkAccessFlags		 lastAccess	   = 0;
		uint32_t			  gpuUseLeases	   = 0;
		uint32_t			  readbackLeases   = 0;
		bool				  destroyRequested = false;
	};

	struct MsaaAttachment {
		VkImage				 image		   = nullptr;
		VkDeviceMemory		 memory		   = nullptr;
		VkImageView			 view		   = nullptr;
		uint32_t			 width		   = 0;
		uint32_t			 height		   = 0;
		uint32_t			 slot		   = 0;
		TextureHandle		 resolveTarget = INVALID_TEXTURE_HANDLE;
		VkFormat			 format		   = VK_FORMAT_UNDEFINED;
		VkSampleCountFlagBits samples	   = VK_SAMPLE_COUNT_1_BIT;
		VkImageAspectFlags	 aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
		VkImageLayout		 currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VkPipelineStageFlags lastStage	   = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkAccessFlags		 lastAccess	   = 0;
	};

	struct SamplerKey {
		SamplerFilter	   minFilter	= SamplerFilter::Nearest;
		SamplerFilter	   magFilter	= SamplerFilter::Nearest;
		SamplerMipmapMode  mipmapMode	= SamplerMipmapMode::Nearest;
		SamplerAddressMode addressU		= SamplerAddressMode::ClampToEdge;
		SamplerAddressMode addressV		= SamplerAddressMode::ClampToEdge;
		SamplerAddressMode addressW		= SamplerAddressMode::ClampToEdge;
		float			   mipLodBias	= 0.0f;
		float			   minLod		= 0.0f;
		float			   maxLod		= 1000.0f;
		bool			   anisotropyEnable = false;
		float			   maxAnisotropy = 1.0f;
		bool			   compareEnable = false;
		CompareOp		   compareOp = CompareOp::Always;
		SamplerBorderColor borderColor = SamplerBorderColor::FloatOpaqueBlack;

		bool operator==(const SamplerKey &other) const {
			return minFilter == other.minFilter && magFilter == other.magFilter &&
				   mipmapMode == other.mipmapMode &&
				   addressU == other.addressU && addressV == other.addressV && addressW == other.addressW &&
				   mipLodBias == other.mipLodBias && minLod == other.minLod && maxLod == other.maxLod &&
				   anisotropyEnable == other.anisotropyEnable && maxAnisotropy == other.maxAnisotropy &&
				   compareEnable == other.compareEnable && compareOp == other.compareOp &&
				   borderColor == other.borderColor;
		}
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
		// Graphics pipeline extensions
		bool							 isGraphics		= false;
		ShaderHandle					 vertexShader	= INVALID_SHADER_HANDLE;
		ShaderHandle					 fragmentShader = INVALID_SHADER_HANDLE;
		PrimitiveTopology				 topology		= PrimitiveTopology::TriangleList;
		PixelFormat						 colorFormat	= PixelFormat::RGBA8;
		std::vector<PixelFormat>		 colorFormats;
		VkSampleCountFlagBits			 samples		= VK_SAMPLE_COUNT_1_BIT;
		bool							 depthEnable	= false;
		std::vector<VertexLayoutEntry>	 vertexLayout;
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
		std::array<SamplerKey, MAX_TEXTURE_BINDINGS>	boundSamplers{};
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
							   uint32_t depth, const void *data, TextureHandle trackedTexture = INVALID_TEXTURE_HANDLE);
	void CopyBufferToTexture(TextureInfo &info, VkBuffer sourceBuffer, size_t sourceOffset, uint32_t x, uint32_t y,
							 uint32_t z, uint32_t width, uint32_t height, uint32_t depth,
							 TextureHandle trackedTexture = INVALID_TEXTURE_HANDLE,
							 BufferHandle  trackedSource  = INVALID_BUFFER_HANDLE);
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
								 uint32_t depth, void *outData, TextureHandle trackedTexture = INVALID_TEXTURE_HANDLE);
	void CopyTextureToBuffer(TextureInfo &info, VkBuffer destinationBuffer, size_t destinationOffset, uint32_t x,
							 uint32_t y, uint32_t z, uint32_t width, uint32_t height, uint32_t depth,
							 TextureHandle trackedTexture	  = INVALID_TEXTURE_HANDLE,
							 BufferHandle  trackedDestination = INVALID_BUFFER_HANDLE);
	void CopyTextureToBufferBlocking(TextureInfo &info, VkBuffer destinationBuffer, size_t destinationOffset,
									 uint32_t x, uint32_t y, uint32_t z, uint32_t width, uint32_t height,
									 uint32_t depth, TextureHandle trackedTexture = INVALID_TEXTURE_HANDLE,
									 BufferHandle trackedDestination = INVALID_BUFFER_HANDLE);
	void DestroyBufferNow(BufferHandle buffer);
	void DestroyTextureNow(TextureHandle texture);
	void TryDestroyDeferredBuffer(BufferHandle buffer);
	void TryDestroyDeferredTexture(TextureHandle texture);
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
	void TransitionMsaaAttachment(MsaaAttachment &info, VkImageLayout newLayout, VkPipelineStageFlags dstStage,
								  VkAccessFlags dstAccess);
	MsaaAttachment &GetOrCreateMsaaAttachment(uint32_t width, uint32_t height, uint32_t slot, VkFormat format,
											  VkSampleCountFlagBits samples, TextureHandle resolveTarget,
											  VkImageUsageFlags usage, VkImageAspectFlags aspectMask);
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
	SubmissionHandle SubmitCommandBuffer(bool wait = false, bool externallyVisible = false);
	/** @brief Ensure a command buffer is available and recording. */
	void EnsureCommandBuffer();
	/** @brief Wait for all submitted GPU work to complete. */
	void WaitForSubmittedWork();
	struct SubmissionInfo;
	bool UpdateSubmissionStatus(SubmissionHandle submission, uint64_t timeoutNanoseconds, bool wait);
	void ReapReleasedSubmissions();
	void RecycleSubmissionResources(SubmissionInfo &submission);
	void DestroySubmissionResources(SubmissionInfo &submission);
	void	 TrackBufferUsage(BufferHandle buffer);
	void	 TrackTextureUsage(TextureHandle texture);
	void	 ReleaseSubmissionResourceLeases(SubmissionInfo &submission);
	void	 ReleaseReadbackLease(SubmissionInfo &submission);

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
	static VkSampleCountFlagBits GetVkSampleCount(SampleCount sampleCount);
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
	static VkShaderStageFlags GetVkResourceStages(uint32_t stageFlags, bool graphicsPipeline);

	/**
	 * @brief Compile GLSL source code to SPIR-V bytecode.
	 * @param glslSource GLSL source string.
	 * @param type Shader type.
	 * @param optimizationLevel SPIR-V optimization preset.
	 * @param preserveInterface Preserve non-IO entry-point interface variables.
	 * @return SPIR-V binary as uint32_t vector.
	 */
	std::vector<uint32_t>	  CompileGLSLToSPIRV(const std::string &glslSource, ShaderType type,
												 ShaderOptimizationLevel optimizationLevel, bool preserveInterface);
	std::optional<std::vector<uint32_t>> LoadMemoryCachedSpirv(const std::filesystem::path &path);
	void StoreMemoryCachedSpirv(const std::filesystem::path &path, const std::vector<uint32_t> &spirv);

	/**
	 * @brief Optimize SPIR-V bytecode with SPIRV-Tools.
	 * @param spirv Unoptimized SPIR-V binary.
	 * @param optimizationLevel SPIR-V optimization preset.
	 * @param preserveInterface Preserve non-IO entry-point interface variables.
	 * @return Optimized SPIR-V binary.
	 */
	std::vector<uint32_t>	  OptimizeSPIRV(const std::vector<uint32_t> &spirv,
											ShaderOptimizationLevel optimizationLevel, bool preserveInterface);

	/**
	 * @brief Convert SPIR-V bytecode back to readable GLSL for inspection.
	 * @param spirv SPIR-V binary.
	 * @param type Shader type.
	 * @return GLSL source generated by SPIRV-Cross.
	 */
	std::string				  DecompileSPIRVToGLSL(const std::vector<uint32_t> &spirv, ShaderType type);
	void					  InitializePipelineCache();
	void					  PersistPipelineCache();

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
	VkSampler				GetOrCreateSampler(const SamplerKey &key);
	SamplerKey				MakeSamplerKey(const SamplerDesc &desc, bool hasMipmaps) const;

private:
	struct SubmissionInfo {
		enum class ReadbackMappingState : uint8_t {
			Available,
			Mapped,
			Consumed,
		};
		struct ReadbackInfo {
			TextureHandle		 texture	   = INVALID_TEXTURE_HANDLE;
			BufferHandle		 stagingBuffer = INVALID_BUFFER_HANDLE;
			size_t				 stagingOffset = 0;
			size_t				 byteSize	   = 0;
			size_t				 rowPitch	   = 0;
			ReadbackMappingState mappingState  = ReadbackMappingState::Available;
		};
		VkCommandPool pool = nullptr;
		VkCommandBuffer commandBuffer = nullptr;
		VkFence fence = nullptr;
		bool completed = false;
		bool released = false;
		bool						failed				   = false;
		bool						resourceLeasesReleased = false;
		std::vector<BufferHandle>	bufferUses;
		std::vector<TextureHandle>	textureUses;
		std::optional<ReadbackInfo> readback;
	};
	static constexpr size_t MAX_CACHED_SUBMISSION_RESOURCES = 64;

	// Vulkan handles
	VkInstance										 _instance				  = nullptr;
	VkDebugUtilsMessengerEXT						 _debugMessenger		  = nullptr;
	VkPhysicalDevice								 _physicalDevice		  = nullptr;
	VkDevice										 _device				  = nullptr;
	VkQueue											 _computeQueue			  = nullptr;
	uint32_t										 _computeQueueFamilyIndex = 0;

	// Command resources
	VkCommandPool									 _commandPool			  = nullptr;
	VkCommandBuffer									 _commandBuffer			  = nullptr;
	VkFence											 _commandFence			  = nullptr;
	bool											 _commandBufferRecording  = false;
	std::unordered_set<BufferHandle>					 _recordingBufferUses;
	std::unordered_set<TextureHandle>					 _recordingTextureUses;
	std::unordered_map<SubmissionHandle, SubmissionInfo> _submissions;
	std::vector<SubmissionInfo> _availableSubmissionResources;
	SubmissionHandle _nextSubmissionHandle = 1;
	SubmissionHandle _completedSubmissionWatermark = INVALID_SUBMISSION_HANDLE;
	BackendOperationCounters							 _operationCounters;

	// Graphics pipeline state
	bool											 _insideRenderPass		  = false;
	BufferHandle									 _currentVertexBuffer	  = INVALID_BUFFER_HANDLE;
	BufferHandle									 _currentIndexBuffer	  = INVALID_BUFFER_HANDLE;

	// Dynamic rendering function pointers (loaded at runtime for Vulkan 1.1 compatibility)
	PFN_vkCmdBeginRenderingKHR						 _vkCmdBeginRenderingKHR  = nullptr;
	PFN_vkCmdEndRenderingKHR						 _vkCmdEndRenderingKHR	  = nullptr;

	// Descriptor resources
	VkDescriptorPool								 _descriptorPool		  = nullptr;
	std::vector<VkDescriptorPool>					 _descriptorPools;
	VkSampler										 _defaultSampler		  = nullptr;
	VkSampler										 _mipmapSampler			  = nullptr;
	std::vector<std::pair<SamplerKey, VkSampler>>	 _samplerCache;
	VkPipelineCache									 _pipelineCache			  = nullptr;
	std::filesystem::path							 _pipelineCachePath;
	bool											 _pipelineCacheDirty	  = false;
	std::vector<VkDescriptorSet>					 _inFlightDescriptorSets;

	// Query pool for timing
	VkQueryPool										 _queryPool		 = nullptr;
	uint32_t										 _nextQueryIndex = 0;

	// Resource maps
	std::unordered_map<BufferHandle, BufferInfo>	 _buffers;
	std::unordered_map<TextureHandle, TextureInfo>	 _textures;
	std::vector<MsaaAttachment>						 _msaaAttachments;
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
	bool											 _samplerAnisotropySupported = false;
	float											 _maxSamplerAnisotropy = 1.0f;
	bool											 _depthClampSupported = false;
	bool											 _fillModeNonSolidSupported = false;
	ShaderCompilationStats						 _shaderCompilationStats;
	PipelineCacheStats							 _pipelineCacheStats;
	struct SpirvMemoryCacheEntry {
		std::vector<uint32_t>			 spirv;
		uintmax_t						 fileSize = 0;
		std::filesystem::file_time_type lastWriteTime{};
		uint64_t						 lastAccess = 0;
	};
	static constexpr size_t MAX_CACHED_SPIRV_MODULES = 256;
	static constexpr size_t MAX_CACHED_SPIRV_MEMORY_BYTES = 64u * 1024u * 1024u;
	std::unordered_map<std::string, SpirvMemoryCacheEntry> _spirvMemoryCache;
	size_t _spirvMemoryCacheBytes = 0;
	uint64_t _spirvMemoryCacheAccess = 0;

	// Thread safety
	mutable std::mutex							 _mutex;
	bool											 _isCurrent = false;
};

Backend *CreateVulkanBackend();

} // namespace GPU::Backend

#endif // EASYGPU_VULKANBACKEND_H
