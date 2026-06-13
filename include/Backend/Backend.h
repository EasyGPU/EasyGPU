#pragma once

/**
 * @file Backend.h
 * @brief Abstract backend interface for GPU compute operations.
 */

#ifndef EASYGPU_BACKEND_H
#define EASYGPU_BACKEND_H

#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace GPU::Backend {

// ============================================================================
// Constants
// ============================================================================

constexpr uint32_t MAX_BUFFER_BINDINGS			 = 32;
constexpr uint32_t MAX_TEXTURE_BINDINGS			 = 32;

// ============================================================================
// Handle Types
// ============================================================================

using BufferHandle								 = uint32_t;
using TextureHandle								 = uint32_t;
using ShaderHandle								 = uint32_t;
using PipelineHandle							 = uint32_t;

constexpr BufferHandle	 INVALID_BUFFER_HANDLE	 = 0;
constexpr TextureHandle	 INVALID_TEXTURE_HANDLE	 = 0;
constexpr ShaderHandle	 INVALID_SHADER_HANDLE	 = 0;
constexpr PipelineHandle INVALID_PIPELINE_HANDLE = 0;

// OpenGL buffer access mode constants (for internal use)
constexpr int			 BUFFER_MODE_READ_ONLY	 = 0x88B8;
constexpr int			 BUFFER_MODE_WRITE_ONLY	 = 0x88B9;
constexpr int			 BUFFER_MODE_READ_WRITE	 = 0x88BA;

// ============================================================================
// Buffer Mode Enum (defined here to avoid forward declaration issues)
// ============================================================================

/** @brief Buffer access mode for GPU buffer operations. */
enum class BufferMode {
	Read,	  // Readonly access
	Write,	  // Writeonly access
	ReadWrite // Read-write access
};

// ============================================================================
// Pixel Format Enum (defined here to avoid forward declaration issues)
// ============================================================================

/** @brief Pixel format specification for textures. */
enum class PixelFormat {
	R8,
	RG8,
	RGBA8,
	R32F,
	RG32F,
	RGBA32F,
	R16F,
	RG16F,
	RGBA16F,
	R32I,
	RG32I,
	RGBA32I,
	R32UI,
	RG32UI,
	RGBA32UI
};

// ============================================================================
// Buffer Description
// ============================================================================

/** @brief Descriptor for buffer creation. */
struct BufferDesc {
	size_t		sizeInBytes = 0;
	BufferMode	mode		= BufferMode::ReadWrite;
	const void *initialData = nullptr;
};

// ============================================================================
// Texture Description
// ============================================================================

/** @brief Descriptor for texture creation. */
struct TextureDesc {
	uint32_t	width		= 0;
	uint32_t	height		= 0;
	uint32_t	depth		= 1;
	uint32_t	mipLevels	= 1;
	PixelFormat format		= PixelFormat::RGBA8;
	const void *initialData = nullptr;
};

// ============================================================================
// Shader Description
// ============================================================================

/** @brief Type of shader stage. */
enum class ShaderType {
	Compute,
	Vertex,
	Fragment
};

/** @brief Descriptor for shader creation. */
struct ShaderDesc {
	ShaderType	type = ShaderType::Compute;
	std::string sourceCode;
	const char *entryPoint = "main";
};

// ============================================================================
// Resource Binding
// ============================================================================

/** @brief Type of resource binding. */
enum class BindingType {
	Buffer,
	Texture,
	Sampler
};

/** @brief Pipeline resource layout entry describing a single binding. */
struct ResourceLayoutEntry {
	uint32_t	binding	 = 0;
	BindingType type	 = BindingType::Buffer;
	PixelFormat format	 = PixelFormat::RGBA8;
	bool		readOnly = false;
};

inline bool operator==(const ResourceLayoutEntry &a, const ResourceLayoutEntry &b) {
	return a.binding == b.binding && a.type == b.type && a.format == b.format && a.readOnly == b.readOnly;
}

inline bool operator<(const ResourceLayoutEntry &a, const ResourceLayoutEntry &b) {
	if (a.binding != b.binding)
		return a.binding < b.binding;
	if (a.type != b.type)
		return a.type < b.type;
	if (a.format != b.format)
		return a.format < b.format;
	return a.readOnly < b.readOnly;
}

// ============================================================================
// Pipeline Description
// ============================================================================

/** @brief Descriptor for compute pipeline creation. */
struct PipelineDesc {
	ShaderHandle					 computeShader	= INVALID_SHADER_HANDLE;
	uint32_t						 workGroupSizeX = 1;
	uint32_t						 workGroupSizeY = 1;
	uint32_t						 workGroupSizeZ = 1;
	std::vector<ResourceLayoutEntry> resources;
	uint32_t						 pushConstantSize = 0;
};

/** @brief Actual resource binding used at dispatch time. */
struct ResourceBinding {
	uint32_t	binding = 0;
	BindingType type	= BindingType::Buffer;
	union {
		BufferHandle  buffer;
		TextureHandle texture;
	};
	PixelFormat format	 = PixelFormat::RGBA8;
	bool		readOnly = false;
};

// ============================================================================
// Memory Barrier Types
// ============================================================================

/** @brief Memory barrier type flags for pipeline synchronization. */
enum class BarrierType : uint32_t {
	None	= 0,
	Buffer	= 1 << 0,
	Texture = 1 << 2,
	Uniform = 1 << 3,
	All		= Buffer | Texture | Uniform
};

inline BarrierType operator|(BarrierType a, BarrierType b) {
	return static_cast<BarrierType>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline BarrierType operator&(BarrierType a, BarrierType b) {
	return static_cast<BarrierType>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline bool HasFlag(BarrierType flags, BarrierType flag) {
	return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
}

// ============================================================================
// Capabilities
// ============================================================================

/** @brief Backend capability information queried after initialization. */
struct BackendCaps {
	std::string versionString;
	uint32_t	maxWorkGroupSizeX		 = 0;
	uint32_t	maxWorkGroupSizeY		 = 0;
	uint32_t	maxWorkGroupSizeZ		 = 0;
	uint32_t	maxBufferBindings		 = 0;
	uint32_t	maxTextureBindings		 = 0;
	bool		supportsComputeShaders	 = false;
	bool		supportsGraphics		 = false;
	bool		supportsAsyncTransfer	 = false;
	bool		supportsMultiQueue		 = false;
	bool		supportsTimestampQueries = false;
};

// ============================================================================
// Graphics Pipeline Types
// ============================================================================

/** @brief Primitive topology for graphics pipeline assembly. */
enum class PrimitiveTopology {
	PointList,
	LineList,
	LineStrip,
	TriangleList,
	TriangleStrip,
	TriangleFan
};

/** @brief Vertex attribute layout entry for vertex buffer binding. */
struct VertexLayoutEntry {
	uint32_t	location = 0;
	PixelFormat format	 = PixelFormat::RGBA32F; // R32G32B32_FLOAT etc.
	uint32_t	offset	 = 0;					 // Byte offset in vertex struct
};

/** @brief Descriptor for creating a graphics pipeline. */
struct GraphicsPipelineDesc {
	ShaderHandle					 vertexShader		   = INVALID_SHADER_HANDLE;
	ShaderHandle					 fragmentShader		   = INVALID_SHADER_HANDLE;
	PrimitiveTopology				 topology			   = PrimitiveTopology::TriangleList;
	PixelFormat						 colorAttachmentFormat = PixelFormat::RGBA8;
	bool							 depthTestEnable	   = false;
	bool							 depthWriteEnable	   = true;
	std::vector<VertexLayoutEntry>	 vertexLayout;
	std::vector<ResourceLayoutEntry> resources;
	uint32_t						 pushConstantSize = 0;
};

/** @brief Descriptor for beginning a render pass. */
struct RenderPassBeginDesc {
	TextureHandle colorAttachment = INVALID_TEXTURE_HANDLE;
	TextureHandle depthAttachment = INVALID_TEXTURE_HANDLE;
	float		  clearColor[4]	  = {0.0f, 0.0f, 0.0f, 1.0f};
	float		  clearDepth	  = 1.0f;
	bool		  clearColorFlag  = true;
	bool		  clearDepthFlag  = true;
};

// ============================================================================
// Backend Factory
// ============================================================================

/** @brief Backend API type enumeration for factory creation. */
enum class BackendType {
	OpenGL,
	Vulkan,
	DirectX12,
	Metal,
	Count
};

inline const char *GetBackendTypeName(BackendType type) {
	switch (type) {
	case BackendType::OpenGL:
		return "OpenGL";
	case BackendType::Vulkan:
		return "Vulkan";
	case BackendType::DirectX12:
		return "DirectX12";
	case BackendType::Metal:
		return "Metal";
	default:
		return "Unknown";
	}
}

// ============================================================================
// Abstract Backend Interface
// ============================================================================

/** @brief Abstract GPU backend interface providing compute and resource operations. */
class Backend {
public:
	virtual ~Backend()																					  = default;

	/** @brief Initialize the backend and create GPU resources. */
	virtual void		  Initialize()																	  = 0;
	/** @brief Shutdown the backend and release all GPU resources. */
	virtual void		  Shutdown()																	  = 0;
	/**
	 * @brief Check if the backend has been initialized.
	 * @return True if initialized.
	 */
	virtual bool		  IsInitialized() const															  = 0;
	/** @brief Make this backend's context current on the calling thread. */
	virtual void		  MakeCurrent()																	  = 0;
	/** @brief Release the current context from the calling thread. */
	virtual void		  MakeNoneCurrent()																  = 0;
	/**
	 * @brief Get backend capabilities.
	 * @return BackendCaps structure with hardware/driver limits.
	 */
	virtual BackendCaps	  GetCaps() const																  = 0;

	/**
	 * @brief Create a GPU buffer.
	 * @param desc Buffer creation descriptor.
	 * @return Handle to the created buffer.
	 */
	virtual BufferHandle  CreateBuffer(const BufferDesc &desc)											  = 0;
	/**
	 * @brief Destroy a GPU buffer.
	 * @param buffer Handle of the buffer to destroy.
	 */
	virtual void		  DestroyBuffer(BufferHandle buffer)											  = 0;
	/**
	 * @brief Upload data to a GPU buffer.
	 * @param buffer Buffer handle.
	 * @param offset Byte offset into the buffer.
	 * @param size Number of bytes to upload.
	 * @param data Source data pointer.
	 */
	virtual void		  UploadBuffer(BufferHandle buffer, size_t offset, size_t size, const void *data) = 0;
	/**
	 * @brief Download data from a GPU buffer.
	 * @param buffer Buffer handle.
	 * @param offset Byte offset into the buffer.
	 * @param size Number of bytes to download.
	 * @param outData Destination buffer pointer.
	 */
	virtual void		  DownloadBuffer(BufferHandle buffer, size_t offset, size_t size, void *outData)  = 0;
	/**
	 * @brief Map a buffer for CPU access.
	 * @param buffer Buffer handle.
	 * @param read True to allow read access.
	 * @param write True to allow write access.
	 * @return Mapped pointer, or nullptr on failure.
	 */
	virtual void		 *MapBuffer(BufferHandle buffer, bool read, bool write)							  = 0;
	/**
	 * @brief Unmap a previously mapped buffer.
	 * @param buffer Buffer handle.
	 */
	virtual void		  UnmapBuffer(BufferHandle buffer)												  = 0;

	/**
	 * @brief Create a GPU texture.
	 * @param desc Texture creation descriptor.
	 * @return Handle to the created texture.
	 */
	virtual TextureHandle CreateTexture(const TextureDesc &desc)										  = 0;
	/**
	 * @brief Destroy a GPU texture.
	 * @param texture Texture handle.
	 */
	virtual void		  DestroyTexture(TextureHandle texture)											  = 0;
	/**
	 * @brief Upload pixel data to a 2D texture region.
	 * @param texture Texture handle.
	 * @param x Destination x offset.
	 * @param y Destination y offset.
	 * @param width Region width in pixels.
	 * @param height Region height in pixels.
	 * @param data Source pixel data.
	 */
	virtual void		  UploadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
										const void *data)												  = 0;
	/** @brief Generate all mip levels from level zero. */
	virtual void		  GenerateMipmaps(TextureHandle texture)										  = 0;
	/**
	 * @brief Upload voxel data to a 3D texture region.
	 * @param texture Texture handle.
	 * @param x Destination x offset.
	 * @param y Destination y offset.
	 * @param z Destination z offset.
	 * @param width Region width in voxels.
	 * @param height Region height in voxels.
	 * @param depth Region depth in voxels.
	 * @param data Source voxel data.
	 */
	virtual void		  UploadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
										  uint32_t height, uint32_t depth, const void *data)			  = 0;
	/**
	 * @brief Download pixel data from a 2D texture region.
	 * @param texture Texture handle.
	 * @param x Source x offset.
	 * @param y Source y offset.
	 * @param width Region width in pixels.
	 * @param height Region height in pixels.
	 * @param outData Destination pixel buffer.
	 */
	virtual void		 DownloadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
										 void *outData)													  = 0;
	/**
	 * @brief Download voxel data from a 3D texture region.
	 * @param texture Texture handle.
	 * @param x Source x offset.
	 * @param y Source y offset.
	 * @param z Source z offset.
	 * @param width Region width in voxels.
	 * @param height Region height in voxels.
	 * @param depth Region depth in voxels.
	 * @param outData Destination voxel buffer.
	 */
	virtual void		 DownloadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
										   uint32_t height, uint32_t depth, void *outData)				  = 0;

	/**
	 * @brief Compile and create a shader module.
	 * @param desc Shader creation descriptor.
	 * @return Handle to the created shader.
	 */
	virtual ShaderHandle CreateShader(const ShaderDesc &desc)											  = 0;
	/**
	 * @brief Destroy a shader module.
	 * @param shader Shader handle.
	 */
	virtual void		 DestroyShader(ShaderHandle shader)												  = 0;

	/**
	 * @brief Create a compute pipeline.
	 * @param desc Pipeline creation descriptor.
	 * @return Handle to the created pipeline.
	 */
	virtual PipelineHandle CreatePipeline(const PipelineDesc &desc)										  = 0;
	/**
	 * @brief Destroy a compute pipeline.
	 * @param pipeline Pipeline handle.
	 */
	virtual void		   DestroyPipeline(PipelineHandle pipeline)										  = 0;

	/**
	 * @brief Bind a compute pipeline for dispatch.
	 * @param pipeline Pipeline handle.
	 */
	virtual void		   BindPipeline(PipelineHandle pipeline)										  = 0;
	/**
	 * @brief Bind resources for the currently bound pipeline.
	 * @param bindings Array of resource bindings.
	 * @param count Number of bindings in the array.
	 */
	virtual void		   BindResources(const ResourceBinding *bindings, uint32_t count)				  = 0;
	/**
	 * @brief Set a named uniform on a pipeline.
	 * @param pipeline Pipeline handle.
	 * @param name Uniform name in the shader.
	 * @param type Uniform type string.
	 * @param data Pointer to uniform data.
	 */
	virtual void		   SetUniform(PipelineHandle pipeline, const std::string &name, const std::string &type,
									  const void *data)													  = 0;
	/**
	 * @brief Set raw uniform data on a pipeline.
	 * @param pipeline Pipeline handle.
	 * @param data Pointer to uniform data.
	 * @param size Size in bytes.
	 */
	virtual void		   SetUniformData(PipelineHandle pipeline, const void *data, size_t size) {
		(void)pipeline;
		(void)data;
		(void)size;
	}
	/**
	 * @brief Dispatch a compute shader with the given work group count.
	 * @param groupX Work groups in X dimension.
	 * @param groupY Work groups in Y dimension.
	 * @param groupZ Work groups in Z dimension.
	 */
	virtual void		   Dispatch(uint32_t groupX, uint32_t groupY, uint32_t groupZ)							  = 0;
	/**
	 * @brief Insert a memory barrier for pipeline synchronization.
	 * @param barrierType Types of barriers to insert.
	 */
	virtual void		   MemoryBarrier(BarrierType barrierType)												  = 0;
	/** @brief Flush and wait for all pending GPU work to complete. */
	virtual void		   Finish()																				  = 0;

	/**
	 * @brief Begin a GPU timestamp query.
	 * @return Query index for passing to EndQuery.
	 */
	virtual uint32_t	   BeginQuery()																			  = 0;
	/**
	 * @brief End a GPU timestamp query and retrieve the result.
	 * @param query Query index from BeginQuery.
	 * @return Timestamp value in nanoseconds.
	 */
	virtual uint64_t	   EndQuery(uint32_t query)																  = 0;

	// ========================================================================
	// Graphics Pipeline Support
	// ========================================================================

	/**
	 * @brief Create a graphics pipeline (vertex + fragment shader).
	 * @param desc Graphics pipeline creation descriptor.
	 * @return Handle to the created pipeline.
	 */
	virtual PipelineHandle CreateGraphicsPipeline(const GraphicsPipelineDesc &desc)								  = 0;

	/**
	 * @brief Begin a dynamic render pass to a texture attachment.
	 * @param desc Render pass begin descriptor with color/depth attachments.
	 */
	virtual void		   BeginRendering(const RenderPassBeginDesc &desc)										  = 0;

	/** @brief End the current dynamic render pass. */
	virtual void		   EndRendering()																		  = 0;

	/**
	 * @brief Set the viewport for rasterization.
	 * @param x Left coordinate.
	 * @param y Top coordinate.
	 * @param width Viewport width.
	 * @param height Viewport height.
	 */
	virtual void		   SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height)					  = 0;

	/**
	 * @brief Set the scissor rectangle.
	 * @param x Left coordinate.
	 * @param y Top coordinate.
	 * @param width Scissor width.
	 * @param height Scissor height.
	 */
	virtual void		   SetScissor(uint32_t x, uint32_t y, uint32_t width, uint32_t height)					  = 0;

	/**
	 * @brief Bind a vertex buffer for indexed or non-indexed drawing.
	 * @param buffer Buffer handle containing vertex data.
	 * @param stride Vertex stride in bytes.
	 */
	virtual void		   BindVertexBuffer(BufferHandle buffer, uint32_t stride)								  = 0;

	/**
	 * @brief Bind an index buffer for indexed drawing.
	 * @param buffer Buffer handle containing index data (uint32_t).
	 */
	virtual void		   BindIndexBuffer(BufferHandle buffer)													  = 0;

	/**
	 * @brief Draw non-indexed primitives.
	 * @param vertexCount Number of vertices to draw.
	 * @param instanceCount Number of instances.
	 * @param firstVertex First vertex index.
	 * @param firstInstance First instance index.
	 */
	virtual void Draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) = 0;

	/**
	 * @brief Draw indexed primitives.
	 * @param indexCount Number of indices to draw.
	 * @param instanceCount Number of instances.
	 * @param firstIndex First index offset.
	 * @param vertexOffset Vertex base offset.
	 * @param firstInstance First instance index.
	 */
	virtual void DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
							 uint32_t firstInstance)															  = 0;

	/**
	 * @brief Create a depth buffer texture for depth testing.
	 * @param width Depth buffer width.
	 * @param height Depth buffer height.
	 * @return Handle to the depth buffer texture.
	 */
	virtual TextureHandle CreateDepthBuffer(uint32_t width, uint32_t height)									  = 0;

	/**
	 * @brief Destroy a depth buffer.
	 * @param texture Depth buffer handle from CreateDepthBuffer.
	 */
	virtual void		  DestroyDepthBuffer(TextureHandle texture)												  = 0;

	// ========================================================================
	// Uniform Buffer Support
	// ========================================================================

	/**
	 * @brief Create a uniform buffer (UBO) with initial data.
	 * @param size Buffer size in bytes.
	 * @param data Initial data pointer (may be nullptr).
	 * @return Buffer handle.
	 */
	virtual BufferHandle  CreateUniformBuffer(size_t size, const void *data) {
		(void)size;
		(void)data;
		return INVALID_BUFFER_HANDLE;
	}

	/**
	 * @brief Upload data to a uniform buffer.
	 * @param handle Buffer handle.
	 * @param data Data pointer.
	 * @param size Data size in bytes.
	 */
	virtual void UploadUniformBuffer(BufferHandle handle, const void *data, size_t size) {
		(void)handle;
		(void)data;
		(void)size;
	}

	// ========================================================================
	// Binary Cache Support (for shader compilation acceleration)
	// ========================================================================

	/**
	 * Create a pipeline from cached binary data
	 * @param desc Pipeline description
	 * @param binaryData Cached binary data from GetPipelineBinary
	 * @param binarySize Size of binary data in bytes
	 * @param format Backend-specific format identifier
	 * @return Pipeline handle, or INVALID_PIPELINE_HANDLE if loading failed
	 */
	virtual PipelineHandle CreatePipelineFromBinary(const PipelineDesc &desc, const void *binaryData, size_t binarySize,
													uint32_t format) {
		(void)desc;
		(void)binaryData;
		(void)binarySize;
		(void)format;
		return INVALID_PIPELINE_HANDLE;
	}

	/**
	 * Get binary representation of a compiled pipeline for caching
	 * @param pipeline Pipeline handle
	 * @param[out] format Backend-specific format identifier
	 * @return Binary data as byte vector, empty if not supported
	 */
	virtual std::vector<uint8_t> GetPipelineBinary(PipelineHandle pipeline, uint32_t &format) {
		(void)pipeline;
		format = 0;
		return {};
	}

	/**
	 * Check if this backend supports binary pipeline caching
	 * @return True if binary caching is supported
	 */
	virtual bool SupportsPipelineCache() const {
		return false;
	}

	/**
	 * Get backend-specific cache format identifier
	 * @return Format version/hash for this backend/driver combination
	 */
	virtual uint32_t GetPipelineCacheFormat() const {
		return 0;
	}

	// ========================================================================
	// Native Handle Access (optional, for platform-specific features)
	// ========================================================================

	/**
	 * Get a native handle from the backend (if applicable)
	 * For OpenGL backend, this returns the GL context handle (HGLRC/GLXContext)
	 * For other backends, this may return nullptr
	 * @return Native handle as void*, or nullptr if not applicable
	 */
	virtual void *GetNativeHandle() const {
		return nullptr;
	}

	/**
	 * Get the backend type identifier
	 * @return The type of this backend
	 */
	virtual BackendType GetType() const {
		return BackendType::Count;
	}
};

// ============================================================================
Backend	   *CreateBackend(BackendType type);
void		DestroyBackend(Backend *backend);
BackendType GetDefaultBackendType();

} // namespace GPU::Backend

#endif // EASYGPU_BACKEND_H
