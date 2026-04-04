#pragma once

/**
 * @file Backend.h
 * @brief Abstract backend interface for GPU compute operations
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

enum class BufferMode {
	Read,	  // Readonly access
	Write,	  // Writeonly access
	ReadWrite // Read-write access
};

// ============================================================================
// Pixel Format Enum (defined here to avoid forward declaration issues)
// ============================================================================

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

struct BufferDesc {
	size_t		sizeInBytes = 0;
	BufferMode	mode		= BufferMode::ReadWrite;
	const void *initialData = nullptr;
};

// ============================================================================
// Texture Description
// ============================================================================

struct TextureDesc {
	uint32_t	width		= 0;
	uint32_t	height		= 0;
	uint32_t	depth		= 1;
	PixelFormat format		= PixelFormat::RGBA8;
	const void *initialData = nullptr;
};

// ============================================================================
// Shader Description
// ============================================================================

enum class ShaderType {
	Compute,
	Vertex,
	Fragment
};

struct ShaderDesc {
	ShaderType	type = ShaderType::Compute;
	std::string sourceCode;
	const char *entryPoint = "main";
};

// ============================================================================
// Resource Binding
// ============================================================================

enum class BindingType {
	Buffer,
	Texture,
	Sampler
};

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

struct PipelineDesc {
	ShaderHandle					 computeShader	= INVALID_SHADER_HANDLE;
	uint32_t						 workGroupSizeX = 1;
	uint32_t						 workGroupSizeY = 1;
	uint32_t						 workGroupSizeZ = 1;
	std::vector<ResourceLayoutEntry> resources;
	uint32_t						 pushConstantSize = 0;
};

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

struct BackendCaps {
	std::string versionString;
	uint32_t	maxWorkGroupSizeX	   = 0;
	uint32_t	maxWorkGroupSizeY	   = 0;
	uint32_t	maxWorkGroupSizeZ	   = 0;
	uint32_t	maxBufferBindings	   = 0;
	uint32_t	maxTextureBindings	   = 0;
	bool		supportsComputeShaders = false;
	bool		supportsAsyncTransfer  = false;
	bool		supportsMultiQueue	   = false;
};

// ============================================================================
// Abstract Backend Interface
// ============================================================================

class Backend {
public:
	virtual ~Backend()																					  = default;

	virtual void		  Initialize()																	  = 0;
	virtual void		  Shutdown()																	  = 0;
	virtual bool		  IsInitialized() const															  = 0;
	virtual void		  MakeCurrent()																	  = 0;
	virtual void		  MakeNoneCurrent()																  = 0;
	virtual BackendCaps	  GetCaps() const																  = 0;

	virtual BufferHandle  CreateBuffer(const BufferDesc &desc)											  = 0;
	virtual void		  DestroyBuffer(BufferHandle buffer)											  = 0;
	virtual void		  UploadBuffer(BufferHandle buffer, size_t offset, size_t size, const void *data) = 0;
	virtual void		  DownloadBuffer(BufferHandle buffer, size_t offset, size_t size, void *outData)  = 0;
	virtual void		 *MapBuffer(BufferHandle buffer, bool read, bool write)							  = 0;
	virtual void		  UnmapBuffer(BufferHandle buffer)												  = 0;

	virtual TextureHandle CreateTexture(const TextureDesc &desc)										  = 0;
	virtual void		  DestroyTexture(TextureHandle texture)											  = 0;
	virtual void		  UploadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
										const void *data)												  = 0;
	virtual void		 DownloadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
										 void *outData)													  = 0;

	virtual ShaderHandle CreateShader(const ShaderDesc &desc)											  = 0;
	virtual void		 DestroyShader(ShaderHandle shader)												  = 0;

	virtual PipelineHandle CreatePipeline(const PipelineDesc &desc)										  = 0;
	virtual void		   DestroyPipeline(PipelineHandle pipeline)										  = 0;

	virtual void		   BindPipeline(PipelineHandle pipeline)										  = 0;
	virtual void		   BindResources(const ResourceBinding *bindings, uint32_t count)				  = 0;
	virtual void		   SetUniform(PipelineHandle pipeline, const std::string &name, const std::string &type,
									  const void *data)													  = 0;
	virtual void		   SetUniformData(PipelineHandle pipeline, const void *data, size_t size) {
		  (void)pipeline;
		  (void)data;
		  (void)size;
	}
	virtual void		   Dispatch(uint32_t groupX, uint32_t groupY, uint32_t groupZ) = 0;
	virtual void		   MemoryBarrier(BarrierType barrierType)					   = 0;
	virtual void		   Finish()													   = 0;

	virtual uint32_t	   BeginQuery()												   = 0;
	virtual uint64_t	   EndQuery(uint32_t query)									   = 0;

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
};

// ============================================================================
// Backend Factory
// ============================================================================

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

Backend	   *CreateBackend(BackendType type);
void		DestroyBackend(Backend *backend);
BackendType GetDefaultBackendType();

} // namespace GPU::Backend

#endif // EASYGPU_BACKEND_H
