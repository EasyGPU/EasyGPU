#pragma once

/**
 * @file OpenGLBackend.h
 * @brief OpenGL implementation of the Backend interface.
 */

#ifndef EASYGPU_OPENGLBACKEND_H
#define EASYGPU_OPENGLBACKEND_H

#include <Backend/Backend.h>

#include <array>
#include <unordered_map>
#include <vector>

// Platform-specific forward declarations
// We use void* to avoid including platform headers here
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace GPU::Backend {

/** @brief OpenGL implementation of the Backend interface using compute shaders. */
class OpenGLBackend : public Backend {
public:
	OpenGLBackend();
	~OpenGLBackend() override;

	OpenGLBackend(const OpenGLBackend &)			= delete;
	OpenGLBackend &operator=(const OpenGLBackend &) = delete;
	OpenGLBackend(OpenGLBackend &&)					= delete;
	OpenGLBackend &operator=(OpenGLBackend &&)		= delete;

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
	void		   GenerateMipmaps(TextureHandle texture) override;
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
	ShaderHandle   CreateShader(const ShaderDesc &desc) override;
	/** @copydoc Backend::DestroyShader */
	void		   DestroyShader(ShaderHandle shader) override;

	/** @copydoc Backend::CreatePipeline */
	PipelineHandle CreatePipeline(const PipelineDesc &desc) override;
	/** @copydoc Backend::DestroyPipeline */
	void		   DestroyPipeline(PipelineHandle pipeline) override;

	/** @copydoc Backend::BindPipeline */
	void		   BindPipeline(PipelineHandle pipeline) override;
	/** @copydoc Backend::BindResources */
	void		   BindResources(const ResourceBinding *bindings, uint32_t count) override;
	/** @copydoc Backend::SetUniform */
	void		   SetUniform(PipelineHandle pipeline, const std::string &name, const std::string &type,
							  const void *data) override;
	/** @copydoc Backend::Dispatch */
	void		   Dispatch(uint32_t groupX, uint32_t groupY, uint32_t groupZ) override;
#ifdef MemoryBarrier
#undef MemoryBarrier
#endif
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

	/** @copydoc Backend::GetNativeHandle */
	void		 *GetNativeHandle() const override {
#ifdef _WIN32
		return _hglrc;
#else
		return _glxContext;
#endif
	}

	/** @copydoc Backend::GetType */
	BackendType GetType() const override {
		return BackendType::OpenGL;
	}

private:
	/** @brief Internal GL buffer resource information. */
	struct BufferInfo {
		uint32_t glHandle = 0;
		size_t	 size	  = 0;
		uint32_t mode	  = 0;
	};

	/** @brief Internal GL texture resource information. */
	struct TextureInfo {
		uint32_t glHandle		= 0;
		uint32_t width			= 0;
		uint32_t height			= 0;
		uint32_t depth			= 1;
		uint32_t internalFormat = 0;
		uint32_t format			= 0;
		uint32_t type			= 0;
		uint32_t mipLevels		= 1;
	};

	/** @brief Internal GL shader resource information. */
	struct ShaderInfo {
		uint32_t   glHandle = 0;
		ShaderType type		= ShaderType::Compute;
	};

	/** @brief Internal GL pipeline resource information. */
	struct PipelineInfo {
		uint32_t glProgram		= 0;
		uint32_t computeShader	= 0;
		uint32_t workGroupSizeX = 1;
		uint32_t workGroupSizeY = 1;
		uint32_t workGroupSizeZ = 1;
	};

	/** @brief Internal GL query resource information. */
	struct QueryInfo {
		uint32_t glQuery = 0;
		bool	 active	 = false;
	};

	/** @brief Initialize platform-specific window and context resources. */
	void	 InitializePlatform();
	/** @brief Clean up platform-specific resources. */
	void	 CleanupPlatform();
	/** @brief Create a hidden window for the GL context. */
	void	 CreateHiddenWindow();
	/** @brief Destroy the hidden window. */
	void	 DestroyHiddenWindow();
	/** @brief Create and make current the GL context. */
	void	 SetupGLContext();
	/** @brief Load OpenGL function pointers via GLAD. */
	void	 LoadGLAD();

	/** @brief Invalidate all cached GL binding state. */
	void	 InvalidateCache();
	/**
	 * @brief Bind a GL program if not already bound.
	 * @param program GL program ID.
	 */
	void	 BindProgram(uint32_t program);
	/**
	 * @brief Bind a shader storage buffer to a binding point.
	 * @param binding Binding point index.
	 * @param buffer GL buffer ID.
	 */
	void	 BindSSBO(uint32_t binding, uint32_t buffer);
	/**
	 * @brief Bind an image texture unit for shader access.
	 * @param binding Binding point index.
	 * @param texture GL texture ID.
	 * @param format GL image format constant.
	 * @param readOnly Whether the texture is read-only.
	 */
	void	 BindImageTexture(uint32_t binding, uint32_t texture, uint32_t format, bool readOnly);

	/**
	 * @brief Convert BufferMode to GL access mode constant.
	 * @param mode Buffer access mode.
	 * @return GL access mode (e.g. GL_READ_ONLY).
	 */
	uint32_t GetGLBufferMode(BufferMode mode);
	/**
	 * @brief Convert BufferMode to GL buffer usage hint.
	 * @param mode Buffer access mode.
	 * @return GL usage constant (e.g. GL_DYNAMIC_DRAW).
	 */
	uint32_t GetGLBufferUsage(BufferMode mode);
	/**
	 * @brief Convert PixelFormat to GL format tuple.
	 * @param format Pixel format.
	 * @return Tuple of internal format, pixel format, and data type.
	 */
	std::tuple<uint32_t, uint32_t, uint32_t> GetGLPixelFormat(PixelFormat format);
	/**
	 * @brief Convert PixelFormat to GL image format constant.
	 * @param format Pixel format.
	 * @return GL image format (e.g. GL_RGBA8).
	 */
	uint32_t								 GetGLImageFormat(PixelFormat format);

	// Platform-specific members
#ifdef _WIN32
	HINSTANCE _hInstance = nullptr;
	HWND	  _hwnd		 = nullptr;
	HDC		  _hdc		 = nullptr;
	HGLRC	  _hglrc	 = nullptr;
#else
	// Linux: use void* to avoid X11 types in header
	void	*_display	 = nullptr;
	uint64_t _window	 = 0;
	void	*_glxContext = nullptr;
#endif

	bool											 _initialized = false;

	std::unordered_map<BufferHandle, BufferInfo>	 _buffers;
	std::unordered_map<TextureHandle, TextureInfo>	 _textures;
	std::unordered_map<ShaderHandle, ShaderInfo>	 _shaders;
	std::unordered_map<PipelineHandle, PipelineInfo> _pipelines;
	std::vector<QueryInfo>							 _queries;

	BufferHandle									 _nextBufferHandle	 = 1;
	TextureHandle									 _nextTextureHandle	 = 1;
	ShaderHandle									 _nextShaderHandle	 = 1;
	PipelineHandle									 _nextPipelineHandle = 1;

	uint32_t										 _currentProgram	 = 0;
	std::array<uint32_t, MAX_BUFFER_BINDINGS>		 _boundBuffers{};
	std::array<uint32_t, MAX_TEXTURE_BINDINGS>		 _boundTextures{};
	std::array<uint32_t, MAX_TEXTURE_BINDINGS>		 _boundTextureFormats{};
};

Backend *CreateOpenGLBackend();

} // namespace GPU::Backend

#endif // EASYGPU_OPENGLBACKEND_H
