#pragma once

/**
 * @file OpenGLBackend.h
 * @brief OpenGL implementation of the Backend interface
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

class OpenGLBackend : public Backend {
public:
	OpenGLBackend();
	~OpenGLBackend() override;

	OpenGLBackend(const OpenGLBackend &)			= delete;
	OpenGLBackend &operator=(const OpenGLBackend &) = delete;
	OpenGLBackend(OpenGLBackend &&)					= delete;
	OpenGLBackend &operator=(OpenGLBackend &&)		= delete;

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
	void				   UploadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
																   uint32_t height, uint32_t depth, const void *data) override;
	void		   DownloadTexture(TextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
								   void *outData) override;
	void				   DownloadTexture3D(TextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
																   uint32_t height, uint32_t depth, void *outData) override;

	ShaderHandle   CreateShader(const ShaderDesc &desc) override;
	void		   DestroyShader(ShaderHandle shader) override;

	PipelineHandle CreatePipeline(const PipelineDesc &desc) override;
	void		   DestroyPipeline(PipelineHandle pipeline) override;

	void		   BindPipeline(PipelineHandle pipeline) override;
	void		   BindResources(const ResourceBinding *bindings, uint32_t count) override;
	void		   SetUniform(PipelineHandle pipeline, const std::string &name, const std::string &type,
							  const void *data) override;
	void		   Dispatch(uint32_t groupX, uint32_t groupY, uint32_t groupZ) override;
#ifdef MemoryBarrier
#undef MemoryBarrier
#endif
	void				 MemoryBarrier(BarrierType barrierType) override;
	void				 Finish() override;

	uint32_t			 BeginQuery() override;
	uint64_t			 EndQuery(uint32_t query) override;

	PipelineHandle		 CreatePipelineFromBinary(const PipelineDesc &desc, const void *binaryData, size_t binarySize,
												  uint32_t format) override;
	std::vector<uint8_t> GetPipelineBinary(PipelineHandle pipeline, uint32_t &format) override;
	bool				 SupportsPipelineCache() const override;
	uint32_t			 GetPipelineCacheFormat() const override;

	void				*GetNativeHandle() const override {
#ifdef _WIN32
		return _hglrc;
#else
		return _glxContext;
#endif
	}

private:
	struct BufferInfo {
		uint32_t glHandle = 0;
		size_t	 size	  = 0;
		uint32_t mode	  = 0;
	};

	struct TextureInfo {
		uint32_t glHandle		= 0;
		uint32_t width			= 0;
		uint32_t height			= 0;
		uint32_t depth			= 1;
		uint32_t internalFormat = 0;
		uint32_t format			= 0;
		uint32_t type			= 0;
	};

	struct ShaderInfo {
		uint32_t   glHandle = 0;
		ShaderType type		= ShaderType::Compute;
	};

	struct PipelineInfo {
		uint32_t glProgram		= 0;
		uint32_t computeShader	= 0;
		uint32_t workGroupSizeX = 1;
		uint32_t workGroupSizeY = 1;
		uint32_t workGroupSizeZ = 1;
	};

	struct QueryInfo {
		uint32_t glQuery = 0;
		bool	 active	 = false;
	};

	void	 InitializePlatform();
	void	 CleanupPlatform();
	void	 CreateHiddenWindow();
	void	 DestroyHiddenWindow();
	void	 SetupGLContext();
	void	 LoadGLAD();

	void	 InvalidateCache();
	void	 BindProgram(uint32_t program);
	void	 BindSSBO(uint32_t binding, uint32_t buffer);
	void	 BindImageTexture(uint32_t binding, uint32_t texture, uint32_t format, bool readOnly);

	uint32_t GetGLBufferMode(BufferMode mode);
	uint32_t GetGLBufferUsage(BufferMode mode);
	std::tuple<uint32_t, uint32_t, uint32_t> GetGLPixelFormat(PixelFormat format);
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
