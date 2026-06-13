#pragma once

/**
 * @file FragmentShader.h
 * @brief FragmentShader — simple fullscreen fragment shader (VS is a hardcoded fullscreen triangle).
 *
 * Usage:
 *   FragmentShader shader([&](Var<Vec2> &fragCoord, Var<Vec4> &fragColor) {
 *       fragColor = MakeFloat4(1.0, 0.0, 0.0, 1.0); // red
 *   }, 1024, 1024);
 *
 *   shader.Render(renderTarget, true);
 */

#ifndef EASYGPU_FRAGMENT_SHADER_H
#define EASYGPU_FRAGMENT_SHADER_H

#include <Backend/Backend.h>
#include <IR/Builder/Builder.h>
#include <IR/Value/Var.h>
#include <Kernel/GraphicsBuildContext.h>
#include <Kernel/Kernel.h>
#include <Runtime/Context.h>
#include <Runtime/Texture.h>

#include <functional>
#include <memory>
#include <string>

namespace GPU::Kernel {

/**
 * @brief Simple fullscreen fragment shader with a hardcoded fullscreen triangle VS.
 *
 * Follows the same conventions as Kernel1D: lambda DSL construction,
 * optional name, GetShaderSource() for debugging.
 */
class FragmentShader : public KernelBase {
public:
	using FragmentFunc =
		std::function<void(IR::Value::Var<GPU::Math::Vec2> &fragCoord, IR::Value::Var<GPU::Math::Vec4> &fragColor)>;

	FragmentShader(const FragmentFunc &func, uint32_t width, uint32_t height);
	FragmentShader(const std::string &name, const FragmentFunc &func, uint32_t width, uint32_t height);

	void		SetName(const std::string &name);
	std::string GetName() const;

	uint32_t	GetWidth() const {
		return _width;
	}
	uint32_t GetHeight() const {
		return _height;
	}
	void										SetResolution(uint32_t width, uint32_t height);

	/** Get generated GLSL for debugging. */
	std::string									GetShaderSource();

	/** Render to a texture (match Kernel::Dispatch semantics). */
	template <Runtime::PixelFormat Format> void Render(Runtime::Texture2D<Format> &renderTarget, bool sync = false) {
		EnsureCompiled();
		RenderToTexture(renderTarget.GetHandle(), renderTarget.GetWidth(), renderTarget.GetHeight(), sync);
	}

private:
	void		BuildShaders(const FragmentFunc &func);
	void		EnsureCompiled();
	void		RenderToTexture(Backend::TextureHandle handle, uint32_t w, uint32_t h, bool sync);

	std::string _name;
	uint32_t	_width, _height;
	std::unique_ptr<GraphicsBuildContext> _context;
	std::string							  _vsSource, _fsSource;

	Backend::ShaderHandle				  _vsHandle		  = Backend::INVALID_SHADER_HANDLE;
	Backend::ShaderHandle				  _fsHandle		  = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle				  _pipelineHandle = Backend::INVALID_PIPELINE_HANDLE;
	bool								  _compiled		  = false;
};

} // namespace GPU::Kernel

#endif
