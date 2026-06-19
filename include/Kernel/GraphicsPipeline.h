#pragma once

/**
 * @file GraphicsPipeline.h
 * @brief Complete rasterization pipeline with vertex + fragment shader DSL.
 *
 * Usage:
 *   Varying<Vec3> v_worldPos("v_worldPos");
 *   Varying<Vec2> v_uv("v_uv");
 *
 *   GraphicsPipeline pipeline(
 *       // Vertex shader
 *       [&](Var<MyVertex> &in_vert, Var<Vec4> &gl_Position) {
 *           gl_Position = mvp * MakeFloat4(in_vert.position, 1.0f);
 *           v_worldPos = in_vert.position;
 *           v_uv = in_vert.texcoord;
 *       },
 *       // Fragment shader
 *       [&](Var<Vec4> &fragColor) {
 *           auto tex = modelTex.BindSampler();
 *           fragColor = tex.Sample(v_uv);
 *       }
 *   );
 */

#ifndef EASYGPU_GRAPHICS_PIPELINE_H
#define EASYGPU_GRAPHICS_PIPELINE_H

#include <Backend/Backend.h>
#include <IR/Builder/Builder.h>
#include <IR/Value/Var.h>
#include <IR/Value/Varying.h>
#include <Kernel/GraphicsBuildContext.h>
#include <Kernel/Kernel.h>
#include <Runtime/Context.h>
#include <Runtime/DepthBuffer.h>
#include <Runtime/Texture.h>
#include <Utility/Meta/StructMeta.h>
#include <Utility/Meta/Std430Layout.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace GPU::Kernel {

using GraphicsPipelineBuilderGuard = IR::Builder::Builder::ScopedBind;

/**
 * @brief Complete rasterization pipeline (vertex shader + fragment shader DSL).
 */
class GraphicsPipeline : public KernelBase {
public:
	template <typename VertexType>
	using VertexFunc =
		std::function<void(IR::Value::Var<VertexType> &in_vert, IR::Value::Var<GPU::Math::Vec4> &gl_Position)>;

	using VertexFuncSimple = std::function<void(IR::Value::Var<GPU::Math::Vec4> &gl_Position)>;
	using FragmentFunc	   = std::function<void(IR::Value::Var<GPU::Math::Vec4> &fragColor)>;

	// === Construction with vertex input ===
	template <typename VertexType>
	GraphicsPipeline(const VertexFunc<VertexType> &vertexFunc, const FragmentFunc &fragmentFunc)
		: _name("GraphicsPipeline") {
		BuildShaders<VertexType>(vertexFunc, fragmentFunc);
	}

	template <typename VertexType>
	GraphicsPipeline(const std::string &name, const VertexFunc<VertexType> &vertexFunc,
					 const FragmentFunc &fragmentFunc)
		: _name(name) {
		BuildShaders<VertexType>(vertexFunc, fragmentFunc);
	}

	// === Construction without explicit vertex input ===
	GraphicsPipeline(const VertexFuncSimple &vertexFunc, const FragmentFunc &fragmentFunc) : _name("GraphicsPipeline") {
		BuildShadersSimple(vertexFunc, fragmentFunc);
	}

	GraphicsPipeline(const std::string &name, const VertexFuncSimple &vertexFunc, const FragmentFunc &fragmentFunc)
		: _name(name) {
		BuildShadersSimple(vertexFunc, fragmentFunc);
	}

	~GraphicsPipeline() override;

	void SetName(const std::string &name) {
		_name = name;
	}
	std::string GetName() const {
		return _name;
	}

	/** Get generated shader source for debugging. */
	std::string GetShaderSource() {
		if (_context)
			return _context->GetCompleteCode();
		return "";
	}

	// === Vertex / Index Buffer Binding ===

	/**
	 * @brief Bind a vertex buffer with the given stride.
	 * @param bufferHandle Backend buffer handle.
	 * @param stride Vertex stride in bytes.
	 */
	void SetVertexBuffer(Backend::BufferHandle bufferHandle, uint32_t stride) {
		if (bufferHandle == Backend::INVALID_BUFFER_HANDLE || stride == 0) {
			throw std::runtime_error("GraphicsPipeline::SetVertexBuffer requires a valid buffer and non-zero stride");
		}
		_vertexBufferHandle = bufferHandle;
		_vertexStride		= stride;
		_hasVertexBuffer	= true;
	}

	/** @brief Bind an index buffer (uint32_t indices). */
	void SetIndexBuffer(Backend::BufferHandle bufferHandle) {
		if (bufferHandle == Backend::INVALID_BUFFER_HANDLE) {
			throw std::runtime_error("GraphicsPipeline::SetIndexBuffer requires a valid buffer");
		}
		_indexBufferHandle = bufferHandle;
		_hasIndexBuffer	   = true;
	}

	void SetIndexCount(uint32_t count) {
		_indexCount = count;
	}

	// === Rendering ===

	/** @brief Draw non-indexed primitives to a render target texture. */
	template <Runtime::PixelFormat Format>
	void Draw(Runtime::Texture2D<Format> &renderTarget, uint32_t vertexCount, bool sync = false) {
		DrawImpl(renderTarget.GetHandle(), Backend::INVALID_TEXTURE_HANDLE, renderTarget.GetWidth(),
				 renderTarget.GetHeight(), vertexCount, 0, false, sync);
	}

	/** @brief Draw non-indexed with depth buffer. */
	template <Runtime::PixelFormat Format>
	void Draw(Runtime::Texture2D<Format> &renderTarget, Runtime::DepthBuffer &depthBuffer, uint32_t vertexCount,
			  bool sync = false) {
		DrawImpl(renderTarget.GetHandle(), depthBuffer.GetHandle(), renderTarget.GetWidth(), renderTarget.GetHeight(),
				 vertexCount, 0, false, sync);
	}

	/** @brief Draw indexed primitives. */
	template <Runtime::PixelFormat Format>
	void DrawIndexed(Runtime::Texture2D<Format> &renderTarget, uint32_t indexCount, bool sync = false) {
		DrawImpl(renderTarget.GetHandle(), Backend::INVALID_TEXTURE_HANDLE, renderTarget.GetWidth(),
				 renderTarget.GetHeight(), 0, indexCount, true, sync);
	}

	/** @brief Draw indexed primitives with depth buffer. */
	template <Runtime::PixelFormat Format>
	void DrawIndexed(Runtime::Texture2D<Format> &renderTarget, Runtime::DepthBuffer &depthBuffer, uint32_t indexCount,
					 bool sync = false) {
		DrawImpl(renderTarget.GetHandle(), depthBuffer.GetHandle(), renderTarget.GetWidth(), renderTarget.GetHeight(),
				 0, indexCount, true, sync);
	}

private:
	template <typename VertexType>
	void BuildShaders(const VertexFunc<VertexType> &vertexFunc, const FragmentFunc &fragmentFunc) {
		_context = std::make_unique<GraphicsBuildContext>();
		{
			GraphicsPipelineBuilderGuard guard(IR::Builder::Builder::Get(), *_context);
			GPU::Meta::RegisterStructWithDependencies<VertexType>();
		}

		// Auto-generate vertex layout
		std::vector<Backend::VertexLayoutEntry> layout;
		GenerateVertexLayout<VertexType>(layout);
		_context->SetVertexLayout(layout);
		_context->SetVertexInputSetupCode(GenerateVertexInputSetupCode<VertexType>());
		_vertexLayout = std::move(layout);

		// Stage 1: Vertex Shader
		{
			_context->BeginVSStage();
			GraphicsPipelineBuilderGuard	guard(IR::Builder::Builder::Get(), *_context);
			IR::Value::Var<VertexType>		in_vert("_in_vertex", true);
			IR::Value::Var<GPU::Math::Vec4> gl_Position("gl_Position", true);
			vertexFunc(in_vert, gl_Position);
		}
		_context->EndVSStage();
		_vsCode = _context->GetVertexShaderCode();

		// Stage 2: Fragment Shader
		{
			_context->BeginFSStage();
			GraphicsPipelineBuilderGuard	guard(IR::Builder::Builder::Get(), *_context);
			IR::Value::Var<GPU::Math::Vec4> fragColor("fragColor", true);
			fragmentFunc(fragColor);
		}
		_context->EndFSStage();
		_fsCode			   = _context->GetFragmentShaderCode();
		_simpleVertexInput = false;
	}

	void BuildShadersSimple(const VertexFuncSimple &vertexFunc, const FragmentFunc &fragmentFunc) {
		_context = std::make_unique<GraphicsBuildContext>();

		{
			_context->BeginVSStage();
			GraphicsPipelineBuilderGuard	guard(IR::Builder::Builder::Get(), *_context);
			IR::Value::Var<GPU::Math::Vec4> gl_Position("gl_Position", true);
			vertexFunc(gl_Position);
		}
		_context->EndVSStage();
		_vsCode = _context->GetVertexShaderCode();

		{
			_context->BeginFSStage();
			GraphicsPipelineBuilderGuard	guard(IR::Builder::Builder::Get(), *_context);
			IR::Value::Var<GPU::Math::Vec4> fragColor("fragColor", true);
			fragmentFunc(fragColor);
		}
		_context->EndFSStage();
		_fsCode			   = _context->GetFragmentShaderCode();
		_simpleVertexInput = true;
	}

	void EnsureCompiled();

	/** Draw implementation shared by all Draw/DrawIndexed variants. */
	void DrawImpl(Backend::TextureHandle colorHandle, Backend::TextureHandle depthHandle, uint32_t width,
				  uint32_t height, uint32_t vertexCount, uint32_t indexCount, bool indexed, bool sync);

	/** Generate vertex layout for a struct type.
	 *  Uses a single vec4 attribute slot covering the full struct. */
	template <typename T> static void GenerateVertexLayout(std::vector<Backend::VertexLayoutEntry> &layout) {
		layout.clear();
		auto formatForGLSLType = [](const std::string &type) -> Backend::PixelFormat {
			if (type == "float")
				return Backend::PixelFormat::R32F;
			if (type == "vec2")
				return Backend::PixelFormat::RG32F;
			if (type == "vec3")
				return Backend::PixelFormat::RGB32F;
			if (type == "vec4")
				return Backend::PixelFormat::RGBA32F;
			if (type == "int")
				return Backend::PixelFormat::R32I;
			if (type == "ivec2")
				return Backend::PixelFormat::RG32I;
			if (type == "ivec3")
				return Backend::PixelFormat::RGB32I;
			if (type == "ivec4")
				return Backend::PixelFormat::RGBA32I;
			if (type == "uint")
				return Backend::PixelFormat::R32UI;
			if (type == "uvec2")
				return Backend::PixelFormat::RG32UI;
			if (type == "uvec3")
				return Backend::PixelFormat::RGB32UI;
			if (type == "uvec4")
				return Backend::PixelFormat::RGBA32UI;
			throw std::runtime_error("GraphicsPipeline: unsupported vertex attribute GLSL type: " + type);
		};

		if constexpr (GPU::Meta::StructMeta<T>::isRegistered) {
			auto fields = GPU::Meta::StructMeta<T>::GetFieldInfos();
			layout.reserve(fields.size());
			for (uint32_t i = 0; i < fields.size(); ++i) {
				Backend::VertexLayoutEntry entry;
				entry.location = i;
				entry.offset	  = static_cast<uint32_t>(fields[i].cppOffset);
				entry.format	  = formatForGLSLType(fields[i].glslType);
				layout.push_back(entry);
			}
		} else {
			Backend::VertexLayoutEntry entry;
			entry.location = 0;
			entry.offset	  = 0;
			entry.format	  = formatForGLSLType(std::string(GPU::Meta::GetGLSLTypeName<T>()));
			layout.push_back(entry);
		}
	}

	template <typename T> static std::string GenerateVertexInputSetupCode() {
		if constexpr (GPU::Meta::StructMeta<T>::isRegistered) {
			std::string code =
				std::format("{} _in_vertex;\n", std::string(GPU::Meta::StructMeta<T>::glslTypeName));
			auto fields = GPU::Meta::StructMeta<T>::GetFieldInfos();
			for (uint32_t i = 0; i < fields.size(); ++i) {
				code += std::format("_in_vertex.{} = a_{};\n", fields[i].name, i);
			}
			return code;
		} else {
			return std::format("{} _in_vertex = a_0;\n", std::string(GPU::Meta::GetGLSLTypeName<T>()));
		}
	}

	std::string								_name;
	std::unique_ptr<GraphicsBuildContext>	_context;
	std::string								_vsCode;
	std::string								_fsCode;
	std::vector<Backend::VertexLayoutEntry> _vertexLayout;
	bool									_simpleVertexInput = true;

	Backend::ShaderHandle					_vsHandle		   = Backend::INVALID_SHADER_HANDLE;
	Backend::ShaderHandle					_fsHandle		   = Backend::INVALID_SHADER_HANDLE;
	Backend::PipelineHandle					_pipeline		   = Backend::INVALID_PIPELINE_HANDLE;
	bool									_compiled		   = false;

	bool									_hasVertexBuffer   = false;
	bool									_hasIndexBuffer	   = false;
	Backend::BufferHandle					_vertexBufferHandle = Backend::INVALID_BUFFER_HANDLE;
	uint32_t								_vertexStride	   = 0;
	Backend::BufferHandle					_indexBufferHandle = Backend::INVALID_BUFFER_HANDLE;
	uint32_t								_indexCount		   = 0;
};

} // namespace GPU::Kernel

#endif
