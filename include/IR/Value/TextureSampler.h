#pragma once

/**
 * TextureSampler.h:
 *      @Descripiton    :   Texture sampler for FragmentKernel (uses texture() instead of imageLoad)
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/19/2026
 *
 * Usage in FragmentKernel:
 *   Texture2D<PixelFormat::RGBA8> tex(1024, 1024);
 *   FragmentKernel2D kernel([&](Var<Vec4>& fragColor) {
 *       auto sampler = tex.BindSampler();
 *       Var<Vec4> color = sampler.Sample(vec2(0.5, 0.5));  // UV sampling
 *       fragColor = color;
 *   });
 */

#ifndef EASYGPU_TEXTURE_SAMPLER_H
#define EASYGPU_TEXTURE_SAMPLER_H

#include <IR/Builder/Builder.h>
#include <IR/Value/Var.h>
#include <Runtime/PixelFormat.h>

#include <format>
#include <string>

// Forward declaration
namespace GPU::Runtime {
template <PixelFormat Format> class Texture2D;
template <PixelFormat Format> class Texture3D;
}

namespace GPU::IR::Value {

namespace Detail {

template <Runtime::PixelFormat Format> struct TextureSamplerValueType {
	using type						= GPU::Math::Vec4;
	static constexpr bool supported = true;
};

template <> struct TextureSamplerValueType<Runtime::PixelFormat::R32I> {
	using type						= GPU::Math::IVec4;
	static constexpr bool supported = true;
};

template <> struct TextureSamplerValueType<Runtime::PixelFormat::RG32I> {
	using type						= GPU::Math::IVec4;
	static constexpr bool supported = true;
};

template <> struct TextureSamplerValueType<Runtime::PixelFormat::RGBA32I> {
	using type						= GPU::Math::IVec4;
	static constexpr bool supported = true;
};

template <> struct TextureSamplerValueType<Runtime::PixelFormat::R32UI> {
	using type						= GPU::Math::IVec4;
	static constexpr bool supported = false;
};

template <> struct TextureSamplerValueType<Runtime::PixelFormat::RG32UI> {
	using type						= GPU::Math::IVec4;
	static constexpr bool supported = false;
};

template <> struct TextureSamplerValueType<Runtime::PixelFormat::RGBA32UI> {
	using type						= GPU::Math::IVec4;
	static constexpr bool supported = false;
};

} // namespace Detail

/**
 * Texture sampler for fragment shader DSL access
 * Uses texture() for sampling instead of imageLoad/imageStore
 * @tparam Format The pixel format of the texture
 */
template <Runtime::PixelFormat Format> class TextureSampler2D {
public:
	using SampleType = typename Detail::TextureSamplerValueType<Format>::type;

	TextureSampler2D(std::string textureName, uint32_t binding, uint32_t width, uint32_t height)
		: _textureName(std::move(textureName)), _binding(binding), _width(width), _height(height) {
		static_assert(
			Detail::TextureSamplerValueType<Format>::supported,
			"Unsigned integer sampled textures require uint/uvec DSL support, which EasyGPU does not provide yet");
	}

	/**
	 * Constructor for function parameter references.
	 * Used when TextureSampler2D is passed as a callable parameter.
	 * Binding/width/height are not needed in function body since only the name is used.
	 */
	explicit TextureSampler2D(std::string textureName)
		: _textureName(std::move(textureName)), _binding(0), _width(0), _height(0) {
		static_assert(
			Detail::TextureSamplerValueType<Format>::supported,
			"Unsigned integer sampled textures require uint/uvec DSL support, which EasyGPU does not provide yet");
	}

	[[nodiscard]] uint32_t GetBinding() const {
		return _binding;
	}
	[[nodiscard]] const std::string &GetTextureName() const {
		return _textureName;
	}
	[[nodiscard]] uint32_t GetTextureWidth() const {
		return _width;
	}
	[[nodiscard]] uint32_t GetTextureHeight() const {
		return _height;
	}
	static constexpr Runtime::PixelFormat GetFormat() {
		return Format;
	}

public:
	// =======================================================================
	// Sample operations - texture(texture, vec2(uv))
	// =======================================================================

	/**
	 * Sample texture at UV coordinates (0-1 range)
	 * @param uv UV coordinates (0,0) to (1,1)
	 * @return Vec4 color value
	 */
	[[nodiscard]] Var<SampleType> Sample(const Var<GPU::Math::Vec2> &uv) const {
		std::string uvStr = Builder::Builder::Get().BuildNode(*uv.Load().get());
		return MakeSampleVar(std::format("texture({}, {})", _textureName, uvStr));
	}

	[[nodiscard]] Var<SampleType> Sample(const Expr<GPU::Math::Vec2> &uv) const {
		std::string uvStr = Builder::Builder::Get().BuildNode(*uv.Node());
		return MakeSampleVar(std::format("texture({}, {})", _textureName, uvStr));
	}

	/**
	 * Sample texture at explicit float coordinates
	 */
	[[nodiscard]] Var<SampleType> Sample(const Var<float> &u, const Var<float> &v) const {
		std::string uStr = Builder::Builder::Get().BuildNode(*u.Load().get());
		std::string vStr = Builder::Builder::Get().BuildNode(*v.Load().get());
		return MakeSampleVar(std::format("texture({}, vec2({}, {}))", _textureName, uStr, vStr));
	}

	[[nodiscard]] Var<SampleType> Sample(const Expr<float> &u, const Var<float> &v) const {
		std::string uStr = Builder::Builder::Get().BuildNode(*u.Node());
		std::string vStr = Builder::Builder::Get().BuildNode(*v.Load().get());
		return MakeSampleVar(std::format("texture({}, vec2({}, {}))", _textureName, uStr, vStr));
	}

	[[nodiscard]] Var<SampleType> Sample(const Var<float> &u, const Expr<float> &v) const {
		std::string uStr = Builder::Builder::Get().BuildNode(*u.Load().get());
		std::string vStr = Builder::Builder::Get().BuildNode(*v.Node());
		return MakeSampleVar(std::format("texture({}, vec2({}, {}))", _textureName, uStr, vStr));
	}

	[[nodiscard]] Var<SampleType> Sample(const Expr<float> &u, const Expr<float> &v) const {
		std::string uStr = Builder::Builder::Get().BuildNode(*u.Node());
		std::string vStr = Builder::Builder::Get().BuildNode(*v.Node());
		return MakeSampleVar(std::format("texture({}, vec2({}, {}))", _textureName, uStr, vStr));
	}

	// Literal float versions
	[[nodiscard]] Var<SampleType> Sample(float u, const Var<float> &v) const {
		std::string vStr = Builder::Builder::Get().BuildNode(*v.Load().get());
		return MakeSampleVar(std::format("texture({}, vec2({}, {}))", _textureName, u, vStr));
	}

	[[nodiscard]] Var<SampleType> Sample(const Var<float> &u, float v) const {
		std::string uStr = Builder::Builder::Get().BuildNode(*u.Load().get());
		return MakeSampleVar(std::format("texture({}, vec2({}, {}))", _textureName, uStr, v));
	}

	[[nodiscard]] Var<SampleType> Sample(float u, float v) const {
		return MakeSampleVar(std::format("texture({}, vec2({}, {}))", _textureName, u, v));
	}

	[[nodiscard]] Var<SampleType> Sample(float u, const Expr<float> &v) const {
		std::string vStr = Builder::Builder::Get().BuildNode(*v.Node());
		return MakeSampleVar(std::format("texture({}, vec2({}, {}))", _textureName, u, vStr));
	}

	[[nodiscard]] Var<SampleType> Sample(const Expr<float> &u, float v) const {
		std::string uStr = Builder::Builder::Get().BuildNode(*u.Node());
		return MakeSampleVar(std::format("texture({}, vec2({}, {}))", _textureName, uStr, v));
	}

	// Vec2 literal
	[[nodiscard]] Var<SampleType> Sample(const GPU::Math::Vec2 &uv) const {
		return MakeSampleVar(std::format("texture({}, vec2({}, {}))", _textureName, uv.x, uv.y));
	}

public:
	// =======================================================================
	// Size accessors - textureSize(texture, lod)
	// =======================================================================

	/**
	 * Get texture size
	 * @return Vec2 containing width and height
	 */
	[[nodiscard]] Var<GPU::Math::Vec2> GetSize() const {
		std::string code = std::format("vec2(textureSize({}, 0))", _textureName);
		return Var<GPU::Math::Vec2>(code);
	}

	/**
	 * Get texture width
	 */
	[[nodiscard]] Var<int> GetWidth() const {
		std::string code = std::format("textureSize({}, 0).x", _textureName);
		return Var<int>(code);
	}

	/**
	 * Get texture height
	 */
	[[nodiscard]] Var<int> GetHeight() const {
		std::string code = std::format("textureSize({}, 0).y", _textureName);
		return Var<int>(code);
	}

private:
	[[nodiscard]] Var<SampleType> MakeSampleVar(std::string code) const {
		return Var<SampleType>(code);
	}

	std::string _textureName;
	uint32_t	_binding;
	uint32_t	_width;
	uint32_t	_height;
};

/**
 * Type alias for convenience
 */
template <Runtime::PixelFormat Format> using sampler2D = TextureSampler2D<Format>;

/**
 * 3D Texture sampler for fragment shader DSL access
 * Uses texture() for sampling instead of imageLoad/imageStore
 * @tparam Format The pixel format of the texture
 */
template <Runtime::PixelFormat Format> class TextureSampler3D {
public:
	using SampleType = typename Detail::TextureSamplerValueType<Format>::type;

	TextureSampler3D(std::string textureName, uint32_t binding, uint32_t width, uint32_t height, uint32_t depth)
		: _textureName(std::move(textureName)), _binding(binding), _width(width), _height(height), _depth(depth) {
		static_assert(
			Detail::TextureSamplerValueType<Format>::supported,
			"Unsigned integer sampled textures require uint/uvec DSL support, which EasyGPU does not provide yet");
	}

	explicit TextureSampler3D(std::string textureName)
		: _textureName(std::move(textureName)), _binding(0), _width(0), _height(0), _depth(0) {
		static_assert(
			Detail::TextureSamplerValueType<Format>::supported,
			"Unsigned integer sampled textures require uint/uvec DSL support, which EasyGPU does not provide yet");
	}

	[[nodiscard]] uint32_t GetBinding() const {
		return _binding;
	}
	[[nodiscard]] const std::string &GetTextureName() const {
		return _textureName;
	}
	[[nodiscard]] uint32_t GetTextureWidth() const {
		return _width;
	}
	[[nodiscard]] uint32_t GetTextureHeight() const {
		return _height;
	}
	[[nodiscard]] uint32_t GetTextureDepth() const {
		return _depth;
	}
	static constexpr Runtime::PixelFormat GetFormat() {
		return Format;
	}

public:
	// Sample operations - texture(texture, vec3(uvw))
	[[nodiscard]] Var<SampleType> Sample(const Var<GPU::Math::Vec3> &uvw) const {
		std::string uvwStr = Builder::Builder::Get().BuildNode(*uvw.Load().get());
		return MakeSampleVar(std::format("texture({}, {})", _textureName, uvwStr));
	}

	[[nodiscard]] Var<SampleType> Sample(const Expr<GPU::Math::Vec3> &uvw) const {
		std::string uvwStr = Builder::Builder::Get().BuildNode(*uvw.Node());
		return MakeSampleVar(std::format("texture({}, {})", _textureName, uvwStr));
	}

	[[nodiscard]] Var<SampleType> Sample(const GPU::Math::Vec3 &uvw) const {
		return MakeSampleVar(std::format("texture({}, vec3({}, {}, {}))", _textureName, uvw.x, uvw.y, uvw.z));
	}

public:
	// Size accessors - textureSize(texture, lod)
	[[nodiscard]] Var<GPU::Math::Vec3> GetSize() const {
		std::string code = std::format("vec3(textureSize({}, 0))", _textureName);
		return Var<GPU::Math::Vec3>(code);
	}

	[[nodiscard]] Var<int> GetWidth() const {
		std::string code = std::format("textureSize({}, 0).x", _textureName);
		return Var<int>(code);
	}

	[[nodiscard]] Var<int> GetHeight() const {
		std::string code = std::format("textureSize({}, 0).y", _textureName);
		return Var<int>(code);
	}

	[[nodiscard]] Var<int> GetDepth() const {
		std::string code = std::format("textureSize({}, 0).z", _textureName);
		return Var<int>(code);
	}

private:
	[[nodiscard]] Var<SampleType> MakeSampleVar(std::string code) const {
		return Var<SampleType>(code);
	}

	std::string _textureName;
	uint32_t    _binding;
	uint32_t    _width;
	uint32_t    _height;
	uint32_t    _depth;
};

/**
 * Type alias for convenience
 */
template <Runtime::PixelFormat Format> using sampler3D = TextureSampler3D<Format>;


} // namespace GPU::IR::Value

// Register TextureSampler2D as a valid ScalarType for Callable support
namespace GPU::Meta {
template <Runtime::PixelFormat Format> struct StructMeta<IR::Value::TextureSampler2D<Format>> {
	static constexpr bool		 isRegistered = true;
	static constexpr const char *glslTypeName = "sampler2D";
};

template <Runtime::PixelFormat Format> struct StructMeta<IR::Value::TextureSampler3D<Format>> {
	static constexpr bool		 isRegistered = true;
	static constexpr const char *glslTypeName = "sampler3D";
};
} // namespace GPU::Meta

#endif // EASYGPU_TEXTURE_SAMPLER_H
