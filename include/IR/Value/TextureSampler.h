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

#include <IR/Value/Var.h>
#include <IR/Value/Expr.h>
#include <IR/Value/ExprVector.h>
#include <IR/Builder/Builder.h>
#include <Runtime/PixelFormat.h>

#include <format>
#include <string>

// Forward declaration
namespace GPU::Runtime {
    template<PixelFormat Format>
    class Texture2D;
}

namespace GPU::IR::Value {

    /**
     * Texture sampler for fragment shader DSL access
     * Uses texture() for sampling instead of imageLoad/imageStore
     * @tparam Format The pixel format of the texture
     */
    template<Runtime::PixelFormat Format>
    class TextureSampler2D {
    public:
        TextureSampler2D(std::string textureName, uint32_t binding, uint32_t width, uint32_t height)
            : _textureName(std::move(textureName))
            , _binding(binding)
            , _width(width)
            , _height(height) {
        }

        [[nodiscard]] uint32_t GetBinding() const { return _binding; }
        [[nodiscard]] const std::string& GetTextureName() const { return _textureName; }
        [[nodiscard]] uint32_t GetTextureWidth() const { return _width; }
        [[nodiscard]] uint32_t GetTextureHeight() const { return _height; }
        static constexpr Runtime::PixelFormat GetFormat() { return Format; }

    public:
        // =======================================================================
        // Sample operations - texture(texture, vec2(uv))
        // =======================================================================

        /**
         * Sample texture at UV coordinates (0-1 range)
         * @param uv UV coordinates (0,0) to (1,1)
         * @return Vec4 color value
         */
        [[nodiscard]] Var<GPU::Math::Vec4> Sample(const Var<GPU::Math::Vec2>& uv) const {
            std::string uvStr = Builder::Builder::Get().BuildNode(*uv.Load().get());
            std::string code = std::format("texture({}, {})", _textureName, uvStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Sample(const Expr<GPU::Math::Vec2>& uv) const {
            std::string uvStr = Builder::Builder::Get().BuildNode(*uv.Node());
            std::string code = std::format("texture({}, {})", _textureName, uvStr);
            return Var<GPU::Math::Vec4>(code);
        }

        /**
         * Sample texture at explicit float coordinates
         */
        [[nodiscard]] Var<GPU::Math::Vec4> Sample(const Var<float>& u, const Var<float>& v) const {
            std::string uStr = Builder::Builder::Get().BuildNode(*u.Load().get());
            std::string vStr = Builder::Builder::Get().BuildNode(*v.Load().get());
            std::string code = std::format("texture({}, vec2({}, {}))", _textureName, uStr, vStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Sample(const Expr<float>& u, const Var<float>& v) const {
            std::string uStr = Builder::Builder::Get().BuildNode(*u.Node());
            std::string vStr = Builder::Builder::Get().BuildNode(*v.Load().get());
            std::string code = std::format("texture({}, vec2({}, {}))", _textureName, uStr, vStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Sample(const Var<float>& u, const Expr<float>& v) const {
            std::string uStr = Builder::Builder::Get().BuildNode(*u.Load().get());
            std::string vStr = Builder::Builder::Get().BuildNode(*v.Node());
            std::string code = std::format("texture({}, vec2({}, {}))", _textureName, uStr, vStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Sample(const Expr<float>& u, const Expr<float>& v) const {
            std::string uStr = Builder::Builder::Get().BuildNode(*u.Node());
            std::string vStr = Builder::Builder::Get().BuildNode(*v.Node());
            std::string code = std::format("texture({}, vec2({}, {}))", _textureName, uStr, vStr);
            return Var<GPU::Math::Vec4>(code);
        }

        // Literal float versions
        [[nodiscard]] Var<GPU::Math::Vec4> Sample(float u, const Var<float>& v) const {
            std::string vStr = Builder::Builder::Get().BuildNode(*v.Load().get());
            std::string code = std::format("texture({}, vec2({}, {}))", _textureName, u, vStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Sample(const Var<float>& u, float v) const {
            std::string uStr = Builder::Builder::Get().BuildNode(*u.Load().get());
            std::string code = std::format("texture({}, vec2({}, {}))", _textureName, uStr, v);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Sample(float u, float v) const {
            std::string code = std::format("texture({}, vec2({}, {}))", _textureName, u, v);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Sample(float u, const Expr<float>& v) const {
            std::string vStr = Builder::Builder::Get().BuildNode(*v.Node());
            std::string code = std::format("texture({}, vec2({}, {}))", _textureName, u, vStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Sample(const Expr<float>& u, float v) const {
            std::string uStr = Builder::Builder::Get().BuildNode(*u.Node());
            std::string code = std::format("texture({}, vec2({}, {}))", _textureName, uStr, v);
            return Var<GPU::Math::Vec4>(code);
        }

        // Vec2 literal
        [[nodiscard]] Var<GPU::Math::Vec4> Sample(const GPU::Math::Vec2& uv) const {
            std::string code = std::format("texture({}, vec2({}, {}))", _textureName, uv.x, uv.y);
            return Var<GPU::Math::Vec4>(code);
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
        std::string _textureName;
        uint32_t _binding;
        uint32_t _width;
        uint32_t _height;
    };

    /**
     * Type alias for convenience
     */
    template<Runtime::PixelFormat Format>
    using sampler2D = TextureSampler2D<Format>;

} // namespace GPU::IR::Value

#endif // EASYGPU_TEXTURE_SAMPLER_H
