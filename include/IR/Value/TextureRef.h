#pragma once

/**
 * TextureRef.h:
 *      @Descripiton    :   Texture reference for DSL access in Kernel
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 * 
 * Usage in Kernel:
 *   Texture2D<PixelFormat::RGBA8> tex(1024, 1024);
 *   Kernel1D kernel([&](Var<int>& id) {
 *       auto img = tex.Bind();
 *       Var<Vec4> color = img.Read(x, y);     // Read pixel
 *       img.Write(x, y, color * 0.5f);         // Write pixel
 *   });
 */
#ifndef EASYGPU_TEXTUREREF_H
#define EASYGPU_TEXTUREREF_H

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
     * Texture reference class for DSL access
     * @tparam Format The pixel format of the texture
     *
     * Usage:
     *   auto img = texture.Bind();
     *   Var<Vec4> color = img.Read(x, y);        // Read
     *   img.Write(x, y, Vec4(1.0f, 0.0f, 0.0f, 1.0f));  // Write
     */
    template<Runtime::PixelFormat Format>
    class TextureRef {
    public:
        TextureRef(std::string textureName, uint32_t binding, uint32_t width, uint32_t height)
                : _textureName(std::move(textureName)), _binding(binding), _width(width), _height(height) {
        }

        [[nodiscard]] uint32_t GetBinding() const { return _binding; }

        [[nodiscard]] const std::string &GetTextureName() const { return _textureName; }

        [[nodiscard]] uint32_t GetWidth() const { return _width; }

        [[nodiscard]] uint32_t GetHeight() const { return _height; }

        static constexpr Runtime::PixelFormat GetFormat() { return Format; }

    public:
        // =======================================================================
        // Read operations - imageLoad(texture, ivec2(x, y))
        // =======================================================================

        /**
         * Read pixel at integer coordinates
         * @param x X coordinate (0 to width-1)
         * @param y Y coordinate (0 to height-1)
         * @return Vec4 color value (automatically converted from format)
         */
        [[nodiscard]] Var<GPU::Math::Vec4> Read(const Var<int> &x, const Var<int> &y) const {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Load().get());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Load().get());
            std::string code = std::format("imageLoad({}, ivec2({}, {}))", _textureName, xStr, yStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Read(const Expr<int> &x, const Var<int> &y) const {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Node());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Load().get());
            std::string code = std::format("imageLoad({}, ivec2({}, {}))", _textureName, xStr, yStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Read(const Var<int> &x, const Expr<int> &y) const {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Load().get());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Node());
            std::string code = std::format("imageLoad({}, ivec2({}, {}))", _textureName, xStr, yStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Read(const Expr<int> &x, const Expr<int> &y) const {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Node());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Node());
            std::string code = std::format("imageLoad({}, ivec2({}, {}))", _textureName, xStr, yStr);
            return Var<GPU::Math::Vec4>(code);
        }

        // Literal integer versions
        [[nodiscard]] Var<GPU::Math::Vec4> Read(int x, const Var<int> &y) const {
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Load().get());
            std::string code = std::format("imageLoad({}, ivec2({}, {}))", _textureName, x, yStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Read(const Var<int> &x, int y) const {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Load().get());
            std::string code = std::format("imageLoad({}, ivec2({}, {}))", _textureName, xStr, y);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Read(int x, int y) const {
            std::string code = std::format("imageLoad({}, ivec2({}, {}))", _textureName, x, y);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Read(int x, const Expr<int> &y) const {
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Node());
            std::string code = std::format("imageLoad({}, ivec2({}, {}))", _textureName, x, yStr);
            return Var<GPU::Math::Vec4>(code);
        }

        [[nodiscard]] Var<GPU::Math::Vec4> Read(const Expr<int> &x, int y) const {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Node());
            std::string code = std::format("imageLoad({}, ivec2({}, {}))", _textureName, xStr, y);
            return Var<GPU::Math::Vec4>(code);
        }

    public:
        // =======================================================================
        // Write operations - imageStore(texture, ivec2(x, y), value)
        // =======================================================================

        /**
         * Write pixel at integer coordinates
         * @param x X coordinate
         * @param y Y coordinate
         * @param color Color value (Vec4 or expression)
         */
        void Write(const Var<int> &x, const Var<int> &y, const Var<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Load().get());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Load().get());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Load().get());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(const Expr<int> &x, const Var<int> &y, const Var<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Node());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Load().get());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Load().get());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(const Var<int> &x, const Expr<int> &y, const Var<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Load().get());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Node());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Load().get());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(const Expr<int> &x, const Expr<int> &y, const Var<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Node());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Node());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Load().get());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        // With Expr<Math::Vec4> color
        void Write(const Var<int> &x, const Var<int> &y, const Expr<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Load().get());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Load().get());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Node());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(const Expr<int> &x, const Var<int> &y, const Expr<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Node());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Load().get());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Node());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(const Var<int> &x, const Expr<int> &y, const Expr<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Load().get());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Node());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Node());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(const Expr<int> &x, const Expr<int> &y, const Expr<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Node());
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Node());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Node());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        // Literal integer coordinates
        void Write(int x, const Var<int> &y, const Var<GPU::Math::Vec4> &color) {
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Load().get());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Load().get());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, x, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(const Var<int> &x, int y, const Var<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Load().get());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Load().get());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, y, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(int x, int y, const Var<GPU::Math::Vec4> &color) {
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Load().get());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, x, y, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(int x, const Expr<int> &y, const Var<GPU::Math::Vec4> &color) {
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Node());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Load().get());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, x, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(const Expr<int> &x, int y, const Var<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Node());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Load().get());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, y, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(int x, int y, const Expr<GPU::Math::Vec4> &color) {
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Node());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, x, y, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(int x, const Var<int> &y, const Expr<GPU::Math::Vec4> &color) {
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Load().get());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Node());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, x, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(const Var<int> &x, int y, const Expr<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Load().get());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Node());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, y, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(int x, const Expr<int> &y, const Expr<GPU::Math::Vec4> &color) {
            std::string yStr = Builder::Builder::Get().BuildNode(*y.Node());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Node());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, x, yStr, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
        }

        void Write(const Expr<int> &x, int y, const Expr<GPU::Math::Vec4> &color) {
            std::string xStr = Builder::Builder::Get().BuildNode(*x.Node());
            std::string colorStr = Builder::Builder::Get().BuildNode(*color.Node());
            std::string code = std::format("imageStore({}, ivec2({}, {}), {});\n", _textureName, xStr, y, colorStr);
            Builder::Builder::Get().Context()->PushTranslatedCode(code);
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
    using image2d = TextureRef<Format>;

} // namespace GPU::IR::Value

#endif // EASYGPU_TEXTUREREF_H
