#pragma once

/**
 * TextureSlot.h:
 *      @Descripiton    :   Dynamic texture slot for runtime resource switching
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/6/2026
 * 
 * This class allows switching texture bindings at runtime without recompiling kernels.
 * 
 * Usage:
 *   TextureSlot<RGBA8> albedoSlot;  // Declare slot for 2D texture
 *   
 *   Kernel2D kernel([&](Int x, Int y) {
 *       auto albedo = albedoSlot.Bind();  // Bind the slot (not specific texture)
 *       Var<Vec4> color = albedo.Read(x, y);
 *   });
 *   
 *   Texture2D<RGBA8> tex1(1024, 1024);
 *   Texture2D<RGBA8> tex2(1024, 1024);
 *   
 *   albedoSlot.Attach(tex1);      // Attach tex1
 *   kernel.Dispatch(64, 64, true); // Execute with tex1
 *   
 *   albedoSlot.Attach(tex2);      // Switch to tex2 (no recompilation!)
 *   kernel.Dispatch(64, 64, true); // Execute with tex2
 * 
 * NOTE: Texture3D and Texture3DSlot have been removed due to OpenGL driver compatibility issues.
 */
#ifndef EASYGPU_TEXTURESLOT_H
#define EASYGPU_TEXTURESLOT_H

#include <IR/Value/TextureRef.h>
#include <IR/Value/TextureSampler.h>
#include <IR/Builder/Builder.h>
#include <Runtime/PixelFormat.h>
#include <Runtime/Texture.h>

#include <string>
#include <stdexcept>

namespace GPU::Runtime {
    // Forward declaration
    template<PixelFormat Format>
    class Texture2D;

    // Forward declaration for friend access
    class KernelBuildContext;

    /**
     * Texture slot base class (non-template)
     * Used for type erasure in KernelBuildContext
     */
    class TextureSlotBase {
    public:
        virtual ~TextureSlotBase() = default;
        
        /**
         * Get the OpenGL texture handle of the attached texture
         * @return The OpenGL texture ID, or 0 if not attached
         */
        virtual uint32_t GetHandle() const = 0;
        
        /**
         * Check if a texture is currently attached
         * @return true if attached, false otherwise
         */
        virtual bool IsAttached() const = 0;
        
        /**
         * Get the pixel format of this slot
         * @return The pixel format
         */
        virtual PixelFormat GetFormat() const = 0;
        
        /**
         * Get the texture dimensions
         * @param[out] width The texture width
         * @param[out] height The texture height
         */
        virtual void GetDimensions(uint32_t& width, uint32_t& height) const = 0;
        
        /**
         * Get the binding slot assigned by KernelBuildContext
         * @return The binding slot index, or -1 if not bound
         */
        int GetBinding() const { return _binding; }
        
        /**
         * Get the variable name in GLSL
         * @return The GLSL variable name
         */
        const std::string& GetName() const { return _name; }
        
        /**
         * Set the binding information (called by KernelBuildContext)
         * @param binding The binding slot
         * @param name The GLSL variable name
         */
        void SetBindingInfo(int binding, const std::string& name) {
            _binding = binding;
            _name = name;
        }
        
    protected:
        int _binding = -1;           // Assigned by KernelBuildContext during Bind()
        std::string _name;           // GLSL variable name
    };

    /**
     * 2D Texture slot for dynamic texture switching at runtime
     * @tparam Format The pixel format of the texture
     * 
     * This class allows you to define a kernel once and switch between different
     * 2D textures at runtime without recompiling the kernel.
     * 
     * Example - Deferred shading with multiple G-Buffer textures:
     *   TextureSlot<RGBA8> gAlbedo;      // Albedo channel
     *   TextureSlot<RGBA32F> gNormal;    // Normal channel
     *   
     *   Kernel2D shade([&](Int x, Int y) {
     *       auto albedo = gAlbedo.Bind();
     *       auto normal = gNormal.Bind();
     *       // ... lighting calculation ...
     *   });
     *   
     *   Texture2D<RGBA8> sceneA_albedo(1920, 1080);
     *   Texture2D<RGBA32F> sceneA_normal(1920, 1080);
     *   Texture2D<RGBA8> sceneB_albedo(1920, 1080);
     *   Texture2D<RGBA32F> sceneB_normal(1920, 1080);
     *   
     *   // Render scene A
     *   gAlbedo.Attach(sceneA_albedo);
     *   gNormal.Attach(sceneA_normal);
     *   shade.Dispatch(120, 68, true);
     *   
     *   // Render scene B (same kernel, no recompilation)
     *   gAlbedo.Attach(sceneB_albedo);
     *   gNormal.Attach(sceneB_normal);
     *   shade.Dispatch(120, 68, true);
     */
    template<PixelFormat Format>
    class TextureSlot : public TextureSlotBase {
    public:
        /**
         * Default constructor - creates an unattached slot
         */
        TextureSlot() = default;
        
        /**
         * Destructor
         */
        ~TextureSlot() override = default;
        
        // Disable copy
        TextureSlot(const TextureSlot&) = delete;
        TextureSlot& operator=(const TextureSlot&) = delete;
        
        // Enable move
        TextureSlot(TextureSlot&&) noexcept = default;
        TextureSlot& operator=(TextureSlot&&) noexcept = default;
        
    public:
        // ===================================================================
        // Runtime API - Called outside kernel definition
        // ===================================================================
        
        /**
         * Attach a 2D texture to this slot
         * The texture will be used when the kernel is next dispatched.
         * @param texture The 2D texture to attach
         */
        void Attach(Texture2D<Format>& texture) {
            _texture = &texture;
        }
        
        /**
         * Detach the current texture
         */
        void Detach() {
            _texture = nullptr;
        }
        
        /**
         * Check if a texture is currently attached
         * @return true if attached, false otherwise
         */
        bool IsAttached() const override {
            return _texture != nullptr;
        }
        
        /**
         * Get the currently attached texture
         * @return Pointer to the attached texture, or nullptr if not attached
         */
        Texture2D<Format>* GetAttached() const {
            return _texture;
        }
        
        /**
         * Get the OpenGL texture handle of the attached texture
         * @return The OpenGL texture ID, or 0 if not attached
         */
        uint32_t GetHandle() const override {
            return _texture ? _texture->GetHandle() : 0;
        }
        
        /**
         * Get the pixel format of this slot
         * @return The pixel format
         */
        PixelFormat GetFormat() const override {
            return Format;
        }
        
        /**
         * Get the texture dimensions
         * @param[out] width The texture width
         * @param[out] height The texture height
         */
        void GetDimensions(uint32_t& width, uint32_t& height) const override {
            if (_texture) {
                width = _texture->GetWidth();
                height = _texture->GetHeight();
            } else {
                width = height = 0;
            }
        }
        
    public:
        // ===================================================================
        // DSL API - Called inside kernel definition
        // ===================================================================
        
        /**
         * Bind this slot to the current kernel being defined
         * This allocates a binding slot and returns a TextureRef for DSL access.
         * The actual texture binding happens at dispatch time.
         * @return TextureRef<Format> for DSL access (imageLoad/imageStore)
         */
        [[nodiscard]] IR::Value::TextureRef<Format> Bind() {
            auto* context = IR::Builder::Builder::Get().Context();
            if (!context) {
                throw std::runtime_error("TextureSlot::Bind() called outside of Kernel definition");
            }
            
            // Register this slot with the context
            // The actual texture binding happens at dispatch time via GetTextureSlots()
            context->RegisterTextureSlot(this);
            
            // Get dimensions (will be 0 if not attached, but that's OK for code generation)
            uint32_t width = 0, height = 0;
            GetDimensions(width, height);
            
            // Return TextureRef using our assigned name and binding
            return IR::Value::TextureRef<Format>(_name, static_cast<uint32_t>(_binding), width, height);
        }
        
        /**
         * Bind this slot as a sampler to the current kernel being defined
         * This allocates a binding slot and returns a TextureSampler2D for DSL access.
         * Use this for texture() sampling in fragment shaders.
         * @return TextureSampler2D<Format> for DSL access (texture sampling)
         */
        [[nodiscard]] IR::Value::TextureSampler2D<Format> BindSampler() {
            auto* context = IR::Builder::Builder::Get().Context();
            if (!context) {
                throw std::runtime_error("TextureSlot::BindSampler() called outside of Kernel definition");
            }
            
            // Register this slot with the context
            context->RegisterTextureSlot(this);
            
            // Get dimensions
            uint32_t width = 0, height = 0, depth = 0;
            GetDimensions(width, height, depth);
            
            // Return TextureSampler2D using our assigned name and binding
            return IR::Value::TextureSampler2D<Format>(_name, static_cast<uint32_t>(_binding), width, height);
        }
        
    private:
        Texture2D<Format>* _texture = nullptr;  // Currently attached texture
        
        // Grant KernelBuildContext access to protected members
        friend class KernelBuildContext;
    };
}

#endif // EASYGPU_TEXTURESLOT_H
