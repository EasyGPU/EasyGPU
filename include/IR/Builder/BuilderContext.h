#pragma once

/**
 * BuilderContext.h:
 *      @Descripiton    :   The context for the builder to bind
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */
#ifndef EASYGPU_BUILDERCONTEXT_H
#define EASYGPU_BUILDERCONTEXT_H

#include <Runtime/PixelFormat.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>

namespace GPU::IR::Builder {
    /**
     * Generation state for callable functions
     */
    struct CallableGenState {
        bool declared = false;  // Forward declaration generated
        bool defined = false;   // Function body generated
    };

    // Forward declaration
    using PixelFormat = GPU::Runtime::PixelFormat;
    /**
     * The context for the builder to bind, which provide a series of abstracted API
     * for the class to accomplish
     */
    class BuilderContext {
    public:
        virtual ~BuilderContext() = default;

    public:
        /**
         * Pushing the translated code to the context
         * @param Code The coded that translated from the builder
         */
        virtual void PushTranslatedCode(std::string Code) = 0;

        /**
         * Assigning the variable name
         * @return The variable name assigned
         */
        virtual std::string AssignVarName() = 0;

    public:
        /**
         * Check if a struct type is already defined
         * @param TypeName The struct type name
         * @return True if already defined
         */
        virtual bool HasStructDefinition(const std::string &TypeName) const = 0;

        /**
         * Add a struct type definition
         * @param TypeName The struct type name
         * @param Definition The GLSL struct definition code
         */
        virtual void AddStructDefinition(const std::string &TypeName, const std::string &Definition) = 0;

        /**
         * Get all struct definitions
         * @return Vector of struct definitions in order of registration
         */
        virtual const std::vector<std::string> &GetStructDefinitions() const = 0;

    public:
        /**
         * Allocate a binding slot for buffer/image
         * @return The allocated binding slot index
         */
        virtual uint32_t AllocateBindingSlot() = 0;

        /**
         * Register a buffer for the kernel
         * @param binding The binding slot
         * @param typeName The element type name in GLSL
         * @param bufferName The buffer variable name
         * @param mode The buffer access mode
         */
        virtual void RegisterBuffer(uint32_t binding, const std::string& typeName,
                                   const std::string& bufferName, int mode) = 0;

        /**
         * Get the buffer declarations for GLSL
         * @return The buffer declaration string
         */
        virtual std::string GetBufferDeclarations() const = 0;

        /**
         * Get all registered buffer bindings
         * @return Vector of binding slots
         */
        virtual const std::vector<uint32_t>& GetBufferBindings() const = 0;

        /**
         * Bind a runtime GPU buffer to a binding slot
         * This is called by Buffer::Bind() to associate the actual GL buffer with the binding
         * @param binding The binding slot
         * @param bufferHandle The OpenGL buffer handle
         */
        virtual void BindRuntimeBuffer(uint32_t binding, uint32_t bufferHandle) = 0;

        /**
         * Get all runtime buffer bindings for dispatch
         * @return Map of binding slot -> OpenGL buffer handle
         */
        virtual const std::unordered_map<uint32_t, uint32_t>& GetRuntimeBufferBindings() const = 0;

    public:
        // ===================================================================
        // Texture Support
        // ===================================================================
        
        /**
         * Allocate a binding slot for texture/image
         * @return The allocated binding slot index
         */
        virtual uint32_t AllocateTextureBinding() = 0;

        /**
         * Register a texture for the kernel
         * @param binding The binding slot
         * @param format The pixel format
         * @param textureName The texture variable name in GLSL
         * @param width Texture width
         * @param height Texture height
         */
        virtual void RegisterTexture(uint32_t binding, PixelFormat format,
                                    const std::string& textureName,
                                    uint32_t width, uint32_t height) = 0;

        /**
         * Get the texture declarations for GLSL
         * @return The texture declaration string
         */
        virtual std::string GetTextureDeclarations() const = 0;

        /**
         * Get all registered texture bindings
         * @return Vector of binding slots
         */
        virtual const std::vector<uint32_t>& GetTextureBindings() const = 0;

        /**
         * Bind a runtime GPU texture to a binding slot
         * This is called by Texture2D::Bind() to associate the actual GL texture with the binding
         * @param binding The binding slot
         * @param textureHandle The OpenGL texture handle
         */
        virtual void BindRuntimeTexture(uint32_t binding, uint32_t textureHandle) = 0;

        /**
         * Get all runtime texture bindings for dispatch
         * @return Map of binding slot -> OpenGL texture handle
         */
        virtual const std::unordered_map<uint32_t, uint32_t>& GetRuntimeTextureBindings() const = 0;

    public:
        // ===================================================================
        // Uniform Support
        // ===================================================================
        
        /**
         * Register a uniform variable for the kernel
         * @param typeName The GLSL type name
         * @param uniformPtr Pointer to the Uniform object (as void* for type erasure)
         * @param uploadFunc Function to upload the uniform value to GPU
         * @return The assigned uniform variable name in GLSL
         */
        virtual std::string RegisterUniform(const std::string& typeName, void* uniformPtr, 
                                            std::function<void(uint32_t program, const std::string& name, void* ptr)> uploadFunc) = 0;

        /**
         * Get the uniform declarations for GLSL
         * @return The uniform declaration string
         */
        virtual std::string GetUniformDeclarations() const = 0;

    public:
        // ===================================================================
        // Callable Function Support
        // ===================================================================
        
        /**
         * Get the generation state for a callable in this context
         * @param callablePtr Pointer to the callable object (as void* to avoid template dependency)
         * @return Reference to the generation state
         */
        virtual CallableGenState &GetCallableState(const void *callablePtr) {
            return _callableStates[callablePtr];
        }

        /**
         * Add a callable function declaration (forward declaration)
         * @param declaration The function prototype string
         */
        virtual void AddCallableDeclaration(const std::string &declaration) = 0;

        /**
         * Register a callable body generator function
         * This will be called later to generate the function body after main()
         * @param generator The function that generates the callable body
         */
        virtual void AddCallableBodyGenerator(std::function<void()> generator) = 0;

        /**
         * Enter callable body generation mode
         * Pushes a new code buffer for collecting callable body code
         */
        virtual void PushCallableBody() = 0;

        /**
         * Exit callable body generation mode
         * Pops the callable body code buffer and stores it for later output
         */
        virtual void PopCallableBody() = 0;

        /**
         * Get all callable function declarations
         * @return Vector of function declarations
         */
        virtual std::vector<std::string> GetCallableDeclarations() const = 0;

        /**
         * Generate all callable function bodies
         * This should be called after main() generation
         * @return The complete callable function definitions string
         */
        virtual std::string GenerateCallableBodies() = 0;

    protected:
        std::unordered_map<const void*, CallableGenState> _callableStates;
    };
}

#endif //EASYGPU_BUILDERCONTEXT_H
