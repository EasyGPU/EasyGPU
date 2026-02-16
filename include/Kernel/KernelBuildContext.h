/**
 * KernelBuildContext.h:
 *      @Descripiton    :   Ther build context for the kernel function class
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_KERNELBUILDCONTEXT_H
#define EASYGPU_KERNELBUILDCONTEXT_H

#include <IR/Builder/BuilderContext.h>

#include <Runtime/Buffer.h>
#include <Runtime/PixelFormat.h>

#include <stdexcept>
#include <unordered_map>
#include <stack>

namespace GPU::Kernel {
    /**
     * The exception for dimension out of range
     */
    class KernelDimensionOutOfRange : public std::out_of_range {
    public:
        KernelDimensionOutOfRange();
    };

    /**
     * The build context for the kernel
     */
    class KernelBuildContext : public IR::Builder::BuilderContext {
    public:
        /**
         * This constructor will construct the work size in default
         * When it is 1 dimension, the local_size_x = 256 while others=1
         * When it is 2 dimension, the local_size_x = 16 and local_size_y = 16 while others = 1
         * When it is 3 dimension, the local_size_x = 8 and local_size_y = 8 and local_size_z = 4
         * @param Dimension The dimension of the kernel
         */
        KernelBuildContext(int Dimension);

    public:
        void PushTranslatedCode(std::string Code) override;

        std::string AssignVarName() override;

        /**
         * Get the complete kernel code including struct definitions
         * @return The complete GLSL code
         */
        std::string GetCompleteCode();

    public:
        /**
         * Check if a struct type is already defined
         * @param TypeName The struct type name
         * @return True if already defined
         */
        bool HasStructDefinition(const std::string &TypeName) const override;

        /**
         * Add a struct type definition
         * @param TypeName The struct type name
         * @param Definition The GLSL struct definition code
         */
        void AddStructDefinition(const std::string &TypeName, const std::string &Definition) override;

        /**
         * Get all struct definitions
         * @return Vector of struct definitions in order of registration
         */
        const std::vector<std::string> &GetStructDefinitions() const override;

    public:
        /**
         * Allocate a binding slot for buffer/image
         * @return The allocated binding slot index
         */
        uint32_t AllocateBindingSlot() override;

        /**
         * Register a buffer for the kernel
         * @param binding The binding slot
         * @param typeName The element type name in GLSL
         * @param bufferName The buffer variable name
         * @param mode The buffer access mode
         */
        void RegisterBuffer(uint32_t binding, const std::string& typeName,
                           const std::string& bufferName, int mode) override;

        /**
         * Get the buffer declarations for GLSL
         * @return The buffer declaration string
         */
        std::string GetBufferDeclarations() const override;

        /**
         * Get all registered buffer bindings
         * @return Vector of binding slots
         */
        const std::vector<uint32_t>& GetBufferBindings() const override {
            return _bufferBindings;
        }

    public:
        /**
         * Bind a runtime GPU buffer to a binding slot
         * This is called by Buffer::Bind() to associate the actual GL buffer with the binding
         * @param binding The binding slot
         * @param bufferHandle The OpenGL buffer handle
         */
        void BindRuntimeBuffer(uint32_t binding, uint32_t bufferHandle) override {
            _runtimeBuffers[binding] = bufferHandle;
        }

        /**
         * Get all runtime buffer bindings for dispatch
         * @return Map of binding slot -> OpenGL buffer handle
         */
        const std::unordered_map<uint32_t, uint32_t>& GetRuntimeBufferBindings() const override {
            return _runtimeBuffers;
        }

    public:
        // ===================================================================
        // Texture Support
        // ===================================================================
        
        /**
         * Allocate a binding slot for texture/image
         * @return The allocated binding slot index
         */
        uint32_t AllocateTextureBinding() override;

        /**
         * Register a texture for the kernel
         * @param binding The binding slot
         * @param format The pixel format
         * @param textureName The texture variable name in GLSL
         * @param width Texture width
         * @param height Texture height
         */
        void RegisterTexture(uint32_t binding, Runtime::PixelFormat format,
                            const std::string& textureName,
                            uint32_t width, uint32_t height) override;

        /**
         * Get the texture declarations for GLSL
         * @return The texture declaration string
         */
        std::string GetTextureDeclarations() const override;

        /**
         * Get all registered texture bindings
         * @return Vector of binding slots
         */
        const std::vector<uint32_t>& GetTextureBindings() const override {
            return _textureBindings;
        }

        /**
         * Bind a runtime GPU texture to a binding slot
         * This is called by Texture2D::Bind() to associate the actual GL texture with the binding
         * @param binding The binding slot
         * @param textureHandle The OpenGL texture handle
         */
        void BindRuntimeTexture(uint32_t binding, uint32_t textureHandle) override {
            _runtimeTextures[binding] = textureHandle;
        }

        /**
         * Get all runtime texture bindings for dispatch
         * @return Map of binding slot -> OpenGL texture handle
         */
        const std::unordered_map<uint32_t, uint32_t>& GetRuntimeTextureBindings() const override {
            return _runtimeTextures;
        }

    public:
        // ===================================================================
        // Callable Function Support
        // ===================================================================
        
        /**
         * Add a callable function declaration (forward declaration)
         * @param declaration The function prototype string
         */
        void AddCallableDeclaration(const std::string &declaration) override;

        /**
         * Register a callable body generator function
         * @param generator The function that generates the callable body
         */
        void AddCallableBodyGenerator(std::function<void()> generator) override;

        /**
         * Enter callable body generation mode
         */
        void PushCallableBody() override;

        /**
         * Exit callable body generation mode
         */
        void PopCallableBody() override;

        /**
         * Get all callable function declarations
         * @return Vector of function declarations
         */
        std::vector<std::string> GetCallableDeclarations() const override;

        /**
         * Generate all callable function bodies
         * @return The complete callable function definitions string
         */
        std::string GenerateCallableBodies() override;

    public:
        int WorkSizeX;
        int WorkSizeY;
        int WorkSizeZ;

    private:
        // Callable support
        std::vector<std::string> _callableDeclarations;
        std::vector<std::function<void()>> _callableBodyGenerators;
        std::vector<std::string> _callableBodies;
        std::stack<std::string> _callableBodyStack;
        std::string _currentCallableBody;
        bool _inCallableBody = false;

    private:
        /**
         * Buffer registration info
         */
        struct BufferInfo {
            uint32_t binding;
            std::string typeName;
            std::string bufferName;
            int mode;  // GL_READ_ONLY, GL_WRITE_ONLY, GL_READ_WRITE
        };

        /**
         * Texture registration info
         */
        struct TextureInfo {
            uint32_t binding;
            Runtime::PixelFormat format;
            std::string textureName;
            uint32_t width;
            uint32_t height;
        };

        uint32_t _nextBinding = 0;
        uint32_t _nextTextureBinding = 0;  // Separate counter for textures
        std::vector<BufferInfo> _buffers;
        std::vector<uint32_t> _bufferBindings;
        std::unordered_map<uint32_t, uint32_t> _runtimeBuffers;  // binding -> GL buffer handle
        
        std::vector<TextureInfo> _textures;
        std::vector<uint32_t> _textureBindings;
        std::unordered_map<uint32_t, uint32_t> _runtimeTextures;  // binding -> GL texture handle
        
        /**
         * The index for the variable name generation
         */
        int _variableIndex;
        int _dimension;
        std::string _code;
        std::unordered_set<std::string> _definedStructs;
        std::vector<std::string> _structNames;  // Keep insertion order for forward declarations
        std::vector<std::string> _structDefinitions;

        friend class Kernel;
    };
}

#endif //EASYGPU_KERNELBUILDCONTEXT_H
