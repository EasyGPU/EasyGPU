#pragma once

/**
 * Buffer.h:
 *      @Descripiton    :   The GPU buffer for compute shader
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_BUFFER_H
#define EASYGPU_BUFFER_H

#include <Utility/Vec.h>
#include <Utility/Matrix.h>
#include <Utility/Meta/StructMeta.h>
#include <Utility/Meta/Std430Layout.h>
#include <Runtime/Context.h>
#include <IR/Value/BufferRef.h>
#include <IR/Builder/Builder.h>

#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <format>

// Forward declaration for GLAD
#include <GLAD/glad.h>

namespace GPU::Runtime {
    /**
     * The access mode for the buffer
     */
    enum class BufferMode {
        Read,       // Readonly access
        Write,      // Writeonly access
        ReadWrite   // Read-write access
    };

    /**
     * Get the GL access mode from BufferMode
     * Returns GL_READ_ONLY (0x88B8), GL_WRITE_ONLY (0x88B9), or GL_READ_WRITE (0x88BA)
     * @param mode The buffer mode
     * @return The GL access mode
     */
    inline int GetGLBufferMode(BufferMode mode) {
        switch (mode) {
            case BufferMode::Read:
                return 0x88B8;  // GL_READ_ONLY
            case BufferMode::Write:
                return 0x88B9;  // GL_WRITE_ONLY
            case BufferMode::ReadWrite:
                return 0x88BA;  // GL_READ_WRITE
        }
        return 0x88BA;  // GL_READ_WRITE
    }

    /**
     * Get the GL usage hint from BufferMode
     * @param mode The buffer mode
     * @return The GL usage hint
     */
    inline int GetGLBufferUsage(BufferMode mode) {
        switch (mode) {
            case BufferMode::Read:
                return 0x88E0;  // GL_STATIC_READ
            case BufferMode::Write:
                return 0x88E4;  // GL_STATIC_DRAW
            case BufferMode::ReadWrite:
                return 0x88E6;  // GL_DYNAMIC_COPY
        }
        return 0x88E6;  // GL_DYNAMIC_COPY
    }

    /**
     * Helper function to get GLSL type name for buffer elements
     * Handles both primitive types and custom structs
     */
    // Helper to detect if T is a registered struct
    template<typename T>
    struct IsStructRegistered {
        static constexpr bool value = GPU::Meta::StructMeta<T>::isRegistered;
    };

    // Overload for registered structs
    template<typename T>
    std::string GetGLSLTypeNameForBuffer(T*) {
        // Explicitly use data and size to avoid string_view to string conversion issues
        auto sv = std::string(GPU::Meta::StructMeta<T>::glslTypeName);
        return std::string(sv.data(), sv.size());
    }

    // Overloads for primitive types (using tag dispatch)
    inline std::string GetGLSLTypeNameForBuffer(float*) { return "float"; }
    inline std::string GetGLSLTypeNameForBuffer(int*) { return "int"; }
    inline std::string GetGLSLTypeNameForBuffer(bool*) { return "bool"; }
    inline std::string GetGLSLTypeNameForBuffer(Math::Vec2*) { return "vec2"; }
    inline std::string GetGLSLTypeNameForBuffer(Math::Vec3*) { return "vec3"; }
    inline std::string GetGLSLTypeNameForBuffer(Math::Vec4*) { return "vec4"; }
    inline std::string GetGLSLTypeNameForBuffer(Math::IVec2*) { return "ivec2"; }
    inline std::string GetGLSLTypeNameForBuffer(Math::IVec3*) { return "ivec3"; }
    inline std::string GetGLSLTypeNameForBuffer(Math::IVec4*) { return "ivec4"; }
    inline std::string GetGLSLTypeNameForBuffer(Math::Mat2*) { return "mat2"; }
    inline std::string GetGLSLTypeNameForBuffer(Math::Mat3*) { return "mat3"; }
    inline std::string GetGLSLTypeNameForBuffer(Math::Mat4*) { return "mat4"; }

    // Main template that dispatches to the correct overload
    template<typename T>
    std::string GetGLSLTypeNameForBuffer() {
        return GetGLSLTypeNameForBuffer(static_cast<T*>(nullptr));
    }
    template<> inline std::string GetGLSLTypeNameForBuffer<Math::Mat3>() { return "mat3"; }
    template<> inline std::string GetGLSLTypeNameForBuffer<Math::Mat4>() { return "mat4"; }

    /**
     * The GPU buffer for compute shader
     * @tparam T The element type of the buffer
     * 
     * AUTOMATIC LAYOUT CONVERSION:
     * This class automatically handles conversion between C++ struct layout and 
     * GLSL std430 layout. Users can use natural C++ structs without manual padding!
     */
    template<typename T>
    class Buffer {
    public:
        /**
         * Creating a buffer with specified count and mode
         * @param Count The element count
         * @param Mode The access mode
         */
        Buffer(size_t Count, BufferMode Mode = BufferMode::ReadWrite)
            : _count(Count)
            , _mode(Mode) {
            InitLayout();
            CreateBuffer();
        }

        /**
         * Creating a buffer from CPU data
         * @param Data The CPU data
         * @param Mode The access mode
         */
        Buffer(const std::vector<T>& Data, BufferMode Mode = BufferMode::ReadWrite)
            : _count(Data.size())
            , _mode(Mode) {
            InitLayout();
            CreateBuffer();
            if (!Data.empty()) {
                Upload(Data.data(), Data.size());
            }
        }

        /**
         * Move constructor
         * @param other The other buffer to move from
         */
        Buffer(Buffer&& other) noexcept
            : _bufferId(other._bufferId)
            , _count(other._count)
            , _elementSize(other._elementSize)
            , _mode(other._mode)
            , _boundBinding(other._boundBinding)
            , _layoutConverter(std::move(other._layoutConverter)) {
            other._bufferId = 0;
            other._count = 0;
            other._elementSize = 0;
            other._boundBinding = -1;
            // _layoutConverter is already moved and nullified
        }

        /**
         * Move assignment
         * @param other The other buffer to move from
         * @return Reference to this
         */
        Buffer& operator=(Buffer&& other) noexcept {
            if (this != &other) {
                DestroyBuffer();
                _bufferId = other._bufferId;
                _count = other._count;
                _elementSize = other._elementSize;
                _mode = other._mode;
                _boundBinding = other._boundBinding;
                _layoutConverter = std::move(other._layoutConverter);
                other._bufferId = 0;
                other._count = 0;
                other._elementSize = 0;
                other._boundBinding = -1;
            }
            return *this;
        }

        /**
         * Destructor
         */
        ~Buffer() {
            DestroyBuffer();
        }

        // Disable copy
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

    private:
        /**
         * Initialize layout converter and element size
         */
        void InitLayout() {
            _layoutConverter = std::make_unique<Meta::Std430Converter<T>>();
            _elementSize = _layoutConverter->GetGPULayoutSize();
            // Ensure minimum element size of sizeof(T) for primitive types
            if (_elementSize < sizeof(T)) {
                _elementSize = sizeof(T);
            }
        }

    public:
        /**
         * Bind this buffer to the current kernel being defined
         * Automatically allocates a binding slot and registers the buffer
         * @return BufferRef<T> for DSL access
         */
        [[nodiscard]] IR::Value::BufferRef<T> Bind() {
            // Get current builder context
            auto* context = IR::Builder::Builder::Get().Context();
            if (!context) {
                throw std::runtime_error("Buffer::Bind() called outside of Kernel definition");
            }

            // Allocate binding slot
            uint32_t binding = context->AllocateBindingSlot();

            // Generate buffer variable name
            std::string bufferName = std::format("buf{}", binding);

            // Get GLSL type name
            std::string typeName = GetGLSLTypeNameForBuffer<T>();

            // Register buffer in context (for GLSL code generation)
            context->RegisterBuffer(binding, typeName, 
                                   bufferName, 
                                   GetGLBufferMode(_mode));

            // Register runtime buffer handle (for actual GPU dispatch)
            context->BindRuntimeBuffer(binding, _bufferId);

            // Store binding info
            _boundBinding = binding;

            // Return BufferRef for DSL access
            return IR::Value::BufferRef<T>(bufferName, binding);
        }

        /**
         * Upload data to GPU buffer with automatic std430 layout conversion
         * @param data The CPU data pointer
         * @param count The element count to upload
         */
        void Upload(const T* data, size_t count) {
            if (_bufferId == 0 || data == nullptr || count == 0) {
                return;
            }
            if (count > _count) {
                count = _count;
            }
            
            // Ensure context is current for this thread
            Runtime::Context::GetInstance().MakeCurrent();
            
            // Check if layout conversion is needed
            if (_layoutConverter && _layoutConverter->NeedsConversion()) {
                // Convert to GPU layout
                std::vector<char> gpuBuffer(count * _elementSize);
                _layoutConverter->ConvertToGPU(data, gpuBuffer.data(), count);
                
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, _bufferId);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, count * _elementSize, gpuBuffer.data());
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            } else {
                // Direct upload (no conversion needed)
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, _bufferId);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, count * _elementSize, data);
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            }
        }

        /**
         * Upload data to GPU buffer (vector version)
         * @param data The CPU data vector
         */
        void Upload(const std::vector<T>& data) {
            if (!data.empty()) {
                Upload(data.data(), data.size());
            }
        }

        /**
         * Download data from GPU buffer with automatic std430 layout conversion
         * @param outData The output CPU data pointer
         * @param count The element count to download
         */
        void Download(T* outData, size_t count) {
            if (_bufferId == 0 || outData == nullptr || count == 0) {
                return;
            }
            if (count > _count) {
                count = _count;
            }
            
            // Ensure context is current for this thread
            Runtime::Context::GetInstance().MakeCurrent();
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, _bufferId);
            
            // Check if layout conversion is needed
            if (_layoutConverter && _layoutConverter->NeedsConversion()) {
                // Download to temporary buffer in GPU layout
                std::vector<char> gpuBuffer(count * _elementSize);
                void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, count * _elementSize, GL_MAP_READ_BIT);
                if (mapped != nullptr) {
                    std::memcpy(gpuBuffer.data(), mapped, count * _elementSize);
                    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                    
                    // Convert from GPU layout to C++ layout
                    _layoutConverter->ConvertFromGPU(gpuBuffer.data(), outData, count);
                }
            } else {
                // Direct download (no conversion needed)
                void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, count * _elementSize, GL_MAP_READ_BIT);
                if (mapped != nullptr) {
                    std::memcpy(outData, mapped, count * _elementSize);
                    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                }
            }

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        /**
         * Download data from GPU buffer (vector version)
         * @param outData The output CPU data vector
         */
        void Download(std::vector<T>& outData) {
            if (outData.size() < _count) {
                outData.resize(_count);
            }
            if (!outData.empty()) {
                Download(outData.data(), outData.size());
            }
        }

    public:
        /**
         * Get the OpenGL buffer handle
         * @return The OpenGL buffer ID
         */
        [[nodiscard]] uint32_t GetHandle() const {
            return _bufferId;
        }

        /**
         * Get the element count
         * @return The element count
         */
        [[nodiscard]] size_t GetCount() const {
            return _count;
        }

        /**
         * Get the buffer mode
         * @return The buffer access mode
         */
        [[nodiscard]] BufferMode GetMode() const {
            return _mode;
        }

        /**
         * Get the element size (in GPU layout)
         * @return The size of each element in bytes
         */
        [[nodiscard]] size_t GetElementSize() const {
            return _elementSize;
        }

        /**
         * Get the total buffer size
         * @return The total buffer size in bytes
         */
        [[nodiscard]] size_t GetBufferSize() const {
            return _count * _elementSize;
        }

        /**
         * Get the binding slot if bound to a kernel
         * @return The binding slot, or -1 if not bound
         */
        [[nodiscard]] int GetBinding() const {
            return _boundBinding;
        }

    private:
        void CreateBuffer() {
            // Auto-initialize OpenGL context on first GPU operation
            Runtime::AutoInitContext();
            // Ensure context is current for this thread
            Runtime::Context::GetInstance().MakeCurrent();
            
            if (_count == 0) {
                return;
            }
            glGenBuffers(1, &_bufferId);
            if (_bufferId == 0) {
                throw std::runtime_error("Failed to create OpenGL buffer");
            }
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, _bufferId);
            glBufferData(GL_SHADER_STORAGE_BUFFER, _count * _elementSize, nullptr, GetGLBufferUsage(_mode));
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        void DestroyBuffer() {
            if (_bufferId != 0) {
                glDeleteBuffers(1, &_bufferId);
                _bufferId = 0;
            }
        }

    private:
        uint32_t _bufferId = 0;
        size_t _count = 0;
        size_t _elementSize = sizeof(T);
        BufferMode _mode = BufferMode::ReadWrite;
        int _boundBinding = -1;  // -1 means not bound
        std::unique_ptr<Meta::LayoutConverter> _layoutConverter = nullptr;
    };
}

#endif //EASYGPU_BUFFER_H
