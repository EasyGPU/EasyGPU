/**
 * @file GLStateCache.h
 * @brief OpenGL state cache for minimizing redundant state changes
 * 
 * This class implements an exclusive-mode OpenGL state manager that assumes
 * EasyGPU is the sole controller of OpenGL state within the current context.
 * It caches bound programs, buffers, and textures to avoid redundant driver calls.
 * 
 * @warning This class assumes exclusive ownership of OpenGL state. If you need
 * to interleave raw OpenGL calls with EasyGPU operations, call Invalidate() 
 * to force state re-binding on the next operation.
 */

#pragma once

#include <GLAD/glad.h>

#include <array>
#include <cstdint>
#include <vector>

namespace GPU::Runtime {

    /**
     * OpenGL state cache for minimizing redundant state changes.
     * 
     * Design principles:
     * - Exclusive mode: Assumes EasyGPU is the only OpenGL user in the context
     * - Lazy binding: Only calls glBind* when state actually changes
     * - No defensive glGet: Trusts the cache, does not verify with driver
     * 
     * This design maximizes performance by eliminating:
     * - Redundant glUseProgram calls
     * - Redundant glBindBufferBase calls
     * - Redundant glBindImageTexture calls
     * - Redundant glBindVertexArray calls
     */
    class GLStateCache {
    public:
        /**
         * Maximum number of SSBO binding points supported
         */
        static constexpr size_t MAX_SSBO_BINDINGS = 16;
        
        /**
         * Maximum number of image/texture binding points supported
         */
        static constexpr size_t MAX_IMAGE_BINDINGS = 16;
        
        /**
         * Maximum number of texture units for sampler bindings
         */
        static constexpr size_t MAX_TEXTURE_UNITS = 16;

    public:
        GLStateCache();
        ~GLStateCache() = default;

        // Non-copyable, non-movable
        GLStateCache(const GLStateCache&) = delete;
        GLStateCache& operator=(const GLStateCache&) = delete;
        GLStateCache(GLStateCache&&) = delete;
        GLStateCache& operator=(GLStateCache&&) = delete;

    public:
        /**
         * Bind a shader program, only if different from current
         * @param program OpenGL program ID, or 0 to unbind
         */
        void BindProgram(GLuint program);
        
        /**
         * Get currently bound program
         * @return Current program ID, or 0 if none
         */
        GLuint GetBoundProgram() const { return _currentProgram; }

    public:
        /**
         * Bind a SSBO to a binding point, only if different from current
         * @param binding Binding point index
         * @param buffer Buffer ID, or 0 to unbind
         */
        void BindSSBO(uint32_t binding, GLuint buffer);
        
        /**
         * Bind multiple SSBOs efficiently
         * Buffers that are already bound at the correct point are skipped
         * @param bindings Vector of (binding_point, buffer_id) pairs
         */
        void BindSSBOs(const std::vector<std::pair<uint32_t, GLuint>>& bindings);
        
        /**
         * Unbind all SSBOs that are currently bound
         */
        void UnbindAllSSBOs();
        
        /**
         * Get buffer bound at specific binding point
         * @param binding Binding point index
         * @return Buffer ID, or 0 if none
         */
        GLuint GetBoundSSBO(uint32_t binding) const;

    public:
        /**
         * Bind an image texture to a binding point, only if different from current
         * @param binding Binding point index
         * @param texture Texture ID, or 0 to unbind
         * @param format Image format (e.g., GL_RGBA8)
         * @param access Access mode (GL_READ_ONLY, GL_WRITE_ONLY, GL_READ_WRITE)
         */
        void BindImageTexture(uint32_t binding, GLuint texture, GLenum format = GL_RGBA8, GLenum access = GL_READ_WRITE);
        
        /**
         * Bind multiple image textures efficiently
         * @param bindings Vector of (binding_point, texture_id) pairs
         * @param format Image format for all textures
         * @param access Access mode for all textures
         */
        void BindImageTextures(const std::vector<std::pair<uint32_t, GLuint>>& bindings, GLenum format = GL_RGBA8, GLenum access = GL_READ_WRITE);
        
        /**
         * Unbind all image textures that are currently bound
         */
        void UnbindAllImageTextures();

    public:
        /**
         * Bind a sampler texture to a texture unit, only if different from current
         * @param unit Texture unit (0-15)
         * @param target Texture target (GL_TEXTURE_2D, etc.)
         * @param texture Texture ID, or 0 to unbind
         */
        void BindTexture(uint32_t unit, GLenum target, GLuint texture);
        
        /**
         * Unbind all sampler textures and reset active texture unit
         */
        void UnbindAllTextures();
        
        /**
         * Set active texture unit, only if different from current
         * @param unit Texture unit (0-15)
         */
        void ActiveTexture(uint32_t unit);

    public:
        /**
         * Bind a VAO, only if different from current
         * @param vao VAO ID, or 0 to unbind
         */
        void BindVAO(GLuint vao);
        
        /**
         * Get currently bound VAO
         * @return Current VAO ID, or 0 if none
         */
        GLuint GetBoundVAO() const { return _currentVAO; }

    public:
        /**
         * Invalidate all cached state.
         * 
         * Call this if you perform raw OpenGL operations outside of EasyGPU
         * and need to force state re-binding on the next EasyGPU operation.
         * 
         * Note: In exclusive mode (default), this should rarely be needed.
         */
        void Invalidate();
        
        /**
         * Invalidate only program binding
         */
        void InvalidateProgram();
        
        /**
         * Invalidate only buffer bindings
         */
        void InvalidateBuffers();
        
        /**
         * Invalidate only texture bindings
         */
        void InvalidateTextures();

    private:
        // Currently bound shader program (0 = none)
        GLuint _currentProgram;
        
        // SSBO bindings - indexed by binding point
        std::array<GLuint, MAX_SSBO_BINDINGS> _ssboBindings;
        
        // Image texture bindings - indexed by binding point
        std::array<GLuint, MAX_IMAGE_BINDINGS> _imageBindings;
        
        // Sampler texture bindings - indexed by texture unit
        std::array<GLuint, MAX_TEXTURE_UNITS> _textureBindings;
        
        // Current active texture unit
        GLuint _activeTextureUnit;
        
        // Currently bound VAO
        GLuint _currentVAO;
        
        // Flags to track if each category has any bindings
        bool _hasAnySSBOBindings;
        bool _hasAnyImageBindings;
        bool _hasAnyTextureBindings;
    };

    /**
     * Get the global GLStateCache instance for the current context.
     * 
     * This is lazily initialized and tied to the current OpenGL context.
     * The cache is automatically invalidated when the context changes.
     */
    GLStateCache& GetStateCache();

    /**
     * RAII guard that temporarily invalidates state cache
     * 
     * Use this when performing raw OpenGL operations that may modify state:
     * 
     * {
     *     StateCacheInvalidateGuard guard;  // Forces re-bind on next EasyGPU op
     *     glUseProgram(myRawProgram);       // Raw GL call
     *     glDrawArrays(...);
     * }  // Guard releases, next EasyGPU op will re-bind its state
     */
    class StateCacheInvalidateGuard {
    public:
        explicit StateCacheInvalidateGuard(bool restoreOnExit = false);
        ~StateCacheInvalidateGuard();

        StateCacheInvalidateGuard(const StateCacheInvalidateGuard&) = delete;
        StateCacheInvalidateGuard& operator=(const StateCacheInvalidateGuard&) = delete;

    private:
        bool _restoreOnExit;
    };

} // namespace GPU::Runtime
