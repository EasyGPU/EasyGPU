/**
 * @file GLStateCache.cpp
 * @brief OpenGL state cache implementation
 */

#include <Runtime/GLStateCache.h>

#include <Runtime/Context.h>

#include <algorithm>
#include <cstring>

namespace GPU::Runtime {

    // ===================================================================================
    // GLStateCache Implementation
    // ===================================================================================

    GLStateCache::GLStateCache()
        : _currentProgram(0)
        , _activeTextureUnit(0)
        , _currentVAO(0)
        , _hasAnySSBOBindings(false)
        , _hasAnyImageBindings(false)
        , _hasAnyTextureBindings(false) {
        
        _ssboBindings.fill(0);
        _imageBindings.fill(0);
        _textureBindings.fill(0);
    }

    // ===================================================================================
    // Program State
    // ===================================================================================

    void GLStateCache::BindProgram(GLuint program) {
        if (_currentProgram != program) {
            glUseProgram(program);
            _currentProgram = program;
        }
    }

    // ===================================================================================
    // SSBO State
    // ===================================================================================

    void GLStateCache::BindSSBO(uint32_t binding, GLuint buffer) {
        if (binding >= MAX_SSBO_BINDINGS) {
            return; // Silently ignore out-of-range bindings
        }
        
        if (_ssboBindings[binding] != buffer) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer);
            _ssboBindings[binding] = buffer;
            
            if (buffer != 0) {
                _hasAnySSBOBindings = true;
            } else {
                // Check if any bindings remain
                _hasAnySSBOBindings = std::any_of(
                    _ssboBindings.begin(), 
                    _ssboBindings.end(),
                    [](GLuint b) { return b != 0; }
                );
            }
        }
    }

    void GLStateCache::BindSSBOs(const std::vector<std::pair<uint32_t, GLuint>>& bindings) {
        for (const auto& [binding, buffer] : bindings) {
            BindSSBO(binding, buffer);
        }
    }

    void GLStateCache::UnbindAllSSBOs() {
        if (!_hasAnySSBOBindings) {
            return; // Nothing to unbind
        }
        
        for (size_t i = 0; i < MAX_SSBO_BINDINGS; ++i) {
            if (_ssboBindings[i] != 0) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, static_cast<GLuint>(i), 0);
                _ssboBindings[i] = 0;
            }
        }
        _hasAnySSBOBindings = false;
    }

    GLuint GLStateCache::GetBoundSSBO(uint32_t binding) const {
        if (binding >= MAX_SSBO_BINDINGS) {
            return 0;
        }
        return _ssboBindings[binding];
    }

    // ===================================================================================
    // Image Texture State
    // ===================================================================================

    void GLStateCache::BindImageTexture(uint32_t binding, GLuint texture, GLenum format, GLenum access) {
        if (binding >= MAX_IMAGE_BINDINGS) {
            return; // Silently ignore out-of-range bindings
        }
        
        if (_imageBindings[binding] != texture) {
            glBindImageTexture(binding, texture, 0, GL_FALSE, 0, access, format);
            _imageBindings[binding] = texture;
            
            if (texture != 0) {
                _hasAnyImageBindings = true;
            } else {
                _hasAnyImageBindings = std::any_of(
                    _imageBindings.begin(),
                    _imageBindings.end(),
                    [](GLuint t) { return t != 0; }
                );
            }
        }
    }

    void GLStateCache::BindImageTextures(const std::vector<std::pair<uint32_t, GLuint>>& bindings, 
                                          GLenum format, GLenum access) {
        for (const auto& [binding, texture] : bindings) {
            BindImageTexture(binding, texture, format, access);
        }
    }

    void GLStateCache::UnbindAllImageTextures() {
        if (!_hasAnyImageBindings) {
            return; // Nothing to unbind
        }
        
        for (size_t i = 0; i < MAX_IMAGE_BINDINGS; ++i) {
            if (_imageBindings[i] != 0) {
                glBindImageTexture(static_cast<GLuint>(i), 0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
                _imageBindings[i] = 0;
            }
        }
        _hasAnyImageBindings = false;
    }

    // ===================================================================================
    // Sampler Texture State
    // ===================================================================================

    void GLStateCache::BindTexture(uint32_t unit, GLenum target, GLuint texture) {
        if (unit >= MAX_TEXTURE_UNITS) {
            return; // Silently ignore out-of-range units
        }
        
        // First ensure correct texture unit is active
        ActiveTexture(unit);
        
        if (_textureBindings[unit] != texture) {
            glBindTexture(target, texture);
            _textureBindings[unit] = texture;
            
            if (texture != 0) {
                _hasAnyTextureBindings = true;
            } else {
                _hasAnyTextureBindings = std::any_of(
                    _textureBindings.begin(),
                    _textureBindings.end(),
                    [](GLuint t) { return t != 0; }
                );
            }
        }
    }

    void GLStateCache::UnbindAllTextures() {
        if (!_hasAnyTextureBindings) {
            return; // Nothing to unbind
        }
        
        for (size_t i = 0; i < MAX_TEXTURE_UNITS; ++i) {
            if (_textureBindings[i] != 0) {
                ActiveTexture(static_cast<uint32_t>(i));
                glBindTexture(GL_TEXTURE_2D, 0);
                _textureBindings[i] = 0;
            }
        }
        
        // Reset to unit 0
        ActiveTexture(0);
        _hasAnyTextureBindings = false;
    }

    void GLStateCache::ActiveTexture(uint32_t unit) {
        if (unit >= MAX_TEXTURE_UNITS) {
            return;
        }
        
        if (_activeTextureUnit != unit) {
            glActiveTexture(GL_TEXTURE0 + unit);
            _activeTextureUnit = unit;
        }
    }

    // ===================================================================================
    // VAO State
    // ===================================================================================

    void GLStateCache::BindVAO(GLuint vao) {
        if (_currentVAO != vao) {
            glBindVertexArray(vao);
            _currentVAO = vao;
        }
    }

    // ===================================================================================
    // Invalidation
    // ===================================================================================

    void GLStateCache::Invalidate() {
        _currentProgram = 0;
        _ssboBindings.fill(0);
        _imageBindings.fill(0);
        _textureBindings.fill(0);
        _activeTextureUnit = 0;
        _currentVAO = 0;
        _hasAnySSBOBindings = false;
        _hasAnyImageBindings = false;
        _hasAnyTextureBindings = false;
    }

    void GLStateCache::InvalidateProgram() {
        _currentProgram = 0;
    }

    void GLStateCache::InvalidateBuffers() {
        _ssboBindings.fill(0);
        _imageBindings.fill(0);
        _hasAnySSBOBindings = false;
        _hasAnyImageBindings = false;
    }

    void GLStateCache::InvalidateTextures() {
        _imageBindings.fill(0);
        _textureBindings.fill(0);
        _activeTextureUnit = 0;
        _hasAnyImageBindings = false;
        _hasAnyTextureBindings = false;
    }

    // ===================================================================================
    // Global Instance
    // ===================================================================================

    GLStateCache& GetStateCache() {
        // Thread-local cache per context would be ideal, but for now
        // we use a simple static that gets invalidated on context changes
        static GLStateCache instance;
        return instance;
    }

    // ===================================================================================
    // Invalidate Guard
    // ===================================================================================

    StateCacheInvalidateGuard::StateCacheInvalidateGuard(bool restoreOnExit)
        : _restoreOnExit(restoreOnExit) {
        // Invalidate on entry to ensure raw GL calls work correctly
        GetStateCache().Invalidate();
    }

    StateCacheInvalidateGuard::~StateCacheInvalidateGuard() {
        if (_restoreOnExit) {
            // Invalidate again on exit to force EasyGPU to re-bind
            GetStateCache().Invalidate();
        }
        // If !_restoreOnExit, we assume the raw GL code cleaned up properly
        // and EasyGPU will continue with its cached state
    }

} // namespace GPU::Runtime
