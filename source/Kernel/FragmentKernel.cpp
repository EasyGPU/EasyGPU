/**
 * FragmentKernel.cpp:
 *      @Descripiton    :   Fragment kernel implementation - rasterization-based GPU rendering
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/19/2026
 */

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <Kernel/FragmentKernel.h>
#include <Kernel/KernelProfiler.h>

#include <Runtime/Context.h>
#include <Runtime/ShaderUtils.h>

#include <glad/glad.h>

#include <iostream>
#include <stdexcept>

namespace GPU::Kernel {

    // ===================================================================================
    // Fragment Kernel Builder Guard
    // ===================================================================================

    /**
     * RAII guard for saving and restoring builder context in FragmentKernel constructors
     */
    class FragmentKernelBuilderGuard {
    public:
        FragmentKernelBuilderGuard(IR::Builder::Builder& builder, FragmentBuildContext& newContext)
            : _builder(builder)
            , _previousContext(builder.Context()) {
            _builder.Bind(newContext);
        }

        ~FragmentKernelBuilderGuard() {
            if (_previousContext != nullptr) {
                _builder.Bind(*_previousContext);
            } else {
                _builder.Unbind();
            }
        }

        FragmentKernelBuilderGuard(const FragmentKernelBuilderGuard&) = delete;
        FragmentKernelBuilderGuard& operator=(const FragmentKernelBuilderGuard&) = delete;
        FragmentKernelBuilderGuard(FragmentKernelBuilderGuard&&) = delete;
        FragmentKernelBuilderGuard& operator=(FragmentKernelBuilderGuard&&) = delete;

    private:
        IR::Builder::Builder& _builder;
        IR::Builder::BuilderContext* _previousContext;
    };

    // ===================================================================================
    // FragmentKernel2D Implementation
    // ===================================================================================

    FragmentKernel2D::FragmentKernel2D(const std::string& name,
                                        const std::function<void(IR::Value::Var<GPU::Math::Vec2>& fragCoord,
                                                                IR::Value::Var<GPU::Math::Vec2>& resolution,
                                                                IR::Value::Var<GPU::Math::Vec4>& fragColor)>& func,
                                        uint32_t width, uint32_t height)
        : _name(name)
        , _width(width)
        , _height(height) {
        
        // Initialize the build context
        _context = std::make_unique<FragmentBuildContext>(width, height);
        
        // Set up builder guard to bind context
        FragmentKernelBuilderGuard guard(IR::Builder::Builder::Get(), *_context);
        
        // Create input/output variables as external (predefined in shader)
        // These are automatically declared in the generated fragment shader
        IR::Value::Var<GPU::Math::Vec2> fragCoord("fragCoord", true);       // vec2 fragCoord = gl_FragCoord.xy
        IR::Value::Var<GPU::Math::Vec2> resolution("u_resolution", true);   // vec2 u_resolution
        IR::Value::Var<GPU::Math::Vec4> fragColor("fragColor", true);       // out vec4 fragColor
        
        // Execute user function to generate IR
        func(fragCoord, resolution, fragColor);
    }

    FragmentKernel2D::~FragmentKernel2D() {
        Detach();
        CleanupResources();
    }

    FragmentKernel2D::FragmentKernel2D(FragmentKernel2D&& other) noexcept
        : _name(std::move(other._name))
        , _context(std::move(other._context))
        , _windowAttachment(std::move(other._windowAttachment))
        , _vao(other._vao)
        , _shaderProgram(other._shaderProgram)
        , _resourcesInitialized(other._resourcesInitialized)
        , _width(other._width)
        , _height(other._height) {
        
        other._vao = 0;
        other._shaderProgram = 0;
        other._resourcesInitialized = false;
    }

    FragmentKernel2D& FragmentKernel2D::operator=(FragmentKernel2D&& other) noexcept {
        if (this != &other) {
            Detach();
            CleanupResources();
            
            _name = std::move(other._name);
            _context = std::move(other._context);
            _windowAttachment = std::move(other._windowAttachment);
            _vao = other._vao;
            _shaderProgram = other._shaderProgram;
            _resourcesInitialized = other._resourcesInitialized;
            _width = other._width;
            _height = other._height;
            
            other._vao = 0;
            other._shaderProgram = 0;
            other._resourcesInitialized = false;
        }
        return *this;
    }

    // ===================================================================================
    // Window Attachment
    // ===================================================================================

    bool FragmentKernel2D::Attach(HWND hwnd) {
        if (!hwnd || !IsWindow(hwnd)) {
            return false;
        }

        // Ensure OpenGL context is initialized
        Runtime::AutoInitContext();

        // Create window attachment
        if (!_windowAttachment) {
            _windowAttachment = std::make_unique<WindowAttachment>();
        }

        // Set up resize callback
        auto resizeCallback = [this](uint32_t width, uint32_t height) {
            this->OnResize(width, height);
        };

        if (!_windowAttachment->Attach(hwnd, resizeCallback)) {
            return false;
        }

        // Initialize OpenGL resources
        InitializeResources();

        return true;
    }

    void FragmentKernel2D::Detach() {
        if (_windowAttachment) {
            _windowAttachment->Detach();
            _windowAttachment.reset();
        }
    }

    bool FragmentKernel2D::IsAttached() const {
        return _windowAttachment && _windowAttachment->IsAttached();
    }

    HWND FragmentKernel2D::GetWindow() const {
        return _windowAttachment ? _windowAttachment->GetHWND() : nullptr;
    }

    // ===================================================================================
    // Rendering
    // ===================================================================================

    void FragmentKernel2D::Flush() {
        if (!IsAttached()) {
            throw std::runtime_error("FragmentKernel2D::Flush() called before Attach()");
        }

        // Get EasyGPU's OpenGL context
        HGLRC hglrc = Runtime::Context::GetInstance().GetGLContext();
        if (!hglrc) {
            throw std::runtime_error("FragmentKernel2D: No OpenGL context available");
        }

        // Switch to window's DC for rendering
        if (!_windowAttachment->MakeCurrent(hglrc)) {
            std::cerr << "Failed to make window DC current" << std::endl;
        }

        // Execute render
        ExecuteRender();

        // Swap buffers on window
        _windowAttachment->SwapBuffers();

        // Restore EasyGPU's original DC (optional, but good practice)
        Runtime::Context::GetInstance().MakeCurrent();
    }

    void FragmentKernel2D::ExecuteRender() {
        // Ensure shader is compiled
        EnsureShaderCompiled();

        // Check for OpenGL errors
        auto checkGLError = [](const char* where) {
            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                std::cerr << "OpenGL error at " << where << ": " << err << std::endl;
            }
        };

        // Begin profiling if enabled (use current context method for FragmentKernel)
        unsigned int queryId = 0;
        if (_profilingEnabled) {
            queryId = KernelProfiler::GetInstance().BeginQueryOnCurrentContext();
        }

        // Set up rendering state
        glViewport(0, 0, _width, _height);
        checkGLError("glViewport");
        
        // Clear background to blue (so we can distinguish between black render and no render)
        glClearColor(0.0f, 0.0f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        checkGLError("glClear");
        
        // Bind shader program
        glUseProgram(_shaderProgram);
        checkGLError("glUseProgram");

        // Upload uniforms (resolution)
        GLint resLoc = glGetUniformLocation(_shaderProgram, "u_resolution");
        if (resLoc >= 0) {
            glUniform2f(resLoc, static_cast<float>(_width), static_cast<float>(_height));
        } else {
            std::cerr << "Warning: u_resolution uniform not found" << std::endl;
        }
        checkGLError("glUniform2f");

        // Upload user uniforms
        if (_context) {
            _context->UploadUniformValues(_shaderProgram);
        }
        checkGLError("UploadUniformValues");

        // Bind textures
        const auto& textureBindings = _context->GetRuntimeTextureBindings();
        for (const auto& [binding, textureHandle] : textureBindings) {
            glActiveTexture(GL_TEXTURE0 + binding);
            glBindTexture(GL_TEXTURE_2D, textureHandle);
        }

        // Bind VAO
        glBindVertexArray(_vao);
        checkGLError("glBindVertexArray");

        // Draw full-screen triangle (3 vertices, no buffers)
        glDrawArrays(GL_TRIANGLES, 0, 3);
        checkGLError("glDrawArrays");

        // Unbind
        glBindVertexArray(0);
        
        // Unbind textures
        for (const auto& [binding, textureHandle] : textureBindings) {
            glActiveTexture(GL_TEXTURE0 + binding);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glActiveTexture(GL_TEXTURE0);
        
        glUseProgram(0);
        checkGLError("glUseProgram(0)");

        // End profiling - fragment kernels use single "work group" (1,1,1)
        // since they're rasterization-based, not compute-based
        if (_profilingEnabled && queryId != 0) {
            KernelProfiler::GetInstance().EndQueryOnCurrentContext(queryId, _name, 1, 1, 1);
        }
    }

    // ===================================================================================
    // Resource Management
    // ===================================================================================

    void FragmentKernel2D::InitializeResources() {
        if (_resourcesInitialized) {
            return;
        }

        Runtime::Context::GetInstance().MakeCurrent();

        // Create VAO (no VBO needed - vertices generated in shader)
        glGenVertexArrays(1, &_vao);

        _resourcesInitialized = true;
    }

    void FragmentKernel2D::CleanupResources() {
        if (!_resourcesInitialized) {
            return;
        }

        Runtime::Context::GetInstance().MakeCurrent();

        if (_vao) {
            glDeleteVertexArrays(1, &_vao);
            _vao = 0;
        }

        if (_shaderProgram) {
            glDeleteProgram(_shaderProgram);
            _shaderProgram = 0;
        }

        _resourcesInitialized = false;
    }

    void FragmentKernel2D::EnsureShaderCompiled() {
        if (!_context) {
            throw std::runtime_error("FragmentKernel2D: No build context");
        }

        // Check if we have a cached program
        if (_context->HasCachedProgram() && _shaderProgram == 0) {
            _shaderProgram = _context->GetCachedProgram();
        }

        // Check if we need to recompile
        if (_shaderProgram != 0 && _context->IsShaderValid()) {
            return;
        }

        // Delete old program if exists
        if (_shaderProgram != 0) {
            glDeleteProgram(_shaderProgram);
            _shaderProgram = 0;
        }

        // Get vertex and fragment shader sources separately
        std::string vsSource = _context->GetVertexShaderSource();
        std::string fsSource = _context->GetFragmentShaderSource();

        // Debug: save shaders to file
        FILE* fp = nullptr;
        fopen_s(&fp, "vertex_shader.glsl", "w");
        if (fp) {
            fprintf(fp, "%s", vsSource.c_str());
            fclose(fp);
        }
        fopen_s(&fp, "fragment_shader.glsl", "w");
        if (fp) {
            fprintf(fp, "%s", fsSource.c_str());
            fclose(fp);
        }

        try {
            // Compile vertex shader
            uint32_t vs = Runtime::ShaderCompiler::CompileShader(GL_VERTEX_SHADER, vsSource);
            
            // Compile fragment shader
            uint32_t fs = Runtime::ShaderCompiler::CompileShader(GL_FRAGMENT_SHADER, fsSource);
            
            // Link program
            _shaderProgram = Runtime::ShaderCompiler::LinkProgram({vs, fs});
            
            // Cache the program
            _context->SetCachedProgram(_shaderProgram);
        } catch (const std::exception& e) {
            std::cerr << "Shader compilation failed: " << e.what() << std::endl;
            throw;
        }
    }

    // ===================================================================================
    // Properties
    // ===================================================================================

    void FragmentKernel2D::SetName(const std::string& name) {
        _name = name;
    }

    std::string FragmentKernel2D::GetName() const {
        return _name;
    }

    std::string FragmentKernel2D::GetShaderSource() {
        if (_context) {
            return "// === Vertex Shader ===\n" + _context->GetVertexShaderSource() + 
                   "\n// === Fragment Shader ===\n" + _context->GetFragmentShaderSource();
        }
        return "";
    }

    uint32_t FragmentKernel2D::GetWidth() const {
        return _width;
    }

    uint32_t FragmentKernel2D::GetHeight() const {
        return _height;
    }

    void FragmentKernel2D::SetResolution(uint32_t width, uint32_t height) {
        if (_width != width || _height != height) {
            _width = width;
            _height = height;
            if (_context) {
                _context->SetResolution(width, height);
            }
        }
    }

    void FragmentKernel2D::OnResize(uint32_t width, uint32_t height) {
        SetResolution(width, height);
    }

    // ===================================================================================
    // Profiling
    // ===================================================================================

    void FragmentKernel2D::SetProfilingEnabled(bool enabled) {
        _profilingEnabled = enabled;
        if (enabled) {
            KernelProfiler::GetInstance().SetEnabled(true);
        }
    }

    bool FragmentKernel2D::IsProfilingEnabled() const {
        return _profilingEnabled;
    }

} // namespace GPU::Kernel

#endif // _WIN32
