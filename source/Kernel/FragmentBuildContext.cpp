/**
 * FragmentBuildContext.cpp:
 *      @Descripiton    :   Fragment shader code generation implementation
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/19/2026
 */

#include <Kernel/FragmentBuildContext.h>

#include <iostream>
#include <format>
#include <sstream>

namespace GPU::Kernel {

    FragmentBuildContext::FragmentBuildContext(uint32_t width, uint32_t height)
        : KernelBuildContext(2)  // Use dimension 2 as base
        , _width(width)
        , _height(height) {
        // Reset work size for fragment shader (not used but set for consistency)
        WorkSizeX = 16;
        WorkSizeY = 16;
        WorkSizeZ = 1;
    }

    void FragmentBuildContext::SetResolution(uint32_t width, uint32_t height) {
        if (_width != width || _height != height) {
            _width = width;
            _height = height;
            InvalidateCachedProgram();
        }
    }

    void FragmentBuildContext::GenerateCommonHeaders(std::ostringstream& oss) {
        // Version directive
        oss << GenerateHeader();
        
        // Output struct definitions
        for (const auto &structDef: GetStructDefinitions()) {
            oss << structDef;
        }
        if (!GetStructDefinitions().empty()) {
            oss << "\n";
        }
        
        // Output uniform declarations
        std::string uniformDecls = GetUniformDeclarations();
        if (!uniformDecls.empty()) {
            oss << uniformDecls << "\n";
        }
        
        // Add built-in resolution uniform if not already present
        if (uniformDecls.find("u_resolution") == std::string::npos) {
            oss << "uniform vec2 u_resolution;\n\n";
        }
        
        // Output callable function forward declarations and definitions
        for (const auto &decl : _callableDeclarations) {
            oss << decl << ";\n";
        }
        if (!_callableDeclarations.empty()) {
            oss << "\n";
        }
        
        std::string callableDefs = GenerateCallableBodies();
        if (!callableDefs.empty()) {
            oss << callableDefs << "\n";
        }
    }

    std::string FragmentBuildContext::GetCompleteCode() {
        // Execute all body generators first
        {
            std::string savedCallableBody = std::move(_currentCallableBody);
            bool savedInCallableBody = _inCallableBody;
            std::stack<std::string> savedBodyStack = std::move(_callableBodyStack);
            
            _currentCallableBody.clear();
            _inCallableBody = false;
            while (!_callableBodyStack.empty()) {
                _callableBodyStack.pop();
            }
            
            auto &builder = IR::Builder::Builder::Get();
            auto *prevContext = builder.Context();
            builder.Bind(*this);
            
            size_t processedCount = 0;
            while (processedCount < _callableBodyGenerators.size()) {
                size_t currentCount = _callableBodyGenerators.size();
                for (size_t i = processedCount; i < currentCount; ++i) {
                    _callableBodyGenerators[i]();
                }
                processedCount = currentCount;
            }
            
            if (prevContext) {
                builder.Bind(*prevContext);
            } else {
                builder.Unbind();
            }
            
            _currentCallableBody = std::move(savedCallableBody);
            _inCallableBody = savedInCallableBody;
            _callableBodyStack = std::move(savedBodyStack);
        }
        
        std::ostringstream oss;
        GenerateCommonHeaders(oss);
        
        // Output texture declarations
        std::string textureDecls = GetTextureDeclarations();
        if (!textureDecls.empty()) {
            oss << textureDecls << "\n";
        }
        
        std::string texture3DDecls = GetTexture3DDeclarations();
        if (!texture3DDecls.empty()) {
            oss << texture3DDecls << "\n";
        }
        
        // Output buffer declarations
        std::string bufferDecls = GetBufferDeclarations();
        if (!bufferDecls.empty()) {
            oss << bufferDecls << "\n";
        }
        
        // Output vertex and fragment shaders
        oss << GenerateVertexShader();
        oss << GenerateFragmentShader();
        
        return oss.str();
    }

    std::string FragmentBuildContext::GetVertexShaderSource() {
        std::ostringstream oss;
        oss << GenerateHeader();
        
        // Output vertex shader only
        oss << GenerateVertexShader();
        
        return oss.str();
    }

    std::string FragmentBuildContext::GetFragmentShaderSource() {
        // Execute all body generators first
        {
            std::string savedCallableBody = std::move(_currentCallableBody);
            bool savedInCallableBody = _inCallableBody;
            std::stack<std::string> savedBodyStack = std::move(_callableBodyStack);
            
            _currentCallableBody.clear();
            _inCallableBody = false;
            while (!_callableBodyStack.empty()) {
                _callableBodyStack.pop();
            }
            
            auto &builder = IR::Builder::Builder::Get();
            auto *prevContext = builder.Context();
            builder.Bind(*this);
            
            size_t processedCount = 0;
            while (processedCount < _callableBodyGenerators.size()) {
                size_t currentCount = _callableBodyGenerators.size();
                for (size_t i = processedCount; i < currentCount; ++i) {
                    _callableBodyGenerators[i]();
                }
                processedCount = currentCount;
            }
            
            if (prevContext) {
                builder.Bind(*prevContext);
            } else {
                builder.Unbind();
            }
            
            _currentCallableBody = std::move(savedCallableBody);
            _inCallableBody = savedInCallableBody;
            _callableBodyStack = std::move(savedBodyStack);
        }
        
        std::ostringstream oss;
        GenerateCommonHeaders(oss);
        
        // Output texture declarations for fragment shader
        std::string textureDecls = GetTextureDeclarations();
        if (!textureDecls.empty()) {
            oss << textureDecls << "\n";
        }
        
        std::string texture3DDecls = GetTexture3DDeclarations();
        if (!texture3DDecls.empty()) {
            oss << texture3DDecls << "\n";
        }
        
        // Output buffer declarations
        std::string bufferDecls = GetBufferDeclarations();
        if (!bufferDecls.empty()) {
            oss << bufferDecls << "\n";
        }
        
        // Output fragment shader
        oss << GenerateFragmentShader();
        
        return oss.str();
    }

    std::string FragmentBuildContext::GetTextureDeclarations() const {
        std::ostringstream oss;
        for (const auto &tex: _textures) {
            // Fragment shaders use sampler2D for reading (not image2D)
            oss << std::format("layout(binding={}) uniform sampler2D {};\n",
                               tex.binding, tex.textureName);
        }
        return oss.str();
    }

    std::string FragmentBuildContext::GetTexture3DDeclarations() const {
        std::ostringstream oss;
        for (const auto &tex: _textures3D) {
            // Fragment shaders use sampler3D for reading
            oss << std::format("layout(binding={}) uniform sampler3D {};\n",
                               tex.binding, tex.textureName);
        }
        return oss.str();
    }

    std::string FragmentBuildContext::GenerateHeader() {
        return "#version 430 core\n\n";
    }

    std::string FragmentBuildContext::GenerateVertexShader() {
        std::ostringstream oss;
        oss << "// Vertex Shader - Full screen triangle\n";
        oss << "const vec2 _verts[3] = vec2[](\n";
        oss << "    vec2(-1.0, -1.0),\n";
        oss << "    vec2( 3.0, -1.0),\n";
        oss << "    vec2(-1.0,  3.0)\n";
        oss << ");\n\n";
        
        oss << "out vec2 vUV;\n\n";
        
        oss << "void main() {\n";
        oss << "    vec2 pos = _verts[gl_VertexID];\n";
        oss << "    vUV = pos * 0.5 + 0.5;\n";
        oss << "    gl_Position = vec4(pos, 0.0, 1.0);\n";
        oss << "}\n\n";
        
        return oss.str();
    }

    std::string FragmentBuildContext::GenerateFragmentShader() {
        std::ostringstream oss;
        oss << "// Fragment Shader\n";
        oss << "in vec2 vUV;\n";
        oss << "out vec4 fragColor;\n\n";
        
        oss << "void main() {\n";
        oss << "    vec2 fragCoord = vUV * u_resolution;\n";
        
        // Debug: check if _code is empty
        if (_code.empty()) {
            std::cerr << "WARNING: _code is empty in GenerateFragmentShader!" << std::endl;
        }
        
        // Add user code with indentation
        if (!_code.empty()) {
            oss << "\n";
            std::istringstream codeStream(_code);
            std::string line;
            while (std::getline(codeStream, line)) {
                oss << "    " << line << "\n";
            }
        }
        
        oss << "}\n";
        
        return oss.str();
    }

} // namespace GPU::Kernel
