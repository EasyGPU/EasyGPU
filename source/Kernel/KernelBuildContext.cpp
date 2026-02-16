/**
 * KernelBuildContext.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/12/2026
 */

#include <Kernel/KernelBuildContext.h>

#include <IR/Builder/Builder.h>

#include <format>
#include <sstream>
#include <iostream>

namespace GPU::Kernel {
    KernelDimensionOutOfRange::KernelDimensionOutOfRange() : std::out_of_range("Kernel dimension out of range!") {
    }

    KernelBuildContext::KernelBuildContext(int Dimension) : _variableIndex(0), _nextBinding(0), _dimension(Dimension) {
        if (_dimension == 1) {
            WorkSizeX = 256;
            WorkSizeY = 1;
            WorkSizeZ = 1;
        } else if (_dimension == 2) {
            WorkSizeX = 16;
            WorkSizeY = 16;
            WorkSizeZ = 1;
        } else if (_dimension == 3) {
            WorkSizeX = 8;
            WorkSizeY = 8;
            WorkSizeZ = 4;
        } else {
            throw KernelDimensionOutOfRange();
        }
    }

    void KernelBuildContext::PushTranslatedCode(std::string Code) {
        if (_inCallableBody) {
            _currentCallableBody.append(Code);
        } else {
            _code.append(Code);
        }
    }

    std::string KernelBuildContext::AssignVarName() {
        ++_variableIndex;

        return std::format("v{}", _variableIndex);
    }

    std::string KernelBuildContext::GetCompleteCode() {
        std::ostringstream oss;

        // Version directive
        oss << "#version 430 core\n\n";

        // Layout for compute shader
        if (_dimension == 1) {
            oss << std::format("layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;\n\n", WorkSizeX,
                               WorkSizeY, WorkSizeZ);
        }
        if (_dimension == 2) {
            oss << std::format("layout(local_size_x = {}, local_size_y = {}) in;\n\n", WorkSizeX, WorkSizeY);
        }
        if (_dimension == 3) {
            oss << std::format("layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;\n\n", WorkSizeX,
                               WorkSizeY, WorkSizeZ);
        }

        // ===================================================================
        // Phase 1: Execute all body generators to collect declarations AND generate bodies
        // We do this before outputting declarations because generators may register
        // new Callables (when one Callable calls another). We need to discover all
        // Callables before outputting their declarations.
        // ===================================================================
        {
            // Save current state
            std::string savedCallableBody = std::move(_currentCallableBody);
            bool savedInCallableBody = _inCallableBody;
            std::stack<std::string> savedBodyStack = std::move(_callableBodyStack);
            
            // Clear state for pre-execution
            _currentCallableBody.clear();
            _inCallableBody = false;
            while (!_callableBodyStack.empty()) {
                _callableBodyStack.pop();
            }
            
            auto &builder = IR::Builder::Builder::Get();
            auto *prevContext = builder.Context();
            builder.Bind(*this);
            
            // Execute all generators to collect declarations and generate bodies
            // Use a while loop because executing generators may register new Callables
            size_t processedCount = 0;
            while (processedCount < _callableBodyGenerators.size()) {
                size_t currentCount = _callableBodyGenerators.size();
                for (size_t i = processedCount; i < currentCount; ++i) {
                    _callableBodyGenerators[i]();
                }
                processedCount = currentCount;
            }
            
            // Restore previous context
            if (prevContext) {
                builder.Bind(*prevContext);
            } else {
                builder.Unbind();
            }
            
            // NOTE: We do NOT clear _callableBodies here because we want to keep
            // the generated bodies. Phase 2 will just output them without re-executing.
            
            // Restore original state
            _currentCallableBody = std::move(savedCallableBody);
            _inCallableBody = savedInCallableBody;
            _callableBodyStack = std::move(savedBodyStack);
        }

        // ===================================================================
        // Output struct declarations AFTER Phase 1 (so all used structs are registered)
        // Structs are registered in dependency order (dependencies first), so output in order
        // ===================================================================
        for (const auto &structDef: GetStructDefinitions()) {
            oss << structDef;
        }
        if (!GetStructDefinitions().empty()) {
            oss << "\n";
        }

        // Output texture declarations (after struct definitions, before buffers)
        std::string textureDecls = GetTextureDeclarations();
        if (!textureDecls.empty()) {
            oss << textureDecls << "\n";
        }

        // Output buffer declarations (after texture declarations)
        std::string bufferDecls = GetBufferDeclarations();
        if (!bufferDecls.empty()) {
            oss << bufferDecls << "\n";
        }

        // Output callable function forward declarations (before main)
        for (const auto &decl : _callableDeclarations) {
            oss << decl << ";\n";
        }
        if (!_callableDeclarations.empty()) {
            oss << "\n";
        }

        // Main function wrapper
        oss << "void main() {\n";

        // Add the kernel code with indentation
        std::istringstream codeStream(_code);
        std::string line;
        while (std::getline(codeStream, line)) {
            if (!line.empty()) {
                oss << "    " << line << "\n";
            } else {
                oss << "\n";
            }
        }

        oss << "}\n";

        // ===================================================================
        // Phase 2: Generate callable bodies and output definitions
        // Now all declarations are known, generate bodies in order
        // ===================================================================
        {
            auto &builder = IR::Builder::Builder::Get();
            auto *prevContext = builder.Context();
            builder.Bind(*this);
            
            std::string callableDefs = GenerateCallableBodies();
            if (!callableDefs.empty()) {
                oss << "\n" << callableDefs;
            }
            
            // Restore previous context
            if (prevContext) {
                builder.Bind(*prevContext);
            } else {
                builder.Unbind();
            }
        }

        return oss.str();
    }

    /**
     * Check if a struct type is already defined
     * @param TypeName The struct type name
     * @return True if already defined
     */
    bool KernelBuildContext::HasStructDefinition(const std::string &TypeName) const {
        return _definedStructs.count(TypeName) > 0;
    }

    /**
     * Add a struct type definition
     * @param TypeName The struct type name
     * @param Definition The GLSL struct definition code
     */
    void KernelBuildContext::AddStructDefinition(const std::string &TypeName, const std::string &Definition) {
        if (_definedStructs.insert(TypeName).second) {
            _structNames.push_back(TypeName);
            _structDefinitions.push_back(Definition);
        }
        // Note: We now output forward declarations for all structs before the actual definitions,
        // so the order of registration no longer matters for GLSL compilation.
    }

    /**
     * Get all struct definitions
     * @return Vector of struct definitions in order of registration
     */
    const std::vector<std::string> &KernelBuildContext::GetStructDefinitions() const {
        return _structDefinitions;
    }

    /**
     * Allocate a binding slot for buffer/image
     * @return The allocated binding slot index
     */
    uint32_t KernelBuildContext::AllocateBindingSlot() {
        return _nextBinding++;
    }

    /**
     * Register a buffer for the kernel
     * @param binding The binding slot
     * @param typeName The element type name in GLSL
     * @param bufferName The buffer variable name
     * @param mode The buffer access mode
     */
    void KernelBuildContext::RegisterBuffer(uint32_t binding, const std::string &typeName,
                                            const std::string &bufferName, int mode) {
        _buffers.push_back({binding, typeName, bufferName, mode});
        _bufferBindings.push_back(binding);
    }

    /**
     * Get the buffer declarations for GLSL
     * @return The buffer declaration string
     */
    std::string KernelBuildContext::GetBufferDeclarations() const {
        std::ostringstream oss;
        for (const auto &buf: _buffers) {
            std::string qualifier;
            if (buf.mode == 0x88B8) {
                // GL_READ_ONLY
                qualifier = "readonly ";
            } else if (buf.mode == 0x88B9) {
                // GL_WRITE_ONLY
                qualifier = "writeonly ";
            }
            // GL_READ_WRITE (0x88BA) has no qualifier

            oss << std::format("layout(std430, binding={}) {}buffer {}_t {{\n",
                               buf.binding, qualifier, buf.bufferName);
            oss << std::format("    {} {}[];\n", buf.typeName, buf.bufferName);
            oss << "};\n";
        }
        return oss.str();
    }

    /**
     * Allocate a binding slot for texture/image
     * @return The allocated binding slot index
     */
    uint32_t KernelBuildContext::AllocateTextureBinding() {
        return _nextTextureBinding++;
    }

    /**
     * Register a texture for the kernel
     * @param binding The binding slot
     * @param format The pixel format
     * @param textureName The texture variable name in GLSL
     * @param width Texture width
     * @param height Texture height
     */
    void KernelBuildContext::RegisterTexture(uint32_t binding, Runtime::PixelFormat format,
                                             const std::string &textureName,
                                             uint32_t width, uint32_t height) {
        _textures.push_back({binding, format, textureName, width, height});
        _textureBindings.push_back(binding);
    }

    /**
     * Get the texture declarations for GLSL
     * @return The texture declaration string
     */
    std::string KernelBuildContext::GetTextureDeclarations() const {
        std::ostringstream oss;
        for (const auto &tex: _textures) {
            std::string formatQualifier = GetGLSLFormatQualifier(tex.format);
            oss << std::format("layout({}, binding={}) uniform image2D {};\n",
                               formatQualifier, tex.binding, tex.textureName);
        }
        return oss.str();
    }

    // ===================================================================
    // Callable Function Support
    // ===================================================================

    void KernelBuildContext::AddCallableDeclaration(const std::string &declaration) {
        _callableDeclarations.push_back(declaration);
    }

    void KernelBuildContext::AddCallableBodyGenerator(std::function<void()> generator) {
        _callableBodyGenerators.push_back(std::move(generator));
    }

    void KernelBuildContext::PushCallableBody() {
        _callableBodyStack.push(_currentCallableBody);
        _currentCallableBody.clear();
        _inCallableBody = true;
    }

    void KernelBuildContext::PopCallableBody() {
        // Store the completed callable body
        _callableBodies.push_back(std::move(_currentCallableBody));
        _currentCallableBody.clear();
        _inCallableBody = false;
        
        // Restore previous state if nested
        if (!_callableBodyStack.empty()) {
            _currentCallableBody = _callableBodyStack.top();
            _callableBodyStack.pop();
            _inCallableBody = true;
        }
    }

    std::vector<std::string> KernelBuildContext::GetCallableDeclarations() const {
        return _callableDeclarations;
    }

    std::string KernelBuildContext::GenerateCallableBodies() {
        std::ostringstream oss;
        
        // NOTE: Bodies were already generated in Phase 1 (GetCompleteCode).
        // We just output them here without re-executing generators.
        // This avoids duplicate generation when Callables call other Callables.
        
        // Output callable function definitions
        // Bodies should be in _callableBodies in the same order as declarations
        size_t bodyCount = _callableBodies.size();
        for (size_t i = 0; i < _callableDeclarations.size(); ++i) {
            // Check if body was produced
            if (i < bodyCount && !_callableBodies[i].empty()) {
                // Output this callable
                oss << _callableDeclarations[i] << " {\n";
                
                // Indent the body
                std::istringstream bodyStream(_callableBodies[i]);
                std::string line;
                while (std::getline(bodyStream, line)) {
                    if (!line.empty()) {
                        oss << "    " << line << "\n";
                    } else {
                        oss << "\n";
                    }
                }
                
                oss << "}\n\n";
            }
        }
        
        return oss.str();
    }
}
