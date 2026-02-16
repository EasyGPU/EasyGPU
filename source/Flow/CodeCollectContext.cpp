/**
 * CodeCollectContext.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */

#include <Flow/CodeCollectContext.h>

namespace GPU::Flow {
    CodeCollectContext::CodeCollectContext()
        : _parentContext(nullptr) {
    }

    void CodeCollectContext::PushTranslatedCode(std::string Code) {
        _collectedCode.push_back(std::move(Code));
    }

    std::string CodeCollectContext::AssignVarName() {
        if (_parentContext) {
            return _parentContext->AssignVarName();
        }
        return "";
    }

    bool CodeCollectContext::HasStructDefinition(const std::string &TypeName) const {
        if (_parentContext) {
            return _parentContext->HasStructDefinition(TypeName);
        }
        return false;
    }

    void CodeCollectContext::AddStructDefinition(const std::string &TypeName, const std::string &Definition) {
        if (_parentContext) {
            _parentContext->AddStructDefinition(TypeName, Definition);
        }
    }

    const std::vector<std::string> &CodeCollectContext::GetStructDefinitions() const {
        if (_parentContext) {
            return _parentContext->GetStructDefinitions();
        }
        static std::vector<std::string> empty;
        return empty;
    }

    uint32_t CodeCollectContext::AllocateBindingSlot() {
        if (_parentContext) {
            return _parentContext->AllocateBindingSlot();
        }
        return 0;
    }

    void CodeCollectContext::RegisterBuffer(uint32_t binding, const std::string& typeName,
                                           const std::string& bufferName, int mode) {
        if (_parentContext) {
            _parentContext->RegisterBuffer(binding, typeName, bufferName, mode);
        }
    }

    std::string CodeCollectContext::GetBufferDeclarations() const {
        if (_parentContext) {
            return _parentContext->GetBufferDeclarations();
        }
        return "";
    }

    const std::vector<uint32_t>& CodeCollectContext::GetBufferBindings() const {
        if (_parentContext) {
            return _parentContext->GetBufferBindings();
        }
        static std::vector<uint32_t> empty;
        return empty;
    }

    void CodeCollectContext::BindRuntimeBuffer(uint32_t binding, uint32_t bufferHandle) {
        if (_parentContext) {
            _parentContext->BindRuntimeBuffer(binding, bufferHandle);
        }
    }

    const std::unordered_map<uint32_t, uint32_t>& CodeCollectContext::GetRuntimeBufferBindings() const {
        if (_parentContext) {
            return _parentContext->GetRuntimeBufferBindings();
        }
        static std::unordered_map<uint32_t, uint32_t> empty;
        return empty;
    }

    uint32_t CodeCollectContext::AllocateTextureBinding() {
        if (_parentContext) {
            return _parentContext->AllocateTextureBinding();
        }
        return 0;
    }

    void CodeCollectContext::RegisterTexture(uint32_t binding, Runtime::PixelFormat format,
                                            const std::string& textureName,
                                            uint32_t width, uint32_t height) {
        if (_parentContext) {
            _parentContext->RegisterTexture(binding, format, textureName, width, height);
        }
    }

    std::string CodeCollectContext::GetTextureDeclarations() const {
        if (_parentContext) {
            return _parentContext->GetTextureDeclarations();
        }
        return "";
    }

    const std::vector<uint32_t>& CodeCollectContext::GetTextureBindings() const {
        if (_parentContext) {
            return _parentContext->GetTextureBindings();
        }
        static std::vector<uint32_t> empty;
        return empty;
    }

    void CodeCollectContext::BindRuntimeTexture(uint32_t binding, uint32_t textureHandle) {
        if (_parentContext) {
            _parentContext->BindRuntimeTexture(binding, textureHandle);
        }
    }

    const std::unordered_map<uint32_t, uint32_t>& CodeCollectContext::GetRuntimeTextureBindings() const {
        if (_parentContext) {
            return _parentContext->GetRuntimeTextureBindings();
        }
        static std::unordered_map<uint32_t, uint32_t> empty;
        return empty;
    }

    void CodeCollectContext::SetParentContext(IR::Builder::BuilderContext* parent) {
        _parentContext = parent;
    }

    const std::vector<std::string>& CodeCollectContext::GetCollectedCode() const {
        return _collectedCode;
    }

    std::vector<std::string> CodeCollectContext::ReleaseCollectedCode() {
        return std::move(_collectedCode);
    }

    void CodeCollectContext::Clear() {
        _collectedCode.clear();
    }

    // Uniform Support - delegate to parent
    std::string CodeCollectContext::RegisterUniform(const std::string& typeName, void* uniformPtr,
                                                    std::function<void(uint32_t program, const std::string& name, void* ptr)> uploadFunc) {
        if (_parentContext) {
            return _parentContext->RegisterUniform(typeName, uniformPtr, uploadFunc);
        }
        return "";
    }

    std::string CodeCollectContext::GetUniformDeclarations() const {
        if (_parentContext) {
            return _parentContext->GetUniformDeclarations();
        }
        return "";
    }

    // Callable Function Support - delegate to parent
    void CodeCollectContext::AddCallableDeclaration(const std::string &declaration) {
        if (_parentContext) {
            _parentContext->AddCallableDeclaration(declaration);
        }
    }

    void CodeCollectContext::AddCallableBodyGenerator(std::function<void()> generator) {
        if (_parentContext) {
            _parentContext->AddCallableBodyGenerator(std::move(generator));
        }
    }

    void CodeCollectContext::PushCallableBody() {
        if (_parentContext) {
            _parentContext->PushCallableBody();
        }
    }

    void CodeCollectContext::PopCallableBody() {
        if (_parentContext) {
            _parentContext->PopCallableBody();
        }
    }

    std::vector<std::string> CodeCollectContext::GetCallableDeclarations() const {
        if (_parentContext) {
            return _parentContext->GetCallableDeclarations();
        }
        return {};
    }

    std::string CodeCollectContext::GenerateCallableBodies() {
        if (_parentContext) {
            return _parentContext->GenerateCallableBodies();
        }
        return "";
    }

    IR::Builder::CallableGenState &CodeCollectContext::GetCallableState(const void *callablePtr) {
        if (_parentContext) {
            return _parentContext->GetCallableState(callablePtr);
        }
        // Fallback to base class implementation (should not reach here in normal usage)
        return BuilderContext::GetCallableState(callablePtr);
    }
}
