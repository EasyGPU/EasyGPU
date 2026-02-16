#pragma once

/**
 * CodeCollectContext.h:
 *      @Descripiton    :   The context for collecting code during control flow lambda execution
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */
#ifndef EASYGPU_CODECOLLECTCONTEXT_H
#define EASYGPU_CODECOLLECTCONTEXT_H

#include <IR/Builder/BuilderContext.h>

#include <string>
#include <vector>
#include <unordered_map>

namespace GPU::Flow {
    /**
     * A temporary context for collecting code during lambda execution.
     * Instead of pushing code to the main output, it stores lines in a vector.
     * Delegates variable naming and other operations to the parent context.
     */
    class CodeCollectContext : public IR::Builder::BuilderContext {
    public:
        CodeCollectContext();

    public:
        void PushTranslatedCode(std::string Code) override;
        std::string AssignVarName() override;

        bool HasStructDefinition(const std::string &TypeName) const override;
        void AddStructDefinition(const std::string &TypeName, const std::string &Definition) override;
        const std::vector<std::string> &GetStructDefinitions() const override;

        uint32_t AllocateBindingSlot() override;
        void RegisterBuffer(uint32_t binding, const std::string& typeName,
                           const std::string& bufferName, int mode) override;
        std::string GetBufferDeclarations() const override;
        const std::vector<uint32_t>& GetBufferBindings() const override;
        void BindRuntimeBuffer(uint32_t binding, uint32_t bufferHandle) override;
        const std::unordered_map<uint32_t, uint32_t>& GetRuntimeBufferBindings() const override;

        uint32_t AllocateTextureBinding() override;
        void RegisterTexture(uint32_t binding, Runtime::PixelFormat format,
                            const std::string& textureName,
                            uint32_t width, uint32_t height) override;
        std::string GetTextureDeclarations() const override;
        const std::vector<uint32_t>& GetTextureBindings() const override;
        void BindRuntimeTexture(uint32_t binding, uint32_t textureHandle) override;
        const std::unordered_map<uint32_t, uint32_t>& GetRuntimeTextureBindings() const override;

        // Callable Function Support
        // Uniform Support
        std::string RegisterUniform(const std::string& typeName, void* uniformPtr,
                                    std::function<void(uint32_t program, const std::string& name, void* ptr)> uploadFunc) override;
        std::string GetUniformDeclarations() const override;

        void AddCallableDeclaration(const std::string &declaration) override;
        void AddCallableBodyGenerator(std::function<void()> generator) override;
        void PushCallableBody() override;
        void PopCallableBody() override;
        std::vector<std::string> GetCallableDeclarations() const override;
        std::string GenerateCallableBodies() override;

    public:
        /**
         * Set the parent context for delegation
         * @param parent The parent builder context
         */
        void SetParentContext(IR::Builder::BuilderContext* parent);

        /**
         * Get the collected code lines
         * @return Vector of collected code lines
         */
        const std::vector<std::string>& GetCollectedCode() const;

        /**
         * Move out the collected code lines
         * @return Vector of collected code lines (moved)
         */
        std::vector<std::string> ReleaseCollectedCode();

        /**
         * Clear the collected code
         */
        void Clear();

        /**
         * Get the generation state for a callable - delegate to parent context
         * This ensures Callable declarations are not duplicated when used inside control flow
         */
        IR::Builder::CallableGenState &GetCallableState(const void *callablePtr) override;

    private:
        IR::Builder::BuilderContext* _parentContext;
        std::vector<std::string> _collectedCode;
    };
}

#endif //EASYGPU_CODECOLLECTCONTEXT_H
