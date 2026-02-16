/**
 * If.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/14/2026
 */

#include <Flow/If.h>

#include <stdexcept>
#include <format>

namespace GPU::Flow {
    // =============================================================================
    // ScopedCodeCollect
    // =============================================================================

    ScopedCodeCollect::ScopedCodeCollect(CodeCollectContext& collectContext)
        : _collectContext(collectContext)
        , _originalContext(nullptr) {
        auto* builder = &IR::Builder::Builder::Get();
        _originalContext = builder->Context();

        // Set parent for delegation
        _collectContext.SetParentContext(_originalContext);

        // Switch to collection context
        builder->Bind(_collectContext);
    }

    ScopedCodeCollect::~ScopedCodeCollect() {
        // Restore original context
        if (_originalContext) {
            IR::Builder::Builder::Get().Bind(*_originalContext);
        }
    }

    // =============================================================================
    // CollectedCodeToNodes
    // =============================================================================

    std::vector<std::unique_ptr<IR::Node::Node>> CollectedCodeToNodes(const std::vector<std::string>& codeLines) {
        std::vector<std::unique_ptr<IR::Node::Node>> nodes;
        for (const auto& line : codeLines) {
            // Remove trailing newline if present
            std::string trimmed = line;
            if (!trimmed.empty() && trimmed.back() == '\n') {
                trimmed.pop_back();
            }
            // Remove trailing semicolon for raw code nodes (will be added back by builder if needed)
            if (!trimmed.empty() && trimmed.back() == ';') {
                trimmed.pop_back();
            }
            nodes.push_back(std::make_unique<IR::Node::RawCodeNode>(trimmed));
        }
        return nodes;
    }

    // =============================================================================
    // IfChain
    // =============================================================================

    IfChain::IfChain(std::unique_ptr<IR::Node::Node> condition,
                     std::vector<std::string> ifCode,
                     IR::Builder::BuilderContext* originalContext)
        : _originalContext(originalContext)
        , _emitted(false) {
        _conditions.push_back(std::move(condition));
        _codeBlocks.push_back(std::move(ifCode));
    }

    IfChain& IfChain::Elif(IR::Value::Expr<bool>&& condition, const std::function<void()>& body) {
        // Collect code for this elif branch
        CodeCollectContext collectContext;
        {
            ScopedCodeCollect guard(collectContext);
            body();
        }

        _conditions.push_back(condition.Release());
        _codeBlocks.push_back(collectContext.ReleaseCollectedCode());

        return *this;
    }

    void IfChain::Else(const std::function<void()>& body) {
        // Collect code for else branch
        CodeCollectContext collectContext;
        {
            ScopedCodeCollect guard(collectContext);
            body();
        }

        _elseCode = collectContext.ReleaseCollectedCode();

        // Finalize - emit the complete if statement
        EmitIfStatement();
    }

    IfChain::~IfChain() {
        // Only emit if we haven't already (i.e., no Else was called)
        if (!_emitted && _originalContext) {
            EmitIfStatement();
        }
    }

    IfChain::IfChain(IfChain&& other) noexcept
        : _conditions(std::move(other._conditions))
        , _codeBlocks(std::move(other._codeBlocks))
        , _elseCode(std::move(other._elseCode))
        , _originalContext(other._originalContext)
        , _emitted(other._emitted) {
        other._originalContext = nullptr;
        other._emitted = true;
    }

    IfChain& IfChain::operator=(IfChain&& other) noexcept {
        if (this != &other) {
            _conditions = std::move(other._conditions);
            _codeBlocks = std::move(other._codeBlocks);
            _elseCode = std::move(other._elseCode);
            _originalContext = other._originalContext;
            _emitted = other._emitted;
            other._originalContext = nullptr;
            other._emitted = true;
        }
        return *this;
    }

    void IfChain::EmitIfStatement() {
        if (_emitted || !_originalContext) {
            return;
        }
        _emitted = true;

        // Build the complete if statement as GLSL code
        std::string ifCode = BuildGLSLIf();

        // Push to original context
        _originalContext->PushTranslatedCode(ifCode);
    }

    std::string IfChain::BuildGLSLIf() {
        if (_conditions.empty() || _codeBlocks.empty()) {
            return "";
        }

        IR::Builder::Builder& builder = IR::Builder::Builder::Get();

        std::string result;

        // Build if branch
        std::string condStr = builder.BuildNode(*_conditions[0]);
        result += std::format("if ({}) {{\n", condStr);
        for (const auto& line : _codeBlocks[0]) {
            result += "    " + line;
        }
        result += "}";

        // Build elif branches
        for (size_t i = 1; i < _conditions.size() && i < _codeBlocks.size(); ++i) {
            std::string elifCondStr = builder.BuildNode(*_conditions[i]);
            result += std::format(" else if ({}) {{\n", elifCondStr);
            for (const auto& line : _codeBlocks[i]) {
                result += "    " + line;
            }
            result += "}";
        }

        // Build else branch
        if (!_elseCode.empty()) {
            result += " else {\n";
            for (const auto& line : _elseCode) {
                result += "    " + line;
            }
            result += "}";
        }

        result += "\n";
        return result;
    }

    // =============================================================================
    // If function
    // =============================================================================

    IfChain If(IR::Value::Expr<bool> condition, const std::function<void()>& body) {
        auto* originalContext = IR::Builder::Builder::Get().Context();
        if (!originalContext) {
            throw std::runtime_error("If() called outside of Kernel definition");
        }

        // Collect code for if branch
        CodeCollectContext collectContext;
        {
            ScopedCodeCollect guard(collectContext);
            body();
        }

        return IfChain(condition.Release(), collectContext.ReleaseCollectedCode(), originalContext);
    }
}
