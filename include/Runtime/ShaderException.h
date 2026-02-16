/**
 * ShaderException.h:
 *      @Descripiton    :   Beautiful exception types for shader compilation errors
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_SHADEREXCEPTION_H
#define EASYGPU_SHADEREXCEPTION_H

#include <stdexcept>
#include <cstdint>
#include <string>
#include <vector>
#include <format>

namespace GPU::Runtime {

    /**
     * Error severity level for syntax highlighting
     */
    enum class ErrorSeverity {
        Info,       // General information
        Warning,    // Warning (compilation succeeded with issues)
        Error,      // Error (compilation failed)
        Fatal       // Fatal error (system/resource issue)
    };

    /**
     * Single diagnostic message with location info
     */
    struct ShaderDiagnostic {
        ErrorSeverity severity;
        std::string message;
        std::string file;
        int line;
        int column;

        ShaderDiagnostic(ErrorSeverity sev, std::string msg,
                         std::string f = "", int l = 0, int c = 0)
                : severity(sev), message(std::move(msg)),
                  file(std::move(f)), line(l), column(c) {}
    };

    /**
     * Base class for shader-related exceptions
     */
    class ShaderException : public std::exception {
    public:
        explicit ShaderException(std::string stage, std::string message)
                : _stage(std::move(stage)), _message(std::move(message)), _formattedMessage(FormatMessage()) {}

        const char *what() const noexcept override {
            return _formattedMessage.c_str();
        }

        [[nodiscard]] const std::string &Stage() const { return _stage; }

        [[nodiscard]] const std::string &RawMessage() const { return _message; }

        /**
         * Get beautifully formatted error message with colors/unicode
         */
        [[nodiscard]] virtual std::string GetBeautifulOutput() const {
            return _formattedMessage;
        }

    protected:
        std::string _stage;
        std::string _message;
        std::string _formattedMessage;

        std::string FormatMessage() const {
            return std::format("[{}] {}", _stage, _message);
        }
    };

    /**
     * Shader compilation failed
     */
    class ShaderCompileException : public ShaderException {
    public:
        ShaderCompileException(uint32_t shaderType, std::string source,
                               std::string log, std::vector<ShaderDiagnostic> diagnostics = {})
                : ShaderException(GetStageName(shaderType), std::move(log)), _shaderType(shaderType),
                  _source(std::move(source)), _diagnostics(std::move(diagnostics)) {
            _formattedMessage = FormatBeautifulError();
        }

        [[nodiscard]] uint32_t GetShaderType() const { return _shaderType; }

        [[nodiscard]] const std::string &GetSource() const { return _source; }

        [[nodiscard]] const std::vector<ShaderDiagnostic> &GetDiagnostics() const { return _diagnostics; }

        [[nodiscard]] std::string GetBeautifulOutput() const override {
            return _formattedMessage;
        }

    private:
        uint32_t _shaderType;
        std::string _source;
        std::vector<ShaderDiagnostic> _diagnostics;

        static std::string GetStageName(uint32_t type);

        std::string FormatBeautifulError() const;
    };

    /**
     * Program linking failed
     */
    class ShaderLinkException : public ShaderException {
    public:
        ShaderLinkException(std::string log,
                            std::vector<std::pair<uint32_t, std::string>> attachedShaders = {})
                : ShaderException("Link", std::move(log)), _attachedShaders(std::move(attachedShaders)) {
            _formattedMessage = FormatBeautifulError();
        }

        [[nodiscard]] std::string GetBeautifulOutput() const override {
            return _formattedMessage;
        }

    private:
        std::vector<std::pair<uint32_t, std::string>> _attachedShaders;

        std::string FormatBeautifulError() const;
    };

    /**
     * Resource creation failed (out of memory, etc.)
     */
    class ShaderResourceException : public ShaderException {
    public:
        explicit ShaderResourceException(std::string resource, std::string reason)
                : ShaderException("Resource", std::format("Failed to create {}: {}", resource, reason)),
                  _resource(std::move(resource)), _reason(std::move(reason)) {}

        [[nodiscard]] const std::string &GetResource() const { return _resource; }

        [[nodiscard]] const std::string &GetReason() const { return _reason; }

    private:
        std::string _resource;
        std::string _reason;
    };

    /**
     * Context not available for shader operations
     */
    class ShaderContextException : public ShaderException {
    public:
        explicit ShaderContextException(std::string reason)
                : ShaderException("Context", std::move(reason)) {}
    };

} // namespace GPU::Runtime

#endif //EASYGPU_SHADEREXCEPTION_H
