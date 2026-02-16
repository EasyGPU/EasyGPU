/**
 * ShaderUtils.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 */

#include "Runtime/ShaderUtils.h"

#include <algorithm>
#include <iomanip>
#include <unordered_set>

namespace GPU::Runtime {

// =============================================================================
// ShaderException Implementation
// =============================================================================

std::string ShaderCompileException::GetStageName(uint32_t type) {
    switch (type) {
        case GL_COMPUTE_SHADER: return "Compute";
        case GL_VERTEX_SHADER: return "Vertex";
        case GL_FRAGMENT_SHADER: return "Fragment";
        case GL_GEOMETRY_SHADER: return "Geometry";
        case GL_TESS_CONTROL_SHADER: return "TessControl";
        case GL_TESS_EVALUATION_SHADER: return "TessEval";
        default: return "Unknown";
    }
}

std::string ShaderCompileException::FormatBeautifulError() const {
    std::ostringstream oss;
    
    using namespace Colors;
    using namespace BoxChars;
    
    auto Repeat = [](const char* ch, int count) {
        std::string result;
        for (int i = 0; i < count; i++) {
            result += ch;
        }
        return result;
    };
    
    // Header with box
    int width = 76;
    std::string title = std::format(" {} SHADER COMPILATION FAILED ", GetStageName(_shaderType));
    int padding = (width - 2 - title.length()) / 2;
    
    // Truncate error message if too long
    std::string displayMessage = _message;
    if (displayMessage.length() > 200) {
        displayMessage = displayMessage.substr(0, 197) + "...";
    }
    
    oss << "\n";
    oss << Bold << Red << TopLeft << Repeat(Horizontal, padding) << title 
        << Repeat(Horizontal, width - 2 - padding - (int)title.length()) << TopRight << Reset << "\n";
    
    // Error summary - ensure we don't create negative string length
    int msgDisplayLen = std::min((int)displayMessage.length(), width - 15);
    oss << Bold << Red << Vertical << Reset << " " << Red << CrossX << " Error: " 
        << Reset << displayMessage.substr(0, msgDisplayLen);
    if (msgDisplayLen < (int)displayMessage.length()) {
        oss << "...";
    }
    int paddingSpaces = width - 12 - msgDisplayLen - (msgDisplayLen < (int)displayMessage.length() ? 3 : 0);
    if (paddingSpaces > 0) {
        oss << std::string(paddingSpaces, ' ');
    }
    oss << Bold << Red << Vertical << Reset << "\n";
    
    oss << Bold << Red << LeftT << Repeat(Horizontal, width - 2) << RightT << Reset << "\n";
    
    // Source code preview (first 20 lines max)
    oss << Bold << Cyan << Vertical << " Source Preview:" 
        << std::string(width - 17, ' ') << Vertical << Reset << "\n";
    
    std::istringstream sourceStream(_source);
    std::string line;
    int lineNum = 1;
    int maxLines = 20;
    
    while (std::getline(sourceStream, line) && lineNum <= maxLines) {
        std::string lineStr = std::format("{:4}", lineNum);
        // Truncate long lines
        if (line.length() > width - 15) {
            line = line.substr(0, width - 18) + "...";
        }
        oss << Dim << Vertical << Reset << " " << Cyan << lineStr << " │ " << Reset
            << line << std::string(width - 10 - lineStr.length() - line.length(), ' ') 
            << Dim << Vertical << Reset << "\n";
        lineNum++;
    }
    
    if (lineNum > maxLines) {
        oss << Dim << Vertical << "     ... " << (lineNum - maxLines) << " more lines ..." 
            << std::string(width - 30, ' ') << Vertical << Reset << "\n";
    }
    
    oss << Bold << Red << BottomLeft << Repeat(Horizontal, width - 2) << BottomRight << Reset << "\n";
    
    return oss.str();
}

std::string ShaderLinkException::FormatBeautifulError() const {
    std::ostringstream oss;
    
    using namespace Colors;
    using namespace BoxChars;
    
    auto Repeat = [](const char* ch, int count) {
        std::string result;
        for (int i = 0; i < count; i++) {
            result += ch;
        }
        return result;
    };
    
    int width = 76;
    std::string title = " PROGRAM LINKING FAILED ";
    int padding = (width - 2 - title.length()) / 2;
    
    oss << "\n";
    oss << Bold << Red << TopLeft << Repeat(Horizontal, padding) << title 
        << Repeat(Horizontal, width - 2 - padding - (int)title.length()) << TopRight << Reset << "\n";
    
    oss << Bold << Red << Vertical << Reset << " " << Red << CrossX << " Error: " 
        << Reset << _message << std::string(width - 12 - _message.length(), ' ') 
        << Bold << Red << Vertical << Reset << "\n";
    
    // Attached shaders info
    oss << Bold << Yellow << LeftT << Repeat(Horizontal, width - 2) << RightT << Reset << "\n";
    oss << Bold << Yellow << Vertical << " Attached Shaders:" 
        << std::string(width - 19, ' ') << Vertical << Reset << "\n";
    
    for (const auto& [type, name] : _attachedShaders) {
        std::string typeName;
        switch (type) {
            case GL_COMPUTE_SHADER: typeName = "COMPUTE"; break;
            case GL_VERTEX_SHADER: typeName = "VERTEX"; break;
            case GL_FRAGMENT_SHADER: typeName = "FRAGMENT"; break;
            default: typeName = "OTHER"; break;
        }
        std::string info = std::format("  • {}: {}", typeName, name);
        oss << Yellow << Vertical << Reset << info 
            << std::string(width - 2 - info.length(), ' ') << Yellow << Vertical << Reset << "\n";
    }
    
    oss << Bold << Red << BottomLeft << Repeat(Horizontal, width - 2) << BottomRight << Reset << "\n";
    
    return oss.str();
}

// =============================================================================
// ShaderCompiler Implementation
// =============================================================================

uint32_t ShaderCompiler::CompileShader(uint32_t type, const std::string& source) {
    uint32_t shader = glCreateShader(type);
    if (shader == 0) {
        throw ShaderResourceException("shader object", "glCreateShader returned 0");
    }
    
    const char* sourcePtr = source.c_str();
    glShaderSource(shader, 1, &sourcePtr, nullptr);
    glCompileShader(shader);
    
    // Check compilation status
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    
    if (!success) {
        // Get error log
        int logLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        
        std::string log;
        if (logLength > 0) {
            log.resize(logLength);
            glGetShaderInfoLog(shader, logLength, nullptr, log.data());
            // Remove null terminator if present
            if (!log.empty() && log.back() == '\0') {
                log.pop_back();
            }
        }
        
        // Parse diagnostics from log
        auto diagnostics = ParseErrorLog(log, source);
        
        // Delete failed shader
        glDeleteShader(shader);
        
        throw ShaderCompileException(type, source, log, diagnostics);
    }
    
    return shader;
}

uint32_t ShaderCompiler::LinkProgram(const std::vector<uint32_t>& shaders) {
    uint32_t program = glCreateProgram();
    if (program == 0) {
        throw ShaderResourceException("program object", "glCreateProgram returned 0");
    }
    
    // Attach shaders
    for (uint32_t shader : shaders) {
        glAttachShader(program, shader);
    }
    
    // Link
    glLinkProgram(program);
    
    // Check link status
    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    
    // Collect attached shader info for error reporting
    std::vector<std::pair<uint32_t, std::string>> attachedInfo;
    for (uint32_t shader : shaders) {
        int shaderType;
        glGetShaderiv(shader, GL_SHADER_TYPE, &shaderType);
        attachedInfo.emplace_back(shaderType, "<compiled>");
    }
    
    if (!success) {
        // Get error log
        int logLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        
        std::string log;
        if (logLength > 0) {
            log.resize(logLength);
            glGetProgramInfoLog(program, logLength, nullptr, log.data());
            if (!log.empty() && log.back() == '\0') {
                log.pop_back();
            }
        }
        
        // Delete program
        glDeleteProgram(program);
        
        throw ShaderLinkException(log, attachedInfo);
    }
    
    // Detach and delete shaders (they're now linked into program)
    for (uint32_t shader : shaders) {
        glDetachShader(program, shader);
        glDeleteShader(shader);
    }
    
    return program;
}

uint32_t ShaderCompiler::CompileComputeShader(const std::string& source) {
    uint32_t shader = CompileShader(GL_COMPUTE_SHADER, source);
    
    try {
        uint32_t program = LinkProgram({shader});
        // Shader is detached and deleted by LinkProgram on success
        return program;
    } catch (...) {
        // Clean up shader if linking failed
        glDeleteShader(shader);
        throw;
    }
}

std::string ShaderCompiler::GetShaderTypeName(uint32_t type) {
    switch (type) {
        case GL_COMPUTE_SHADER: return "Compute Shader";
        case GL_VERTEX_SHADER: return "Vertex Shader";
        case GL_FRAGMENT_SHADER: return "Fragment Shader";
        case GL_GEOMETRY_SHADER: return "Geometry Shader";
        case GL_TESS_CONTROL_SHADER: return "Tessellation Control Shader";
        case GL_TESS_EVALUATION_SHADER: return "Tessellation Evaluation Shader";
        default: return "Unknown Shader";
    }
}

std::vector<ShaderDiagnostic> ShaderCompiler::ParseErrorLog(const std::string& log, const std::string& source) {
    std::vector<ShaderDiagnostic> diagnostics;
    
    // Parse NVIDIA/AMD style errors: "0(123) : error C0000: message"
    // Parse Intel style: "ERROR: 0:123: message"
    
    std::regex nvidiaRegex(R"((\d+)\((\d+)\)\s*:\s*(error|warning)\s*([\w\s]+):\s*(.+))");
    std::regex intelRegex(R"((ERROR|WARNING):\s*\d+:(\d+):\s*(.+))");
    
    std::istringstream stream(log);
    std::string line;
    
    while (std::getline(stream, line)) {
        std::smatch match;
        
        if (std::regex_search(line, match, nvidiaRegex)) {
            int lineNum = std::stoi(match[2].str());
            std::string severity = match[3].str();
            std::string message = match[5].str();
            
            ErrorSeverity sev = (severity == "error") ? ErrorSeverity::Error : ErrorSeverity::Warning;
            diagnostics.emplace_back(sev, message, "", lineNum, 0);
        }
        else if (std::regex_search(line, match, intelRegex)) {
            std::string severity = match[1].str();
            int lineNum = std::stoi(match[3].str());
            std::string message = match[4].str();
            
            ErrorSeverity sev = (severity == "ERROR") ? ErrorSeverity::Error : ErrorSeverity::Warning;
            diagnostics.emplace_back(sev, message, "", lineNum, 0);
        }
    }
    
    return diagnostics;
}

// =============================================================================
// ShaderErrorFormatter Implementation
// =============================================================================

void ShaderErrorFormatter::PrintError(std::ostream& out, const ShaderException& ex) {
    out << ex.GetBeautifulOutput();
}

std::string ShaderErrorFormatter::FormatSourceWithErrors(const std::string& source,
                                                          const std::vector<ShaderDiagnostic>& diagnostics) {
    std::ostringstream oss;
    std::istringstream sourceStream(source);
    std::string line;
    int lineNum = 1;
    
    // Collect error lines
    std::unordered_set<int> errorLines;
    for (const auto& diag : diagnostics) {
        if (diag.line > 0) {
            errorLines.insert(diag.line);
        }
    }
    
    while (std::getline(sourceStream, line)) {
        bool hasError = errorLines.count(lineNum) > 0;
        
        if (hasError) {
            oss << Colors::BGRed << Colors::White;
        } else {
            oss << Colors::Dim;
        }
        
        oss << std::setw(4) << lineNum << " │ " << Colors::Reset;
        
        if (hasError) {
            oss << Colors::Red << line << Colors::Reset;
        } else {
            oss << line;
        }
        
        oss << "\n";
        lineNum++;
    }
    
    return oss.str();
}

std::string ShaderErrorFormatter::MakeBox(const std::string& title, const std::vector<std::string>& lines,
                                           const char* titleColor) {
    using namespace BoxChars;
    using namespace Colors;
    
    int maxWidth = title.length() + 4;
    for (const auto& line : lines) {
        maxWidth = std::max(maxWidth, (int)line.length() + 4);
    }
    
    std::ostringstream oss;
    
    // Top border
    oss << titleColor << Bold << TopLeft << Repeat(Horizontal, maxWidth - 2) << TopRight << Reset << "\n";
    
    // Title
    int titlePadding = (maxWidth - 2 - title.length()) / 2;
    oss << titleColor << Bold << Vertical << Reset << std::string(titlePadding, ' ') 
        << Bold << title << Reset << std::string(maxWidth - 2 - titlePadding - title.length(), ' ') 
        << titleColor << Bold << Vertical << Reset << "\n";
    
    // Separator
    oss << titleColor << Bold << LeftT << Repeat(Horizontal, maxWidth - 2) << RightT << Reset << "\n";
    
    // Content
    for (const auto& line : lines) {
        oss << Vertical << " " << line << std::string(maxWidth - 3 - line.length(), ' ') << Vertical << "\n";
    }
    
    // Bottom border
    oss << titleColor << Bold << BottomLeft << Repeat(Horizontal, maxWidth - 2) << BottomRight << Reset << "\n";
    
    return oss.str();
}

std::string ShaderErrorFormatter::FormatSeverity(ErrorSeverity severity) {
    using namespace Colors;
    using namespace BoxChars;
    
    switch (severity) {
        case ErrorSeverity::Info:
            return std::format("{}{} {}{}", Blue, Info, Reset, "Info");
        case ErrorSeverity::Warning:
            return std::format("{}{} {}{}", Yellow, Warning, Reset, "Warning");
        case ErrorSeverity::Error:
            return std::format("{}{} {}{}", Red, CrossX, Reset, "Error");
        case ErrorSeverity::Fatal:
            return std::format("{}{} {}{}", Magenta, "💥", Reset, "Fatal");
        default:
            return "Unknown";
    }
}

std::string ShaderErrorFormatter::Repeat(const char* ch, int count) {
    std::string result;
    for (int i = 0; i < count; i++) {
        result += ch;
    }
    return result;
}

std::string ShaderErrorFormatter::PadRight(const std::string& s, int width) {
    if ((int)s.length() >= width) return s;
    return s + std::string(width - s.length(), ' ');
}

} // namespace GPU::Runtime
