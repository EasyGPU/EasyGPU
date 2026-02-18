#pragma once

/**
 * ShaderUtils.h:
 *      @Descripiton    :   Shader compilation utilities and beautiful error formatting
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_SHADERUTILS_H
#define EASYGPU_SHADERUTILS_H

#include <Runtime/ShaderException.h>

#include <GLAD/glad.h>

#include <string>
#include <vector>
#include <sstream>
#include <format>
#include <regex>
#include <iostream>

namespace GPU::Runtime {
    /**
     * ANSI color codes for beautiful terminal output
     */
    namespace Colors {
        constexpr const char *Reset = "\033[0m";
        constexpr const char *Bold = "\033[1m";
        constexpr const char *Dim = "\033[2m";
        constexpr const char *Red = "\033[31m";
        constexpr const char *Green = "\033[32m";
        constexpr const char *Yellow = "\033[33m";
        constexpr const char *Blue = "\033[34m";
        constexpr const char *Magenta = "\033[35m";
        constexpr const char *Cyan = "\033[36m";
        constexpr const char *White = "\033[37m";
        constexpr const char *BGRed = "\033[41m";
        constexpr const char *BGGreen = "\033[42m";
        constexpr const char *BGYellow = "\033[43m";
        constexpr const char *BGBlue = "\033[44m";
        constexpr const char *BGMagenta = "\033[45m";
        constexpr const char *BGCyan = "\033[46m";
    }

    /**
     * Box drawing characters for beautiful frames
     */
    namespace BoxChars {
        constexpr const char *TopLeft = "╔";
        constexpr const char *TopRight = "╗";
        constexpr const char *BottomLeft = "╚";
        constexpr const char *BottomRight = "╝";
        constexpr const char *Horizontal = "═";
        constexpr const char *Vertical = "║";
        constexpr const char *LeftT = "╠";
        constexpr const char *RightT = "╣";
        constexpr const char *Cross = "╬";
        constexpr const char *Bullet = "*";
        constexpr const char *Arrow = "->";
        constexpr const char *Check = "[OK]";
        constexpr const char *CrossX = "[X]";
        constexpr const char *Warning = "[!]";
        constexpr const char *Info = "[i]";
    }

    /**
     * Utility class for shader operations
     */
    class ShaderCompiler {
    public:
        /**
         * Compile a single shader stage
         * @param type GL_COMPUTE_SHADER, GL_VERTEX_SHADER, etc.
         * @param source GLSL source code
         * @return Shader handle
         * @throw ShaderCompileException on failure
         */
        static uint32_t CompileShader(uint32_t type, const std::string &source);

        /**
         * Link shader program
         * @param shaders Vector of compiled shader handles
         * @return Program handle
         * @throw ShaderLinkException on failure
         */
        static uint32_t LinkProgram(const std::vector<uint32_t> &shaders);

        /**
         * Compile compute shader from source
         * @param source GLSL compute shader source
         * @return Linked program handle
         * @throw ShaderCompileException or ShaderLinkException on failure
         */
        static uint32_t CompileComputeShader(const std::string &source);

    private:
        static std::string GetShaderTypeName(uint32_t type);

        static std::vector<ShaderDiagnostic> ParseErrorLog(const std::string &log, const std::string &source);
    };

    /**
     * Beautiful output formatter
     */
    class ShaderErrorFormatter {
    public:
        /**
         * Print beautiful error to output stream
         */
        static void PrintError(std::ostream &out, const ShaderException &ex);

        /**
         * Format source code with line numbers and error highlighting
         */
        static std::string FormatSourceWithErrors(const std::string &source,
                                                  const std::vector<ShaderDiagnostic> &diagnostics);

        /**
         * Create a framed box with title
         */
        static std::string MakeBox(const std::string &title, const std::vector<std::string> &lines,
                                   const char *titleColor = Colors::Red);

        /**
         * Format severity with icon and color
         */
        static std::string FormatSeverity(ErrorSeverity severity);

    private:
        static std::string Repeat(const char *ch, int count);

        static std::string PadRight(const std::string &s, int width);
    };

} // namespace GPU::Runtime

#endif //EASYGPU_SHADERUTILS_H
