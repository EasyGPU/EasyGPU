#pragma once

/**
 * @file ShaderUtils.h
 * @brief Shader compilation utilities and beautiful error formatting.
 */

#ifndef EASYGPU_SHADERUTILS_H
#define EASYGPU_SHADERUTILS_H

#include <Runtime/ShaderException.h>

#include <GLAD/glad.h>

#include <format>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace GPU::Runtime {
/**
 * @brief ANSI color codes for beautiful terminal output.
 */
namespace Colors {
constexpr const char *Reset		= "\033[0m";
constexpr const char *Bold		= "\033[1m";
constexpr const char *Dim		= "\033[2m";
constexpr const char *Red		= "\033[31m";
constexpr const char *Green		= "\033[32m";
constexpr const char *Yellow	= "\033[33m";
constexpr const char *Blue		= "\033[34m";
constexpr const char *Magenta	= "\033[35m";
constexpr const char *Cyan		= "\033[36m";
constexpr const char *White		= "\033[37m";
constexpr const char *BGRed		= "\033[41m";
constexpr const char *BGGreen	= "\033[42m";
constexpr const char *BGYellow	= "\033[43m";
constexpr const char *BGBlue	= "\033[44m";
constexpr const char *BGMagenta = "\033[45m";
constexpr const char *BGCyan	= "\033[46m";
} // namespace Colors

/**
 * @brief Box drawing characters for beautiful terminal frames.
 */
namespace BoxChars {
constexpr const char *TopLeft	  = "╔";
constexpr const char *TopRight	  = "╗";
constexpr const char *BottomLeft  = "╚";
constexpr const char *BottomRight = "╝";
constexpr const char *Horizontal  = "═";
constexpr const char *Vertical	  = "║";
constexpr const char *LeftT		  = "╠";
constexpr const char *RightT	  = "╣";
constexpr const char *Cross		  = "╬";
constexpr const char *Bullet	  = "*";
constexpr const char *Arrow		  = "->";
constexpr const char *Check		  = "[OK]";
constexpr const char *CrossX	  = "[X]";
constexpr const char *Warning	  = "[!]";
constexpr const char *Info		  = "[i]";
} // namespace BoxChars

/**
 * @brief Utility class for shader compilation and linking.
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
	static std::string					 GetShaderTypeName(uint32_t type);

	static std::vector<ShaderDiagnostic> ParseErrorLog(const std::string &log, const std::string &source);
};

/**
 * @brief Beautiful output formatter for shader errors.
 *
 * Renders shader compilation/linking errors with ANSI-colored framed boxes,
 * source code highlighting, and severity icons for terminal output.
 */
class ShaderErrorFormatter {
public:
	/**
	 * @brief Print a beautifully formatted error to an output stream.
	 * @param out The output stream to write to.
	 * @param ex The shader exception to format and print.
	 */
	static void		   PrintError(std::ostream &out, const ShaderException &ex);

	/**
	 * @brief Format source code with line numbers and error highlighting.
	 * @param source The GLSL source code string.
	 * @param diagnostics List of diagnostics to highlight in the source.
	 * @return Formatted source string with ANSI color codes.
	 */
	static std::string FormatSourceWithErrors(const std::string					  &source,
											  const std::vector<ShaderDiagnostic> &diagnostics);

	/**
	 * @brief Create a framed box with a title.
	 * @param title The title text for the box.
	 * @param lines The content lines to place inside the box.
	 * @param titleColor ANSI color code for the title.
	 * @return Formatted string with box-drawing characters.
	 */
	static std::string MakeBox(const std::string &title, const std::vector<std::string> &lines,
							   const char *titleColor = Colors::Red);

	/**
	 * @brief Format a severity level with icon and color.
	 * @param severity The error severity level.
	 * @return Formatted string with ANSI color and icon.
	 */
	static std::string FormatSeverity(ErrorSeverity severity);

private:
	static std::string Repeat(const char *ch, int count);

	static std::string PadRight(const std::string &s, int width);
};

} // namespace GPU::Runtime

#endif // EASYGPU_SHADERUTILS_H
