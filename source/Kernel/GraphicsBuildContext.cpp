/**
 * @file GraphicsBuildContext.cpp
 * @brief Graphics pipeline build context — VS + FS GLSL code generation.
 */

#include <IR/Value/Varying.h>
#include <Kernel/GraphicsBuildContext.h>

#include <Runtime/PixelFormat.h>

#include <sstream>
#include <stdexcept>

namespace GPU::Kernel {

GraphicsBuildContext::GraphicsBuildContext() : KernelBuildContext(2) {
	WorkSizeX = 1;
	WorkSizeY = 1;
	WorkSizeZ = 1;

	// Drain the global varying registry — Varying<T> instances constructed
	// before the GraphicsPipeline register here since there's no active context.
	for (auto &entry : IR::Value::DrainVaryingRegistry()) {
		RegisterVarying(entry.name, entry.glslType);
	}
}

void GraphicsBuildContext::RegisterVarying(const std::string &name, const std::string &glslType) {
	// Check for duplicates
	for (const auto &v : _varyings) {
		if (v.name == name) {
			return; // Already registered
		}
	}
	VaryingInfo info;
	info.name	  = name;
	info.glslType = glslType;
	info.location = static_cast<uint32_t>(_varyings.size());
	_varyings.push_back(info);
}

void GraphicsBuildContext::SetVertexLayout(const std::vector<Backend::VertexLayoutEntry> &layout) {
	_vertexLayout = layout;
}

void GraphicsBuildContext::PushTranslatedCode(std::string Code) {
	if (_stage != ShaderStage::VS && _stage != ShaderStage::FS) {
		throw std::runtime_error("GraphicsBuildContext: translated code emitted outside a shader stage");
	}
	KernelBuildContext::PushTranslatedCode(std::move(Code));
}

// ===================================================================
// Code Generation
// ===================================================================

std::string GraphicsBuildContext::GenerateCommonHeaders() {
	std::ostringstream oss;

	// GLSL version — use 450 for Vulkan compatibility
	oss << "#version 450 core\n\n";

	// Float atomic extensions
	if (!_floatAtomicBuffers.empty()) {
		oss << "#extension GL_NV_shader_atomic_float : enable\n"
			<< "#extension GL_EXT_shader_atomic_float : enable\n\n";
	}

	// Struct definitions
	for (const auto &structDef : GetStructDefinitions()) {
		oss << structDef;
	}
	if (!GetStructDefinitions().empty()) {
		oss << "\n";
	}

	// Callable forward declarations
	for (const auto &decl : _callableDeclarations) {
		oss << decl << ";\n";
	}
	if (!_callableDeclarations.empty()) {
		oss << "\n";
	}

	return oss.str();
}

std::string GraphicsBuildContext::GenerateVertexInputs() {
	std::ostringstream oss;

	if (_vertexLayout.empty()) {
		// No vertex layout — user expects no vertex inputs (fullscreen triangle via gl_VertexIndex)
		return "";
	}

	for (const auto &entry : _vertexLayout) {
		// Map pixel format to GLSL type for vertex input
		std::string glslType;
		switch (entry.format) {
		case Backend::PixelFormat::R32F:
			glslType = "float";
			break;
		case Backend::PixelFormat::RG32F:
			glslType = "vec2";
			break;
		case Backend::PixelFormat::RGB32F:
			glslType = "vec3";
			break;
		case Backend::PixelFormat::RGBA32F:
			glslType = "vec4";
			break;
		case Backend::PixelFormat::RGBA8:
			glslType = "vec4";
			break;
		case Backend::PixelFormat::R32I:
			glslType = "int";
			break;
		case Backend::PixelFormat::RG32I:
			glslType = "ivec2";
			break;
		case Backend::PixelFormat::RGB32I:
			glslType = "ivec3";
			break;
		case Backend::PixelFormat::RGBA32I:
			glslType = "ivec4";
			break;
		case Backend::PixelFormat::R32UI:
			glslType = "uint";
			break;
		case Backend::PixelFormat::RG32UI:
			glslType = "uvec2";
			break;
		case Backend::PixelFormat::RGB32UI:
			glslType = "uvec3";
			break;
		case Backend::PixelFormat::RGBA32UI:
			glslType = "uvec4";
			break;
		default:
			glslType = "vec4";
			break;
		}
		oss << "layout(location=" << entry.location << ") in " << glslType << " a_" << entry.location << ";\n";
	}
	oss << "\n";
	return oss.str();
}

std::string GraphicsBuildContext::GenerateVaryingOutputs() {
	std::ostringstream oss;

	for (const auto &v : _varyings) {
		oss << "layout(location=" << v.location << ") out " << v.glslType << " " << v.name << ";\n";
	}

	if (!_varyings.empty()) {
		oss << "\n";
	}
	return oss.str();
}

std::string GraphicsBuildContext::GenerateVaryingInputs() {
	std::ostringstream oss;

	for (const auto &v : _varyings) {
		oss << "layout(location=" << v.location << ") in " << v.glslType << " " << v.name << ";\n";
	}

	if (!_varyings.empty()) {
		oss << "\n";
	}
	return oss.str();
}

std::string GraphicsBuildContext::GenerateVertexShaderMain() {
	std::ostringstream oss;
	oss << "void main() {\n";

	if (!_vertexInputSetupCode.empty()) {
		std::istringstream setupStream(_vertexInputSetupCode);
		std::string		   line;
		while (std::getline(setupStream, line)) {
			if (!line.empty()) {
				oss << "\t" << line << "\n";
			}
		}
	}

	if (!_vsBodyCode.empty()) {
		std::istringstream codeStream(_vsBodyCode);
		std::string		   line;
		while (std::getline(codeStream, line)) {
			if (!line.empty()) {
				oss << "\t" << line << "\n";
			}
		}
	}

	oss << "}\n\n";
	return oss.str();
}

std::string GraphicsBuildContext::GenerateFragmentShaderMain() {
	std::ostringstream oss;
	oss << "void main() {\n";

	// Declare fragColor as output
	oss << "\tvec4 fragColor;\n\n";

	if (!_fsBodyCode.empty()) {
		std::istringstream codeStream(_fsBodyCode);
		std::string		   line;
		while (std::getline(codeStream, line)) {
			if (!line.empty()) {
				oss << "\t" << line << "\n";
			}
		}
	}

	// Output the fragment color
	oss << "\n\toutColor = fragColor;\n";
	oss << "}\n";
	return oss.str();
}

std::string GraphicsBuildContext::GetVertexShaderCode() {
	std::ostringstream oss;

	oss << GenerateCommonHeaders();
	oss << GenerateVertexInputs();
	oss << GenerateVaryingOutputs();

	// Buffer declarations (SSBOs)
	std::string bufDecls = GetBufferDeclarations();
	if (!bufDecls.empty()) {
		oss << bufDecls << "\n";
	}

	// Uniform declarations
	std::string uniformDecls = GetUniformDeclarations();
	if (!uniformDecls.empty()) {
		oss << uniformDecls << "\n";
	}

	oss << GenerateVertexShaderMain();
	return oss.str();
}

std::string GraphicsBuildContext::GetFragmentShaderCode() {
	std::ostringstream oss;

	oss << GenerateCommonHeaders();

	// Fragment shader output declaration
	oss << "layout(location=0) out vec4 outColor;\n\n";

	oss << GenerateVaryingInputs();

	// Texture declarations (sampler2D)
	std::string texDecls = GetTextureDeclarations();
	if (!texDecls.empty()) {
		oss << texDecls << "\n";
	}

	// Buffer declarations
	std::string bufDecls = GetBufferDeclarations();
	if (!bufDecls.empty()) {
		oss << bufDecls << "\n";
	}

	// Uniform declarations
	std::string uniformDecls = GetUniformDeclarations();
	if (!uniformDecls.empty()) {
		oss << uniformDecls << "\n";
	}

	oss << GenerateFragmentShaderMain();

	// Callable function bodies
	std::string callableDefs = GenerateCallableBodies();
	if (!callableDefs.empty()) {
		oss << callableDefs << "\n";
	}

	return oss.str();
}

std::string GraphicsBuildContext::GetCompleteCode() {
	std::ostringstream oss;
	oss << "// === Vertex Shader ===\n";
	oss << GetVertexShaderCode();
	oss << "// === Fragment Shader ===\n";
	oss << GetFragmentShaderCode();
	return oss.str();
}

std::string GraphicsBuildContext::GetTextureDeclarations() const {
	std::ostringstream oss;
	for (const auto &tex : _textures) {
		// Fragment shaders use sampler2D for reading (not image2D)
		if (tex.sampled) {
			oss << "layout(binding=" << tex.binding << ") uniform sampler2D " << tex.textureName << ";\n";
		}
	}
	return oss.str();
}

} // namespace GPU::Kernel
