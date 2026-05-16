/**
 * @file AdjointTable.cpp
 * @brief Implementation of the forward-to-adjoint variable name mapping.
 */

#include <AD/AdjointTable.h>

#include <format>

namespace GPU::AD {

std::string AdjointTable::GetOrCreate(const std::string &varName, const std::string &glslType) {
	auto it = _map.find(varName);
	if (it != _map.end()) return it->second;

	std::string adjName = MakeAdjointName(varName);
	_map[varName] = adjName;
	_types[adjName] = glslType;
	_insertionOrder.push_back(adjName);
	return adjName;
}

std::string AdjointTable::Get(const std::string &varName) const {
	auto it = _map.find(varName);
	return it != _map.end() ? it->second : "";
}

bool AdjointTable::Has(const std::string &varName) const {
	return _map.count(varName) > 0;
}

std::vector<std::pair<std::string, std::string>> AdjointTable::AllDeclarations() const {
	std::vector<std::pair<std::string, std::string>> decls;
	for (const auto &adjName : _insertionOrder) {
		auto typeIt = _types.find(adjName);
		if (typeIt != _types.end()) {
			decls.emplace_back(adjName, typeIt->second);
		}
	}
	return decls;
}

void AdjointTable::Clear() {
	_map.clear();
	_types.clear();
	_insertionOrder.clear();
}

std::string AdjointTable::MakeAdjointName(const std::string &varName) {
	// Buffer element access "buf2[0]" -> "grad_buf2_0" (include constant index
	// to disambiguate multiple parameters from the same buffer).
	if (auto bracketPos = varName.find('['); bracketPos != std::string::npos) {
		auto closePos = varName.find(']', bracketPos);
		std::string base = varName.substr(0, bracketPos);
		if (closePos != std::string::npos && closePos > bracketPos + 1) {
			std::string idx = varName.substr(bracketPos + 1, closePos - bracketPos - 1);
			if (idx.find('(') == std::string::npos && idx.find('v') == std::string::npos) {
				return "grad_" + base + "_" + idx;
			}
		}
		return "grad_" + base;
	}
	// Swizzle access "v3.xyz" -> "d_v3_xyz"
	std::string sanitized;
	for (char c : varName) {
		if (c == '.') sanitized += '_';
		else sanitized += c;
	}
	return std::format("d_{}", sanitized);
}

} // namespace GPU::AD
