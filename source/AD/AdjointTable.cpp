/**
 * @file AdjointTable.cpp
 * @brief Implementation of the forward-to-adjoint variable name mapping.
 */

#include <AD/AdjointTable.h>

#include <cctype>
#include <format>

namespace GPU::AD {

std::string AdjointTable::GetOrCreate(const std::string &varName, const std::string &glslType) {
	auto it = _map.find(varName);
	if (it != _map.end())
		return it->second;

	std::string adjName = MakeAdjointName(varName);

	_map[varName]		= adjName;

	// Also index by base buffer name for variable-indexed lookups
	auto bpos			= varName.find('[');
	if (bpos != std::string::npos) {
		std::string base = varName.substr(0, bpos);
		if (!_baseMap.count(base)) {
			_baseMap[base] = adjName;
		}
	}

	// Only track this adjoint for declaration if it hasn't been seen before.
	// Multiple forward varNames (e.g., buf1[idx1], buf1[idx2]) can map to the
	// same adjoint name (e.g., "grad_buf1"), and it should be declared once.
	if (!_types.count(adjName)) {
		_types[adjName] = glslType;
		_insertionOrder.push_back(adjName);
	}
	return adjName;
}

std::string AdjointTable::Get(const std::string &varName) const {
	auto it = _map.find(varName);
	if (it == _map.end()) {
		// Fall back to base-name matching for buffer-type names with
		// variable indices (e.g. buf5[v115*16+v116] -> lookup "buf5")
		auto bpos = varName.find('[');
		if (bpos != std::string::npos) {
			std::string base   = varName.substr(0, bpos);
			auto		baseIt = _baseMap.find(base);
			if (baseIt == _baseMap.end())
				return "";
			std::string adjBase = baseIt->second;
			auto		epos	= varName.rfind(']');
			if (epos == std::string::npos || epos <= bpos)
				return "";
			std::string idxExpr = varName.substr(bpos + 1, epos - bpos - 1);
			return adjBase + "[" + idxExpr + "]";
		}
		return "";
	}
	// For buffer-type names (containing [...]), reconstruct the array-indexed
	// adjoint reference so callers get the full expression like grad_buf[0]
	// or grad_buf[tokenId*E+d] without needing to parse the index themselves.
	auto bpos = varName.find('[');
	if (bpos != std::string::npos) {
		auto		epos	= varName.rfind(']');
		if (epos == std::string::npos || epos <= bpos)
			return it->second;
		std::string idxExpr = varName.substr(bpos + 1, epos - bpos - 1);
		return it->second + "[" + idxExpr + "]";
	}
	return it->second;
}

bool AdjointTable::Has(const std::string &varName) const {
	return _map.count(varName) > 0;
}

std::string AdjointTable::GetTypeForAdjoint(const std::string &adjName) const {
	std::string baseName = adjName;
	if (auto bracketPos = baseName.find('['); bracketPos != std::string::npos) {
		baseName = baseName.substr(0, bracketPos);
	}
	auto it = _types.find(baseName);
	return it != _types.end() ? it->second : "";
}

std::vector<std::pair<std::string, std::string>> AdjointTable::AllDeclarations() const {
	std::vector<std::pair<std::string, std::string>> decls;
	for (const auto &adjName : _insertionOrder) {
		auto typeIt = _types.find(adjName);
		if (typeIt != _types.end()) {
			auto sizeIt = _arraySizes.find(adjName);
			if (sizeIt != _arraySizes.end() && sizeIt->second > 0) {
				// Array adjoint for buffer parameters
				decls.emplace_back(adjName, std::format("{}[{}]", typeIt->second, sizeIt->second));
			} else {
				decls.emplace_back(adjName, typeIt->second);
			}
		}
	}
	return decls;
}

void AdjointTable::DeclareAdjoint(const std::string &adjName, const std::string &glslType) {
	if (adjName.empty() || glslType.empty())
		return;

	if (!_types.count(adjName)) {
		_types[adjName] = glslType;
		_insertionOrder.push_back(adjName);
	}
}

void AdjointTable::SetArraySize(const std::string &adjName, size_t arraySize) {
	_arraySizes[adjName] = arraySize;
}

size_t AdjointTable::GetArraySize(const std::string &adjName) const {
	auto it = _arraySizes.find(adjName);
	return it != _arraySizes.end() ? it->second : 0;
}

void AdjointTable::Clear() {
	_map.clear();
	_baseMap.clear();
	_types.clear();
	_arraySizes.clear();
	_insertionOrder.clear();
}

std::string AdjointTable::MakeAdjointName(const std::string &varName) {
	auto sanitizeIdentifierPart = [](const std::string &name) {
		std::string sanitized;
		sanitized.reserve(name.size());
		for (unsigned char c : name) {
			char next = '_';
			if (std::isalnum(c) || c == '_') {
				next = static_cast<char>(c);
			}
			if (next == '_' && (sanitized.empty() || sanitized.back() == '_')) {
				continue;
			}
			sanitized += next;
		}
		while (!sanitized.empty() && sanitized.front() == '_') {
			sanitized.erase(sanitized.begin());
		}
		if (sanitized.empty()) {
			return std::string("v");
		}
		if (std::isdigit(static_cast<unsigned char>(sanitized.front()))) {
			sanitized.insert(0, "v_");
		}
		return sanitized;
	};

	// Buffer element access "buf2[0]" or "buf2[expr]" -> "grad_buf2"
	// All accesses to the same buffer share a single adjoint array.
	// Index expressions are preserved in the generated GLSL code by
	// reconstructing "grad_buf2[index]" from the base name and index.
	if (auto bracketPos = varName.find('['); bracketPos != std::string::npos) {
		return "grad_" + sanitizeIdentifierPart(varName.substr(0, bracketPos));
	}

	return std::format("d_{}", sanitizeIdentifierPart(varName));
}

} // namespace GPU::AD
