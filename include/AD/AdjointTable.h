#pragma once

/**
 * @file AdjointTable.h
 * @brief Maps forward variable names to their adjoint (gradient) variable names.
 *
 * During backward pass generation, every forward variable that requires a
 * gradient gets a corresponding adjoint accumulator variable. The AdjointTable
 * manages this mapping and ensures each adjoint variable is declared exactly
 * once with zero-initialization.
 */

#ifndef EASYGPU_AD_ADJOINTTABLE_H
#define EASYGPU_AD_ADJOINTTABLE_H

#include <string>
#include <unordered_map>
#include <vector>

namespace GPU::AD {

/**
 * Maintains the mapping from forward variable names to adjoint variable names.
 *
 * Example: forward variable "v5" (float) maps to adjoint "d_v5" (float).
 * The adjoint prefix "d_" is used to distinguish adjoint variables.
 */
class AdjointTable {
public:
	/**
	 * Get or create an adjoint variable for a forward variable.
	 * @param varName  The forward variable name (e.g., "v5")
	 * @param glslType The GLSL type of the forward variable (e.g., "float")
	 * @return The adjoint variable name (e.g., "d_v5")
	 */
	std::string GetOrCreate(const std::string &varName, const std::string &glslType);

	/**
	 * Get the adjoint variable name if it exists, empty string otherwise.
	 */
	std::string Get(const std::string &varName) const;

	/**
	 * Check if an adjoint exists for a forward variable.
	 */
	bool Has(const std::string &varName) const;

	/**
	 * Get all (adjointName, glslType) pairs for GLSL variable declarations.
	 * Each pair represents: "glslType adjointName = glslType(0);"
	 */
	std::vector<std::pair<std::string, std::string>> AllDeclarations() const;

	/**
	 * Clear all mappings (for reuse).
	 */
	void Clear();

	/**
	 * Generate a uniquely-named adjoint variable from a forward variable name.
	 */
	static std::string MakeAdjointName(const std::string &varName);

private:
	// Forward var name -> adjoint var name
	std::unordered_map<std::string, std::string> _map;
	// Adjoint var name -> GLSL type
	std::unordered_map<std::string, std::string> _types;
	// Insertion order tracking for deterministic declarations
	std::vector<std::string> _insertionOrder;
};

} // namespace GPU::AD

#endif // EASYGPU_AD_ADJOINTTABLE_H
