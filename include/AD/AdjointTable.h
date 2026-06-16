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
	bool		Has(const std::string &varName) const;

	/**
	 * Get the GLSL type for an adjoint variable, empty string if unknown.
	 */
	std::string GetTypeForAdjoint(const std::string &adjName) const;

	/**
	 * Get all (adjointName, glslType) pairs for GLSL variable declarations.
	 * Each pair represents: "glslType adjointName = glslType(0);"
	 */
	std::vector<std::pair<std::string, std::string>> AllDeclarations() const;

	/**
	 * Clear all mappings (for reuse).
	 */
	void											 Clear();

	/**
	 * Generate a uniquely-named adjoint variable from a forward variable name.
	 */
	static std::string								 MakeAdjointName(const std::string &varName);

	/**
	 * Set the array size for a buffer-type adjoint.
	 * Buffer adjoints are declared as arrays (float grad_buf[N]) so that
	 * per-element gradient indexing works with both constant and variable indices.
	 */
	void											 SetArraySize(const std::string &adjName, size_t arraySize);

	/**
	 * Get the array size for an adjoint name, or 0 if it's a scalar.
	 */
	size_t											 GetArraySize(const std::string &adjName) const;

private:
	// Forward var name -> adjoint var name
	std::unordered_map<std::string, std::string> _map;
	// Base buffer name -> adjoint name (e.g. "buf5" -> "grad_buf5")
	std::unordered_map<std::string, std::string> _baseMap;
	// Adjoint var name -> GLSL type (element type for arrays)
	std::unordered_map<std::string, std::string> _types;
	// Adjoint var name -> array size (0 = scalar)
	std::unordered_map<std::string, size_t>		 _arraySizes;
	// Insertion order tracking for deterministic declarations
	std::vector<std::string>					 _insertionOrder;
};

} // namespace GPU::AD

#endif // EASYGPU_AD_ADJOINTTABLE_H
