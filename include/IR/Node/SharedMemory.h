#pragma once

/**
 * @file SharedMemory.h
 * @brief The node for shared memory declaration.
 */

#ifndef EASYGPU_SHAREDMEMORY_H
#define EASYGPU_SHAREDMEMORY_H

#include <IR/Node/Node.h>
#include <string>

namespace GPU::IR::Node {
/**
 * The node for shared memory array declaration (workgroup-local memory)
 * Maps to GLSL: shared Type Name[Size];
 */
class SharedMemoryNode : public Node {
public:
	SharedMemoryNode(std::string Name, std::string Type, int Size);

public:
	NodeType Type() const override;

public:
	/**
	 * Getting name of the shared memory array
	 * @return The name of the shared memory array
	 */
	[[nodiscard]] std::string			VarName() const;

	/**
	 * Getting type of the shared memory array
	 * @return The type of the shared memory array
	 */
	[[nodiscard]] std::string			VarType() const;

	/**
	 * Getting the size of the shared memory array
	 * @return The size of the shared memory array
	 */
	[[nodiscard]] int					Size() const;

	[[nodiscard]] std::unique_ptr<Node> Clone() const override;

private:
	std::string _name;
	std::string _type;
	int			_size;
};
} // namespace GPU::IR::Node

#endif // EASYGPU_SHAREDMEMORY_H
