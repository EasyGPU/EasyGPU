#pragma once

/**
 * @file Barrier.h
 * @brief IR node for shader synchronization barriers.
 */

#ifndef EASYGPU_BARRIER_H
#define EASYGPU_BARRIER_H

#include <IR/Node/Node.h>

namespace GPU::IR::Node {

enum class BarrierCode {
	Workgroup,
	Memory,
	Full,
};

/**
 * Shader synchronization barrier statement.
 */
class BarrierNode : public Node {
public:
	explicit BarrierNode(BarrierCode code);

	[[nodiscard]] NodeType Type() const override;
	[[nodiscard]] BarrierCode Code() const;
	[[nodiscard]] std::unique_ptr<Node> Clone() const override;

private:
	BarrierCode _code;
};

} // namespace GPU::IR::Node

#endif // EASYGPU_BARRIER_H
