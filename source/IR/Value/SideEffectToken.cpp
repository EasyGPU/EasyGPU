/**
 * SideEffectToken.cpp:
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/6/2026
 */
#include <IR/Value/SideEffectToken.h>

namespace GPU::IR::Value {
SideEffectToken::SideEffectToken(std::unique_ptr<Node::Node> node) : _node(std::move(node)), _dismissed(false) {
}

SideEffectToken::~SideEffectToken() {
	// If not dismissed and we have a node, commit the side effect
	if (!_dismissed && _node) {
		// Check if builder has a valid context before building
		auto *context = Builder::Builder::Get().Context();
		if (context) {
			// Build the node as a statement
			Builder::Builder::Get().Build(*_node, true);
		}
	}
}

SideEffectToken::SideEffectToken(SideEffectToken &&other) noexcept
	: _node(std::move(other._node)), _dismissed(other._dismissed) {
	// Mark other as dismissed to prevent double-commit
	other._dismissed = true;
}

SideEffectToken &SideEffectToken::operator=(SideEffectToken &&other) noexcept {
	if (this != &other) {
		// Commit our current node if not dismissed
		if (!_dismissed && _node) {
			Builder::Builder::Get().Build(*_node, true);
		}

		_node			 = std::move(other._node);
		_dismissed		 = other._dismissed;
		other._dismissed = true;
	}
	return *this;
}

void SideEffectToken::dismiss() const {
	_dismissed = true;
}

SideEffectToken::operator bool() const {
	dismiss();
	return true;
}
} // namespace GPU::IR::Value
