/**
 * Builder.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/11/2026
 */

#include <IR/Builder/Builder.h>

#include <sstream>
#include <format>

namespace GPU::IR::Builder {
    Builder &Builder::Get() {
        thread_local static Builder builder;

        return builder;
    }

    void Builder::Bind(BuilderContext &Context) {
        _context = &Context;
    }

    void Builder::Unbind() {
        _context = nullptr;
    }

    BuilderContext *Builder::Context() {
        return _context;
    }

    void Builder::Build(const Node::Node &Node, bool IsStatement) {
        if (_context != nullptr) {
            if (IsStatement) {
                _context->PushTranslatedCode(std::format("{};\n", BuildNode(Node)));
            } else {
                _context->PushTranslatedCode(BuildNode(Node));
            }
        }
    }

    std::string Builder::BuildNode(const Node::Node &Node) {
        switch (Node.Type()) {
            case Node::NodeType::CallInst: {
                return BuildCallInst(dynamic_cast<const Node::IntrinsicCallNode &>(Node));
            }
            case Node::NodeType::Operation: {
                return BuildOperation(dynamic_cast<const Node::OperationNode &>(Node));
            }
            case Node::NodeType::LocalVariable: {
                return BuildLocalVariable(dynamic_cast<const Node::LocalVariableNode &>(Node));
            }
            case Node::NodeType::Load: {
                return BuildLoad(dynamic_cast<const Node::LoadNode &>(Node));
            }
            case Node::NodeType::Store: {
                return BuildStore(dynamic_cast<const Node::StoreNode &>(Node));
            }
            case Node::NodeType::LocalArray: {
                return BuildLocalVariableArray(dynamic_cast<const Node::LocalVariableArrayNode &>(Node));
            }
            case Node::NodeType::ArrayAccess: {
                return BuildArrayAccess(dynamic_cast<const Node::ArrayAccessNode &>(Node));
            }
            case Node::NodeType::CompoundAssignment: {
                return BuildCompoundAssignment(dynamic_cast<const Node::CompoundAssignmentNode &>(Node));
            }
            case Node::NodeType::Increment: {
                return BuildIncrement(dynamic_cast<const Node::IncrementNode &>(Node));
            }
            case Node::NodeType::MemberAccess: {
                return BuildMemberAccess(dynamic_cast<const Node::MemberAccessNode &>(Node));
            }
            case Node::NodeType::If: {
                return BuildIf(dynamic_cast<const Node::IfNode &>(Node));
            }
            case Node::NodeType::While: {
                return BuildWhile(dynamic_cast<const Node::WhileNode &>(Node));
            }
            case Node::NodeType::DoWhile: {
                return BuildDoWhile(dynamic_cast<const Node::DoWhileNode &>(Node));
            }
            case Node::NodeType::For: {
                return BuildFor(dynamic_cast<const Node::ForNode &>(Node));
            }
            case Node::NodeType::Break: {
                return BuildBreak(dynamic_cast<const Node::BreakNode &>(Node));
            }
            case Node::NodeType::Continue: {
                return BuildContinue(dynamic_cast<const Node::ContinueNode &>(Node));
            }
            case Node::NodeType::Return: {
                return BuildReturn(dynamic_cast<const Node::ReturnNode &>(Node));
            }
            case Node::NodeType::Call: {
                return BuildCall(dynamic_cast<const Node::CallNode &>(Node));
            }
            case Node::NodeType::RawCode: {
                return BuildRawCode(dynamic_cast<const Node::RawCodeNode &>(Node));
            }
            default: {
                return "";
            }
        }
    }

    std::string Builder::BuildCallInst(const Node::IntrinsicCallNode &Node) {
        std::ostringstream stream;

        stream << Node.Name() << "(";
        stream << BuildNode(*Node.Parameter()[0]);
        for (size_t index = 1; index < Node.Parameter().size(); ++index) {
            stream << "," << BuildNode(*Node.Parameter()[index]);
        }
        stream << ")";

        return stream.str();
    }

    std::string Builder::BuildOperation(const Node::OperationNode &Node) {
        switch (Node.Code()) {
            case Node::OperationCode::Add: {
                return std::format("({})+({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::Sub: {
                return std::format("({})-({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::Mul: {
                return std::format("({})*({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::Div: {
                return std::format("({})/({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::Mod: {
                return std::format("({})%({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::Neg: {
                return std::format("-({})", BuildNode(*Node.LHS()));
            }
            case Node::OperationCode::BitAnd: {
                return std::format("({})&({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::BitOr: {
                return std::format("({})|({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::BitXor: {
                return std::format("({})^({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::BitNot: {
                return std::format("~({})", BuildNode(*Node.LHS()));
            }
            case Node::OperationCode::Shl: {
                return std::format("({})<<({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::Shr: {
                return std::format("({})>>({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::Less: {
                return std::format("({})<({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::Greater: {
                return std::format("({})>({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::Equal: {
                return std::format("({})==({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::NotEqual: {
                return std::format("({})!=({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::LessEqual: {
                return std::format("({})<=({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::GreaterEqual: {
                return std::format("({})>=({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::LogicalAnd: {
                return std::format("({})&&({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::LogicalOr: {
                return std::format("({})||({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::OperationCode::LogicalNot: {
                return std::format("!({})", BuildNode(*Node.LHS()));
            }
            default: {
                return "";
            }
        }
    }

    std::string Builder::BuildLocalVariable(const Node::LocalVariableNode &Node) {
        // External variables (e.g., uniforms) are declared outside main(),
        // so we don't need to declare them in the main function body
        if (Node.IsExternal()) {
            return "";
        }
        return std::format("{} {}", Node.VarType(), Node.VarName());
    }

    std::string Builder::BuildLocalVariableArray(const Node::LocalVariableArrayNode &Node) {
        return std::format("{} {}[{}]", Node.VarType(), Node.VarName(), Node.Size());
    }

    std::string Builder::BuildLoad(const Node::LoadNode &Node) {
        return Node.Unwarp();
    }

    std::string Builder::BuildStore(const Node::StoreNode &Node) {
        return std::format("({})=({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
    }

    std::string Builder::BuildArrayAccess(const Node::ArrayAccessNode &Node) {
        return std::format("({})[{}]", BuildNode(*Node.Target()), BuildNode(*Node.Index()));
    }

    std::string Builder::BuildCompoundAssignment(const Node::CompoundAssignmentNode &Node) {
        switch (Node.Code()) {
            case Node::CompoundAssignmentCode::AddAssign: {
                return std::format("({}) += ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::CompoundAssignmentCode::SubAssign: {
                return std::format("({}) -= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::CompoundAssignmentCode::MulAssign: {
                return std::format("({}) *= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::CompoundAssignmentCode::DivAssign: {
                return std::format("({}) /= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::CompoundAssignmentCode::ModAssign: {
                return std::format("({}) %= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::CompoundAssignmentCode::BitAndAssign: {
                return std::format("({}) &= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::CompoundAssignmentCode::BitOrAssign: {
                return std::format("({}) |= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::CompoundAssignmentCode::BitXorAssign: {
                return std::format("({}) ^= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::CompoundAssignmentCode::ShlAssign: {
                return std::format("({}) <<= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            case Node::CompoundAssignmentCode::ShrAssign: {
                return std::format("({}) >>= ({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
            }
            default: {
                return "";
            }
        }
    }

    std::string Builder::BuildIncrement(const Node::IncrementNode &Node) {
        switch (Node.Direction()) {
            case Node::IncrementDirection::Decrement: {
                if (Node.IsPrefix()) {
                    return std::format("--({})", BuildNode(*Node.Target()));
                } else {
                    return std::format("({})--", BuildNode(*Node.Target()));
                }
            }
            case Node::IncrementDirection::Increment: {
                if (Node.IsPrefix()) {
                    return std::format("++({})", BuildNode(*Node.Target()));
                } else {
                    return std::format("({})++", BuildNode(*Node.Target()));
                }
            }
            default: {
                return "";
            }
        }
    }

    std::string Builder::BuildMemberAccess(const Node::MemberAccessNode &Node) {
        return std::format("({}).({})", BuildNode(*Node.LHS()), BuildNode(*Node.RHS()));
    }

    std::string Builder::BuildIf(const Node::IfNode &Node) {
        std::string code = std::format("if ({}) {{\n", BuildNode(*Node.Condition()));
        for (auto &node : Node.Do()) {
            code.append(BuildNode(*node));
            code.append("\n");
        }
        code.append("}");

        // Build elif branches
        for (const auto &[elifCond, elifBody] : Node.Elifs()) {
            code.append(std::format(" else if ({}) {{\n", BuildNode(*elifCond)));
            for (auto &node : elifBody) {
                code.append(BuildNode(*node));
                code.append("\n");
            }
            code.append("}");
        }

        // Build else branch
        if (!Node.Else().empty()) {
            code.append(" else {\n");
            for (auto &node : Node.Else()) {
                code.append(BuildNode(*node));
                code.append("\n");
            }
            code.append("}");
        }

        return code;
    }

    std::string Builder::BuildWhile(const Node::WhileNode &Node) {
        std::string code = std::format("while ({}) {{\n", BuildNode(*Node.Condition()));
        for (auto &node : Node.Body()) {
            code.append(BuildNode(*node));
            code.append("\n");
        }
        code.append("}");
        return code;
    }

    std::string Builder::BuildDoWhile(const Node::DoWhileNode &Node) {
        std::string code = "do {\n";
        for (auto &node : Node.Body()) {
            code.append(BuildNode(*node));
            code.append("\n");
        }
        code.append(std::format("}} while ({});", BuildNode(*Node.Condition())));
        return code;
    }

    std::string Builder::BuildFor(const Node::ForNode &Node) {
        std::string code = std::format("for (int {} = {}; {} < {}; {} += {}) {{\n",
                                       Node.VarName(), Node.Start(),
                                       Node.VarName(), Node.End(),
                                       Node.VarName(), Node.Step());
        for (auto &node : Node.Body()) {
            code.append(BuildNode(*node));
            code.append("\n");
        }
        code.append("}");
        return code;
    }

    std::string Builder::BuildBreak(const Node::BreakNode &Node) {
        return "break";
    }

    std::string Builder::BuildContinue(const Node::ContinueNode &Node) {
        return "continue";
    }

    std::string Builder::BuildReturn(const Node::ReturnNode &Node) {
        if (Node.HasValue()) {
            return std::format("return {}", BuildNode(*Node.Value()));
        }
        return "return";
    }

    std::string Builder::BuildCall(const Node::CallNode &Node) {
        std::ostringstream stream;
        stream << Node.FuncName() << "(";
        const auto &args = Node.Arguments();
        if (!args.empty()) {
            stream << BuildNode(*args[0]);
            for (size_t i = 1; i < args.size(); ++i) {
                stream << ", " << BuildNode(*args[i]);
            }
        }
        stream << ")";
        return stream.str();
    }

    std::string Builder::BuildRawCode(const Node::RawCodeNode &Node) {
        return Node.Code();
    }
}
