/**
 * VarStruct.h:
 *      @Descripiton    :   Var<StructType> specialization support
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_VAR_STRUCT_H
#define EASYGPU_VAR_STRUCT_H

#include <Utility/Meta/StructMeta.h>
#include <IR/Value/Var.h>
#include <IR/Value/Expr.h>
#include <IR/Node/LocalVariable.h>
#include <IR/Node/LoadLocalVariable.h>
#include <IR/Node/Store.h>
#include <IR/Builder/Builder.h>
#include <format>

namespace GPU::IR::Value {
    /**
     * Base class for struct-based Var specializations
     */
    class VarStructBase : public Value {
    protected:
        Node::LocalVariableNode* _varNode = nullptr;
        
        std::unique_ptr<Node::LoadLocalVariableNode> Load() const {
            return std::make_unique<Node::LoadLocalVariableNode>(_varNode->VarName());
        }
    };
}

/************************************************
 * Internal macros for EASYGPU_VAR_SPEC
 ************************************************/

// Member access function from pair
#define EASYGPU_MEMBER_ACCESS(type, name) \
    [[nodiscard]] GPU::IR::Value::Var<type> name() { \
        return GPU::IR::Value::Var<type>(std::format("{}.{}", _varNode->VarName(), #name)); \
    }

// Register a type if it's a struct (generic version for non-structs)
template<typename T>
void EASYGPU_RegisterIfStruct() {}

/************************************************
 * Main EASYGPU_VAR_SPEC macro - Manual unroll for reliable expansion
 ************************************************/

#define EASYGPU_VAR_SPEC_1(StructType, P1) \
    template<> inline void EASYGPU_RegisterIfStruct<StructType>() { \
        auto& ctx = *GPU::IR::Builder::Builder::Get().Context(); \
        std::string typeName(GPU::Meta::StructMeta<StructType>::glslTypeName); \
        if (!ctx.HasStructDefinition(typeName)) { \
            ctx.AddStructDefinition(typeName, GPU::Meta::StructMeta<StructType>::GetGLSLDefinition()); \
        } \
    } \
    namespace GPU::IR::Value { \
        template<> class Var<StructType> : public VarStructBase { \
        public: Var() { EASYGPU_RegisterDep_##StructType(); auto name = Builder::Builder::Get().Context()->AssignVarName(); \
            _node = std::make_unique<Node::LocalVariableNode>(name, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); \
            _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); Builder::Builder::Get().Build(*_varNode, true); } \
        explicit Var(const std::string& varName) { EASYGPU_RegisterDep_##StructType(); _node = std::make_unique<Node::LocalVariableNode>(varName, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); } \
        EASYGPU_MEMBER_ACCESS P1 \
        Var& operator=(const Var& other) { if (this != &other) { auto store = std::make_unique<Node::StoreNode>(Load(), other.Load()); Builder::Builder::Get().Build(*store, true); } return *this; } \
        operator Expr<StructType>() { return Expr<StructType>(Load()); } \\
        }; } \
    template<> inline void EASYGPU_RegisterDep_##StructType() {}

#define EASYGPU_VAR_SPEC_2(StructType, P1, P2) \
    template<> inline void EASYGPU_RegisterIfStruct<StructType>() { \
        auto& ctx = *GPU::IR::Builder::Builder::Get().Context(); \
        std::string typeName(GPU::Meta::StructMeta<StructType>::glslTypeName); \
        if (!ctx.HasStructDefinition(typeName)) { \
            ctx.AddStructDefinition(typeName, GPU::Meta::StructMeta<StructType>::GetGLSLDefinition()); \
        } \
    } \
    namespace GPU::IR::Value { \
        template<> class Var<StructType> : public VarStructBase { \
        public: Var() { EASYGPU_RegisterDep_##StructType(); auto name = Builder::Builder::Get().Context()->AssignVarName(); \
            _node = std::make_unique<Node::LocalVariableNode>(name, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); \
            _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); Builder::Builder::Get().Build(*_varNode, true); } \
        explicit Var(const std::string& varName) { EASYGPU_RegisterDep_##StructType(); _node = std::make_unique<Node::LocalVariableNode>(varName, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); } \
        EASYGPU_MEMBER_ACCESS P1 EASYGPU_MEMBER_ACCESS P2 \
        Var& operator=(const Var& other) { if (this != &other) { auto store = std::make_unique<Node::StoreNode>(Load(), other.Load()); Builder::Builder::Get().Build(*store, true); } return *this; } \
        operator Expr<StructType>() { return Expr<StructType>(Load()); } \\
        }; } \
    template<> inline void EASYGPU_RegisterDep_##StructType() {}

#define EASYGPU_VAR_SPEC_3(StructType, P1, P2, P3) \
    template<> inline void EASYGPU_RegisterIfStruct<StructType>() { \
        auto& ctx = *GPU::IR::Builder::Builder::Get().Context(); \
        std::string typeName(GPU::Meta::StructMeta<StructType>::glslTypeName); \
        if (!ctx.HasStructDefinition(typeName)) { \
            ctx.AddStructDefinition(typeName, GPU::Meta::StructMeta<StructType>::GetGLSLDefinition()); \
        } \
    } \
    namespace GPU::IR::Value { \
        template<> class Var<StructType> : public VarStructBase { \
        public: Var() { EASYGPU_RegisterDep_##StructType(); auto name = Builder::Builder::Get().Context()->AssignVarName(); \
            _node = std::make_unique<Node::LocalVariableNode>(name, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); \
            _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); Builder::Builder::Get().Build(*_varNode, true); } \
        explicit Var(const std::string& varName) { EASYGPU_RegisterDep_##StructType(); _node = std::make_unique<Node::LocalVariableNode>(varName, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); } \
        EASYGPU_MEMBER_ACCESS P1 EASYGPU_MEMBER_ACCESS P2 EASYGPU_MEMBER_ACCESS P3 \
        Var& operator=(const Var& other) { if (this != &other) { auto store = std::make_unique<Node::StoreNode>(Load(), other.Load()); Builder::Builder::Get().Build(*store, true); } return *this; } \
        operator Expr<StructType>() { return Expr<StructType>(Load()); } \\
        }; } \
    template<> inline void EASYGPU_RegisterDep_##StructType() {}

#define EASYGPU_VAR_SPEC_4(StructType, P1, P2, P3, P4) \
    template<> inline void EASYGPU_RegisterIfStruct<StructType>() { \
        auto& ctx = *GPU::IR::Builder::Builder::Get().Context(); \
        std::string typeName(GPU::Meta::StructMeta<StructType>::glslTypeName); \
        if (!ctx.HasStructDefinition(typeName)) { \
            ctx.AddStructDefinition(typeName, GPU::Meta::StructMeta<StructType>::GetGLSLDefinition()); \
        } \
    } \
    namespace GPU::IR::Value { \
        template<> class Var<StructType> : public VarStructBase { \
        public: Var() { EASYGPU_RegisterDep_##StructType(); auto name = Builder::Builder::Get().Context()->AssignVarName(); \
            _node = std::make_unique<Node::LocalVariableNode>(name, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); \
            _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); Builder::Builder::Get().Build(*_varNode, true); } \
        explicit Var(const std::string& varName) { EASYGPU_RegisterDep_##StructType(); _node = std::make_unique<Node::LocalVariableNode>(varName, std::string(GPU::Meta::StructMeta<StructType>::glslTypeName)); _varNode = dynamic_cast<Node::LocalVariableNode*>(_node.get()); } \
        EASYGPU_MEMBER_ACCESS P1 EASYGPU_MEMBER_ACCESS P2 EASYGPU_MEMBER_ACCESS P3 EASYGPU_MEMBER_ACCESS P4 \
        Var& operator=(const Var& other) { if (this != &other) { auto store = std::make_unique<Node::StoreNode>(Load(), other.Load()); Builder::Builder::Get().Build(*store, true); } return *this; } \
        operator Expr<StructType>() { return Expr<StructType>(Load()); } \
        }; } \
    template<> inline void EASYGPU_RegisterDep_##StructType() {}

#define EASYGPU_CAT(a, b) a ## b

// Indirection to force expansion before concatenation  
#define EASYGPU_VAR_SPEC_IMPL(StructType, N, ...) EASYGPU_CAT(EASYGPU_VAR_SPEC_, N)(StructType, __VA_ARGS__)
#define EASYGPU_VAR_SPEC(StructType, ...) EASYGPU_VAR_SPEC_IMPL(StructType, EASYGPU_COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

#endif // EASYGPU_VAR_STRUCT_H
