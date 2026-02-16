/**
 * ExprMatrix.h:
 *      @Descripiton    :   The specified expression API for matrix with column access
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   2/13/2026
 */
#ifndef EASYGPU_EXPRMATRIX_H
#define EASYGPU_EXPRMATRIX_H

#include <IR/Value/Expr.h>
#include <IR/Builder/Builder.h>

#include <Utility/Matrix.h>

#include <format>

namespace GPU::IR::Value {
    // Specialization for Mat2 expressions with column access
    template<>
    class Expr<Math::Mat2> : public ExprBase {
    public:
        using ValueType = Math::Mat2;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        ~Expr() = default;

        // Column access via subscript
        template<CountableType IndexType>
        Expr<Math::Vec2> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<Math::Vec2>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<Math::Vec2> operator[](IndexType index) & = delete;

        Expr<Math::Vec2> operator[](ExprBase index) && {
            return Expr<Math::Vec2>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<Math::Vec2> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<Math::Vec2> operator[](Expr<IndexT> index) && {
            return Expr<Math::Vec2>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<Math::Vec2> operator[](Expr<IndexT> index) & = delete;

    public:
        // Arithmetic operations
        friend Expr<Math::Mat2> operator+(Expr<Math::Mat2> lhs, Expr<Math::Mat2> rhs) {
            return Expr<Math::Mat2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Mat2> operator-(Expr<Math::Mat2> lhs, Expr<Math::Mat2> rhs) {
            return Expr<Math::Mat2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Release()));
        }

        // Matrix-Vector multiplication (broadcast)
        friend Expr<Math::Vec2> operator*(Expr<Math::Mat2> lhs, Expr<Math::Vec2> rhs) {
            return Expr<Math::Vec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }
    };

    // Specialization for Mat3 expressions
    template<>
    class Expr<Math::Mat3> : public ExprBase {
    public:
        using ValueType = Math::Mat3;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        ~Expr() = default;

        // Column access via subscript
        template<CountableType IndexType>
        Expr<Math::Vec3> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<Math::Vec3>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<Math::Vec3> operator[](IndexType index) & = delete;

        Expr<Math::Vec3> operator[](ExprBase index) && {
            return Expr<Math::Vec3>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<Math::Vec3> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<Math::Vec3> operator[](Expr<IndexT> index) && {
            return Expr<Math::Vec3>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<Math::Vec3> operator[](Expr<IndexT> index) & = delete;

    public:
        // Arithmetic operations
        friend Expr<Math::Mat3> operator+(Expr<Math::Mat3> lhs, Expr<Math::Mat3> rhs) {
            return Expr<Math::Mat3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Mat3> operator-(Expr<Math::Mat3> lhs, Expr<Math::Mat3> rhs) {
            return Expr<Math::Mat3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Release()));
        }

        // Scalar operations with Var<float>
        friend Expr<Math::Mat3> operator*(Expr<Math::Mat3> lhs, const Var<float>& rhs) {
            return Expr<Math::Mat3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::Mat3> operator*(const Var<float>& lhs, Expr<Math::Mat3> rhs) {
            return Expr<Math::Mat3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Load(), rhs.Release()));
        }
        friend Expr<Math::Mat3> operator/(Expr<Math::Mat3> lhs, const Var<float>& rhs) {
            return Expr<Math::Mat3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Load()));
        }

        // Matrix-Vector multiplication (broadcast)
        friend Expr<Math::Vec3> operator*(Expr<Math::Mat3> lhs, Expr<Math::Vec3> rhs) {
            return Expr<Math::Vec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }
    };

    // Specialization for Mat4 expressions
    template<>
    class Expr<Math::Mat4> : public ExprBase {
    public:
        using ValueType = Math::Mat4;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        ~Expr() = default;

        // Column access via subscript
        template<CountableType IndexType>
        Expr<Math::Vec4> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<Math::Vec4>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<Math::Vec4> operator[](IndexType index) & = delete;

        Expr<Math::Vec4> operator[](ExprBase index) && {
            return Expr<Math::Vec4>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<Math::Vec4> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<Math::Vec4> operator[](Expr<IndexT> index) && {
            return Expr<Math::Vec4>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<Math::Vec4> operator[](Expr<IndexT> index) & = delete;

    public:
        // Arithmetic operations
        friend Expr<Math::Mat4> operator+(Expr<Math::Mat4> lhs, Expr<Math::Mat4> rhs) {
            return Expr<Math::Mat4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Add, lhs.Release(), rhs.Release()));
        }

        friend Expr<Math::Mat4> operator-(Expr<Math::Mat4> lhs, Expr<Math::Mat4> rhs) {
            return Expr<Math::Mat4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Sub, lhs.Release(), rhs.Release()));
        }

        // Scalar operations with Var<float>
        friend Expr<Math::Mat4> operator*(Expr<Math::Mat4> lhs, const Var<float>& rhs) {
            return Expr<Math::Mat4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Load()));
        }
        friend Expr<Math::Mat4> operator*(const Var<float>& lhs, Expr<Math::Mat4> rhs) {
            return Expr<Math::Mat4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Load(), rhs.Release()));
        }
        friend Expr<Math::Mat4> operator/(Expr<Math::Mat4> lhs, const Var<float>& rhs) {
            return Expr<Math::Mat4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Div, lhs.Release(), rhs.Load()));
        }

        // Matrix-Vector multiplication (broadcast)
        friend Expr<Math::Vec4> operator*(Expr<Math::Mat4> lhs, Expr<Math::Vec4> rhs) {
            return Expr<Math::Vec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }
    };

    // Mat2x3: 2 columns x 3 rows -> multiplies Vec2 -> Vec3
    template<>
    class Expr<Math::Mat2x3> : public ExprBase {
    public:
        using ValueType = Math::Mat2x3;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        ~Expr() = default;

        // Column access via subscript
        template<CountableType IndexType>
        Expr<Math::Vec3> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<Math::Vec3>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<Math::Vec3> operator[](IndexType index) & = delete;

        Expr<Math::Vec3> operator[](ExprBase index) && {
            return Expr<Math::Vec3>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<Math::Vec3> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<Math::Vec3> operator[](Expr<IndexT> index) && {
            return Expr<Math::Vec3>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<Math::Vec3> operator[](Expr<IndexT> index) & = delete;

        // Matrix-Vector multiplication (broadcast) - Mat2x3 * Vec2 -> Vec3
        friend Expr<Math::Vec3> operator*(Expr<Math::Mat2x3> lhs, Expr<Math::Vec2> rhs) {
            return Expr<Math::Vec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }
    };

    // Mat3x2: 3 columns x 2 rows -> multiplies Vec3 -> Vec2
    template<>
    class Expr<Math::Mat3x2> : public ExprBase {
    public:
        using ValueType = Math::Mat3x2;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        ~Expr() = default;

        // Column access via subscript
        template<CountableType IndexType>
        Expr<Math::Vec2> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<Math::Vec2>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<Math::Vec2> operator[](IndexType index) & = delete;

        Expr<Math::Vec2> operator[](ExprBase index) && {
            return Expr<Math::Vec2>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<Math::Vec2> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<Math::Vec2> operator[](Expr<IndexT> index) && {
            return Expr<Math::Vec2>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<Math::Vec2> operator[](Expr<IndexT> index) & = delete;

        // Matrix-Vector multiplication (broadcast) - Mat3x2 * Vec3 -> Vec2
        friend Expr<Math::Vec2> operator*(Expr<Math::Mat3x2> lhs, Expr<Math::Vec3> rhs) {
            return Expr<Math::Vec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }
    };

    // Mat2x4: 2 columns x 4 rows -> multiplies Vec2 -> Vec4
    template<>
    class Expr<Math::Mat2x4> : public ExprBase {
    public:
        using ValueType = Math::Mat2x4;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        ~Expr() = default;

        // Column access via subscript
        template<CountableType IndexType>
        Expr<Math::Vec4> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<Math::Vec4>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<Math::Vec4> operator[](IndexType index) & = delete;

        Expr<Math::Vec4> operator[](ExprBase index) && {
            return Expr<Math::Vec4>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<Math::Vec4> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<Math::Vec4> operator[](Expr<IndexT> index) && {
            return Expr<Math::Vec4>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<Math::Vec4> operator[](Expr<IndexT> index) & = delete;

        // Matrix-Vector multiplication (broadcast) - Mat2x4 * Vec2 -> Vec4
        friend Expr<Math::Vec4> operator*(Expr<Math::Mat2x4> lhs, Expr<Math::Vec2> rhs) {
            return Expr<Math::Vec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }
    };

    // Mat4x2: 4 columns x 2 rows -> multiplies Vec4 -> Vec2
    template<>
    class Expr<Math::Mat4x2> : public ExprBase {
    public:
        using ValueType = Math::Mat4x2;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        ~Expr() = default;

        // Column access via subscript
        template<CountableType IndexType>
        Expr<Math::Vec2> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<Math::Vec2>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<Math::Vec2> operator[](IndexType index) & = delete;

        Expr<Math::Vec2> operator[](ExprBase index) && {
            return Expr<Math::Vec2>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<Math::Vec2> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<Math::Vec2> operator[](Expr<IndexT> index) && {
            return Expr<Math::Vec2>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<Math::Vec2> operator[](Expr<IndexT> index) & = delete;

        // Matrix-Vector multiplication (broadcast) - Mat4x2 * Vec4 -> Vec2
        friend Expr<Math::Vec2> operator*(Expr<Math::Mat4x2> lhs, Expr<Math::Vec4> rhs) {
            return Expr<Math::Vec2>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }
    };

    // Mat3x4: 3 columns x 4 rows -> multiplies Vec3 -> Vec4
    template<>
    class Expr<Math::Mat3x4> : public ExprBase {
    public:
        using ValueType = Math::Mat3x4;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        ~Expr() = default;

        // Column access via subscript
        template<CountableType IndexType>
        Expr<Math::Vec4> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<Math::Vec4>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<Math::Vec4> operator[](IndexType index) & = delete;

        Expr<Math::Vec4> operator[](ExprBase index) && {
            return Expr<Math::Vec4>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<Math::Vec4> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<Math::Vec4> operator[](Expr<IndexT> index) && {
            return Expr<Math::Vec4>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<Math::Vec4> operator[](Expr<IndexT> index) & = delete;

        // Matrix-Vector multiplication (broadcast) - Mat3x4 * Vec3 -> Vec4
        friend Expr<Math::Vec4> operator*(Expr<Math::Mat3x4> lhs, Expr<Math::Vec3> rhs) {
            return Expr<Math::Vec4>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }
    };

    // Mat4x3: 4 columns x 3 rows -> multiplies Vec4 -> Vec3
    template<>
    class Expr<Math::Mat4x3> : public ExprBase {
    public:
        using ValueType = Math::Mat4x3;
        using ElementType_t = float;

        Expr() = default;
        Expr(std::unique_ptr<Node::Node> Node) : ExprBase(std::move(Node)) {}
        Expr(const ExprBase& base) : ExprBase(std::unique_ptr<Node::Node>(const_cast<ExprBase&>(base).Release().release())) {}
        Expr(ExprBase&& base) : ExprBase(base.Release()) {}
        
        ~Expr() = default;

        // Column access via subscript
        template<CountableType IndexType>
        Expr<Math::Vec3> operator[](IndexType index) && {
            auto uniform = std::make_unique<Node::LoadUniformNode>(ValueToString(index));
            return Expr<Math::Vec3>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), std::move(uniform)));
        }
        
        template<CountableType IndexType>
        Expr<Math::Vec3> operator[](IndexType index) & = delete;

        Expr<Math::Vec3> operator[](ExprBase index) && {
            return Expr<Math::Vec3>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        Expr<Math::Vec3> operator[](ExprBase index) & = delete;

        template<ScalarType IndexT>
        Expr<Math::Vec3> operator[](Expr<IndexT> index) && {
            return Expr<Math::Vec3>(
                std::make_unique<Node::ArrayAccessNode>(this->Release(), index.Release()));
        }
        
        template<ScalarType IndexT>
        Expr<Math::Vec3> operator[](Expr<IndexT> index) & = delete;

        // Matrix-Vector multiplication (broadcast) - Mat4x3 * Vec4 -> Vec3
        friend Expr<Math::Vec3> operator*(Expr<Math::Mat4x3> lhs, Expr<Math::Vec4> rhs) {
            return Expr<Math::Vec3>(
                std::make_unique<Node::OperationNode>(
                    Node::OperationCode::Mul, lhs.Release(), rhs.Release()));
        }
    };

    // ==================== Var Matrix Scalar Operations ====================
    // Mat2 * float
    [[nodiscard]] inline Expr<Math::Mat2> operator*(const VarBase<Math::Mat2> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat2> operator*(float lhs, const VarBase<Math::Mat2> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // Mat2 / float
    [[nodiscard]] inline Expr<Math::Mat2> operator/(const VarBase<Math::Mat2> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // Mat3 * float
    [[nodiscard]] inline Expr<Math::Mat3> operator*(const VarBase<Math::Mat3> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat3> operator*(float lhs, const VarBase<Math::Mat3> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // Mat3 / float
    [[nodiscard]] inline Expr<Math::Mat3> operator/(const VarBase<Math::Mat3> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // Mat4 * float
    [[nodiscard]] inline Expr<Math::Mat4> operator*(const VarBase<Math::Mat4> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat4> operator*(float lhs, const VarBase<Math::Mat4> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    // Mat4 / float
    [[nodiscard]] inline Expr<Math::Mat4> operator/(const VarBase<Math::Mat4> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // Rectangular matrices * float
    [[nodiscard]] inline Expr<Math::Mat2x3> operator*(const VarBase<Math::Mat2x3> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat2x3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat2x3> operator*(float lhs, const VarBase<Math::Mat2x3> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat2x3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    [[nodiscard]] inline Expr<Math::Mat2x4> operator*(const VarBase<Math::Mat2x4> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat2x4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat2x4> operator*(float lhs, const VarBase<Math::Mat2x4> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat2x4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    [[nodiscard]] inline Expr<Math::Mat3x2> operator*(const VarBase<Math::Mat3x2> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat3x2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat3x2> operator*(float lhs, const VarBase<Math::Mat3x2> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat3x2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    [[nodiscard]] inline Expr<Math::Mat3x4> operator*(const VarBase<Math::Mat3x4> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat3x4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat3x4> operator*(float lhs, const VarBase<Math::Mat3x4> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat3x4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    [[nodiscard]] inline Expr<Math::Mat4x2> operator*(const VarBase<Math::Mat4x2> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat4x2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat4x2> operator*(float lhs, const VarBase<Math::Mat4x2> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat4x2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    [[nodiscard]] inline Expr<Math::Mat4x3> operator*(const VarBase<Math::Mat4x3> &lhs, float rhs) {
        auto lhsLoad = lhs.Load();
        auto rhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(rhs));
        return Expr<Math::Mat4x3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }
    [[nodiscard]] inline Expr<Math::Mat4x3> operator*(float lhs, const VarBase<Math::Mat4x3> &rhs) {
        auto lhsLoad = std::make_unique<Node::LoadUniformNode>(ValueToString(lhs));
        auto rhsLoad = rhs.Load();
        return Expr<Math::Mat4x3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, std::move(lhsLoad), std::move(rhsLoad)));
    }

    // ==================== Var Matrix with Var<float> Operations ====================
    // Mat2 * Var<float>
    [[nodiscard]] inline Expr<Math::Mat2> operator*(const VarBase<Math::Mat2> &lhs, const Var<float> &rhs) {
        return Expr<Math::Mat2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Mat2> operator*(const Var<float> &lhs, const VarBase<Math::Mat2> &rhs) {
        return Expr<Math::Mat2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), rhs.Load()));
    }
    // Mat2 / Var<float>
    [[nodiscard]] inline Expr<Math::Mat2> operator/(const VarBase<Math::Mat2> &lhs, const Var<float> &rhs) {
        return Expr<Math::Mat2>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, lhs.Load(), rhs.Load()));
    }

    // Mat3 * Var<float>
    [[nodiscard]] inline Expr<Math::Mat3> operator*(const VarBase<Math::Mat3> &lhs, const Var<float> &rhs) {
        return Expr<Math::Mat3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Mat3> operator*(const Var<float> &lhs, const VarBase<Math::Mat3> &rhs) {
        return Expr<Math::Mat3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), rhs.Load()));
    }
    // Mat3 / Var<float>
    [[nodiscard]] inline Expr<Math::Mat3> operator/(const VarBase<Math::Mat3> &lhs, const Var<float> &rhs) {
        return Expr<Math::Mat3>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, lhs.Load(), rhs.Load()));
    }

    // Mat4 * Var<float>
    [[nodiscard]] inline Expr<Math::Mat4> operator*(const VarBase<Math::Mat4> &lhs, const Var<float> &rhs) {
        return Expr<Math::Mat4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), rhs.Load()));
    }
    [[nodiscard]] inline Expr<Math::Mat4> operator*(const Var<float> &lhs, const VarBase<Math::Mat4> &rhs) {
        return Expr<Math::Mat4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Mul, lhs.Load(), rhs.Load()));
    }
    // Mat4 / Var<float>
    [[nodiscard]] inline Expr<Math::Mat4> operator/(const VarBase<Math::Mat4> &lhs, const Var<float> &rhs) {
        return Expr<Math::Mat4>(
            std::make_unique<Node::OperationNode>(Node::OperationCode::Div, lhs.Load(), rhs.Load()));
    }
}

#endif //EASYGPU_EXPRMATRIX_H
