/**
 * Matrix.h
 *      Matrix types for GPU Math (non-templated explicit sizes)
 *      @author Margoo
 *      @date 2026-02-12
 */
#ifndef EASYGPU_MATRIX_H
#define EASYGPU_MATRIX_H

#include <Utility/Vec.h>

// Forward declarations for rectangular matrix types used in signatures
namespace GPU::Math {
    class Mat2x3;
    class Mat3x2;
    class Mat2x4;
    class Mat4x2;
    class Mat3x4;
    class Mat4x3;

    /**
     * Mat2: 2x2 matrix (2 rows x 2 cols) stored column-major.
     */
    class Mat2 {
    public:
        /**
         * Default constructs to identity matrix.
         * @return Mat2 identity
         */
        Mat2();

        /**
         * Construct from 4 components in column-major order.
         * @param m00 element row0,col0
         * @param m10 element row1,col0
         * @param m01 element row0,col1
         * @param m11 element row1,col1
         * @return Constructed Mat2
         */
        Mat2(float m00, float m10, float m01, float m11);

    public:
        /**
         * Return identity matrix.
         * @return Mat2 identity
         */
        static Mat2 Identity();

        /**
         * Return zero matrix.
         * @return Mat2 zero
         */
        static Mat2 Zero();

        /**
         * Transpose. Complexity: O(1).
         * @return Transposed matrix */
        [[nodiscard]] Mat2 Transposed() const;

        /**
         * Determinant. Complexity: O(1).
         *@return Determinant
         */
        [[nodiscard]] float Determinant() const;

        /**
         * Inverse if invertible. Throws std::runtime_error on singular.
         */
        [[nodiscard]] Mat2 Inverse() const;

        /** Multiply by `Vec2`. Complexity: O(1). */
        Vec2 operator*(const Vec2 &V) const;

    public: // Operators
        Mat2 operator+(const Mat2 &Rhs) const;

        Mat2 operator-(const Mat2 &Rhs) const;

        Mat2 operator*(float S) const;

        Mat2 operator/(float S) const;

        Mat2 &operator+=(const Mat2 &Rhs);

        Mat2 &operator-=(const Mat2 &Rhs);

        Mat2 &operator*=(float S);

        Mat2 &operator/=(float S);

    public:
        // Column-major: col0 = (m00,m10), col1 = (m01,m11)
        float m00 = 1.0f;
        float m10 = 0.0f;
        float m01 = 0.0f;
        float m11 = 1.0f;
    };

    /**
     * Mat3: 3x3 matrix
     */
    class Mat3 {
    public:
        Mat3();

        Mat3(float m00, float m10, float m20, float m01, float m11, float m21, float m02, float m12, float m22);

    public:
        static Mat3 Identity();

        static Mat3 Zero();

        /**
         * Transpose. Complexity: O(1).
         * @return Transposed matrix (3x3)
         */
        [[nodiscard]] Mat3 Transposed() const;

        [[nodiscard]] float Determinant() const;

        [[nodiscard]] Mat3 Inverse() const; // may throw
        [[nodiscard]] Vec3 operator*(const Vec3 &V) const;

    public:
        Mat3 operator+(const Mat3 &Rhs) const;

        Mat3 operator-(const Mat3 &Rhs) const;

        Mat3 operator*(float S) const;

        Mat3 operator/(float S) const;

        Mat3 &operator+=(const Mat3 &Rhs);

        Mat3 &operator-=(const Mat3 &Rhs);

        Mat3 &operator*=(float S);

        Mat3 &operator/=(float S);

    public:
        // column-major: c0=(m00,m10,m20), c1=(m01,m11,m21), c2=(m02,m12,m22)
        float m00 = 1.0f, m10 = 0.0f, m20 = 0.0f;
        float m01 = 0.0f, m11 = 1.0f, m21 = 0.0f;
        float m02 = 0.0f, m12 = 0.0f, m22 = 1.0f;
    };

    /**
     * Mat4: 4x4 matrix
     */
    class Mat4 {
    public:
        Mat4();

        Mat4(float m00, float m10, float m20, float m30,
             float m01, float m11, float m21, float m31,
             float m02, float m12, float m22, float m32,
             float m03, float m13, float m23, float m33);

    public:
        static Mat4 Identity();

        static Mat4 Zero();

        [[nodiscard]] Mat4 Transposed() const;

        [[nodiscard]] float Determinant() const;

        [[nodiscard]] Mat4 Inverse() const; // may throw

        Vec4 operator*(const Vec4 &V) const;

    public:
        Mat4 operator+(const Mat4 &Rhs) const;

        Mat4 operator-(const Mat4 &Rhs) const;

        Mat4 operator*(float S) const;

        Mat4 operator/(float S) const;

        Mat4 &operator+=(const Mat4 &Rhs);

        Mat4 &operator-=(const Mat4 &Rhs);

        Mat4 &operator*=(float S);

        Mat4 &operator/=(float S);

    public:
        // column-major layout: elements mrc where r=row c=col
        float m00 = 1.0f, m10 = 0.0f, m20 = 0.0f, m30 = 0.0f;
        float m01 = 0.0f, m11 = 1.0f, m21 = 0.0f, m31 = 0.0f;
        float m02 = 0.0f, m12 = 0.0f, m22 = 1.0f, m32 = 0.0f;
        float m03 = 0.0f, m13 = 0.0f, m23 = 0.0f, m33 = 1.0f;
    };

    class Mat3x2;

    // Rectangular matrices (columns x rows as requested):
    // Mat2x3: 2 columns x 3 rows -> multiplies Vec2 -> Vec3
    class Mat2x3 {
    public:
        Mat2x3();

        Mat2x3(const Vec3 &col0, const Vec3 &col1);

    public:
        static Mat2x3 Zero();

        [[nodiscard]] Mat3x2 Transposed() const;

        Vec3 operator*(const Vec2 &V) const;

    public:
        Mat2x3 operator+(const Mat2x3 &Rhs) const;

        Mat2x3 operator-(const Mat2x3 &Rhs) const;

        Mat2x3 operator*(float S) const;

        Mat2x3 &operator+=(const Mat2x3 &Rhs);

        Mat2x3 &operator-=(const Mat2x3 &Rhs);

    public:
        Vec3 c0;
        Vec3 c1;
    };

    class Mat3x2 {
    public:
        Mat3x2();

        Mat3x2(const Vec2 &c0, const Vec2 &c1, const Vec2 &c2);

    public:
        static Mat3x2 Zero();

        /**
         * Transpose.
         * @return Mat2x3 (transpose of 3x2 is 2x3)
         */
        [[nodiscard]] Mat2x3 Transposed() const;

        Vec2 operator*(const Vec3 &V) const;

    public:
        Mat3x2 operator+(const Mat3x2 &Rhs) const;

        Mat3x2 operator-(const Mat3x2 &Rhs) const;

        Mat3x2 operator*(float S) const;

        Mat3x2 &operator+=(const Mat3x2 &Rhs);

        Mat3x2 &operator-=(const Mat3x2 &Rhs);

    public:
        Vec2 c0;
        Vec2 c1;
        Vec2 c2;
    };

    class Mat2x4 {
    public:
        Mat2x4();

        Mat2x4(const Vec4 &c0, const Vec4 &c1);

    public:
        static Mat2x4 Zero();

        Vec4 operator*(const Vec2 &V) const;

        [[nodiscard]] Mat4x2 Transposed() const;

    public:
        Vec4 c0;
        Vec4 c1;
    };

    class Mat4x2 {
    public:
        Mat4x2();

        Mat4x2(const Vec2 &c0, const Vec2 &c1, const Vec2 &c2, const Vec2 &c3);

    public:
        static Mat4x2 Zero();

        Vec2 operator*(const Vec4 &V) const;

        [[nodiscard]] Mat2x4 Transposed() const;

    public:
        Vec2 c0;
        Vec2 c1;
        Vec2 c2;
        Vec2 c3;
    };

    class Mat3x4 {
    public:
        Mat3x4();

        Mat3x4(const Vec4 &c0, const Vec4 &c1, const Vec4 &c2);

    public:
        static Mat3x4 Zero();

        Vec4 operator*(const Vec3 &V) const;

        [[nodiscard]] Mat4x3 Transposed() const;

    public:
        Vec4 c0;
        Vec4 c1;
        Vec4 c2;
    };

    class Mat4x3 {
    public:
        Mat4x3();

        Mat4x3(const Vec3 &c0, const Vec3 &c1, const Vec3 &c2, const Vec3 &c3);

    public:
        static Mat4x3 Zero();

        Vec3 operator*(const Vec4 &V) const;

        [[nodiscard]] Mat3x4 Transposed() const;

    public:
        Vec3 c0;
        Vec3 c1;
        Vec3 c2;
        Vec3 c3;
    };
} // namespace GPU::Math


#endif //EASYGPU_MATRIX_H
