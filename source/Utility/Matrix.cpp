/**
 * Matrix.cpp
 * Implementation for Matrix.h
 */
#include <Utility/Matrix.h>
#include <Utility/Vec.h>

#include <stdexcept>
#include <cmath>

namespace GPU::Math {
    // ---------------- Mat2 ----------------
    Mat2::Mat2() = default;

    Mat2::Mat2(float a00, float a10, float a01, float a11)
        : m00(a00), m10(a10), m01(a01), m11(a11) {
    }

    Mat2 Mat2::Identity() {
        return {1.0f, 0.0f, 0.0f, 1.0f};
    }

    Mat2 Mat2::Zero() {
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }

    Mat2 Mat2::Transposed() const {
        return {m00, m01, m10, m11};
    }

    float Mat2::Determinant() const {
        return m00 * m11 - m01 * m10;
    }

    Mat2 Mat2::Inverse() const {
        const float det = Determinant();
        if (std::fabs(det) < 1e-12f) {
            throw std::runtime_error("Mat2: singular matrix");
        }

        const float inv = 1.0f / det;
        return {m11 * inv, -m10 * inv, -m01 * inv, m00 * inv};
    }

    Vec2 Mat2::operator*(const Vec2 &V) const {
        Vec2 r;
        r.x = m00 * V.x + m01 * V.y;
        r.y = m10 * V.x + m11 * V.y;
        return r;
    }

    Mat2 Mat2::operator+(const Mat2 &Rhs) const {
        return {m00 + Rhs.m00, m10 + Rhs.m10, m01 + Rhs.m01, m11 + Rhs.m11};
    }

    Mat2 Mat2::operator-(const Mat2 &Rhs) const {
        return {m00 - Rhs.m00, m10 - Rhs.m10, m01 - Rhs.m01, m11 - Rhs.m11};
    }

    Mat2 Mat2::operator*(float S) const {
        return {m00 * S, m10 * S, m01 * S, m11 * S};
    }

    Mat2 Mat2::operator/(float S) const {
        return {m00 / S, m10 / S, m01 / S, m11 / S};
    }

    Mat2 &Mat2::operator+=(const Mat2 &Rhs) {
        m00 += Rhs.m00;
        m10 += Rhs.m10;
        m01 += Rhs.m01;
        m11 += Rhs.m11;
        return *this;
    }

    Mat2 &Mat2::operator-=(const Mat2 &Rhs) {
        m00 -= Rhs.m00;
        m10 -= Rhs.m10;
        m01 -= Rhs.m01;
        m11 -= Rhs.m11;
        return *this;
    }

    Mat2 &Mat2::operator*=(float S) {
        m00 *= S;
        m10 *= S;
        m01 *= S;
        m11 *= S;
        return *this;
    }

    Mat2 &Mat2::operator/=(float S) {
        m00 /= S;
        m10 /= S;
        m01 /= S;
        m11 /= S;
        return *this;
    }

    // Mat2 matrix multiplication
    Mat2 Mat2::operator*(const Mat2 &Rhs) const {
        return {
            m00 * Rhs.m00 + m01 * Rhs.m10,
            m10 * Rhs.m00 + m11 * Rhs.m10,
            m00 * Rhs.m01 + m01 * Rhs.m11,
            m10 * Rhs.m01 + m11 * Rhs.m11
        };
    }

    // ---------------- Mat3 ----------------
    Mat3::Mat3() = default;

    Mat3::Mat3(float a00, float a10, float a20, float a01, float a11, float a21, float a02, float a12, float a22)
        : m00(a00), m10(a10), m20(a20), m01(a01), m11(a11), m21(a21), m02(a02), m12(a12), m22(a22) {
    }

    Mat3 Mat3::Identity() {
        return {1, 0, 0, 0, 1, 0, 0, 0, 1};
    }

    Mat3 Mat3::Zero() {
        return {0, 0, 0, 0, 0, 0, 0, 0, 0};
    }

    Mat3 Mat3::Transposed() const {
        return {m00, m01, m02, m10, m11, m12, m20, m21, m22};
    }

    float Mat3::Determinant() const {
        // Compute using rule of Sarrus
        return m00 * (m11 * m22 - m12 * m21)
               - m01 * (m10 * m22 - m12 * m20)
               + m02 * (m10 * m21 - m11 * m20);
    }

    Mat3 Mat3::Inverse() const {
        const float det = Determinant();
        if (std::fabs(det) < 1e-12f) throw std::runtime_error("Mat3: singular matrix");
        const float inv = 1.0f / det;
        Mat3        adj;
        // Cofactors (adjugate = transpose of cofactor matrix)
        adj.m00 = (m11 * m22 - m12 * m21);
        adj.m01 = -(m01 * m22 - m02 * m21);
        adj.m02 = (m01 * m12 - m02 * m11);

        adj.m10 = -(m10 * m22 - m12 * m20);
        adj.m11 = (m00 * m22 - m02 * m20);
        adj.m12 = -(m00 * m12 - m02 * m10);

        adj.m20 = (m10 * m21 - m11 * m20);
        adj.m21 = -(m00 * m21 - m01 * m20);
        adj.m22 = (m00 * m11 - m01 * m10);

        // adj currently is cofactor matrix; need transpose to get adjugate in column-major storage
        Mat3 res = adj.Transposed();
        return res * inv;
    }

    Vec3 Mat3::operator*(const Vec3 &V) const {
        Vec3 r;
        r.x = m00 * V.x + m01 * V.y + m02 * V.z;
        r.y = m10 * V.x + m11 * V.y + m12 * V.z;
        r.z = m20 * V.x + m21 * V.y + m22 * V.z;
        return r;
    }

    Mat3 Mat3::operator+(const Mat3 &Rhs) const {
        return {
            m00 + Rhs.m00, m10 + Rhs.m10, m20 + Rhs.m20, m01 + Rhs.m01, m11 + Rhs.m11, m21 + Rhs.m21,
            m02 + Rhs.m02, m12 + Rhs.m12, m22 + Rhs.m22
        };
    }

    Mat3 Mat3::operator-(const Mat3 &Rhs) const {
        return {
            m00 - Rhs.m00, m10 - Rhs.m10, m20 - Rhs.m20, m01 - Rhs.m01, m11 - Rhs.m11, m21 - Rhs.m21,
            m02 - Rhs.m02, m12 - Rhs.m12, m22 - Rhs.m22
        };
    }

    Mat3 Mat3::operator*(float S) const {
        return {m00 * S, m10 * S, m20 * S, m01 * S, m11 * S, m21 * S, m02 * S, m12 * S, m22 * S};
    }

    Mat3 Mat3::operator/(float S) const {
        return (*this) * (1.0f / S);
    }

    Mat3 &Mat3::operator+=(const Mat3 &Rhs) {
        m00 += Rhs.m00;
        m10 += Rhs.m10;
        m20 += Rhs.m20;
        m01 += Rhs.m01;
        m11 += Rhs.m11;
        m21 += Rhs.m21;
        m02 += Rhs.m02;
        m12 += Rhs.m12;
        m22 += Rhs.m22;
        return *this;
    }

    Mat3 &Mat3::operator-=(const Mat3 &Rhs) {
        m00 -= Rhs.m00;
        m10 -= Rhs.m10;
        m20 -= Rhs.m20;
        m01 -= Rhs.m01;
        m11 -= Rhs.m11;
        m21 -= Rhs.m21;
        m02 -= Rhs.m02;
        m12 -= Rhs.m12;
        m22 -= Rhs.m22;
        return *this;
    }

    Mat3 &Mat3::operator*=(float S) {
        m00 *= S;
        m10 *= S;
        m20 *= S;
        m01 *= S;
        m11 *= S;
        m21 *= S;
        m02 *= S;
        m12 *= S;
        m22 *= S;
        return *this;
    }

    Mat3 &Mat3::operator/=(float S) { return (*this) *= (1.0f / S); }

    // Mat3 matrix multiplication
    Mat3 Mat3::operator*(const Mat3 &Rhs) const {
        return {
            m00 * Rhs.m00 + m01 * Rhs.m10 + m02 * Rhs.m20,
            m10 * Rhs.m00 + m11 * Rhs.m10 + m12 * Rhs.m20,
            m20 * Rhs.m00 + m21 * Rhs.m10 + m22 * Rhs.m20,
            m00 * Rhs.m01 + m01 * Rhs.m11 + m02 * Rhs.m21,
            m10 * Rhs.m01 + m11 * Rhs.m11 + m12 * Rhs.m21,
            m20 * Rhs.m01 + m21 * Rhs.m11 + m22 * Rhs.m21,
            m00 * Rhs.m02 + m01 * Rhs.m12 + m02 * Rhs.m22,
            m10 * Rhs.m02 + m11 * Rhs.m12 + m12 * Rhs.m22,
            m20 * Rhs.m02 + m21 * Rhs.m12 + m22 * Rhs.m22
        };
    }

    // ---------------- Mat4 ----------------
    Mat4::Mat4() = default;

    Mat4::Mat4(float a00, float a10, float a20, float a30,
               float a01, float a11, float a21, float a31,
               float a02, float a12, float a22, float a32,
               float a03, float a13, float a23, float a33)
        : m00(a00), m10(a10), m20(a20), m30(a30), m01(a01), m11(a11), m21(a21), m31(a31), m02(a02), m12(a12), m22(a22),
          m32(a32), m03(a03), m13(a13), m23(a23), m33(a33) {
    }

    Mat4 Mat4::Identity() {
        return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    }

    Mat4 Mat4::Zero() {
        return {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    }

    Mat4 Mat4::Transposed() const {
        return {m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33};
    }

    float Mat4::Determinant() const {
        // Expand along first row for clarity
        float cofactor00 = m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (
                               m21 * m32 - m22 * m31);
        float cofactor01 = -(m10 * (m22 * m33 - m23 * m32) - m12 * (m20 * m33 - m23 * m30) + m13 * (
                                 m20 * m32 - m22 * m30));
        float cofactor02 = m10 * (m21 * m33 - m23 * m31) - m11 * (m20 * m33 - m23 * m30) + m13 * (
                               m20 * m31 - m21 * m30);
        float cofactor03 = -(m10 * (m21 * m32 - m22 * m31) - m11 * (m20 * m32 - m22 * m30) + m12 * (
                                 m20 * m31 - m21 * m30));
        return m00 * cofactor00 + m01 * cofactor01 + m02 * cofactor02 + m03 * cofactor03;
    }

    Mat4 Mat4::Inverse() const {
        // Compute inverse via classical adjugate method (explicit)
        float       inv[16];
        const float a00 = m00, a01 = m01, a02 = m02, a03 = m03;
        const float a10 = m10, a11 = m11, a12 = m12, a13 = m13;
        const float a20 = m20, a21 = m21, a22 = m22, a23 = m23;
        const float a30 = m30, a31 = m31, a32 = m32, a33 = m33;

        inv[0] = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
        inv[1] = -a01 * (a22 * a33 - a23 * a32) + a02 * (a21 * a33 - a23 * a31) - a03 * (a21 * a32 - a22 * a31);
        inv[2] = a01 * (a12 * a33 - a13 * a32) - a02 * (a11 * a33 - a13 * a31) + a03 * (a11 * a32 - a12 * a31);
        inv[3] = -a01 * (a12 * a23 - a13 * a22) + a02 * (a11 * a23 - a13 * a21) - a03 * (a11 * a22 - a12 * a21);

        inv[4] = -a10 * (a22 * a33 - a23 * a32) + a12 * (a20 * a33 - a23 * a30) - a13 * (a20 * a32 - a22 * a30);
        inv[5] = a00 * (a22 * a33 - a23 * a32) - a02 * (a20 * a33 - a23 * a30) + a03 * (a20 * a32 - a22 * a30);
        inv[6] = -a00 * (a12 * a33 - a13 * a32) + a02 * (a10 * a33 - a13 * a30) - a03 * (a10 * a32 - a12 * a30);
        inv[7] = a00 * (a12 * a23 - a13 * a22) - a02 * (a10 * a23 - a13 * a20) + a03 * (a10 * a22 - a12 * a20);

        inv[8]  = a10 * (a21 * a33 - a23 * a31) - a11 * (a20 * a33 - a23 * a30) + a13 * (a20 * a31 - a21 * a30);
        inv[9]  = -a00 * (a21 * a33 - a23 * a31) + a01 * (a20 * a33 - a23 * a30) - a03 * (a20 * a31 - a21 * a30);
        inv[10] = a00 * (a11 * a33 - a13 * a31) - a01 * (a10 * a33 - a13 * a30) + a03 * (a10 * a31 - a11 * a30);
        inv[11] = -a00 * (a11 * a23 - a13 * a21) + a01 * (a10 * a23 - a13 * a20) - a03 * (a10 * a21 - a11 * a20);

        inv[12] = -a10 * (a21 * a32 - a22 * a31) + a11 * (a20 * a32 - a22 * a30) - a12 * (a20 * a31 - a21 * a30);
        inv[13] = a00 * (a21 * a32 - a22 * a31) - a01 * (a20 * a32 - a22 * a30) + a02 * (a20 * a31 - a21 * a30);
        inv[14] = -a00 * (a11 * a32 - a12 * a31) + a01 * (a10 * a32 - a12 * a30) - a02 * (a10 * a31 - a11 * a30);
        inv[15] = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);

        float det = a00 * inv[0] + a01 * inv[4] + a02 * inv[8] + a03 * inv[12];
        if (std::fabs(det) < 1e-12f) throw std::runtime_error("Mat4: singular matrix");
        const float invDet = 1.0f / det;

        Mat4 out;
        out.m00 = inv[0] * invDet;
        out.m01 = inv[1] * invDet;
        out.m02 = inv[2] * invDet;
        out.m03 = inv[3] * invDet;
        out.m10 = inv[4] * invDet;
        out.m11 = inv[5] * invDet;
        out.m12 = inv[6] * invDet;
        out.m13 = inv[7] * invDet;
        out.m20 = inv[8] * invDet;
        out.m21 = inv[9] * invDet;
        out.m22 = inv[10] * invDet;
        out.m23 = inv[11] * invDet;
        out.m30 = inv[12] * invDet;
        out.m31 = inv[13] * invDet;
        out.m32 = inv[14] * invDet;
        out.m33 = inv[15] * invDet;
        return out;
    }

    Vec4 Mat4::operator*(const Vec4 &V) const {
        Vec4 r;
        r.x = m00 * V.x + m01 * V.y + m02 * V.z + m03 * V.w;
        r.y = m10 * V.x + m11 * V.y + m12 * V.z + m13 * V.w;
        r.z = m20 * V.x + m21 * V.y + m22 * V.z + m23 * V.w;
        r.w = m30 * V.x + m31 * V.y + m32 * V.z + m33 * V.w;
        return r;
    }

    Mat4 Mat4::operator+(const Mat4 &Rhs) const {
        return {
            m00 + Rhs.m00, m10 + Rhs.m10, m20 + Rhs.m20, m30 + Rhs.m30, m01 + Rhs.m01, m11 + Rhs.m11,
            m21 + Rhs.m21, m31 + Rhs.m31, m02 + Rhs.m02, m12 + Rhs.m12, m22 + Rhs.m22, m32 + Rhs.m32,
            m03 + Rhs.m03, m13 + Rhs.m13, m23 + Rhs.m23, m33 + Rhs.m33
        };
    }

    Mat4 Mat4::operator-(const Mat4 &Rhs) const {
        return {
            m00 - Rhs.m00, m10 - Rhs.m10, m20 - Rhs.m20, m30 - Rhs.m30, m01 - Rhs.m01, m11 - Rhs.m11,
            m21 - Rhs.m21, m31 - Rhs.m31, m02 - Rhs.m02, m12 - Rhs.m12, m22 - Rhs.m22, m32 - Rhs.m32,
            m03 - Rhs.m03, m13 - Rhs.m13, m23 - Rhs.m23, m33 - Rhs.m33
        };
    }

    Mat4 Mat4::operator*(float S) const {
        return {
            m00 * S, m10 * S, m20 * S, m30 * S, m01 * S, m11 * S, m21 * S, m31 * S, m02 * S, m12 * S, m22 * S,
            m32 * S, m03 * S, m13 * S, m23 * S, m33 * S
        };
    }

    Mat4 Mat4::operator/(float S) const { return (*this) * (1.0f / S); }

    Mat4 &Mat4::operator+=(const Mat4 &Rhs) {
        m00 += Rhs.m00;
        m10 += Rhs.m10;
        m20 += Rhs.m20;
        m30 += Rhs.m30;
        m01 += Rhs.m01;
        m11 += Rhs.m11;
        m21 += Rhs.m21;
        m31 += Rhs.m31;
        m02 += Rhs.m02;
        m12 += Rhs.m12;
        m22 += Rhs.m22;
        m32 += Rhs.m32;
        m03 += Rhs.m03;
        m13 += Rhs.m13;
        m23 += Rhs.m23;
        m33 += Rhs.m33;
        return *this;
    }

    Mat4 &Mat4::operator-=(const Mat4 &Rhs) {
        m00 -= Rhs.m00;
        m10 -= Rhs.m10;
        m20 -= Rhs.m20;
        m30 -= Rhs.m30;
        m01 -= Rhs.m01;
        m11 -= Rhs.m11;
        m21 -= Rhs.m21;
        m31 -= Rhs.m31;
        m02 -= Rhs.m02;
        m12 -= Rhs.m12;
        m22 -= Rhs.m22;
        m32 -= Rhs.m32;
        m03 -= Rhs.m03;
        m13 -= Rhs.m13;
        m23 -= Rhs.m23;
        m33 -= Rhs.m33;
        return *this;
    }

    Mat4 &Mat4::operator*=(float S) {
        m00 *= S;
        m10 *= S;
        m20 *= S;
        m30 *= S;
        m01 *= S;
        m11 *= S;
        m21 *= S;
        m31 *= S;
        m02 *= S;
        m12 *= S;
        m22 *= S;
        m32 *= S;
        m03 *= S;
        m13 *= S;
        m23 *= S;
        m33 *= S;
        return *this;
    }

    Mat4 &Mat4::operator/=(float S) { return (*this) *= (1.0f / S); }

    // Mat4 matrix multiplication
    Mat4 Mat4::operator*(const Mat4 &Rhs) const {
        return {
            m00 * Rhs.m00 + m01 * Rhs.m10 + m02 * Rhs.m20 + m03 * Rhs.m30,
            m10 * Rhs.m00 + m11 * Rhs.m10 + m12 * Rhs.m20 + m13 * Rhs.m30,
            m20 * Rhs.m00 + m21 * Rhs.m10 + m22 * Rhs.m20 + m23 * Rhs.m30,
            m30 * Rhs.m00 + m31 * Rhs.m10 + m32 * Rhs.m20 + m33 * Rhs.m30,
            m00 * Rhs.m01 + m01 * Rhs.m11 + m02 * Rhs.m21 + m03 * Rhs.m31,
            m10 * Rhs.m01 + m11 * Rhs.m11 + m12 * Rhs.m21 + m13 * Rhs.m31,
            m20 * Rhs.m01 + m21 * Rhs.m11 + m22 * Rhs.m21 + m23 * Rhs.m31,
            m30 * Rhs.m01 + m31 * Rhs.m11 + m32 * Rhs.m21 + m33 * Rhs.m31,
            m00 * Rhs.m02 + m01 * Rhs.m12 + m02 * Rhs.m22 + m03 * Rhs.m32,
            m10 * Rhs.m02 + m11 * Rhs.m12 + m12 * Rhs.m22 + m13 * Rhs.m32,
            m20 * Rhs.m02 + m21 * Rhs.m12 + m22 * Rhs.m22 + m23 * Rhs.m32,
            m30 * Rhs.m02 + m31 * Rhs.m12 + m32 * Rhs.m22 + m33 * Rhs.m32,
            m00 * Rhs.m03 + m01 * Rhs.m13 + m02 * Rhs.m23 + m03 * Rhs.m33,
            m10 * Rhs.m03 + m11 * Rhs.m13 + m12 * Rhs.m23 + m13 * Rhs.m33,
            m20 * Rhs.m03 + m21 * Rhs.m13 + m22 * Rhs.m23 + m23 * Rhs.m33,
            m30 * Rhs.m03 + m31 * Rhs.m13 + m32 * Rhs.m23 + m33 * Rhs.m33
        };
    }

    // ---------------- Rectangular implementations ----------------
    Mat2x3::Mat2x3() : c0(Vec3::Zero()), c1(Vec3::Zero()) {
    }

    Mat2x3::Mat2x3(const Vec3 &col0, const Vec3 &col1) : c0(col0), c1(col1) {
    }

    Mat2x3 Mat2x3::Zero() {
        return {Vec3::Zero(), Vec3::Zero()};
    }

    Mat3x2 Mat2x3::Transposed() const {
        return {Vec2(c0.x, c1.x), Vec2(c0.y, c1.y), Vec2(c0.z, c1.z)};
    }

    Vec3 Mat2x3::operator*(const Vec2 &V) const {
        return c0 * V.x + c1 * V.y;
    }

    Mat2x3 Mat2x3::operator+(const Mat2x3 &Rhs) const {
        return {c0 + Rhs.c0, c1 + Rhs.c1};
    }

    Mat2x3 Mat2x3::operator-(const Mat2x3 &Rhs) const {
        return {c0 - Rhs.c0, c1 - Rhs.c1};
    }

    Mat2x3 Mat2x3::operator*(float S) const {
        return {c0 * S, c1 * S};
    }

    Mat2x3 &Mat2x3::operator+=(const Mat2x3 &Rhs) {
        c0 += Rhs.c0;
        c1 += Rhs.c1;
        return *this;
    }

    Mat2x3 &Mat2x3::operator-=(const Mat2x3 &Rhs) {
        c0 -= Rhs.c0;
        c1 -= Rhs.c1;
        return *this;
    }

    Mat3x2::Mat3x2() : c0(Vec2::Zero()), c1(Vec2::Zero()), c2(Vec2::Zero()) {
    }

    Mat3x2::Mat3x2(const Vec2 &a, const Vec2 &b, const Vec2 &c) : c0(a), c1(b), c2(c) {
    }

    Mat3x2 Mat3x2::Zero() {
        return {Vec2::Zero(), Vec2::Zero(), Vec2::Zero()};
    }

    Mat2x3 Mat3x2::Transposed() const {
        return {Vec3(c0.x, c1.x, c2.x), Vec3(c0.y, c1.y, c2.y)};
    }

    Vec2 Mat3x2::operator*(const Vec3 &V) const {
        return c0 * V.x + c1 * V.y + c2 * V.z;
    }

    Mat3x2 Mat3x2::operator+(const Mat3x2 &Rhs) const {
        return {c0 + Rhs.c0, c1 + Rhs.c1, c2 + Rhs.c2};
    }

    Mat3x2 Mat3x2::operator-(const Mat3x2 &Rhs) const {
        return {c0 - Rhs.c0, c1 - Rhs.c1, c2 - Rhs.c2};
    }

    Mat3x2 Mat3x2::operator*(float S) const {
        return {c0 * S, c1 * S, c2 * S};
    }

    Mat3x2 &Mat3x2::operator+=(const Mat3x2 &Rhs) {
        c0 += Rhs.c0;
        c1 += Rhs.c1;
        c2 += Rhs.c2;
        return *this;
    }

    Mat3x2 &Mat3x2::operator-=(const Mat3x2 &Rhs) {
        c0 -= Rhs.c0;
        c1 -= Rhs.c1;
        c2 -= Rhs.c2;
        return *this;
    }

    // Mat2x3 * Mat3x2 = Mat3
    // Mat2x3 has 2 Vec3 columns (c0, c1), Mat3x2 has 3 Vec2 columns (c0, c1, c2)
    // Result is 3x3
    Mat3 Mat2x3::operator*(const Mat3x2 &Rhs) const {
        return {
            // Column 0
            c0.x * Rhs.c0.x + c1.x * Rhs.c0.y,
            c0.y * Rhs.c0.x + c1.y * Rhs.c0.y,
            c0.z * Rhs.c0.x + c1.z * Rhs.c0.y,
            // Column 1
            c0.x * Rhs.c1.x + c1.x * Rhs.c1.y,
            c0.y * Rhs.c1.x + c1.y * Rhs.c1.y,
            c0.z * Rhs.c1.x + c1.z * Rhs.c1.y,
            // Column 2
            c0.x * Rhs.c2.x + c1.x * Rhs.c2.y,
            c0.y * Rhs.c2.x + c1.y * Rhs.c2.y,
            c0.z * Rhs.c2.x + c1.z * Rhs.c2.y
        };
    }

    // Mat3x2 * Mat2x3 = Mat2
    // Mat3x2 has 3 Vec2 columns (c0, c1, c2), Mat2x3 has 2 Vec3 columns (c0, c1)
    // Result is 2x2
    Mat2 Mat3x2::operator*(const Mat2x3 &Rhs) const {
        return {
            c0.x * Rhs.c0.x + c1.x * Rhs.c0.y + c2.x * Rhs.c0.z,
            c0.y * Rhs.c0.x + c1.y * Rhs.c0.y + c2.y * Rhs.c0.z,
            c0.x * Rhs.c1.x + c1.x * Rhs.c1.y + c2.x * Rhs.c1.z,
            c0.y * Rhs.c1.x + c1.y * Rhs.c1.y + c2.y * Rhs.c1.z
        };
    }

    Mat2x4::Mat2x4() : c0(Vec4::Zero()), c1(Vec4::Zero()) {
    }

    Mat2x4::Mat2x4(const Vec4 &a, const Vec4 &b) : c0(a), c1(b) {
    }

    Mat2x4 Mat2x4::Zero() {
        return {Vec4::Zero(), Vec4::Zero()};
    }

    Vec4 Mat2x4::operator*(const Vec2 &V) const {
        return c0 * V.x + c1 * V.y;
    }

    Mat4x2 Mat2x4::Transposed() const {
        return {Vec2(c0.x, c1.x), Vec2(c0.y, c1.y), Vec2(c0.z, c1.z), Vec2(c0.w, c1.w)};
    }

    Mat4x2::Mat4x2() : c0(Vec2::Zero()), c1(Vec2::Zero()), c2(Vec2::Zero()), c3(Vec2::Zero()) {
    }

    Mat4x2::Mat4x2(const Vec2 &a, const Vec2 &b, const Vec2 &c, const Vec2 &d) : c0(a), c1(b), c2(c), c3(d) {
    }

    Mat4x2 Mat4x2::Zero() {
        return {Vec2::Zero(), Vec2::Zero(), Vec2::Zero(), Vec2::Zero()};
    }

    Vec2 Mat4x2::operator*(const Vec4 &V) const {
        return c0 * V.x + c1 * V.y + c2 * V.z + c3 * V.w;
    }

    Mat2x4 Mat4x2::Transposed() const {
        return {Vec4(c0.x, c1.x, c2.x, c3.x), Vec4(c0.y, c1.y, c2.y, c3.y)};
    }

    // Mat2x4 * Mat4x2 = Mat4
    // Mat2x4 has 2 Vec4 columns (c0, c1), Mat4x2 has 4 Vec2 columns (c0, c1, c2, c3)
    // Result is 4x4
    Mat4 Mat2x4::operator*(const Mat4x2 &Rhs) const {
        return {
            // Column 0
            c0.x * Rhs.c0.x + c1.x * Rhs.c0.y,
            c0.y * Rhs.c0.x + c1.y * Rhs.c0.y,
            c0.z * Rhs.c0.x + c1.z * Rhs.c0.y,
            c0.w * Rhs.c0.x + c1.w * Rhs.c0.y,
            // Column 1
            c0.x * Rhs.c1.x + c1.x * Rhs.c1.y,
            c0.y * Rhs.c1.x + c1.y * Rhs.c1.y,
            c0.z * Rhs.c1.x + c1.z * Rhs.c1.y,
            c0.w * Rhs.c1.x + c1.w * Rhs.c1.y,
            // Column 2
            c0.x * Rhs.c2.x + c1.x * Rhs.c2.y,
            c0.y * Rhs.c2.x + c1.y * Rhs.c2.y,
            c0.z * Rhs.c2.x + c1.z * Rhs.c2.y,
            c0.w * Rhs.c2.x + c1.w * Rhs.c2.y,
            // Column 3
            c0.x * Rhs.c3.x + c1.x * Rhs.c3.y,
            c0.y * Rhs.c3.x + c1.y * Rhs.c3.y,
            c0.z * Rhs.c3.x + c1.z * Rhs.c3.y,
            c0.w * Rhs.c3.x + c1.w * Rhs.c3.y
        };
    }

    // Mat4x2 * Mat2x4 = Mat2
    Mat2 Mat4x2::operator*(const Mat2x4 &Rhs) const {
        return {
            c0.x * Rhs.c0.x + c1.x * Rhs.c0.y + c2.x * Rhs.c0.z + c3.x * Rhs.c0.w,
            c0.y * Rhs.c0.x + c1.y * Rhs.c0.y + c2.y * Rhs.c0.z + c3.y * Rhs.c0.w,
            c0.x * Rhs.c1.x + c1.x * Rhs.c1.y + c2.x * Rhs.c1.z + c3.x * Rhs.c1.w,
            c0.y * Rhs.c1.x + c1.y * Rhs.c1.y + c2.y * Rhs.c1.z + c3.y * Rhs.c1.w
        };
    }

    Mat3x4::Mat3x4() : c0(Vec4::Zero()), c1(Vec4::Zero()), c2(Vec4::Zero()) {
    }

    Mat3x4::Mat3x4(const Vec4 &a, const Vec4 &b, const Vec4 &c) : c0(a), c1(b), c2(c) {
    }

    Mat3x4 Mat3x4::Zero() {
        return {Vec4::Zero(), Vec4::Zero(), Vec4::Zero()};
    }

    Vec4 Mat3x4::operator*(const Vec3 &V) const {
        return c0 * V.x + c1 * V.y + c2 * V.z;
    }

    Mat4x3 Mat3x4::Transposed() const {
        return {Vec3(c0.x, c1.x, c2.x), Vec3(c0.y, c1.y, c2.y), Vec3(c0.z, c1.z, c2.z), Vec3(c0.w, c1.w, c2.w)};
    }

    Mat4x3::Mat4x3() : c0(Vec3::Zero()), c1(Vec3::Zero()), c2(Vec3::Zero()), c3(Vec3::Zero()) {
    }

    Mat4x3::Mat4x3(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &d) : c0(a), c1(b), c2(c), c3(d) {
    }

    Mat4x3 Mat4x3::Zero() {
        return {Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
    }

    Vec3 Mat4x3::operator*(const Vec4 &V) const {
        return c0 * V.x + c1 * V.y + c2 * V.z + c3 * V.w;
    }

    Mat3x4 Mat4x3::Transposed() const {
        return {Vec4(c0.x, c1.x, c2.x, c3.x), Vec4(c0.y, c1.y, c2.y, c3.y), Vec4(c0.z, c1.z, c2.z, c3.z)};
    }

    // Mat3x4 * Mat4x3 = Mat4
    // Mat3x4 has 3 Vec4 columns (c0, c1, c2), Mat4x3 has 4 Vec3 columns (c0, c1, c2, c3)
    // Result is 4x4
    Mat4 Mat3x4::operator*(const Mat4x3 &Rhs) const {
        return {
            // Column 0
            c0.x * Rhs.c0.x + c1.x * Rhs.c0.y + c2.x * Rhs.c0.z,
            c0.y * Rhs.c0.x + c1.y * Rhs.c0.y + c2.y * Rhs.c0.z,
            c0.z * Rhs.c0.x + c1.z * Rhs.c0.y + c2.z * Rhs.c0.z,
            c0.w * Rhs.c0.x + c1.w * Rhs.c0.y + c2.w * Rhs.c0.z,
            // Column 1
            c0.x * Rhs.c1.x + c1.x * Rhs.c1.y + c2.x * Rhs.c1.z,
            c0.y * Rhs.c1.x + c1.y * Rhs.c1.y + c2.y * Rhs.c1.z,
            c0.z * Rhs.c1.x + c1.z * Rhs.c1.y + c2.z * Rhs.c1.z,
            c0.w * Rhs.c1.x + c1.w * Rhs.c1.y + c2.w * Rhs.c1.z,
            // Column 2
            c0.x * Rhs.c2.x + c1.x * Rhs.c2.y + c2.x * Rhs.c2.z,
            c0.y * Rhs.c2.x + c1.y * Rhs.c2.y + c2.y * Rhs.c2.z,
            c0.z * Rhs.c2.x + c1.z * Rhs.c2.y + c2.z * Rhs.c2.z,
            c0.w * Rhs.c2.x + c1.w * Rhs.c2.y + c2.w * Rhs.c2.z,
            // Column 3
            c0.x * Rhs.c3.x + c1.x * Rhs.c3.y + c2.x * Rhs.c3.z,
            c0.y * Rhs.c3.x + c1.y * Rhs.c3.y + c2.y * Rhs.c3.z,
            c0.z * Rhs.c3.x + c1.z * Rhs.c3.y + c2.z * Rhs.c3.z,
            c0.w * Rhs.c3.x + c1.w * Rhs.c3.y + c2.w * Rhs.c3.z
        };
    }

    // Mat4x3 * Mat3x4 = Mat3
    Mat3 Mat4x3::operator*(const Mat3x4 &Rhs) const {
        return {
            c0.x * Rhs.c0.x + c1.x * Rhs.c0.y + c2.x * Rhs.c0.z + c3.x * Rhs.c0.w,
            c0.y * Rhs.c0.x + c1.y * Rhs.c0.y + c2.y * Rhs.c0.z + c3.y * Rhs.c0.w,
            c0.z * Rhs.c0.x + c1.z * Rhs.c0.y + c2.z * Rhs.c0.z + c3.z * Rhs.c0.w,
            c0.x * Rhs.c1.x + c1.x * Rhs.c1.y + c2.x * Rhs.c1.z + c3.x * Rhs.c1.w,
            c0.y * Rhs.c1.x + c1.y * Rhs.c1.y + c2.y * Rhs.c1.z + c3.y * Rhs.c1.w,
            c0.z * Rhs.c1.x + c1.z * Rhs.c1.y + c2.z * Rhs.c1.z + c3.z * Rhs.c1.w,
            c0.x * Rhs.c2.x + c1.x * Rhs.c2.y + c2.x * Rhs.c2.z + c3.x * Rhs.c2.w,
            c0.y * Rhs.c2.x + c1.y * Rhs.c2.y + c2.y * Rhs.c2.z + c3.y * Rhs.c2.w,
            c0.z * Rhs.c2.x + c1.z * Rhs.c2.y + c2.z * Rhs.c2.z + c3.z * Rhs.c2.w
        };
    }

    // ---------------- Hadamard Product (Element-wise multiplication) ----------------
    
    // Mat2 Hadamard product
    Mat2 Mat2::operator%(const Mat2 &Rhs) const {
        return {m00 * Rhs.m00, m10 * Rhs.m10, m01 * Rhs.m01, m11 * Rhs.m11};
    }
    Mat2 &Mat2::operator%=(const Mat2 &Rhs) {
        m00 *= Rhs.m00; m10 *= Rhs.m10;
        m01 *= Rhs.m01; m11 *= Rhs.m11;
        return *this;
    }
    
    // Mat3 Hadamard product
    Mat3 Mat3::operator%(const Mat3 &Rhs) const {
        return {
            m00 * Rhs.m00, m10 * Rhs.m10, m20 * Rhs.m20,
            m01 * Rhs.m01, m11 * Rhs.m11, m21 * Rhs.m21,
            m02 * Rhs.m02, m12 * Rhs.m12, m22 * Rhs.m22
        };
    }
    Mat3 &Mat3::operator%=(const Mat3 &Rhs) {
        m00 *= Rhs.m00; m10 *= Rhs.m10; m20 *= Rhs.m20;
        m01 *= Rhs.m01; m11 *= Rhs.m11; m21 *= Rhs.m21;
        m02 *= Rhs.m02; m12 *= Rhs.m12; m22 *= Rhs.m22;
        return *this;
    }
    
    // Mat4 Hadamard product
    Mat4 Mat4::operator%(const Mat4 &Rhs) const {
        return {
            m00 * Rhs.m00, m10 * Rhs.m10, m20 * Rhs.m20, m30 * Rhs.m30,
            m01 * Rhs.m01, m11 * Rhs.m11, m21 * Rhs.m21, m31 * Rhs.m31,
            m02 * Rhs.m02, m12 * Rhs.m12, m22 * Rhs.m22, m32 * Rhs.m32,
            m03 * Rhs.m03, m13 * Rhs.m13, m23 * Rhs.m23, m33 * Rhs.m33
        };
    }
    Mat4 &Mat4::operator%=(const Mat4 &Rhs) {
        m00 *= Rhs.m00; m10 *= Rhs.m10; m20 *= Rhs.m20; m30 *= Rhs.m30;
        m01 *= Rhs.m01; m11 *= Rhs.m11; m21 *= Rhs.m21; m31 *= Rhs.m31;
        m02 *= Rhs.m02; m12 *= Rhs.m12; m22 *= Rhs.m22; m32 *= Rhs.m32;
        m03 *= Rhs.m03; m13 *= Rhs.m13; m23 *= Rhs.m23; m33 *= Rhs.m33;
        return *this;
    }
    
    // Mat2x3 Hadamard product
    Mat2x3 Mat2x3::operator%(const Mat2x3 &Rhs) const {
        return {
            Vec3(c0.x * Rhs.c0.x, c0.y * Rhs.c0.y, c0.z * Rhs.c0.z),
            Vec3(c1.x * Rhs.c1.x, c1.y * Rhs.c1.y, c1.z * Rhs.c1.z)
        };
    }
    Mat2x3 &Mat2x3::operator%=(const Mat2x3 &Rhs) {
        c0.x *= Rhs.c0.x; c0.y *= Rhs.c0.y; c0.z *= Rhs.c0.z;
        c1.x *= Rhs.c1.x; c1.y *= Rhs.c1.y; c1.z *= Rhs.c1.z;
        return *this;
    }
    
    // Mat3x2 Hadamard product
    Mat3x2 Mat3x2::operator%(const Mat3x2 &Rhs) const {
        return {
            Vec2(c0.x * Rhs.c0.x, c0.y * Rhs.c0.y),
            Vec2(c1.x * Rhs.c1.x, c1.y * Rhs.c1.y),
            Vec2(c2.x * Rhs.c2.x, c2.y * Rhs.c2.y)
        };
    }
    Mat3x2 &Mat3x2::operator%=(const Mat3x2 &Rhs) {
        c0.x *= Rhs.c0.x; c0.y *= Rhs.c0.y;
        c1.x *= Rhs.c1.x; c1.y *= Rhs.c1.y;
        c2.x *= Rhs.c2.x; c2.y *= Rhs.c2.y;
        return *this;
    }
    
    // Mat2x4 Hadamard product
    Mat2x4 Mat2x4::operator%(const Mat2x4 &Rhs) const {
        return {
            Vec4(c0.x * Rhs.c0.x, c0.y * Rhs.c0.y, c0.z * Rhs.c0.z, c0.w * Rhs.c0.w),
            Vec4(c1.x * Rhs.c1.x, c1.y * Rhs.c1.y, c1.z * Rhs.c1.z, c1.w * Rhs.c1.w)
        };
    }
    Mat2x4 &Mat2x4::operator%=(const Mat2x4 &Rhs) {
        c0.x *= Rhs.c0.x; c0.y *= Rhs.c0.y; c0.z *= Rhs.c0.z; c0.w *= Rhs.c0.w;
        c1.x *= Rhs.c1.x; c1.y *= Rhs.c1.y; c1.z *= Rhs.c1.z; c1.w *= Rhs.c1.w;
        return *this;
    }
    
    // Mat4x2 Hadamard product
    Mat4x2 Mat4x2::operator%(const Mat4x2 &Rhs) const {
        return {
            Vec2(c0.x * Rhs.c0.x, c0.y * Rhs.c0.y),
            Vec2(c1.x * Rhs.c1.x, c1.y * Rhs.c1.y),
            Vec2(c2.x * Rhs.c2.x, c2.y * Rhs.c2.y),
            Vec2(c3.x * Rhs.c3.x, c3.y * Rhs.c3.y)
        };
    }
    Mat4x2 &Mat4x2::operator%=(const Mat4x2 &Rhs) {
        c0.x *= Rhs.c0.x; c0.y *= Rhs.c0.y;
        c1.x *= Rhs.c1.x; c1.y *= Rhs.c1.y;
        c2.x *= Rhs.c2.x; c2.y *= Rhs.c2.y;
        c3.x *= Rhs.c3.x; c3.y *= Rhs.c3.y;
        return *this;
    }
    
    // Mat3x4 Hadamard product
    Mat3x4 Mat3x4::operator%(const Mat3x4 &Rhs) const {
        return {
            Vec4(c0.x * Rhs.c0.x, c0.y * Rhs.c0.y, c0.z * Rhs.c0.z, c0.w * Rhs.c0.w),
            Vec4(c1.x * Rhs.c1.x, c1.y * Rhs.c1.y, c1.z * Rhs.c1.z, c1.w * Rhs.c1.w),
            Vec4(c2.x * Rhs.c2.x, c2.y * Rhs.c2.y, c2.z * Rhs.c2.z, c2.w * Rhs.c2.w)
        };
    }
    Mat3x4 &Mat3x4::operator%=(const Mat3x4 &Rhs) {
        c0.x *= Rhs.c0.x; c0.y *= Rhs.c0.y; c0.z *= Rhs.c0.z; c0.w *= Rhs.c0.w;
        c1.x *= Rhs.c1.x; c1.y *= Rhs.c1.y; c1.z *= Rhs.c1.z; c1.w *= Rhs.c1.w;
        c2.x *= Rhs.c2.x; c2.y *= Rhs.c2.y; c2.z *= Rhs.c2.z; c2.w *= Rhs.c2.w;
        return *this;
    }
    
    // Mat4x3 Hadamard product
    Mat4x3 Mat4x3::operator%(const Mat4x3 &Rhs) const {
        return {
            Vec3(c0.x * Rhs.c0.x, c0.y * Rhs.c0.y, c0.z * Rhs.c0.z),
            Vec3(c1.x * Rhs.c1.x, c1.y * Rhs.c1.y, c1.z * Rhs.c1.z),
            Vec3(c2.x * Rhs.c2.x, c2.y * Rhs.c2.y, c2.z * Rhs.c2.z),
            Vec3(c3.x * Rhs.c3.x, c3.y * Rhs.c3.y, c3.z * Rhs.c3.z)
        };
    }
    Mat4x3 &Mat4x3::operator%=(const Mat4x3 &Rhs) {
        c0.x *= Rhs.c0.x; c0.y *= Rhs.c0.y; c0.z *= Rhs.c0.z;
        c1.x *= Rhs.c1.x; c1.y *= Rhs.c1.y; c1.z *= Rhs.c1.z;
        c2.x *= Rhs.c2.x; c2.y *= Rhs.c2.y; c2.z *= Rhs.c2.z;
        c3.x *= Rhs.c3.x; c3.y *= Rhs.c3.y; c3.z *= Rhs.c3.z;
        return *this;
    }
} // namespace GPU::Math
