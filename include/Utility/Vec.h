#pragma once

/**
 * Vec.h:
 *      @Descripiton    :   The vector library for GPU programing
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#ifndef EASYGPU_VEC_H
#define EASYGPU_VEC_H

namespace GPU::Math {
    /**
     * 2D floating point vector
     */
    class Vec2 {
    public:
        Vec2();

        Vec2(float S);

        /**
         * Construct from components
         * @param X The x component
         * @param Y The y component
         */
        Vec2(float X, float Y);

        /** Arithmetic ops */
        /**
         * Add two vectors
         * @param Rhs The right-hand vector
         * @return The addition result
         */
        Vec2 operator+(const Vec2 &Rhs) const;

        /**
         * Subtract two vectors
         * @param Rhs The right-hand vector
         * @return The subtraction result
         */
        Vec2 operator-(const Vec2 &Rhs) const;

        /**
         * Multiply vector by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        Vec2 operator*(float S) const;

        /**
         * Divide vector by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        Vec2 operator/(float S) const;

        /** Compound assignment */
        /**
         * Add and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        Vec2 &operator+=(const Vec2 &Rhs);

        /**
         * Subtract and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        Vec2 &operator-=(const Vec2 &Rhs);

        /**
         * Multiply and assign
         * @param S The scalar
         * @return Reference to this
         */
        Vec2 &operator*=(float S);

        /**
         * Divide and assign
         * @param S The scalar
         * @return Reference to this
         */
        Vec2 &operator/=(float S);

        /**
         * Negate
         * @return The negated vector
         */
        Vec2 operator-() const;

        /**
         * Dot product
         * @param Rhs The other vector
         * @return The dot product scalar
         */
        [[nodiscard]] float Dot(const Vec2 &Rhs) const;

        /**
         * Squared length
         * @return Squared length
         */
        [[nodiscard]] float Length2() const;

        /**
         * Length (magnitude)
         * @return Length
         */
        [[nodiscard]] float Length() const;

        /**
         * Normalize in-place
         * @return void
         */
        void Normalize();

        /**
         * Return a normalized copy
         * @return Normalized vector
         */
        [[nodiscard]] Vec2 Normalized() const;

        /**
         * Zero vector
         * @return Vector with all components zero
         */
        static Vec2 Zero();

    public:
        float x = 0.0f;
        float y = 0.0f;
    };

    /**
     * 3D floating point vector
     */
    class Vec3 {
    public:
        /** Default constructor */
        Vec3();

        /**
         * Construct all components from S
         * @param S The scalar to set for all components
         * @return Constructed Vec3
         */
        Vec3(float S);

        /**
         * Construct from components
         * @param X The x component
         * @param Y The y component
         * @param Z The z component
         * @return Constructed Vec3
         */
        Vec3(float X, float Y, float Z);

        /**
         * Add two vectors
         * @param Rhs The right-hand vector
         * @return The addition result
         */
        Vec3 operator+(const Vec3 &Rhs) const;

        /**
         * Subtract two vectors
         * @param Rhs The right-hand vector
         * @return The subtraction result
         */
        Vec3 operator-(const Vec3 &Rhs) const;

        /**
         * Multiply vector by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        Vec3 operator*(float S) const;

        /**
         * Divide vector by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        Vec3 operator/(float S) const;


    public:
        /**
         * Add and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        Vec3 &operator+=(const Vec3 &Rhs);

        /**
         * Subtract and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        Vec3 &operator-=(const Vec3 &Rhs);

        /**
         * Multiply and assign
         * @param S The scalar
         * @return Reference to this
         */
        Vec3 &operator*=(float S);

        /**
         * Divide and assign
         * @param S The scalar
         * @return Reference to this
         */
        Vec3 &operator/=(float S);

        /**
         * Negate
         * @return The negated vector
         */
        Vec3 operator-() const;

        /**
         * Dot product
         * @param Rhs The other vector
         * @return The dot product scalar
         */
        [[nodiscard]] float Dot(const Vec3 &Rhs) const;

    public:
        /**
         * Cross product
         * @param Rhs The other vector
         * @return The cross product vector
         */
        [[nodiscard]] Vec3 Cross(const Vec3 &Rhs) const;

        /**
         * Squared length
         * @return Squared length
         */
        [[nodiscard]] float Length2() const;

        /**
         * Length (magnitude)
         * @return Length
         */
        [[nodiscard]] float Length() const;

        /**
         * Normalize in-place
         * @return void
         */
        void Normalize();

        /**
         * Return a normalized copy
         * @return Normalized vector
         */
        [[nodiscard]] Vec3 Normalized() const;

        /**
         * Zero vector
         * @return Vector with all components zero
         */
        static Vec3 Zero();

    public:
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    /**
     * 4D floating point vector
     */
    class Vec4 {
    public:
        Vec4();

        /**
         * Construct all components from S
         * @param S The scalar to set for all components
         * @return Constructed Vec4
         */
        Vec4(float S);

        /**
         * Construct from components
         * @param X The x component
         * @param Y The y component
         * @param Z The z component
         * @param W The w component
         * @return Constructed Vec4
         */
        Vec4(float X, float Y, float Z, float W);

        /**
         * Construct from Vec3 (implicit conversion like GLSL)
         * Creates vec4(vec3.xyz, w) with default w=0.0 for vectors
         * @param V The Vec3 to convert
         * @param W The w component (default 0.0 for vectors, use 1.0 for positions)
         */
        Vec4(const Vec3& V, float W = 0.0f);

        /**
         * Add two vectors
         * @param Rhs The right-hand vector
         * @return The addition result
         */
        Vec4 operator+(const Vec4 &Rhs) const;

        /**
         * Subtract two vectors
         * @param Rhs The right-hand vector
         * @return The subtraction result
         */
        Vec4 operator-(const Vec4 &Rhs) const;

        /**
         * Multiply vector by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        Vec4 operator*(float S) const;

        /**
         * Divide vector by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        Vec4 operator/(float S) const;

        /**
         * Add and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        Vec4 &operator+=(const Vec4 &Rhs);

        /**
         * Subtract and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        Vec4 &operator-=(const Vec4 &Rhs);

        /**
         * Multiply and assign
         * @param S The scalar
         * @return Reference to this
         */
        Vec4 &operator*=(float S);

        /**
         * Divide and assign
         * @param S The scalar
         * @return Reference to this
         */
        Vec4 &operator/=(float S);

        /**
         * Negate
         * @return The negated vector
         */
        Vec4 operator-() const;

        /**
         * Dot product
         * @param Rhs The other vector
         * @return The dot product scalar
         */
        [[nodiscard]] float Dot(const Vec4 &Rhs) const;

        /**
         * Squared length
         * @return Squared length
         */
        [[nodiscard]] float Length2() const;

        /**
         * Length (magnitude)
         * @return Length
         */
        [[nodiscard]] float Length() const;

        /**
         * Normalize in-place
         * @return void
         */
        void Normalize();

    public:
        [[nodiscard]] Vec4 Normalized() const;

        /**
         * Zero vector
         * @return Vector with all components zero
         */
        static Vec4 Zero();

    public:
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float w = 0.0f;
    };

    /**
     * IVec2:
     *      2D integer vector
     */
    class IVec2 {
    public:
        /**
         * @return Constructed IVec2
         */
        IVec2();

        /**
         * Construct all components from S
         * @param S Scalar to set
         * @return Constructed IVec2
         */
        IVec2(int S);

        /**
         * Construct from components
         * @param X The x component
         * @param Y The y component
         * @return Constructed IVec2
         */
        IVec2(int X, int Y);

        /**
         * Construct from float vector (truncates)
         * @param V The float vector
         * @return Constructed IVec2
         */
        explicit IVec2(const Vec2 &V);

        /**
         * Add two integer vectors
         * @param Rhs The right-hand vector
         * @return The addition result
         */
        IVec2 operator+(const IVec2 &Rhs) const;


    public:
        /**
         * Subtract two integer vectors
         * @param Rhs The right-hand vector
         * @return The subtraction result
         */
        IVec2 operator-(const IVec2 &Rhs) const;

    public: // operators
        /**
         * Multiply by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        IVec2 operator*(int S) const;

        /**
         * Divide by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        IVec2 operator/(int S) const;

        /**
         * Add and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        IVec2 &operator+=(const IVec2 &Rhs);

        /**
         * Subtract and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        IVec2 &operator-=(const IVec2 &Rhs);

        /**
         * Multiply and assign
         * @param S The scalar
         * @return Reference to this
         */
        IVec2 &operator*=(int S);

        /**
         * Divide and assign
         * @param S The scalar
         * @return Reference to this
         */
        IVec2 &operator/=(int S);

        /**
         * Zero vector
         * @return Vector with all components zero
         */
        static IVec2 Zero();

    public:
        int x = 0;
        int y = 0;
    };

    /**
     * 3D integer vector
     */
    class IVec3 {
    public:
        IVec3();

        /**
         * Construct all components from S
         * @param S Scalar to set
         * @return Constructed IVec3
         */
        IVec3(int S);

        /**
         * Construct from components
         * @param X The x component
         * @param Y The y component
         * @param Z The z component
         * @return Constructed IVec3
         */
        IVec3(int X, int Y, int Z);

        /**
         * Construct from float vector (truncates)
         * @param V The float vector
         * @return Constructed IVec3
         */
        explicit IVec3(const Vec3 &V);

        /**
         * Add two integer vectors
         * @param Rhs The right-hand vector
         * @return The addition result
         */
        IVec3 operator+(const IVec3 &Rhs) const;

        /**
         * Subtract two integer vectors
         * @param Rhs The right-hand vector
         * @return The subtraction result
         */
        IVec3 operator-(const IVec3 &Rhs) const;

        /**
         * Multiply by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        IVec3 operator*(int S) const;

        /**
         * Divide by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        IVec3 operator/(int S) const;

        /**
         * Add and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        IVec3 &operator+=(const IVec3 &Rhs);

        /**
         * Subtract and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        IVec3 &operator-=(const IVec3 &Rhs);

        /**
         * Multiply and assign
         * @param S The scalar
         * @return Reference to this
         */
        IVec3 &operator*=(int S);

        /**
         * Divide and assign
         * @param S The scalar
         * @return Reference to this
         */
        IVec3 &operator/=(int S);

        /**
         * Zero vector
         * @return Vector with all components zero
         */
        static IVec3 Zero();

    public:
        int x = 0;
        int y = 0;
        int z = 0;
    };

    /**
     * 4D integer vector
     */
    class IVec4 {
    public:
        /** Default constructor
         * @return Constructed IVec4
         */
        IVec4();

        /**
         * Construct all components from S
         * @param S Scalar to set
         * @return Constructed IVec4
         */
        IVec4(int S);

        /**
         * Construct from components
         * @param X The x component
         * @param Y The y component
         * @param Z The z component
         * @param W The w component
         * @return Constructed IVec4
         */
        IVec4(int X, int Y, int Z, int W);

        /**
         * Construct from float vector (truncates)
         * @param V The float vector
         * @return Constructed IVec4
         */
        explicit IVec4(const Vec4 &V);

        /**
         * Add two integer vectors
         * @param Rhs The right-hand vector
         * @return The addition result
         */
        IVec4 operator+(const IVec4 &Rhs) const;

        /**
         * Subtract two integer vectors
         * @param Rhs The right-hand vector
         * @return The subtraction result
         */
        IVec4 operator-(const IVec4 &Rhs) const;

        /**
         * Multiply by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        IVec4 operator*(int S) const;

        /**
         * Divide by scalar
         * @param S The scalar
         * @return The scaled vector
         */
        IVec4 operator/(int S) const;

        /**
         * Add and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        IVec4 &operator+=(const IVec4 &Rhs);

        /**
         * Subtract and assign
         * @param Rhs The right-hand vector
         * @return Reference to this
         */
        IVec4 &operator-=(const IVec4 &Rhs);

        /**
         * Multiply and assign
         * @param S The scalar
         * @return Reference to this
         */
        IVec4 &operator*=(int S);

        /**
         * Divide and assign
         * @param S The scalar
         * @return Reference to this
         */
        IVec4 &operator/=(int S);

        /**
         * Zero vector
         * @return Vector with all components zero
         */
        static IVec4 Zero();

    public:
        int x = 0;
        int y = 0;
        int z = 0;
        int w = 0;
    };
}

#endif //EASYGPU_VEC_H
