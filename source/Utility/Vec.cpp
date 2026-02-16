/**
 * Vec.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */

#include <Utility/Vec.h>
#include <cmath>

namespace GPU::Math {
    // Vec2
    Vec2::Vec2() = default;

    Vec2::Vec2(float S) : x(S), y(S) {
    }

    Vec2::Vec2(float X, float Y) : x(X), y(Y) {
    }

    Vec2 Vec2::operator+(const Vec2 &Rhs) const {
        return {x + Rhs.x, y + Rhs.y};
    }

    Vec2 Vec2::operator-(const Vec2 &Rhs) const {
        return {x - Rhs.x, y - Rhs.y};
    }

    Vec2 Vec2::operator*(float S) const {
        return {x * S, y * S};
    }

    Vec2 Vec2::operator/(float S) const {
        return {x / S, y / S};
    }

    Vec2 &Vec2::operator+=(const Vec2 &Rhs) {
        x += Rhs.x;
        y += Rhs.y;
        return *this;
    }

    Vec2 &Vec2::operator-=(const Vec2 &Rhs) {
        x -= Rhs.x;
        y -= Rhs.y;
        return *this;
    }

    Vec2 &Vec2::operator*=(float S) {
        x *= S;
        y *= S;
        return *this;
    }

    Vec2 &Vec2::operator/=(float S) {
        x /= S;
        y /= S;
        return *this;
    }

    Vec2 Vec2::operator-() const {
        return {-x, -y};
    }

    float Vec2::Dot(const Vec2 &Rhs) const {
        return x * Rhs.x + y * Rhs.y;
    }
    float Vec2::Length2() const { return Dot(*this); }
    float Vec2::Length() const { return std::sqrt(Length2()); }

    void Vec2::Normalize() {
        float l = Length();
        if (l != 0.0f) {
            x /= l;
            y /= l;
        }
    }

    Vec2 Vec2::Normalized() const {
        float l = Length();
        return (l == 0.0f) ? Vec2(0.0f) : Vec2(x / l, y / l);
    }

    Vec2 Vec2::Zero() {
        return {0.0f};
    }

    // Vec3
    Vec3::Vec3() = default;

    Vec3::Vec3(float S) : x(S), y(S), z(S) {
    }

    Vec3::Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {
    }

    Vec3 Vec3::operator+(const Vec3 &Rhs) const {
        return {x + Rhs.x, y + Rhs.y, z + Rhs.z};
    }
    Vec3 Vec3::operator-(const Vec3 &Rhs) const {
        return {x - Rhs.x, y - Rhs.y, z - Rhs.z};
    }
    Vec3 Vec3::operator*(float S) const {
        return {x * S, y * S, z * S};
    }
    Vec3 Vec3::operator/(float S) const {
        return {x / S, y / S, z / S};
    }

    Vec3 &Vec3::operator+=(const Vec3 &Rhs) {
        x += Rhs.x;
        y += Rhs.y;
        z += Rhs.z;
        return *this;
    }

    Vec3 &Vec3::operator-=(const Vec3 &Rhs) {
        x -= Rhs.x;
        y -= Rhs.y;
        z -= Rhs.z;
        return *this;
    }

    Vec3 &Vec3::operator*=(float S) {
        x *= S;
        y *= S;
        z *= S;
        return *this;
    }

    Vec3 &Vec3::operator/=(float S) {
        x /= S;
        y /= S;
        z /= S;
        return *this;
    }

    Vec3 Vec3::operator-() const {
        return {-x, -y, -z};
    }

    float Vec3::Dot(const Vec3 &Rhs) const {
        return x * Rhs.x + y * Rhs.y + z * Rhs.z;
    }

    Vec3 Vec3::Cross(const Vec3 &Rhs) const {
        return {y * Rhs.z - z * Rhs.y, z * Rhs.x - x * Rhs.z, x * Rhs.y - y * Rhs.x};
    }

    float Vec3::Length2() const { return Dot(*this); }
    float Vec3::Length() const { return std::sqrt(Length2()); }

    void Vec3::Normalize() {
        float l = Length();
        if (l != 0.0f) {
            x /= l;
            y /= l;
            z /= l;
        }
    }

    Vec3 Vec3::Normalized() const {
        float l = Length();
        return (l == 0.0f) ? Vec3(0.0f) : Vec3(x / l, y / l, z / l);
    }

    Vec3 Vec3::Zero() {
        return {0.0f};
    }

    // Vec4
    Vec4::Vec4() = default;

    Vec4::Vec4(float S) : x(S), y(S), z(S), w(S) {
    }

    Vec4::Vec4(float X, float Y, float Z, float W) : x(X), y(Y), z(Z), w(W) {
    }

    Vec4::Vec4(const Vec3& V, float W) : x(V.x), y(V.y), z(V.z), w(W) {
    }

    Vec4 Vec4::operator+(const Vec4 &Rhs) const {
        return {x + Rhs.x, y + Rhs.y, z + Rhs.z, w + Rhs.w};
    }
    Vec4 Vec4::operator-(const Vec4 &Rhs) const {
        return {x - Rhs.x, y - Rhs.y, z - Rhs.z, w - Rhs.w};
    }
    Vec4 Vec4::operator*(float S) const {
        return {x * S, y * S, z * S, w * S};
    }
    Vec4 Vec4::operator/(float S) const {
        return {x / S, y / S, z / S, w / S};
    }

    Vec4 &Vec4::operator+=(const Vec4 &Rhs) {
        x += Rhs.x;
        y += Rhs.y;
        z += Rhs.z;
        w += Rhs.w;
        return *this;
    }

    Vec4 &Vec4::operator-=(const Vec4 &Rhs) {
        x -= Rhs.x;
        y -= Rhs.y;
        z -= Rhs.z;
        w -= Rhs.w;
        return *this;
    }

    Vec4 &Vec4::operator*=(float S) {
        x *= S;
        y *= S;
        z *= S;
        w *= S;
        return *this;
    }

    Vec4 &Vec4::operator/=(float S) {
        x /= S;
        y /= S;
        z /= S;
        w /= S;
        return *this;
    }

    Vec4 Vec4::operator-() const {
        return {-x, -y, -z, -w};
    }

    float Vec4::Dot(const Vec4 &Rhs) const {
        return x * Rhs.x + y * Rhs.y + z * Rhs.z + w * Rhs.w;
    }
    float Vec4::Length2() const {
        return Dot(*this);
    }
    float Vec4::Length() const {
        return std::sqrt(Length2());
    }

    void Vec4::Normalize() {
        float l = Length();
        if (l != 0.0f) {
            x /= l;
            y /= l;
            z /= l;
            w /= l;
        }
    }

    Vec4 Vec4::Normalized() const {
        float l = Length();
        return (l == 0.0f) ? Vec4(0.0f) : Vec4(x / l, y / l, z / l, w / l);
    }

    Vec4 Vec4::Zero() {
        return {0.0f};
    }

    // IVec2
    IVec2::IVec2() = default;

    IVec2::IVec2(int S) : x(S), y(S) {
    }

    IVec2::IVec2(int X, int Y) : x(X), y(Y) {
    }

    IVec2::IVec2(const Vec2 &V) : x(static_cast<int>(V.x)), y(static_cast<int>(V.y)) {
    }

    IVec2 IVec2::operator+(const IVec2 &Rhs) const {
        return {x + Rhs.x, y + Rhs.y};
    }
    IVec2 IVec2::operator-(const IVec2 &Rhs) const {
        return {x - Rhs.x, y - Rhs.y};
    }
    IVec2 IVec2::operator*(int S) const {
        return {x * S, y * S};
    }
    IVec2 IVec2::operator/(int S) const {
        return {x / S, y / S};
    }

    IVec2 &IVec2::operator+=(const IVec2 &Rhs) {
        x += Rhs.x;
        y += Rhs.y;
        return *this;
    }

    IVec2 &IVec2::operator-=(const IVec2 &Rhs) {
        x -= Rhs.x;
        y -= Rhs.y;
        return *this;
    }

    IVec2 &IVec2::operator*=(int S) {
        x *= S;
        y *= S;
        return *this;
    }

    IVec2 &IVec2::operator/=(int S) {
        x /= S;
        y /= S;
        return *this;
    }

    IVec2 IVec2::Zero() {
        return {0};
    }

    // IVec3
    IVec3::IVec3() = default;

    IVec3::IVec3(int S) : x(S), y(S), z(S) {
    }

    IVec3::IVec3(int X, int Y, int Z) : x(X), y(Y), z(Z) {
    }

    IVec3::IVec3(const Vec3 &V) : x(static_cast<int>(V.x)), y(static_cast<int>(V.y)), z(static_cast<int>(V.z)) {
    }

    IVec3 IVec3::operator+(const IVec3 &Rhs) const {
        return {x + Rhs.x, y + Rhs.y, z + Rhs.z};
    }

    IVec3 IVec3::operator-(const IVec3 &Rhs) const {
        return {x - Rhs.x, y - Rhs.y, z - Rhs.z};
    }

    IVec3 IVec3::operator*(int S) const {
        return {x * S, y * S, z * S};
    }

    IVec3 IVec3::operator/(int S) const {
        return {x / S, y / S, z / S};
    }

    IVec3 &IVec3::operator+=(const IVec3 &Rhs) {
        x += Rhs.x;
        y += Rhs.y;
        z += Rhs.z;
        return *this;
    }

    IVec3 &IVec3::operator-=(const IVec3 &Rhs) {
        x -= Rhs.x;
        y -= Rhs.y;
        z -= Rhs.z;
        return *this;
    }

    IVec3 &IVec3::operator*=(int S) {
        x *= S;
        y *= S;
        z *= S;
        return *this;
    }

    IVec3 &IVec3::operator/=(int S) {
        x /= S;
        y /= S;
        z /= S;
        return *this;
    }

    IVec3 IVec3::Zero() {
        return {0};
    }

    // IVec4
    IVec4::IVec4() = default;

    IVec4::IVec4(int S) : x(S), y(S), z(S), w(S) {
    }

    IVec4::IVec4(int X, int Y, int Z, int W) : x(X), y(Y), z(Z), w(W) {
    }

    IVec4::IVec4(const Vec4 &V) : x(static_cast<int>(V.x)), y(static_cast<int>(V.y)), z(static_cast<int>(V.z)),
                                  w(static_cast<int>(V.w)) {
    }

    IVec4 IVec4::operator+(const IVec4 &Rhs) const {
        return {x + Rhs.x, y + Rhs.y, z + Rhs.z, w + Rhs.w};
    }

    IVec4 IVec4::operator-(const IVec4 &Rhs) const {
        return {x - Rhs.x, y - Rhs.y, z - Rhs.z, w - Rhs.w};
    }

    IVec4 IVec4::operator*(int S) const {
        return {x * S, y * S, z * S, w * S};
    }

    IVec4 IVec4::operator/(int S) const {
        return {x / S, y / S, z / S, w / S};
    }

    IVec4 &IVec4::operator+=(const IVec4 &Rhs) {
        x += Rhs.x;
        y += Rhs.y;
        z += Rhs.z;
        w += Rhs.w;
        return *this;
    }

    IVec4 &IVec4::operator-=(const IVec4 &Rhs) {
        x -= Rhs.x;
        y -= Rhs.y;
        z -= Rhs.z;
        w -= Rhs.w;
        return *this;
    }

    IVec4 &IVec4::operator*=(int S) {
        x *= S;
        y *= S;
        z *= S;
        w *= S;
        return *this;
    }

    IVec4 &IVec4::operator/=(int S) {
        x /= S;
        y /= S;
        z /= S;
        w /= S;
        return *this;
    }

    IVec4 IVec4::Zero() {
        return {0};
    }
} // namespace GPU::Math
