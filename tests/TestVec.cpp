#include <iostream>
#include <cmath>

#include <Utility/Vec.h>
#include <Utility/Matrix.h>

using namespace GPU::Math;

static int g_pass = 0;
static int g_fail = 0;

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

static void expect(bool cond, const char *what) {
    if (cond) { ++g_pass; std::cout << "PASS: " << what << "\n"; }
    else { ++g_fail; std::cout << "FAIL: " << what << "\n"; }
}

static void expectVec2(const Vec2 &a, const Vec2 &b, const char *what) {
    expect(approx(a.x,b.x) && approx(a.y,b.y), what);
}

static void expectVec3(const Vec3 &a, const Vec3 &b, const char *what) {
    expect(approx(a.x,b.x) && approx(a.y,b.y) && approx(a.z,b.z), what);
}

static void expectVec4(const Vec4 &a, const Vec4 &b, const char *what) {
    expect(approx(a.x,b.x) && approx(a.y,b.y) && approx(a.z,b.z) && approx(a.w,b.w), what);
}

int main() {
    // Vec2 tests
    Vec2 v2a; // default ctor
    expectVec2(v2a, Vec2::Zero(), "Vec2 default == zero");

    Vec2 v2b(2.0f); // all components set
    expectVec2(v2b, Vec2(2.0f,2.0f), "Vec2 scalar ctor");

    Vec2 v2c(1.0f, 3.0f);
    expectVec2(v2c + Vec2(2.0f, -1.0f), Vec2(3.0f,2.0f), "Vec2 add/sub ops");
    expectVec2(v2c - Vec2(0.5f,1.0f), Vec2(0.5f,2.0f), "Vec2 subtract");
    expect(approx(v2c.Dot(Vec2(2.0f,0.0f)), 2.0f), "Vec2 dot");
    expect(approx(v2c.Length2(), 10.0f), "Vec2 length squared");

    Vec2 n2 = v2c.Normalized();
    expect(approx(n2.Length(), 1.0f), "Vec2 normalized length == 1");

    // Vec3 tests
    Vec3 a3(1.0f,2.0f,3.0f);
    Vec3 b3(2.0f,3.0f,4.0f);
    expect(approx(a3.Dot(b3), 20.0f), "Vec3 dot");
    Vec3 c3 = a3.Cross(b3);
    // cross of (1,2,3)x(2,3,4) = (2*4-3*3, 3*2-1*4, 1*3-2*2) = (8-9,6-4,3-4)=(-1,2,-1)
    expectVec3(c3, Vec3(-1.0f,2.0f,-1.0f), "Vec3 cross");

    // Vec4 tests
    Vec4 a4(1,2,3,4);
    expect(approx(a4.Dot(a4), 30.0f), "Vec4 dot");

    // Mat2 tests
    Mat2 m2(1,2,3,4); // columns: c0=(1,2), c1=(3,4)
    Vec2 e0(1,0), e1(0,1);
    expectVec2(m2*e0, Vec2(1,2), "Mat2 * e0 == col0");
    expectVec2(m2*e1, Vec2(3,4), "Mat2 * e1 == col1");

    Mat2 t2 = m2.Transposed();
    // t2 * e0 should equal first row of m2 = (m00,m01) = (1,3)
    expectVec2(t2*e0, Vec2(1,3), "Mat2 Transposed column0 == row0");

    expect(approx(m2.Determinant(), 1.0f*4.0f - 3.0f*2.0f), "Mat2 determinant");

    // inverse round-trip: m * m^{-1} -> identity on basis
    Mat2 inv2 = m2.Inverse();
    // multiplication Mat*Mat is not defined; verify inverse by basis-roundtrip
    Vec2 r0 = m2 * (inv2 * e0);
    Vec2 r1 = m2 * (inv2 * e1);
    expectVec2(r0, e0, "Mat2 inverse round-trip e0");
    expectVec2(r1, e1, "Mat2 inverse round-trip e1");

    // Mat3 tests
    Mat3 m3(1,2,3, 4,5,6, 7,8,9); // columns: c0=(1,2,3), c1=(4,5,6), c2=(7,8,9)
    expectVec3(m3*Vec3(1,0,0), Vec3(1,2,3), "Mat3 * e0 == col0");
    Mat3 t3 = m3.Transposed();
    expectVec3(t3*Vec3(1,0,0), Vec3(1,4,7), "Mat3 Transposed column0 == row0");

    // Singular determinant for this m3 (rows linearly dependent), expect inverse to throw
    bool threw = false;
    try { Mat3 inv3 = m3.Inverse(); (void)inv3; } catch(...) { threw = true; }
    expect(threw, "Mat3 inverse throws on singular matrix");

    // Mat4 tests
    Mat4 m4 = Mat4::Identity();
    Vec4 v4(1,2,3,1);
    expectVec4(m4 * v4, v4, "Mat4 identity * v == v");

    // Rectangular matrix tests
    Mat2x3 m2x3(Vec3(1,0,0), Vec3(0,1,0));
    Vec3 r23 = m2x3 * Vec2(3,4); // should be 3*col0 + 4*col1 = (3,4,0)
    expectVec3(r23, Vec3(3,4,0), "Mat2x3 * Vec2 combination");

    // Transpose twice returns original (for rectangular types)
    Mat3x2 t32 = m2x3.Transposed();
    Mat2x3 back = t32.Transposed();
    // compare by applying to basis vectors
    expectVec3(back * Vec2(1,0), m2x3 * Vec2(1,0), "Mat2x3 transpose twice e0");
    expectVec3(back * Vec2(0,1), m2x3 * Vec2(0,1), "Mat2x3 transpose twice e1");

    // Mat2x4 / Mat4x2 transpose relationship
    Mat2x4 a2x4(Vec4(1,0,0,0), Vec4(0,1,0,0));
    Mat4x2 t4x2 = a2x4.Transposed();
    Mat2x4 back2 = t4x2.Transposed();
    expectVec4(back2 * Vec2(2,3), a2x4 * Vec2(2,3), "Mat2x4 transpose twice correctness");

    // Mat3x4 / Mat4x3 transpose relationship
    Mat3x4 a3x4(Vec4(1,0,0,0), Vec4(0,1,0,0), Vec4(0,0,1,0));
    Mat4x3 t4x3 = a3x4.Transposed();
    Mat3x4 back3 = t4x3.Transposed();
    expectVec4(back3 * Vec3(5,6,7), a3x4 * Vec3(5,6,7), "Mat3x4 transpose twice correctness");

    // Summary
    std::cout << "\nTest summary: " << g_pass << " passed, " << g_fail << " failed.\n";
    return (g_fail == 0) ? 0 : 1;
}
