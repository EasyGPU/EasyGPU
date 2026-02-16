// Test file for compound assignment operators with Expr (scalar only)
#include <GPU.h>

// Test function to verify compound assignment operators compile correctly
void testCompoundAssignment() {
    using namespace GPU;
    using namespace GPU::IR::Value;
    
    // Test scalar types with Expr
    {
        Var<float> f = 1.0f;
        Expr<float> f_expr = 2.0f;
        f += f_expr;  // Var<float> += Expr<float>
        f -= f_expr;  // Var<float> -= Expr<float>
        f *= f_expr;  // Var<float> *= Expr<float>
        f /= f_expr;  // Var<float> /= Expr<float>
    }
    
    {
        Var<int> i = 1;
        Expr<int> i_expr = 2;
        i += i_expr;  // Var<int> += Expr<int>
        i -= i_expr;  // Var<int> -= Expr<int>
        i *= i_expr;  // Var<int> *= Expr<int>
        i /= i_expr;  // Var<int> /= Expr<int>
    }
    
    // Test Vector types with scalar Expr<float>
    {
        Expr<float> f_expr = 2.0f;
        
        Var<Math::Vec2> v2 = Math::Vec2(1.0f, 2.0f);
        v2 *= f_expr;  // Var<Vec2> *= Expr<float>
        v2 /= f_expr;  // Var<Vec2> /= Expr<float>
        
        Var<Math::Vec3> v3 = Math::Vec3(1.0f, 2.0f, 3.0f);
        v3 *= f_expr;  // Var<Vec3> *= Expr<float>
        v3 /= f_expr;  // Var<Vec3> /= Expr<float>
        
        Var<Math::Vec4> v4 = Math::Vec4(1.0f, 2.0f, 3.0f, 4.0f);
        v4 *= f_expr;  // Var<Vec4> *= Expr<float>
        v4 /= f_expr;  // Var<Vec4> /= Expr<float>
    }
    
    // Test IVector types with scalar Expr<int>
    {
        Expr<int> i_expr = 2;
        
        Var<Math::IVec2> iv2 = Math::IVec2(1, 2);
        iv2 *= i_expr;  // Var<IVec2> *= Expr<int>
        iv2 /= i_expr;  // Var<IVec2> /= Expr<int>
        
        Var<Math::IVec3> iv3 = Math::IVec3(1, 2, 3);
        iv3 *= i_expr;  // Var<IVec3> *= Expr<int>
        iv3 /= i_expr;  // Var<IVec3> /= Expr<int>
        
        Var<Math::IVec4> iv4 = Math::IVec4(1, 2, 3, 4);
        iv4 *= i_expr;  // Var<IVec4> *= Expr<int>
        iv4 /= i_expr;  // Var<IVec4> /= Expr<int>
    }
    
    // Test Matrix types with scalar Expr<float>
    {
        Expr<float> f_expr = 2.0f;
        
        Var<Math::Mat2> m2;
        m2 *= f_expr;  // Var<Mat2> *= Expr<float>
        m2 /= f_expr;  // Var<Mat2> /= Expr<float>
        
        Var<Math::Mat3> m3;
        m3 *= f_expr;  // Var<Mat3> *= Expr<float>
        m3 /= f_expr;  // Var<Mat3> /= Expr<float>
        
        Var<Math::Mat4> m4;
        m4 *= f_expr;  // Var<Mat4> *= Expr<float>
        m4 /= f_expr;  // Var<Mat4> /= Expr<float>
        
        Var<Math::Mat2x3> m2x3;
        m2x3 *= f_expr;  // Var<Mat2x3> *= Expr<float>
        m2x3 /= f_expr;  // Var<Mat2x3> /= Expr<float>
        
        Var<Math::Mat2x4> m2x4;
        m2x4 *= f_expr;  // Var<Mat2x4> *= Expr<float>
        m2x4 /= f_expr;  // Var<Mat2x4> /= Expr<float>
        
        Var<Math::Mat3x2> m3x2;
        m3x2 *= f_expr;  // Var<Mat3x2> *= Expr<float>
        m3x2 /= f_expr;  // Var<Mat3x2> /= Expr<float>
        
        Var<Math::Mat3x4> m3x4;
        m3x4 *= f_expr;  // Var<Mat3x4> *= Expr<float>
        m3x4 /= f_expr;  // Var<Mat3x4> /= Expr<float>
        
        Var<Math::Mat4x2> m4x2;
        m4x2 *= f_expr;  // Var<Mat4x2> *= Expr<float>
        m4x2 /= f_expr;  // Var<Mat4x2> /= Expr<float>
        
        Var<Math::Mat4x3> m4x3;
        m4x3 *= f_expr;  // Var<Mat4x3> *= Expr<float>
        m4x3 /= f_expr;  // Var<Mat4x3> /= Expr<float>
    }
}

int main() {
    // This test file is only for compilation verification
    return 0;
}
