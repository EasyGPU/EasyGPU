// Test file for void return statement
#include <GPU.h>

// Test function to verify void return compiles correctly
void testVoidReturn() {
    using namespace GPU;
    using namespace GPU::IR::Value;
    using namespace GPU::Flow;
    
    // Test void return in a void callable function
    // Note: This is just compilation test, actual usage would be within Callable
    
    // Test 1: Simple void return
    Return();
    
    // Test 2: Return with expression (existing functionality)
    Var<float> f = 1.0f;
    Return(f);
    
    Var<int> i = 1;
    Return(i);
    
    // Test 3: Return with literal
    Return(3.14f);
    Return(42);
    
    // Test 4: Return with expression
    Expr<float> expr = 2.0f;
    Return(expr);
}

int main() {
    // This test file is only for compilation verification
    return 0;
}
