/**
 * StructTest.cpp:
*      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/12/2026
 */
#include <iostream>
#include <fstream>
#include <sstream>

#include <Utility/Meta/StructMeta.h>
#include <Utility/Helpers.h>
#include <IR/Value/Var.h>
#include <Kernel/Kernel.h>

// Define leaf structures first (no struct dependencies)
EASYGPU_STRUCT(Material,
    (GPU::Math::Vec3, albedo),
    (float, roughness),
    (float, metallic)
);

EASYGPU_STRUCT(Particle,
    (GPU::Math::Vec3, position),
    (GPU::Math::Vec3, velocity),
    (float, life),
    (int, type)
);

// Define structures that depend on leaf structures
EASYGPU_STRUCT(RenderObject,
    (Particle, particle),
    (Material, material),
    (int, objectId)
);

int main() {
    std::cout << "Testing EASYGPU_STRUCT...\n";

    // Test 1: GLSL definition generation
    std::cout << "\n=== Generated GLSL Definitions ===\n";
    std::cout << GPU::Meta::StructMeta<Particle>::ExpandedDefinition();
    std::cout << GPU::Meta::StructMeta<Material>::ExpandedDefinition();
    std::cout << GPU::Meta::StructMeta<RenderObject>::ExpandedDefinition();

    // Test 2: Kernel with CPU struct capture
    std::cout << "\n=== Kernel Test with CPU Struct Capture ===\n";
    
    // Create CPU structure
    Particle cpuParticle;
    cpuParticle.position = GPU::Math::Vec3(10.0f, 20.0f, 30.0f);
    cpuParticle.velocity = GPU::Math::Vec3(1.0f, 2.0f, 3.0f);
    cpuParticle.life = 5.0f;
    cpuParticle.type = 2;
    
    GPU::Kernel::InspectorKernel kernel([&cpuParticle](GPU::IR::Value::Var<int>& id) {
        // Capture CPU struct into GPU
        GPU::IR::Value::Var<Particle> p = cpuParticle;
        
        // Modify on GPU
        p.position() = p.position() + p.velocity() * GPU::MakeFloat(0.016f);
        p.life() = p.life() - 0.01f;
        
        // Swizzle on vector members
        p.position().x() = p.position().x() + 1.0f;
        
        // Nested struct test with CPU capture
        Material cpuMaterial;
        cpuMaterial.albedo = GPU::Math::Vec3(1.0f, 0.5f, 0.0f);
        cpuMaterial.roughness = 0.3f;
        cpuMaterial.metallic = 0.8f;
        
        GPU::IR::Value::Var<RenderObject> obj;
        obj.particle().position() = p.position();
        obj.material() = cpuMaterial;  // Assignment from CPU struct
        obj.objectId() = 42;
        
        // Chain: nested struct -> vector -> swizzle
        obj.particle().velocity().y() = 0.0f;
        
        // Struct assignment
        GPU::IR::Value::Var<Particle> p2;
        p2 = p;
    });
    
    std::cout << "Dispatching kernel...\n";
    kernel.PrintCode();

    std::cout << "\nTest completed successfully!\n";
    return 0;
}
