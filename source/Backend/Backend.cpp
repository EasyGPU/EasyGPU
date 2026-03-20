/**
 * @file Backend.cpp
 * @brief Backend factory implementation
 */

#include <Backend/Backend.h>

#include <stdexcept>

namespace GPU::Backend {

// Forward declarations for backend implementations
Backend* CreateOpenGLBackend();

Backend* CreateBackend(BackendType type) {
    switch (type) {
        case BackendType::OpenGL:
            return CreateOpenGLBackend();
        case BackendType::Vulkan:
        case BackendType::DirectX12:
        case BackendType::Metal:
            // TODO: Implement other backends
            throw std::runtime_error("Backend type not yet implemented");
        default:
            throw std::runtime_error("Unknown backend type");
    }
}

void DestroyBackend(Backend* backend) {
    if (backend) {
        backend->Shutdown();
        delete backend;
    }
}

BackendType GetDefaultBackendType() {
    // Default to OpenGL for now
    // TODO: Detect platform capabilities and choose best backend
    return BackendType::OpenGL;
}

} // namespace GPU::Backend
