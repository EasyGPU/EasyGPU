/**
 * @file Backend.cpp
 * @brief Backend factory implementation.
 */

#include <Backend/Backend.h>

#include <stdexcept>

namespace GPU::Backend {

// Forward declarations for backend implementations - conditionally compiled
#if defined(EASYGPU_BACKEND_OPENGL) || !defined(EASYGPU_BACKEND_VULKAN)
Backend *CreateOpenGLBackend();
#endif
#if defined(EASYGPU_BACKEND_VULKAN)
Backend *CreateVulkanBackend();
#endif

Backend *CreateBackend(BackendType type) {
	switch (type) {
	case BackendType::OpenGL:
#if defined(EASYGPU_BACKEND_OPENGL) || !defined(EASYGPU_BACKEND_VULKAN)
		return CreateOpenGLBackend();
#else
		throw std::runtime_error("OpenGL backend not available in this build");
#endif
	case BackendType::Vulkan:
#if defined(EASYGPU_BACKEND_VULKAN)
		return CreateVulkanBackend();
#else
		throw std::runtime_error("Vulkan backend not available in this build");
#endif
	case BackendType::DirectX12:
	case BackendType::Metal:
		// TODO: Implement other backends
		throw std::runtime_error("Backend type not yet implemented");
	default:
		throw std::runtime_error("Unknown backend type");
	}
}

void DestroyBackend(Backend *backend) {
	if (backend) {
		backend->Shutdown();
		delete backend;
	}
}

BackendType GetDefaultBackendType() {
// Select backend based on compile-time definition
#if defined(EASYGPU_BACKEND_VULKAN)
	return BackendType::Vulkan;
#else
	// Default to OpenGL
	return BackendType::OpenGL;
#endif
}

} // namespace GPU::Backend
