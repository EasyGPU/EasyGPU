/**
 * @file Context.cpp
 * @brief GPU context implementation with backend abstraction
 */

#include <Runtime/Context.h>

#include <Backend/Backend.h>
#include <Backend/OpenGLBackend.h>
#ifdef EASYGPU_BACKEND_VULKAN
#include <Backend/VulkanBackend.h>
#endif

#include <sstream>

namespace GPU::Runtime {

// Static members
Context *Context::_instance	 = nullptr;
bool	 Context::_destroyed = false;

Context &Context::GetInstance() {
	if (_instance == nullptr) {
		if (_destroyed) {
			throw std::runtime_error("Context was destroyed and cannot be recreated");
		}
		_instance = new Context();
	}
	// Auto-initialize on first access
	if (!_instance->_initialized) {
		_instance->Initialize();
	}
	return *_instance;
}

Backend::Backend *Context::GetBackend() {
	return GetInstance()._backend.get();
}

Context::~Context() {
	DestroyBackend();
	_destroyed = true;
	_instance  = nullptr;
}

void Context::Initialize() {
	if (_initialized) {
		return;
	}

	try {
		CreateBackend();
		_initialized = true;
	} catch (const std::exception &e) {
		DestroyBackend();
		throw std::runtime_error(std::string("Failed to initialize GPU context: ") + e.what());
	}
}

bool Context::IsInitialized() const {
	return _initialized;
}

void Context::MakeCurrent() {
	if (!_initialized || !_backend) {
		throw std::runtime_error("Context not initialized");
	}
	_backend->MakeCurrent();
}

void Context::MakeNoneCurrent() {
	if (_backend) {
		_backend->MakeNoneCurrent();
	}
}

std::string Context::GetVersionString() const {
	if (!_initialized || !_backend) {
		return "Not initialized";
	}
	return _backend->GetCaps().versionString;
}

bool Context::HasComputeShadersSupport() const {
	if (!_initialized || !_backend) {
		return false;
	}
	return _backend->GetCaps().supportsComputeShaders;
}

void Context::GetMaxWorkGroupSize(int &x, int &y, int &z) const {
	if (!_initialized || !_backend) {
		x = y = z = 0;
		return;
	}
	auto caps = _backend->GetCaps();
	x		  = static_cast<int>(caps.maxWorkGroupSizeX);
	y		  = static_cast<int>(caps.maxWorkGroupSizeY);
	z		  = static_cast<int>(caps.maxWorkGroupSizeZ);
}

void *Context::GetNativeGLContext() const {
	if (!_initialized || !_backend) {
		return nullptr;
	}
	// Return the native handle from the backend
	// This is a specific feature of the OpenGL backend
	return _backend->GetNativeHandle();
}

Backend::BackendType Context::GetBackendType() const {
	if (!_initialized || !_backend) {
		return Backend::GetDefaultBackendType();
	}
	// Determine backend type from the backend pointer
	// This is a bit hacky but works for now
	if (dynamic_cast<Backend::OpenGLBackend *>(_backend.get()) != nullptr) {
		return Backend::BackendType::OpenGL;
	}
#ifdef EASYGPU_BACKEND_VULKAN
	if (dynamic_cast<Backend::VulkanBackend *>(_backend.get()) != nullptr) {
		return Backend::BackendType::Vulkan;
	}
#endif
	return Backend::GetDefaultBackendType();
}

void Context::CreateBackend() {
	// Create the default backend (OpenGL for now)
	Backend::BackendType backendType = Backend::GetDefaultBackendType();

	_backend.reset(Backend::CreateBackend(backendType));
	if (!_backend) {
		throw std::runtime_error("Failed to create backend");
	}

	_backend->Initialize();
}

void Context::DestroyBackend() {
	_backend.reset();
}

} // namespace GPU::Runtime
