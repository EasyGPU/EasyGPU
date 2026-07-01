/**
 * @file Context.cpp
 * @brief GPU context implementation with backend abstraction.
 */

#include <Runtime/Context.h>

#include <Backend/Backend.h>
#include <Backend/OpenGLBackend.h>
#ifdef EASYGPU_BACKEND_VULKAN
#include <Backend/VulkanBackend.h>
#endif

#include <sstream>

namespace GPU::Runtime {

Context					&Context::GetInstance() {
	static Context *instance = new Context();
	return *instance;
}

Backend::Backend *Context::GetBackend() {
	return GetInstance()._backend.get();
}

Context::~Context() {
	DestroyBackend();
}

void Context::Initialize() {
	std::lock_guard<std::mutex> lock(_mutex);
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
	std::lock_guard<std::mutex> lock(_mutex);
	return _initialized;
}

void Context::AbandonBackendForProcessExit() noexcept {
	std::lock_guard<std::mutex> lock(_mutex);
	(void)_backend.release();
	_initialized = false;
}

void Context::MakeCurrent() {
	std::lock_guard<std::mutex> lock(_mutex);
	if (!_initialized || !_backend) {
		throw std::runtime_error("Context not initialized");
	}
	_backend->MakeCurrent();
}

void Context::MakeNoneCurrent() {
	std::lock_guard<std::mutex> lock(_mutex);
	if (_backend) {
		_backend->MakeNoneCurrent();
	}
}

std::string Context::GetVersionString() const {
	std::lock_guard<std::mutex> lock(_mutex);
	if (!_initialized || !_backend) {
		return "Not initialized";
	}
	return _backend->GetCaps().versionString;
}

bool Context::HasComputeShadersSupport() const {
	std::lock_guard<std::mutex> lock(_mutex);
	if (!_initialized || !_backend) {
		return false;
	}
	return _backend->GetCaps().supportsComputeShaders;
}

void Context::GetMaxWorkGroupSize(int &x, int &y, int &z) const {
	std::lock_guard<std::mutex> lock(_mutex);
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
	std::lock_guard<std::mutex> lock(_mutex);
	if (!_initialized || !_backend) {
		return nullptr;
	}
	// Return the native handle from the backend
	// This is a specific feature of the OpenGL backend
	return _backend->GetNativeHandle();
}

Backend::BackendType Context::GetBackendType() const {
	std::lock_guard<std::mutex> lock(_mutex);
	if (!_initialized || !_backend) {
		return Backend::GetDefaultBackendType();
	}
	return _backend->GetType();
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
	if (_backend) {
		try {
			_backend->Shutdown();
		} catch (...) {
			// Context destruction can run during process teardown; keep it noexcept.
		}
	}
	_backend.reset();
	_initialized = false;
}

} // namespace GPU::Runtime
