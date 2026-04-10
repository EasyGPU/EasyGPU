#pragma once

/**
 * @file BufferSlot.h
 * @brief Dynamic buffer slot for runtime resource switching with backend support
 */
#ifndef EASYGPU_BUFFERSLOT_H
#define EASYGPU_BUFFERSLOT_H

#include <Backend/Backend.h>
#include <IR/Builder/Builder.h>
#include <IR/Value/BufferRef.h>
#include <Runtime/Buffer.h>

#include <stdexcept>
#include <string>

namespace GPU::Runtime {

// Forward declaration for friend access
class KernelBuildContext;

/**
 * Buffer slot base class (non-template)
 * Used for type erasure in KernelBuildContext
 */
class BufferSlotBase {
public:
	virtual ~BufferSlotBase()						  = default;

	/**
	 * Get the backend buffer handle of the attached buffer
	 * @return The backend buffer handle, or INVALID_BUFFER_HANDLE if not attached
	 */
	virtual Backend::BufferHandle GetHandle() const	  = 0;

	/**
	 * Check if a buffer is currently attached
	 * @return true if attached, false otherwise
	 */
	virtual bool				  IsAttached() const  = 0;

	/**
	 * Get the GLSL type name for this buffer's element type
	 * @return The GLSL type name (e.g., "float", "vec4")
	 */
	virtual const char			 *GetTypeName() const = 0;

	/**
	 * Get the binding slot assigned by KernelBuildContext
	 * @return The binding slot index, or -1 if not bound
	 */
	int							  GetBinding() const {
		  return _binding;
	}

	/**
	 * Get the variable name in GLSL
	 * @return The GLSL variable name
	 */
	const std::string &GetName() const {
		return _name;
	}

	/**
	 * Set the binding information (called by KernelBuildContext)
	 * @param binding The binding slot
	 * @param name The GLSL variable name
	 */
	void SetBindingInfo(int binding, const std::string &name) {
		_binding = binding;
		_name	 = name;
	}

	/**
	 * Set the buffer access mode
	 * @param mode The buffer mode
	 */
	void SetMode(int mode) {
		_mode = mode;
	}

	/**
	 * Get the buffer access mode
	 * @return The buffer mode
	 */
	int GetMode() const {
		return _mode;
	}

protected:
	int			_binding = -1;	// Assigned by KernelBuildContext during Bind()
	std::string _name;			// GLSL variable name
	int			_mode = 0x88BA; // READ_WRITE equivalent (default)
};

/**
 * Buffer slot for dynamic buffer switching at runtime
 * @tparam T The element type of the buffer
 */
template <typename T> class BufferSlot : public BufferSlotBase {
public:
	/**
	 * Default constructor - creates an unattached slot
	 */
	BufferSlot()								  = default;

	/**
	 * Destructor
	 */
	~BufferSlot() override						  = default;

	// Disable copy
	BufferSlot(const BufferSlot &)				  = delete;
	BufferSlot &operator=(const BufferSlot &)	  = delete;

	// Enable move
	BufferSlot(BufferSlot &&) noexcept			  = default;
	BufferSlot &operator=(BufferSlot &&) noexcept = default;

public:
	// ===================================================================
	// Runtime API - Called outside kernel definition
	// ===================================================================

	/**
	 * Attach a buffer to this slot
	 * @param buffer The buffer to attach
	 */
	void Attach(Buffer<T> &buffer) {
		_bufferHandle = buffer.GetHandle();
		_mode		  = static_cast<int>(buffer.GetMode());
	}

	/**
	 * Detach the current buffer
	 */
	void Detach() {
		_bufferHandle = Backend::INVALID_BUFFER_HANDLE;
	}

	/**
	 * Check if a buffer is currently attached
	 * @return true if attached, false otherwise
	 */
	bool IsAttached() const override {
		return _bufferHandle != Backend::INVALID_BUFFER_HANDLE;
	}

	/**
	 * Get the currently attached buffer
	 * @return Pointer to the attached buffer, or nullptr if not attached
	 */
	Buffer<T> *GetAttached() const {
		return nullptr;
	}

	/**
	 * Get the backend buffer handle of the attached buffer
	 * @return The backend buffer handle, or INVALID_BUFFER_HANDLE if not attached
	 */
	Backend::BufferHandle GetHandle() const override {
		return _bufferHandle;
	}

public:
	// ===================================================================
	// DSL API - Called inside kernel definition
	// ===================================================================

	/**
	 * Bind this slot to the current kernel being defined
	 * @return BufferRef<T> for DSL access
	 */
	[[nodiscard]] IR::Value::BufferRef<T> Bind() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("BufferSlot::Bind() called outside of Kernel definition");
		}

		// Register this slot with the context
		context->RegisterBufferSlot(this);

		// Return BufferRef using our assigned name and binding
		return IR::Value::BufferRef<T>(_name, static_cast<uint32_t>(_binding));
	}

public:
	/**
	 * Get the GLSL type name for this buffer's element type
	 * @return The GLSL type name
	 */
	const char *GetTypeName() const override {
		return GetGLSLTypeName<T>();
	}

protected:
	/**
	 * Helper to get GLSL type name
	 */
	template <typename Type> static const char *GetGLSLTypeName() {
		if constexpr (std::same_as<Type, float>)
			return "float";
		else if constexpr (std::same_as<Type, int>)
			return "int";
		else if constexpr (std::same_as<Type, bool>)
			return "bool";
		else if constexpr (std::same_as<Type, Math::Vec2>)
			return "vec2";
		else if constexpr (std::same_as<Type, Math::Vec3>)
			return "vec3";
		else if constexpr (std::same_as<Type, Math::Vec4>)
			return "vec4";
		else if constexpr (std::same_as<Type, Math::IVec2>)
			return "ivec2";
		else if constexpr (std::same_as<Type, Math::IVec3>)
			return "ivec3";
		else if constexpr (std::same_as<Type, Math::IVec4>)
			return "ivec4";
		else if constexpr (std::same_as<Type, Math::Mat2>)
			return "mat2";
		else if constexpr (std::same_as<Type, Math::Mat3>)
			return "mat3";
		else if constexpr (std::same_as<Type, Math::Mat4>)
			return "mat4";
		else
			return "unknown";
	}

private:
	Backend::BufferHandle _bufferHandle = Backend::INVALID_BUFFER_HANDLE; // Currently attached buffer handle

	// Grant KernelBuildContext access to protected members
	friend class KernelBuildContext;
};

} // namespace GPU::Runtime

#endif // EASYGPU_BUFFERSLOT_H
