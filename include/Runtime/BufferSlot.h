#pragma once

/**
 * BufferSlot.h:
 *      @Descripiton    :   Dynamic buffer slot for runtime resource switching
 *      @Author         :   Margoo(qiuzhengyu@sigraph.org)
 *      @Date           :   3/6/2026
 *
 * This class allows switching buffer bindings at runtime without recompiling kernels.
 *
 * Usage:
 *   BufferSlot<float> inputSlot;  // Declare slot (not attached to any buffer yet)
 *
 *   Kernel1D kernel([&](Int i) {
 *       auto buf = inputSlot.Bind();  // Bind the slot (not specific buffer)
 *       out[i] = buf[i] * 2.0f;
 *   });
 *
 *   Buffer<float> buf1(data1);
 *   Buffer<float> buf2(data2);
 *
 *   inputSlot.Attach(buf1);       // Attach buf1
 *   kernel.Dispatch(16, true);    // Execute with buf1
 *
 *   inputSlot.Attach(buf2);       // Switch to buf2 (no recompilation!)
 *   kernel.Dispatch(16, true);    // Execute with buf2
 */
#ifndef EASYGPU_BUFFERSLOT_H
#define EASYGPU_BUFFERSLOT_H

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
	virtual ~BufferSlotBase()				= default;

	/**
	 * Get the OpenGL buffer handle of the attached buffer
	 * @return The OpenGL buffer ID, or 0 if not attached
	 */
	virtual uint32_t	GetHandle() const	= 0;

	/**
	 * Check if a buffer is currently attached
	 * @return true if attached, false otherwise
	 */
	virtual bool		IsAttached() const	= 0;

	/**
	 * Get the GLSL type name for this buffer's element type
	 * @return The GLSL type name (e.g., "float", "vec4")
	 */
	virtual const char *GetTypeName() const = 0;

	/**
	 * Get the binding slot assigned by KernelBuildContext
	 * @return The binding slot index, or -1 if not bound
	 */
	int					GetBinding() const {
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
	 * @param mode The GL buffer mode
	 */
	void SetMode(int mode) {
		_mode = mode;
	}

protected:
	int			_binding = -1;	// Assigned by KernelBuildContext during Bind()
	std::string _name;			// GLSL variable name
	int			_mode = 0x88BA; // GL_READ_WRITE (default)
};

/**
 * Buffer slot for dynamic buffer switching at runtime
 * @tparam T The element type of the buffer
 *
 * This class allows you to define a kernel once and switch between different
 * buffers at runtime without recompiling the kernel.
 *
 * Example - Ping-pong rendering:
 *   BufferSlot<Vec4> readSlot;
 *   BufferSlot<Vec4> writeSlot;
 *
 *   Kernel2D iterate([&](Int x, Int y) {
 *       auto src = readSlot.Bind();
 *       auto dst = writeSlot.Bind();
 *       // ... compute ...
 *   });
 *
 *   Buffer<Vec4> bufA(width * height);
 *   Buffer<Vec4> bufB(width * height);
 *
 *   // Iteration 1: A -> B
 *   readSlot.Attach(bufA);
 *   writeSlot.Attach(bufB);
 *   iterate.Dispatch(groupsX, groupsY, true);
 *
 *   // Iteration 2: B -> A (same kernel, no recompilation)
 *   readSlot.Attach(bufB);
 *   writeSlot.Attach(bufA);
 *   iterate.Dispatch(groupsX, groupsY, true);
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
	 * The buffer will be used when the kernel is next dispatched.
	 * @param buffer The buffer to attach
	 */
	void Attach(Buffer<T> &buffer) {
		_buffer = &buffer;
		_mode	= GetGLBufferMode(buffer.GetMode());
	}

	/**
	 * Detach the current buffer
	 */
	void Detach() {
		_buffer = nullptr;
	}

	/**
	 * Check if a buffer is currently attached
	 * @return true if attached, false otherwise
	 */
	bool IsAttached() const override {
		return _buffer != nullptr;
	}

	/**
	 * Get the currently attached buffer
	 * @return Pointer to the attached buffer, or nullptr if not attached
	 */
	Buffer<T> *GetAttached() const {
		return _buffer;
	}

	/**
	 * Get the OpenGL buffer handle of the attached buffer
	 * @return The OpenGL buffer ID, or 0 if not attached
	 */
	uint32_t GetHandle() const override {
		return _buffer ? _buffer->GetHandle() : 0;
	}

public:
	// ===================================================================
	// DSL API - Called inside kernel definition
	// ===================================================================

	/**
	 * Bind this slot to the current kernel being defined
	 * This allocates a binding slot and returns a BufferRef for DSL access.
	 * The actual buffer binding happens at dispatch time.
	 * @return BufferRef<T> for DSL access
	 */
	[[nodiscard]] IR::Value::BufferRef<T> Bind() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("BufferSlot::Bind() called outside of Kernel definition");
		}

		// Register this slot with the context
		// The actual buffer binding happens at dispatch time via GetBufferSlots()
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

	/**
	 * Get GL buffer mode from BufferMode
	 */
	static int GetGLBufferMode(BufferMode mode) {
		switch (mode) {
		case BufferMode::Read:
			return 0x88B8; // GL_READ_ONLY
		case BufferMode::Write:
			return 0x88B9; // GL_WRITE_ONLY
		case BufferMode::ReadWrite:
			return 0x88BA; // GL_READ_WRITE
		}
		return 0x88BA;
	}

private:
	Buffer<T> *_buffer = nullptr; // Currently attached buffer

	// Grant KernelBuildContext access to protected members
	friend class KernelBuildContext;
};
} // namespace GPU::Runtime

#endif // EASYGPU_BUFFERSLOT_H
