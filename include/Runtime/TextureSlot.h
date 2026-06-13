#pragma once

/**
 * @file TextureSlot.h
 * @brief Dynamic texture slot for runtime resource switching with backend support.
 */

#ifndef EASYGPU_TEXTURESLOT_H
#define EASYGPU_TEXTURESLOT_H

#include <Backend/Backend.h>
#include <IR/Builder/Builder.h>
#include <IR/Value/TextureRef.h>
#include <IR/Value/TextureSampler.h>
#include <Runtime/PixelFormat.h>
#include <Runtime/Texture.h>

#include <stdexcept>
#include <string>

namespace GPU::Runtime {

// Forward declaration
template <PixelFormat Format> class Texture2D;
template <PixelFormat Format> class Texture3D;

// Forward declaration for friend access
class KernelBuildContext;

/**
 * Texture slot base class (non-template)
 * Used for type erasure in KernelBuildContext
 */
class TextureSlotBase {
public:
	virtual ~TextureSlotBase()															  = default;

	/**
	 * Get the backend texture handle of the attached texture
	 * @return The backend texture handle, or INVALID_TEXTURE_HANDLE if not attached
	 */
	virtual Backend::TextureHandle GetHandle() const									  = 0;

	/**
	 * Check if a texture is currently attached
	 * @return true if attached, false otherwise
	 */
	virtual bool				   IsAttached() const									  = 0;

	/**
	 * Get the pixel format of this slot
	 * @return The pixel format
	 */
	virtual PixelFormat			   GetFormat() const									  = 0;

	/**
	 * Get the texture dimensions
	 * @param[out] width The texture width
	 * @param[out] height The texture height
	 */
	virtual void				   GetDimensions(uint32_t &width, uint32_t &height) const = 0;
	virtual uint32_t			   GetDepth() const {
		return 1;
	}

	/**
	 * Get the binding slot assigned by KernelBuildContext
	 * @return The binding slot index, or -1 if not bound
	 */
	int GetBinding() const {
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
	 * @brief Check if this slot was bound as a sampler rather than as an image.
	 * @return true if bound via BindSampler(), false if bound via Bind().
	 */
	bool UsesSamplerBinding() const {
		return _sampledBinding;
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

protected:
	int			_binding = -1; // Assigned by KernelBuildContext during Bind()
	std::string _name;		   // GLSL variable name
	bool		_sampledBinding = false;
};

/**
 * 2D Texture slot for dynamic texture switching at runtime
 * @tparam Format The pixel format of the texture
 */
template <PixelFormat Format> class TextureSlot : public TextureSlotBase {
public:
	/**
	 * Default constructor - creates an unattached slot
	 */
	TextureSlot()									= default;

	/**
	 * Destructor
	 */
	~TextureSlot() override							= default;

	// Disable copy
	TextureSlot(const TextureSlot &)				= delete;
	TextureSlot &operator=(const TextureSlot &)		= delete;

	// Enable move
	TextureSlot(TextureSlot &&) noexcept			= default;
	TextureSlot &operator=(TextureSlot &&) noexcept = default;

public:
	// ===================================================================
	// Runtime API - Called outside kernel definition
	// ===================================================================

	/**
	 * Attach a 2D texture to this slot
	 * @param texture The 2D texture to attach
	 */
	void Attach(Texture2D<Format> &texture) {
		_texture	   = &texture;
		_textureHandle = texture.GetHandle();
		_lifetimeToken = texture.GetLifetimeToken();
		_width		   = texture.GetWidth();
		_height		   = texture.GetHeight();
	}

	/**
	 * Detach the current texture
	 */
	void Detach() {
		_texture	   = nullptr;
		_textureHandle = Backend::INVALID_TEXTURE_HANDLE;
		_lifetimeToken.reset();
		_width = _height = 0;
	}

	/**
	 * Check if a texture is currently attached
	 * @return true if attached, false otherwise
	 */
	bool IsAttached() const override {
		return _textureHandle != Backend::INVALID_TEXTURE_HANDLE && !_lifetimeToken.expired();
	}

	/**
	 * Get the currently attached texture
	 * @return Pointer to the attached texture, or nullptr if not attached
	 */
	Texture2D<Format> *GetAttached() const {
		return _texture;
	}

	/**
	 * Get the backend texture handle of the attached texture
	 * @return The backend texture handle, or INVALID_TEXTURE_HANDLE if not attached
	 */
	Backend::TextureHandle GetHandle() const override {
		return _textureHandle;
	}

	/**
	 * Get the pixel format of this slot
	 * @return The pixel format
	 */
	PixelFormat GetFormat() const override {
		return Format;
	}

	/**
	 * Get the texture dimensions
	 * @param[out] width The texture width
	 * @param[out] height The texture height
	 */
	void GetDimensions(uint32_t &width, uint32_t &height) const override {
		width  = _width;
		height = _height;
	}

public:
	// ===================================================================
	// DSL API - Called inside kernel definition
	// ===================================================================

	/**
	 * Bind this slot to the current kernel being defined
	 * @return TextureRef<Format> for DSL access (imageLoad/imageStore)
	 */
	[[nodiscard]] IR::Value::TextureRef<Format> Bind() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("TextureSlot::Bind() called outside of Kernel definition");
		}
		if (_texture != nullptr && _lifetimeToken.expired()) {
			throw std::runtime_error("TextureSlot::Bind() attached texture has been destroyed");
		}

		_sampledBinding = false;

		// Register this slot with the context
		context->RegisterTextureSlot(this);

		// Get dimensions (will be 0 if not attached, but that's OK for code generation)
		uint32_t width = 0, height = 0;
		GetDimensions(width, height);

		// Return TextureRef using our assigned name and binding
		return IR::Value::TextureRef<Format>(_name, static_cast<uint32_t>(_binding), width, height);
	}

	/**
	 * Bind this slot as a sampler to the current kernel being defined
	 * @return TextureSampler2D<Format> for DSL access (texture sampling)
	 */
	[[nodiscard]] IR::Value::TextureSampler2D<Format> BindSampler() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("TextureSlot::BindSampler() called outside of Kernel definition");
		}
		if (_texture != nullptr && _lifetimeToken.expired()) {
			throw std::runtime_error("TextureSlot::BindSampler() attached texture has been destroyed");
		}

		_sampledBinding = true;

		// Register this slot with the context
		context->RegisterTextureSlot(this);

		// Get dimensions
		uint32_t width = 0, height = 0;
		GetDimensions(width, height);

		// Return TextureSampler2D using our assigned name and binding
		return IR::Value::TextureSampler2D<Format>(_name, static_cast<uint32_t>(_binding), width, height);
	}

private:
	Texture2D<Format>	  *_texture		  = nullptr;
	Backend::TextureHandle _textureHandle = Backend::INVALID_TEXTURE_HANDLE; // Currently attached texture handle
	uint32_t			   _width		  = 0;
	uint32_t			   _height		  = 0;
	std::weak_ptr<void>	   _lifetimeToken;

	// Grant KernelBuildContext access to protected members
	friend class KernelBuildContext;
};

/**
 * 3D Texture slot for dynamic texture switching at runtime
 * @tparam Format The pixel format of the texture
 */
template <PixelFormat Format> class Texture3DSlot : public TextureSlotBase {
public:
	Texture3DSlot()										= default;
	~Texture3DSlot() override							= default;

	Texture3DSlot(const Texture3DSlot &)				= delete;
	Texture3DSlot &operator=(const Texture3DSlot &)		= delete;
	Texture3DSlot(Texture3DSlot &&) noexcept			= default;
	Texture3DSlot &operator=(Texture3DSlot &&) noexcept = default;

public:
	// ===================================================================
	// Runtime API - Called outside kernel definition
	// ===================================================================

	/**
	 * @brief Attach a 3D texture to this slot.
	 * @param texture The 3D texture to attach.
	 */
	void Attach(Texture3D<Format> &texture) {
		_texture	   = &texture;
		_lifetimeToken = texture.GetLifetimeToken();
	}

	void Detach() {
		_texture = nullptr;
		_lifetimeToken.reset();
	}

	bool IsAttached() const override {
		return _texture != nullptr && !_lifetimeToken.expired();
	}

	/**
	 * @brief Get the currently attached 3D texture.
	 * @return Pointer to the attached texture, or nullptr if not attached.
	 */
	Texture3D<Format> *GetAttached() const {
		return _texture;
	}

	Backend::TextureHandle GetHandle() const override {
		return _texture ? _texture->GetHandle() : Backend::INVALID_TEXTURE_HANDLE;
	}

	PixelFormat GetFormat() const override {
		return Format;
	}

	void GetDimensions(uint32_t &width, uint32_t &height) const override {
		if (_texture) {
			width  = _texture->GetWidth();
			height = _texture->GetHeight();
		} else {
			width = height = 0;
		}
	}

	uint32_t GetDepth() const override {
		return _texture ? _texture->GetDepth() : 1;
	}

public:
	// ===================================================================
	// DSL API - Called inside kernel definition
	// ===================================================================

	/**
	 * @brief Bind this slot to the current kernel being defined.
	 * @return TextureRef3D<Format> for DSL access (imageLoad/imageStore on 3D texture).
	 */
	[[nodiscard]] IR::Value::TextureRef3D<Format> Bind() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("Texture3DSlot::Bind() called outside of Kernel definition");
		}
		if (_texture != nullptr && _lifetimeToken.expired()) {
			throw std::runtime_error("Texture3DSlot::Bind() attached texture has been destroyed");
		}

		_sampledBinding = false;
		context->RegisterTextureSlot(this);
		uint32_t width = 0, height = 0;
		GetDimensions(width, height);
		return IR::Value::TextureRef3D<Format>(_name, static_cast<uint32_t>(_binding), width, height, GetDepth());
	}

	/**
	 * @brief Bind this slot as a sampler to the current kernel being defined.
	 * @return TextureSampler3D<Format> for DSL access (texture sampling on 3D texture).
	 */
	[[nodiscard]] IR::Value::TextureSampler3D<Format> BindSampler() {
		auto *context = IR::Builder::Builder::Get().Context();
		if (!context) {
			throw std::runtime_error("Texture3DSlot::BindSampler() called outside of Kernel definition");
		}
		if (_texture != nullptr && _lifetimeToken.expired()) {
			throw std::runtime_error("Texture3DSlot::BindSampler() attached texture has been destroyed");
		}

		_sampledBinding = true;
		context->RegisterTextureSlot(this);
		uint32_t width = 0, height = 0;
		GetDimensions(width, height);
		return IR::Value::TextureSampler3D<Format>(_name, static_cast<uint32_t>(_binding), width, height, GetDepth());
	}

private:
	Texture3D<Format>  *_texture = nullptr;
	std::weak_ptr<void> _lifetimeToken;

	friend class KernelBuildContext;
};

} // namespace GPU::Runtime

#endif // EASYGPU_TEXTURESLOT_H
