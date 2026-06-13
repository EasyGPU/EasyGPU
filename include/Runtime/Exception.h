#pragma once

/**
 * @file Exception.h
 * @brief Base exception types for the EasyGPU runtime.
 *
 * Provides a root exception class that all EasyGPU-specific errors derive from,
 * and concrete exception types for common error categories: IR construction
 * failures, invalid operations, and resource exhaustion.
 */

#ifndef EASYGPU_EXCEPTION_H
#define EASYGPU_EXCEPTION_H

#include <format>
#include <stdexcept>
#include <string>

namespace GPU::Runtime {

/**
 * @brief Root exception class for all EasyGPU-specific errors.
 *
 * All runtime exceptions thrown by the library derive from this type,
 * so users can catch GPU::Runtime::Exception to handle any library error
 * uniformly.  The what() string is formatted once at construction and
 * cached, making it safe to call from a noexcept context.
 */
class Exception : public std::runtime_error {
public:
	/**
	 * @brief Construct an exception with a component tag and message.
	 * @param component Name of the subsystem that raised the error (e.g. "IR", "Backend").
	 * @param message  Human-readable description of the error.
	 */
	Exception(std::string component, std::string message)
		: std::runtime_error(std::format("[GPU::{}] {}", component, message)), _component(std::move(component)),
		  _message(std::move(message)) {
	}

	/** @brief Return the subsystem component that raised this error. */
	[[nodiscard]] const std::string &Component() const noexcept {
		return _component;
	}

	/** @brief Return the raw message without the component prefix. */
	[[nodiscard]] const std::string &RawMessage() const noexcept {
		return _message;
	}

private:
	std::string _component;
	std::string _message;
};

/**
 * @brief Exception thrown when an IR node or variable cannot be built because
 * the Builder is not bound to a valid context.
 *
 * This typically indicates that a GPU variable was constructed outside of a
 * Kernel definition or after the kernel's build context was released.
 */
class BuilderContextException : public Exception {
public:
	explicit BuilderContextException(std::string reason) : Exception("Builder", std::move(reason)) {
	}
};

/**
 * @brief Exception thrown when an internal invariant is violated during IR
 * construction.
 *
 * These errors indicate a bug in EasyGPU itself rather than a user mistake.
 * In debug builds the library may also trip an assertion before throwing.
 */
class InternalIRException : public Exception {
public:
	explicit InternalIRException(std::string detail) : Exception("IR", std::move(detail)) {
	}
};

/**
 * @brief Exception thrown when a GPU resource (buffer, texture, pipeline)
 * cannot be allocated, typically due to out-of-memory conditions.
 */
class ResourceExhaustionException : public Exception {
public:
	ResourceExhaustionException(std::string resourceType, std::string detail)
		: Exception("Resource", std::format("Failed to allocate {}: {}", resourceType, detail)),
		  _resourceType(std::move(resourceType)), _detail(std::move(detail)) {
	}

	/** @brief The type of resource that could not be allocated. */
	[[nodiscard]] const std::string &ResourceType() const noexcept {
		return _resourceType;
	}

	/** @brief Detail about the allocation failure. */
	[[nodiscard]] const std::string &Detail() const noexcept {
		return _detail;
	}

private:
	std::string _resourceType;
	std::string _detail;
};

} // namespace GPU::Runtime

#endif // EASYGPU_EXCEPTION_H
