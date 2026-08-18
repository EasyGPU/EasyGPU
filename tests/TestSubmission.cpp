#include <Backend/Backend.h>
#include <Runtime/Context.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
	GPU::Runtime::AutoInitContext();
	auto *backend = GPU::Runtime::Context::GetBackend();
	assert(backend != nullptr);

	constexpr size_t byteCount = sizeof(uint32_t) * 8;
	const std::array<uint32_t, 8> sourceData = {3, 5, 8, 13, 21, 34, 55, 89};
	GPU::Backend::BufferDesc sourceDesc{};
	sourceDesc.sizeInBytes = byteCount;
	sourceDesc.mode = GPU::Backend::BufferMode::Read;
	sourceDesc.initialData = sourceData.data();
	GPU::Backend::BufferDesc destinationDesc{};
	destinationDesc.sizeInBytes = byteCount;
	destinationDesc.mode = GPU::Backend::BufferMode::ReadWrite;

	const auto source = backend->CreateBuffer(sourceDesc);
	const auto intermediate = backend->CreateBuffer(destinationDesc);
	const auto destination = backend->CreateBuffer(destinationDesc);

	backend->CopyBuffer(source, 0, intermediate, 0, byteCount);
	const auto first = backend->Submit();
	assert(first != GPU::Backend::INVALID_SUBMISSION_HANDLE);

	backend->CopyBuffer(intermediate, 0, destination, 0, byteCount);
	const auto second = backend->Submit();
	assert(second != GPU::Backend::INVALID_SUBMISSION_HANDLE);
	assert(second != first);
	(void)backend->WaitForSubmission(second, 0);

	assert(backend->WaitForSubmission(second, std::numeric_limits<uint64_t>::max()));
	assert(backend->IsSubmissionComplete(first));
	assert(backend->IsSubmissionComplete(second));

	std::array<uint32_t, 8> result{};
	backend->DownloadBuffer(destination, 0, byteCount, result.data());
	assert(result == sourceData);

	backend->ReleaseSubmission(first);
	backend->ReleaseSubmission(second);
	bool rejectedReleasedHandle = false;
	try {
		(void)backend->IsSubmissionComplete(first);
	} catch (const std::runtime_error &) {
		rejectedReleasedHandle = true;
	}
	assert(rejectedReleasedHandle);

	const auto releasedInFlight = backend->Submit();
	backend->ReleaseSubmission(releasedInFlight);
	bool rejectedInFlightReleasedHandle = false;
	try {
		(void)backend->WaitForSubmission(releasedInFlight, 0);
	} catch (const std::runtime_error &) {
		rejectedInFlightReleasedHandle = true;
	}
	assert(rejectedInFlightReleasedHandle);

	backend->DestroyBuffer(destination);
	backend->DestroyBuffer(intermediate);
	backend->DestroyBuffer(source);
	std::cout << "Submission fences and queued buffer copies passed\n";
	return 0;
}
