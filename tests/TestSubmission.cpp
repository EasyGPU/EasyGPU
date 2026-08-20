#include <Backend/Backend.h>
#include <Runtime/Context.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

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
	const auto expectInvalidSubmission = [](auto &&operation) {
		bool rejected = false;
		try {
			operation();
		} catch (const std::runtime_error &) {
			rejected = true;
		}
		assert(rejected);
	};
	expectInvalidSubmission([&] { (void)backend->IsSubmissionComplete(first); });
	expectInvalidSubmission([&] { (void)backend->IsSubmissionComplete(GPU::Backend::INVALID_SUBMISSION_HANDLE); });
	expectInvalidSubmission([&] {
		(void)backend->WaitForSubmission(GPU::Backend::INVALID_SUBMISSION_HANDLE, 0);
	});
	expectInvalidSubmission([&] { backend->ReleaseSubmission(GPU::Backend::INVALID_SUBMISSION_HANDLE); });
	expectInvalidSubmission([&] {
		(void)backend->IsSubmissionComplete(std::numeric_limits<GPU::Backend::SubmissionHandle>::max());
	});

	const auto releasedInFlight = backend->Submit();
	backend->ReleaseSubmission(releasedInFlight);
	expectInvalidSubmission([&] { (void)backend->WaitForSubmission(releasedInFlight, 0); });

	std::vector<GPU::Backend::SubmissionHandle> burst;
	burst.reserve(80);
	for (size_t i = 0; i < 80; ++i) {
		burst.push_back(backend->Submit());
	}
	assert(backend->WaitForSubmission(burst.back(), std::numeric_limits<uint64_t>::max()));
	for (const auto submission : burst) {
		assert(backend->IsSubmissionComplete(submission));
		backend->ReleaseSubmission(submission);
	}
	const auto reusedAfterBurst = backend->Submit();
	assert(backend->WaitForSubmission(reusedAfterBurst, std::numeric_limits<uint64_t>::max()));
	backend->ReleaseSubmission(reusedAfterBurst);

	if (backend->GetCaps().supportsTimestampQueries) {
		// More than MAX_QUERIES sequential intervals proves completed submission ownership
		// returns query slots instead of wrapping over an in-flight pair.
		for (uint32_t iteration = 0; iteration < 300; ++iteration) {
			const uint32_t query = backend->BeginSubmissionTimestamp();
			assert(query != 0);
			backend->CopyBuffer(source, 0, destination, 0, byteCount);
			const auto submission = backend->SubmitTimestamped(query);
			uint64_t elapsedNanoseconds = 0;
			(void)backend->TryGetSubmissionTimestamp(submission, elapsedNanoseconds);
			assert(backend->WaitForSubmission(submission, std::numeric_limits<uint64_t>::max()));
			assert(backend->TryGetSubmissionTimestamp(submission, elapsedNanoseconds));
			assert(elapsedNanoseconds > 0);
			backend->ReleaseSubmission(submission);
		}
	} else {
		assert(backend->BeginSubmissionTimestamp() == 0);
	}

	if (const char *value = std::getenv("EASYGPU_SUBMISSION_BENCHMARK_ITERATIONS")) {
		const auto iterations = std::strtoull(value, nullptr, 10);
		if (iterations != 0) {
			const auto start = std::chrono::steady_clock::now();
			for (uint64_t i = 0; i < iterations; ++i) {
				const auto submission = backend->Submit();
				assert(backend->WaitForSubmission(submission, std::numeric_limits<uint64_t>::max()));
				backend->ReleaseSubmission(submission);
			}
			const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start);
			std::cout << "Submission benchmark: " << iterations << " iterations, "
					  << elapsed.count() / static_cast<double>(iterations) << " us/submission\n";
		}
	}
	if (const char *value = std::getenv("EASYGPU_BURST_SUBMISSION_BENCHMARK_ITERATIONS")) {
		const auto iterations = std::strtoull(value, nullptr, 10);
		if (iterations != 0) {
			constexpr uint64_t submissionsPerBurst = 64;
			std::vector<GPU::Backend::SubmissionHandle> submissions;
			submissions.reserve(submissionsPerBurst);
			const auto start = std::chrono::steady_clock::now();
			for (uint64_t i = 0; i < iterations; ++i) {
				submissions.clear();
				for (uint64_t j = 0; j < submissionsPerBurst; ++j) {
					submissions.push_back(backend->Submit());
				}
				assert(backend->WaitForSubmission(submissions.back(), std::numeric_limits<uint64_t>::max()));
				for (const auto submission : submissions) {
					assert(backend->IsSubmissionComplete(submission));
					backend->ReleaseSubmission(submission);
				}
			}
			const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start);
			std::cout << "Burst submission benchmark: " << iterations << " x " << submissionsPerBurst
					  << " submissions, "
					  << elapsed.count() / static_cast<double>(iterations * submissionsPerBurst)
					  << " us/submission\n";
		}
	}
	if (const char *value = std::getenv("EASYGPU_SYNC_SUBMISSION_BENCHMARK_ITERATIONS")) {
		const auto iterations = std::strtoull(value, nullptr, 10);
		if (iterations != 0) {
			const auto start = std::chrono::steady_clock::now();
			for (uint64_t i = 0; i < iterations; ++i) {
				backend->CopyBuffer(source, 0, destination, 0, byteCount);
				backend->Finish();
			}
			const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start);
			std::cout << "Synchronous submission benchmark: " << iterations << " iterations, "
					  << elapsed.count() / static_cast<double>(iterations) << " us/submission\n";
		}
	}

	backend->DestroyBuffer(destination);
	backend->DestroyBuffer(intermediate);
	backend->DestroyBuffer(source);
	std::cout << "Submission fences and queued buffer copies passed\n";
	return 0;
}
