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
		if (backend->GetType() == GPU::Backend::BackendType::Vulkan) {
			// A synchronous cold upload must fail before recording or submitting anything;
			// otherwise one timestamp pair could be split across two Vulkan submissions.
			const uint32_t guardedInterval = backend->BeginTimestampInterval();
			assert(guardedInterval != 0);
			const auto operationsBeforeGuard = backend->GetOperationCounters();
			const auto resourcesBeforeGuard = backend->GetResourceCounters();
			GPU::Backend::BufferDesc coldDesc{};
			coldDesc.sizeInBytes = byteCount;
			coldDesc.mode = GPU::Backend::BufferMode::Read;
			coldDesc.initialData = sourceData.data();
			expectInvalidSubmission([&] { (void)backend->CreateBuffer(coldDesc); });
			const auto operationsAfterGuard = backend->GetOperationCounters();
			const auto resourcesAfterGuard = backend->GetResourceCounters();
			assert(operationsAfterGuard.blockingSubmissionWaitCalls ==
				   operationsBeforeGuard.blockingSubmissionWaitCalls);
			assert(resourcesAfterGuard.liveBufferHandles == resourcesBeforeGuard.liveBufferHandles);

			backend->CopyBuffer(source, 0, destination, 0, byteCount);
			backend->EndTimestampInterval(guardedInterval);
			const auto guardedSubmission = backend->SubmitProfiled({guardedInterval});
			assert(backend->WaitForSubmission(guardedSubmission, std::numeric_limits<uint64_t>::max()));
			std::vector<uint64_t> guardedNanoseconds;
			assert(backend->TryGetSubmissionTimestamps(guardedSubmission, guardedNanoseconds));
			assert(guardedNanoseconds.size() == 1);
			assert(guardedNanoseconds[0] > 0);
			backend->ReleaseSubmission(guardedSubmission);
		}

		const uint32_t graphInterval = backend->BeginTimestampInterval();
		const uint32_t passInterval = backend->BeginTimestampInterval();
		assert(graphInterval != 0);
		assert(passInterval != 0);
		expectInvalidSubmission([&] { backend->EndTimestampInterval(graphInterval); });

		const uint32_t commandInterval = backend->BeginTimestampInterval();
		assert(commandInterval != 0);
		for (uint32_t copy = 0; copy < 64; ++copy) {
			backend->CopyBuffer(source, 0, destination, 0, byteCount);
		}
		backend->EndTimestampInterval(commandInterval);
		backend->EndTimestampInterval(passInterval);
		backend->EndTimestampInterval(graphInterval);

		const std::vector<uint32_t> nestedIntervals = {graphInterval, passInterval, commandInterval};
		expectInvalidSubmission([&] {
			(void)backend->SubmitProfiled({graphInterval, passInterval});
		});
		const auto nestedSubmission = backend->SubmitProfiled(nestedIntervals);
		std::vector<uint64_t> nestedNanoseconds;
		(void)backend->TryGetSubmissionTimestamps(nestedSubmission, nestedNanoseconds);
		assert(backend->WaitForSubmission(nestedSubmission, std::numeric_limits<uint64_t>::max()));
		assert(backend->TryGetSubmissionTimestamps(nestedSubmission, nestedNanoseconds));
		assert(nestedNanoseconds.size() == nestedIntervals.size());
		assert(nestedNanoseconds[0] > 0);
		assert(nestedNanoseconds[1] > 0);
		assert(nestedNanoseconds[2] > 0);
		assert(nestedNanoseconds[0] >= nestedNanoseconds[1]);
		assert(nestedNanoseconds[1] >= nestedNanoseconds[2]);
		std::vector<GPU::Backend::SubmissionTimestampInterval> nestedTimeline;
		assert(backend->TryGetSubmissionTimestampIntervals(nestedSubmission, nestedTimeline));
		assert(nestedTimeline.size() == nestedIntervals.size());
		assert(nestedTimeline[0].startOffsetNanoseconds == 0);
		assert(nestedTimeline[0].durationNanoseconds == nestedNanoseconds[0]);
		assert(nestedTimeline[1].durationNanoseconds == nestedNanoseconds[1]);
		assert(nestedTimeline[2].durationNanoseconds == nestedNanoseconds[2]);
		assert(nestedTimeline[1].startOffsetNanoseconds >= nestedTimeline[0].startOffsetNanoseconds);
		assert(nestedTimeline[2].startOffsetNanoseconds >= nestedTimeline[1].startOffsetNanoseconds);
		assert(nestedTimeline[1].startOffsetNanoseconds + nestedTimeline[1].durationNanoseconds <=
			   nestedTimeline[0].durationNanoseconds);
		assert(nestedTimeline[2].startOffsetNanoseconds + nestedTimeline[2].durationNanoseconds <=
			   nestedTimeline[1].startOffsetNanoseconds + nestedTimeline[1].durationNanoseconds);
		backend->ReleaseSubmission(nestedSubmission);

		// Releasing an in-flight timestamp submission invalidates its public handle immediately,
		// but its query slot stays leased until the GPU fence completes.
		const uint32_t releasedQuery = backend->BeginSubmissionTimestamp();
		assert(releasedQuery != 0);
		for (uint32_t copy = 0; copy < 64; ++copy) {
			backend->CopyBuffer(source, 0, destination, 0, byteCount);
		}
		const auto releasedTimestampSubmission = backend->SubmitTimestamped(releasedQuery);
		backend->ReleaseSubmission(releasedTimestampSubmission);
		expectInvalidSubmission([&] {
			(void)backend->WaitForSubmission(releasedTimestampSubmission, 0);
		});
		backend->Finish();

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
