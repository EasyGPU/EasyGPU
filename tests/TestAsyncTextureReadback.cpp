#include <Backend/Backend.h>
#include <Runtime/Context.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

size_t							   runtimeErrorExpectation = 0;

template <typename Operation> void ExpectRuntimeError(Operation &&operation) {
	const size_t currentExpectation = ++runtimeErrorExpectation;
	bool		 rejected			= false;
	try {
		operation();
	} catch (const std::runtime_error &) {
		rejected = true;
	}
	if (!rejected) {
		std::cerr << "Expected runtime_error at expectation " << currentExpectation << "\n";
	}
	assert(rejected);
}

GPU::Backend::BufferHandle CreateStaging(GPU::Backend::Backend *backend, size_t byteSize) {
	GPU::Backend::BufferDesc desc{};
	desc.sizeInBytes = byteSize;
	desc.mode		 = GPU::Backend::BufferMode::ReadWrite;
	return backend->CreateBuffer(desc);
}

} // namespace

int main() {
	GPU::Runtime::AutoInitContext();
	auto *backend = GPU::Runtime::Context::GetBackend();
	assert(backend != nullptr);
	if (GPU::Runtime::Context::GetInstance().GetBackendType() != GPU::Backend::BackendType::Vulkan) {
		std::cout << "Asynchronous texture readback is Vulkan-only; skipped\n";
		return 0;
	}

	constexpr uint32_t				width		  = 5;
	constexpr uint32_t				height		  = 3;
	constexpr size_t				pixelBytes	  = 4;
	constexpr size_t				rowPitch	  = width * pixelBytes;
	constexpr size_t				imageBytes	  = rowPitch * height;
	constexpr size_t				stagingOffset = 12;

	std::array<uint8_t, imageBytes> source{};
	for (size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
	}

	GPU::Backend::TextureDesc textureDesc{};
	textureDesc.width  = width;
	textureDesc.height = height;
	textureDesc.format = GPU::Backend::PixelFormat::RGBA8;
	const auto texture = backend->CreateTexture(textureDesc);
	backend->UploadTexture(texture, 0, 0, width, height, source.data());
	const auto staging = CreateStaging(backend, stagingOffset + imageBytes);

	// An ordinary buffer mapping owns the same persistent staging allocation.
	assert(backend->MapBuffer(staging, false, true) != nullptr);
	ExpectRuntimeError(
		[&] { (void)backend->BeginTextureReadback(texture, 0, 0, width, height, staging, stagingOffset); });
	ExpectRuntimeError([&] { backend->UploadBuffer(staging, 0, imageBytes, source.data()); });
	std::array<uint8_t, imageBytes> scratch{};
	ExpectRuntimeError([&] { backend->DownloadBuffer(staging, 0, imageBytes, scratch.data()); });
	backend->UnmapBuffer(staging);

	const auto before	= backend->GetOperationCounters();
	const auto readback = backend->BeginTextureReadback(texture, 0, 0, width, height, staging, stagingOffset);
	ExpectRuntimeError(
		[&] { (void)backend->BeginTextureReadback(texture, 0, 0, width, height, staging, stagingOffset); });
	const auto afterBegin = backend->GetOperationCounters();
	assert(afterBegin.asyncTextureReadbackCalls == before.asyncTextureReadbackCalls + 1);
	assert(afterBegin.finishCalls == before.finishCalls);
	assert(afterBegin.deviceWaitIdleCalls == before.deviceWaitIdleCalls);
	assert(afterBegin.globalDrainCalls == before.globalDrainCalls);
	assert(afterBegin.blockingSubmissionWaitCalls == before.blockingSubmissionWaitCalls);
	assert(afterBegin.blockingTextureDownloadCalls == before.blockingTextureDownloadCalls);

	// The operation may already have completed on a fast device. If not, Map must
	// reject without consuming its one allowed mapping.
	GPU::Backend::TextureReadbackMapping mapped{};
	try {
		mapped = backend->MapTextureReadback(readback);
	} catch (const std::runtime_error &) {
		assert(backend->WaitForSubmission(readback, std::numeric_limits<uint64_t>::max()));
		mapped = backend->MapTextureReadback(readback);
	}
	assert(mapped.data != nullptr);
	assert(mapped.byteSize == imageBytes);
	assert(mapped.rowPitch == rowPitch);
	assert(std::memcmp(mapped.data, source.data(), imageBytes) == 0);
	ExpectRuntimeError([&] { (void)backend->MapTextureReadback(readback); });
	ExpectRuntimeError([&] { backend->UploadBuffer(staging, 0, imageBytes, source.data()); });
	ExpectRuntimeError([&] { backend->DownloadBuffer(staging, 0, imageBytes, scratch.data()); });
	ExpectRuntimeError([&] { (void)backend->MapBuffer(staging, true, false); });
	ExpectRuntimeError([&] { backend->ReleaseSubmission(readback); });
	backend->UnmapTextureReadback(readback);
	ExpectRuntimeError([&] { backend->UnmapTextureReadback(readback); });
	ExpectRuntimeError([&] { (void)backend->MapTextureReadback(readback); });
	backend->ReleaseSubmission(readback);
	ExpectRuntimeError([&] { (void)backend->MapTextureReadback(readback); });

	ExpectRuntimeError([&] { (void)backend->BeginTextureReadback(texture, 0, 0, 0, height, staging, 0); });
	ExpectRuntimeError([&] { (void)backend->BeginTextureReadback(texture, 4, 0, 2, height, staging, 0); });
	ExpectRuntimeError([&] { (void)backend->BeginTextureReadback(texture, 0, 0, width, height, staging, 2); });
	ExpectRuntimeError([&] {
		(void)backend->BeginTextureReadback(texture, 0, 0, width, height, staging,
											std::numeric_limits<size_t>::max() - 3);
	});

	GPU::Backend::TextureDesc noTransferDesc{};
	noTransferDesc.width		 = width;
	noTransferDesc.height		 = height;
	noTransferDesc.format		 = GPU::Backend::PixelFormat::RGBA8;
	noTransferDesc.usage		 = GPU::Backend::TextureUsageSampled;
	const auto noTransferTexture = backend->CreateTexture(noTransferDesc);
	ExpectRuntimeError(
		[&] { (void)backend->BeginTextureReadback(noTransferTexture, 0, 0, width, height, staging, stagingOffset); });
	backend->DestroyTexture(noTransferTexture);

	GPU::Backend::TextureDesc depthDesc{};
	depthDesc.width			= width;
	depthDesc.height		= height;
	depthDesc.format		= GPU::Backend::PixelFormat::D32F;
	depthDesc.usage			= GPU::Backend::TextureUsageTransferSrc | GPU::Backend::TextureUsageDepthStencilAttachment;
	const auto depthTexture = backend->CreateTexture(depthDesc);
	ExpectRuntimeError(
		[&] { (void)backend->BeginTextureReadback(depthTexture, 0, 0, width, height, staging, stagingOffset); });
	backend->DestroyTexture(depthTexture);

	if (backend->GetCaps().supportsGraphics) {
		GPU::Backend::RenderPassBeginDesc renderPass{};
		renderPass.colorAttachment = texture;
		renderPass.colorLoadOp	   = GPU::Backend::AttachmentLoadOp::Load;
		backend->BeginRendering(renderPass);
		ExpectRuntimeError(
			[&] { (void)backend->BeginTextureReadback(texture, 0, 0, width, height, staging, stagingOffset); });
		backend->EndRendering();
		const auto renderSubmission = backend->Submit();
		assert(backend->WaitForSubmission(renderSubmission, std::numeric_limits<uint64_t>::max()));
		backend->ReleaseSubmission(renderSubmission);
	}

	const auto completedUnconsumed =
		backend->BeginTextureReadback(texture, 0, 0, width, height, staging, stagingOffset);
	assert(backend->WaitForSubmission(completedUnconsumed, std::numeric_limits<uint64_t>::max()));
	ExpectRuntimeError([&] { backend->UploadBuffer(staging, 0, imageBytes, source.data()); });
	ExpectRuntimeError([&] { backend->DownloadBuffer(staging, 0, imageBytes, scratch.data()); });
	ExpectRuntimeError([&] { (void)backend->MapBuffer(staging, true, false); });
	const auto completedUnconsumedMap = backend->MapTextureReadback(completedUnconsumed);
	assert(std::memcmp(completedUnconsumedMap.data, source.data(), imageBytes) == 0);
	backend->UnmapTextureReadback(completedUnconsumed);
	backend->ReleaseSubmission(completedUnconsumed);

	std::array<uint8_t, imageBytes> recordedSource{};
	for (size_t i = 0; i < recordedSource.size(); ++i) {
		recordedSource[i] = static_cast<uint8_t>((i * 19 + 5) & 0xff);
	}
	GPU::Backend::BufferDesc recordedSourceDesc{};
	recordedSourceDesc.sizeInBytes	= imageBytes;
	recordedSourceDesc.mode			= GPU::Backend::BufferMode::Read;
	recordedSourceDesc.initialData	= recordedSource.data();
	const auto recordedSourceBuffer = backend->CreateBuffer(recordedSourceDesc);
	const auto recordedDestination	= CreateStaging(backend, imageBytes);
	backend->CopyBuffer(recordedSourceBuffer, 0, recordedDestination, 0, imageBytes);
	const auto beforeRecordedDestroy = backend->GetOperationCounters();
	backend->DestroyBuffer(recordedSourceBuffer);
	const auto afterRecordedDestroy = backend->GetOperationCounters();
	assert(afterRecordedDestroy.globalDrainCalls == beforeRecordedDestroy.globalDrainCalls);
	assert(afterRecordedDestroy.blockingSubmissionWaitCalls == beforeRecordedDestroy.blockingSubmissionWaitCalls);
	ExpectRuntimeError([&] { backend->CopyBuffer(recordedSourceBuffer, 0, recordedDestination, 0, imageBytes); });
	const auto recordedUse = backend->Submit();
	assert(backend->WaitForSubmission(recordedUse, std::numeric_limits<uint64_t>::max()));
	backend->ReleaseSubmission(recordedUse);
	std::array<uint8_t, imageBytes> recordedResult{};
	backend->DownloadBuffer(recordedDestination, 0, imageBytes, recordedResult.data());
	assert(recordedResult == recordedSource);
	backend->DestroyBuffer(recordedDestination);

	std::array<GPU::Backend::BufferHandle, 3>	  overlapBuffers{};
	std::array<GPU::Backend::SubmissionHandle, 3> overlaps{};
	for (size_t i = 0; i < overlaps.size(); ++i) {
		overlapBuffers[i] = CreateStaging(backend, imageBytes);
		overlaps[i]		  = backend->BeginTextureReadback(texture, 0, 0, width, height, overlapBuffers[i], 0);
	}
	assert(backend->WaitForSubmission(overlaps.back(), std::numeric_limits<uint64_t>::max()));
	for (size_t i = 0; i < overlaps.size(); ++i) {
		const auto overlap = backend->MapTextureReadback(overlaps[i]);
		assert(std::memcmp(overlap.data, source.data(), imageBytes) == 0);
		backend->UnmapTextureReadback(overlaps[i]);
		backend->ReleaseSubmission(overlaps[i]);
		backend->DestroyBuffer(overlapBuffers[i]);
	}

	const auto cancelledStaging	   = CreateStaging(backend, imageBytes);
	const auto cancelled		   = backend->BeginTextureReadback(texture, 0, 0, width, height, cancelledStaging, 0);
	const auto beforeCancelRelease = backend->GetOperationCounters();
	backend->ReleaseSubmission(cancelled);
	const auto afterCancelRelease = backend->GetOperationCounters();
	assert(afterCancelRelease.globalDrainCalls == beforeCancelRelease.globalDrainCalls);
	assert(afterCancelRelease.blockingSubmissionWaitCalls == beforeCancelRelease.blockingSubmissionWaitCalls);
	backend->Finish();
	const auto reusedAfterCancel = backend->BeginTextureReadback(texture, 0, 0, width, height, cancelledStaging, 0);
	assert(backend->WaitForSubmission(reusedAfterCancel, std::numeric_limits<uint64_t>::max()));
	const auto cancelledReuseMap = backend->MapTextureReadback(reusedAfterCancel);
	assert(std::memcmp(cancelledReuseMap.data, source.data(), imageBytes) == 0);
	backend->UnmapTextureReadback(reusedAfterCancel);
	backend->ReleaseSubmission(reusedAfterCancel);
	backend->DestroyBuffer(cancelledStaging);

	for (size_t iteration = 0; iteration < 500; ++iteration) {
		const auto operation = backend->BeginTextureReadback(texture, 0, 0, width, height, staging, stagingOffset);
		assert(backend->WaitForSubmission(operation, std::numeric_limits<uint64_t>::max()));
		const auto reuse = backend->MapTextureReadback(operation);
		assert(std::memcmp(reuse.data, source.data(), imageBytes) == 0);
		backend->UnmapTextureReadback(operation);
		backend->ReleaseSubmission(operation);
	}

	const auto deferredStaging	= CreateStaging(backend, imageBytes);
	const auto laterDestination = CreateStaging(backend, imageBytes);
	const auto deferred			= backend->BeginTextureReadback(texture, 0, 0, width, height, deferredStaging, 0);
	if (backend->GetCaps().supportsGraphics) {
		GPU::Backend::RenderPassBeginDesc renderPass{};
		renderPass.colorAttachment = texture;
		backend->BeginRendering(renderPass);
		backend->EndRendering();
	}
	backend->CopyBuffer(deferredStaging, 0, laterDestination, 0, imageBytes);
	const auto laterUse		 = backend->Submit();
	const auto beforeDestroy = backend->GetOperationCounters();
	backend->DestroyTexture(texture);
	backend->DestroyBuffer(deferredStaging);
	const auto afterDestroy = backend->GetOperationCounters();
	assert(afterDestroy.globalDrainCalls == beforeDestroy.globalDrainCalls);
	assert(afterDestroy.blockingSubmissionWaitCalls == beforeDestroy.blockingSubmissionWaitCalls);
	ExpectRuntimeError([&] { backend->UploadBuffer(deferredStaging, 0, imageBytes, source.data()); });
	ExpectRuntimeError([&] { backend->DownloadBuffer(deferredStaging, 0, imageBytes, scratch.data()); });
	assert(backend->MapBuffer(deferredStaging, true, false) == nullptr);
	ExpectRuntimeError(
		[&] { (void)backend->BeginTextureReadback(texture, 0, 0, width, height, staging, stagingOffset); });
	assert(backend->WaitForSubmission(deferred, std::numeric_limits<uint64_t>::max()));
	const auto deferredMap = backend->MapTextureReadback(deferred);
	assert(std::memcmp(deferredMap.data, source.data(), imageBytes) == 0);
	backend->UnmapTextureReadback(deferred);
	backend->ReleaseSubmission(deferred);
	assert(backend->WaitForSubmission(laterUse, std::numeric_limits<uint64_t>::max()));
	backend->ReleaseSubmission(laterUse);
	ExpectRuntimeError(
		[&] { (void)backend->BeginTextureReadback(texture, 0, 0, width, height, staging, stagingOffset); });

	backend->DestroyBuffer(laterDestination);
	backend->DestroyBuffer(staging);
	backend->MakeNoneCurrent();
	std::cout << "Asynchronous texture readback ownership and mapping passed\n";
	return 0;
}
