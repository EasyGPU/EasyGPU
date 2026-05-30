/**
 * @file Optimizer.cpp
 * @brief GPU optimizer method implementations.
 */

#include <NN/Optimizer.h>

#include <AD/ADKernel.h>
#include <Backend/Backend.h>
#include <Runtime/Buffer.h>
#include <Runtime/Context.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace GPU::NN {

// =============================================================================
// Adam
// =============================================================================

Adam::~Adam() = default;

void Adam::Step(AD::ADKernel1D &kernel) {
    if (!gpu_) {
        gpu_ = std::make_unique<GPUAdam>(lr_, beta1_, beta2_, eps_);
        for (const auto &ps : params_) {
            if (!ps.buffer)
                throw std::runtime_error("Adam::Step now requires GPU-backed parameters; use AddTensor or pass a Buffer");
            gpu_->AddParameter(ps.size, ps.buffer->GetHandle());
        }
    }
    gpu_->SetWeightDecay(weightDecay_);
    gpu_->SetGradClip(gradClip_);
    gpu_->Step(kernel, false);
    step_ = gpu_->GetStep();
}

// =============================================================================
// GPUAdam
// =============================================================================

void GPUAdam::AddParameter(size_t size, Backend::BufferHandle weightHandle) {
    if (size == 0 || weightHandle == Backend::INVALID_BUFFER_HANDLE)
        throw std::invalid_argument("GPUAdam::AddParameter requires a live GPU weight buffer");

    ParamSlotGPU ps;
    ps.size = size;
    ps.weightHandle = weightHandle;
    std::vector<float> zeros(size, 0.0f);
    ps.m = std::make_unique<Runtime::Buffer<float>>(zeros, Runtime::BufferMode::ReadWrite);
    ps.v = std::make_unique<Runtime::Buffer<float>>(zeros, Runtime::BufferMode::ReadWrite);
    params_.push_back(std::move(ps));
}

void GPUAdam::Step(AD::ADKernel1D &kernel, bool sync) {
    step_++;
    auto gradParams = kernel.GradientParams();

    if (params_.size() * 2 + 3 <= Backend::MAX_BUFFER_BINDINGS) {
        StepCombined(gradParams, sync);
        return;
    }

    size_t paramBase = 0;
    for (auto &ps : params_) {
        if (paramBase + ps.size > gradParams.size())
            throw std::runtime_error("GPUAdam gradient count does not match registered parameters");

        const auto &first = gradParams[paramBase];
        if (first.gradHandle == Backend::INVALID_BUFFER_HANDLE)
            throw std::runtime_error("GPUAdam called before ADKernel1D::Backward created gradient buffers");

        for (size_t i = 0; i < ps.size; i++) {
            const auto &gp = gradParams[paramBase + i];
            if (gp.gradHandle != first.gradHandle ||
                gp.gradStride != first.gradStride ||
                gp.sampleCount != first.sampleCount ||
                gp.gradOffset != first.gradOffset + static_cast<int>(i)) {
                throw std::runtime_error("GPUAdam requires tensor parameters to map to one contiguous gradient group");
            }
        }

        EnsurePipeline(ps, first);
        UploadHyperParams(ps, first.sampleCount);
        DispatchSlot(ps, first.gradHandle, sync);
        paramBase += ps.size;
    }
}

std::vector<GPUAdam::CombinedSlot>
GPUAdam::BuildCombinedSlots(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
                             size_t &totalSize) const {
    std::vector<CombinedSlot> slots;
    slots.reserve(params_.size());
    totalSize = 0;

    size_t paramBase = 0;
    for (const auto &ps : params_) {
        if (paramBase + ps.size > gradParams.size())
            throw std::runtime_error("GPUAdam gradient count does not match registered parameters");

        const auto &first = gradParams[paramBase];
        if (first.gradHandle == Backend::INVALID_BUFFER_HANDLE)
            throw std::runtime_error("GPUAdam called before ADKernel1D::Backward created gradient buffers");

        for (size_t i = 0; i < ps.size; i++) {
            const auto &gp = gradParams[paramBase + i];
            if (gp.gradHandle != first.gradHandle ||
                gp.gradStride != first.gradStride ||
                gp.sampleCount != first.sampleCount ||
                gp.gradOffset != first.gradOffset + static_cast<int>(i)) {
                throw std::runtime_error("GPUAdam requires tensor parameters to map to one contiguous gradient group");
            }
        }

        CombinedSlot slot;
        slot.size = ps.size;
        slot.base = totalSize;
        slot.sampleCount = first.sampleCount;
        slot.gradOffset = first.gradOffset;
        slot.gradStride = first.gradStride;
        slot.weightHandle = ps.weightHandle;
        slot.gradHandle = first.gradHandle;
        slots.push_back(slot);

        totalSize += ps.size;
        paramBase += ps.size;
    }
    return slots;
}

std::string GPUAdam::CombinedSignature(const std::vector<CombinedSlot> &slots,
                                        size_t totalSize) const {
    std::string sig = std::to_string(totalSize);
    for (const auto &s : slots) {
        sig += std::format("|{}:{}:{}:{}:{}", s.base, s.size,
                           s.sampleCount, s.gradOffset, s.gradStride);
    }
    return sig;
}

std::string GPUAdam::BuildCombinedAdamShader(const std::vector<CombinedSlot> &slots,
                                              size_t totalSize) {
    const size_t n = slots.size();
    std::string src = "#version 430\nlayout(local_size_x = 256) in;\n\n";
    for (size_t i = 0; i < n; i++) {
        src += std::format("layout(std430, binding = {}) buffer WeightBuf{} {{ float weight{}[]; }};\n",
                           i, i, i);
    }
    for (size_t i = 0; i < n; i++) {
        src += std::format("layout(std430, binding = {}) readonly buffer GradBuf{} {{ float grad{}[]; }};\n",
                           n + i, i, i);
    }
    src += std::format("layout(std430, binding = {}) buffer FirstMomentBuf {{ float m[]; }};\n", 2 * n);
    src += std::format("layout(std430, binding = {}) buffer SecondMomentBuf {{ float v[]; }};\n", 2 * n + 1);
    src += std::format("layout(std430, binding = {}) readonly buffer HyperBuf {{ float h[]; }};\n\n", 2 * n + 2);
    src += "void main() {\n";
    src += "    uint i = gl_GlobalInvocationID.x;\n";
    src += std::format("    if (i >= {}u) return;\n\n", totalSize);

    for (size_t si = 0; si < n; si++) {
        const auto &s = slots[si];
        src += std::format("    if (i >= {}u && i < {}u) {{\n",
                           s.base, s.base + s.size);
        src += std::format("        uint j = i - {}u;\n", s.base);
        src += "        float g = 0.0;\n";
        src += std::format("        for (uint sidx = 0u; sidx < {}u; ++sidx) {{\n",
                           s.sampleCount);
        src += std::format("            g += grad{}[sidx * {}u + ({}u + j)];\n",
                           si, s.gradStride, s.gradOffset);
        src += "        }\n";
        src += "        g *= h[8];\n";
        src += std::format("        if (h[4] > 0.0) g += 2.0 * h[4] * weight{}[j];\n", si);
        src += "        if (h[5] > 0.0) g = clamp(g, -h[5], h[5]);\n";
        src += "        m[i] = h[1] * m[i] + (1.0 - h[1]) * g;\n";
        src += "        v[i] = h[2] * v[i] + (1.0 - h[2]) * g * g;\n";
        src += "        float mHat = m[i] / h[6];\n";
        src += "        float vHat = v[i] / h[7];\n";
        src += std::format("        weight{}[j] -= h[0] * mHat / (sqrt(vHat) + h[3]);\n", si);
        src += "        return;\n";
        src += "    }\n";
    }

    src += "}\n";
    return src;
}

void GPUAdam::StepCombined(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
                            bool sync) {
    size_t totalSize = 0;
    auto slots = BuildCombinedSlots(gradParams, totalSize);
    if (totalSize == 0) return;

    EnsureCombinedPipeline(slots, totalSize);
    UploadCombinedHyperParams(totalSize, slots.empty() ? 0 : slots[0].sampleCount);
    DispatchCombined(slots, totalSize, sync);
}

void GPUAdam::EnsureCombinedPipeline(const std::vector<CombinedSlot> &slots, size_t totalSize) {
    std::string sig = CombinedSignature(slots, totalSize);
    if (_combinedPipeline != Backend::INVALID_PIPELINE_HANDLE &&
        _combinedSignature == sig) return;

    Runtime::Context::GetInstance().MakeCurrent();
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPUAdam backend not available");

    if (_combinedPipeline != Backend::INVALID_PIPELINE_HANDLE)
        backend->DestroyPipeline(_combinedPipeline);
    if (_combinedShader != Backend::INVALID_SHADER_HANDLE)
        backend->DestroyShader(_combinedShader);

    if (!_flatM || _flatMSize != totalSize) {
        std::vector<float> zeros(totalSize, 0.0f);
        _flatM = std::make_unique<Runtime::Buffer<float>>(zeros, Runtime::BufferMode::ReadWrite);
        _flatV = std::make_unique<Runtime::Buffer<float>>(zeros, Runtime::BufferMode::ReadWrite);
        _flatMSize = totalSize;
    }

    Backend::ShaderDesc shaderDesc;
    shaderDesc.type = Backend::ShaderType::Compute;
    shaderDesc.sourceCode = BuildCombinedAdamShader(slots, totalSize);
    _combinedShader = backend->CreateShader(shaderDesc);

    Backend::PipelineDesc pipelineDesc;
    pipelineDesc.computeShader = _combinedShader;
    pipelineDesc.workGroupSizeX = 256;
    const uint32_t bindingCount = static_cast<uint32_t>(slots.size() * 2 + 3);
    for (uint32_t binding = 0; binding < bindingCount; binding++) {
        Backend::ResourceLayoutEntry entry;
        entry.binding = binding;
        entry.type = Backend::BindingType::Buffer;
        entry.readOnly = binding >= slots.size() && binding < slots.size() * 2
            || binding == bindingCount - 1;
        pipelineDesc.resources.push_back(entry);
    }
    _combinedPipeline = backend->CreatePipeline(pipelineDesc);
    if (_combinedPipeline == Backend::INVALID_PIPELINE_HANDLE)
        throw std::runtime_error("GPUAdam failed to create combined update pipeline");

    _combinedSignature = std::move(sig);
}

void GPUAdam::UploadCombinedHyperParams(size_t totalSize, size_t sampleCount) {
    (void)totalSize;
    if (!_combinedHyper || _combinedHyper->GetHandle() == Backend::INVALID_BUFFER_HANDLE) {
        std::vector<float> init(9, 0.0f);
        _combinedHyper = std::make_unique<Runtime::Buffer<float>>(init, Runtime::BufferMode::Read);
    }

    const double biasCorr1 = 1.0 - std::pow(beta1_, step_);
    const double biasCorr2 = 1.0 - std::pow(beta2_, step_);
    std::vector<float> h = {
        lr_, beta1_, beta2_, eps_, weightDecay_, gradClip_,
        static_cast<float>(biasCorr1),
        static_cast<float>(biasCorr2),
        sampleCount == 0 ? 0.0f : 1.0f / static_cast<float>(sampleCount)
    };
    _combinedHyper->Upload(h);
}

void GPUAdam::DispatchCombined(const std::vector<CombinedSlot> &slots, size_t totalSize, bool sync) {
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPUAdam backend not available");

    backend->BindPipeline(_combinedPipeline);
    std::vector<Backend::ResourceBinding> bindings;
    auto addBuffer = [&](uint32_t binding, Backend::BufferHandle handle, bool readOnly) {
        Backend::ResourceBinding rb;
        rb.binding = binding;
        rb.type = Backend::BindingType::Buffer;
        rb.buffer = handle;
        rb.readOnly = readOnly;
        bindings.push_back(rb);
    };
    const uint32_t n = static_cast<uint32_t>(slots.size());
    for (uint32_t i = 0; i < n; i++) addBuffer(i, slots[i].weightHandle, false);
    for (uint32_t i = 0; i < n; i++) addBuffer(n + i, slots[i].gradHandle, true);
    addBuffer(2 * n, _flatM->GetHandle(), false);
    addBuffer(2 * n + 1, _flatV->GetHandle(), false);
    addBuffer(2 * n + 2, _combinedHyper->GetHandle(), true);

    backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
    backend->Dispatch(static_cast<uint32_t>((totalSize + 255) / 256), 1, 1);
    backend->MemoryBarrier(Backend::BarrierType::Buffer);
    if (sync) backend->Finish();
}

std::string GPUAdam::BuildAdamShader(size_t paramSize, size_t sampleCount,
                                      int gradOffset, int gradStride) {
    return std::format(R"GLSL(#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer WeightBuf {{ float weight[]; }};
layout(std430, binding = 1) buffer FirstMomentBuf {{ float m[]; }};
layout(std430, binding = 2) buffer SecondMomentBuf {{ float v[]; }};
layout(std430, binding = 3) readonly buffer GradBuf {{ float grad[]; }};
layout(std430, binding = 4) readonly buffer HyperBuf {{ float h[]; }};

void main() {{
    uint i = gl_GlobalInvocationID.x;
    if (i >= {0}u) return;

    float g = 0.0;
    for (uint s = 0u; s < {1}u; ++s) {{
        g += grad[s * {2}u + ({3}u + i)];
    }}
    g *= h[8];

    if (h[4] > 0.0) g += 2.0 * h[4] * weight[i];
    if (h[5] > 0.0) g = clamp(g, -h[5], h[5]);

    m[i] = h[1] * m[i] + (1.0 - h[1]) * g;
    v[i] = h[2] * v[i] + (1.0 - h[2]) * g * g;

    float mHat = m[i] / h[6];
    float vHat = v[i] / h[7];
    weight[i] -= h[0] * mHat / (sqrt(vHat) + h[3]);
}}
)GLSL", paramSize, sampleCount, gradStride, gradOffset);
}

void GPUAdam::EnsurePipeline(ParamSlotGPU &ps, const AD::ADKernel1D::GradientParamInfo &first) {
    if (ps.pipeline != Backend::INVALID_PIPELINE_HANDLE &&
        ps.compiledSamples == first.sampleCount &&
        ps.compiledGradOffset == first.gradOffset &&
        ps.compiledGradStride == first.gradStride) {
        return;
    }

    Runtime::Context::GetInstance().MakeCurrent();
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPUAdam backend not available");

    if (ps.pipeline != Backend::INVALID_PIPELINE_HANDLE) backend->DestroyPipeline(ps.pipeline);
    if (ps.shader != Backend::INVALID_SHADER_HANDLE) backend->DestroyShader(ps.shader);

    Backend::ShaderDesc shaderDesc;
    shaderDesc.type = Backend::ShaderType::Compute;
    shaderDesc.sourceCode = BuildAdamShader(ps.size, first.sampleCount,
                                            first.gradOffset, first.gradStride);
    ps.shader = backend->CreateShader(shaderDesc);

    Backend::PipelineDesc pipelineDesc;
    pipelineDesc.computeShader = ps.shader;
    pipelineDesc.workGroupSizeX = 256;
    for (uint32_t binding = 0; binding <= 4; binding++) {
        Backend::ResourceLayoutEntry entry;
        entry.binding = binding;
        entry.type = Backend::BindingType::Buffer;
        entry.readOnly = binding >= 3;
        pipelineDesc.resources.push_back(entry);
    }
    ps.pipeline = backend->CreatePipeline(pipelineDesc);
    if (ps.pipeline == Backend::INVALID_PIPELINE_HANDLE)
        throw std::runtime_error("GPUAdam failed to create update pipeline");

    ps.compiledSamples = first.sampleCount;
    ps.compiledGradOffset = first.gradOffset;
    ps.compiledGradStride = first.gradStride;
}

void GPUAdam::UploadHyperParams(ParamSlotGPU &ps, size_t sampleCount) {
    if (!ps.hyper || ps.hyper->GetHandle() == Backend::INVALID_BUFFER_HANDLE) {
        std::vector<float> init(9, 0.0f);
        ps.hyper = std::make_unique<Runtime::Buffer<float>>(init, Runtime::BufferMode::Read);
    }

    const double biasCorr1 = 1.0 - std::pow(beta1_, step_);
    const double biasCorr2 = 1.0 - std::pow(beta2_, step_);
    std::vector<float> h = {
        lr_, beta1_, beta2_, eps_, weightDecay_, gradClip_,
        static_cast<float>(biasCorr1),
        static_cast<float>(biasCorr2),
        sampleCount == 0 ? 0.0f : 1.0f / static_cast<float>(sampleCount)
    };
    ps.hyper->Upload(h);
}

void GPUAdam::DispatchSlot(ParamSlotGPU &ps, Backend::BufferHandle gradHandle, bool sync) {
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPUAdam backend not available");

    backend->BindPipeline(ps.pipeline);

    std::vector<Backend::ResourceBinding> bindings;
    auto addBuffer = [&](uint32_t binding, Backend::BufferHandle handle, bool readOnly) {
        Backend::ResourceBinding rb;
        rb.binding = binding;
        rb.type = Backend::BindingType::Buffer;
        rb.buffer = handle;
        rb.readOnly = readOnly;
        bindings.push_back(rb);
    };
    addBuffer(0, ps.weightHandle, false);
    addBuffer(1, ps.m->GetHandle(), false);
    addBuffer(2, ps.v->GetHandle(), false);
    addBuffer(3, gradHandle, true);
    addBuffer(4, ps.hyper->GetHandle(), true);

    backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
    backend->Dispatch(static_cast<uint32_t>((ps.size + 255) / 256), 1, 1);
    backend->MemoryBarrier(Backend::BarrierType::Buffer);
    if (sync) backend->Finish();
}

void GPUAdam::ReleasePipelines() {
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) return;
    if (_combinedPipeline != Backend::INVALID_PIPELINE_HANDLE) {
        backend->DestroyPipeline(_combinedPipeline);
        _combinedPipeline = Backend::INVALID_PIPELINE_HANDLE;
    }
    if (_combinedShader != Backend::INVALID_SHADER_HANDLE) {
        backend->DestroyShader(_combinedShader);
        _combinedShader = Backend::INVALID_SHADER_HANDLE;
    }
    for (auto &ps : params_) {
        if (ps.pipeline != Backend::INVALID_PIPELINE_HANDLE) {
            backend->DestroyPipeline(ps.pipeline);
            ps.pipeline = Backend::INVALID_PIPELINE_HANDLE;
        }
        if (ps.shader != Backend::INVALID_SHADER_HANDLE) {
            backend->DestroyShader(ps.shader);
            ps.shader = Backend::INVALID_SHADER_HANDLE;
        }
    }
}

// =============================================================================
// GPUSGD
// =============================================================================

void GPUSGD::AddParameter(size_t size, Backend::BufferHandle weightHandle) {
    if (size == 0 || weightHandle == Backend::INVALID_BUFFER_HANDLE)
        throw std::invalid_argument("GPUSGD::AddParameter requires a live GPU weight buffer");

    ParamSlotGPU ps;
    ps.size = size;
    ps.weightHandle = weightHandle;
    std::vector<float> zeros(size, 0.0f);
    ps.velocity = std::make_unique<Runtime::Buffer<float>>(zeros, Runtime::BufferMode::ReadWrite);
    params_.push_back(std::move(ps));
}

void GPUSGD::Step(AD::ADKernel1D &kernel, bool sync) {
    step_++;
    auto gradParams = kernel.GradientParams();

    if (params_.size() * 2 + 2 <= Backend::MAX_BUFFER_BINDINGS) {
        StepCombined(gradParams, sync);
        return;
    }

    size_t paramBase = 0;
    for (auto &ps : params_) {
        if (paramBase + ps.size > gradParams.size())
            throw std::runtime_error("GPUSGD gradient count does not match registered parameters");

        const auto &first = gradParams[paramBase];
        ValidateGradientGroup("GPUSGD", gradParams, paramBase, ps.size, first);
        EnsurePipeline(ps, first);
        UploadHyperParams(ps, first.sampleCount);
        DispatchSlot(ps, first.gradHandle, sync);
        paramBase += ps.size;
    }
}

void GPUSGD::ValidateGradientGroup(const char *name,
                                    const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
                                    size_t paramBase, size_t size,
                                    const AD::ADKernel1D::GradientParamInfo &first) {
    if (first.gradHandle == Backend::INVALID_BUFFER_HANDLE)
        throw std::runtime_error(std::string(name) + " called before ADKernel1D::Backward created gradient buffers");
    for (size_t i = 0; i < size; i++) {
        const auto &gp = gradParams[paramBase + i];
        if (gp.gradHandle != first.gradHandle ||
            gp.gradStride != first.gradStride ||
            gp.sampleCount != first.sampleCount ||
            gp.gradOffset != first.gradOffset + static_cast<int>(i)) {
            throw std::runtime_error(std::string(name) + " requires tensor parameters to map to one contiguous gradient group");
        }
    }
}

std::vector<GPUSGD::CombinedSlot>
GPUSGD::BuildCombinedSlots(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
                            size_t &totalSize) const {
    std::vector<CombinedSlot> slots;
    slots.reserve(params_.size());
    totalSize = 0;

    size_t paramBase = 0;
    for (const auto &ps : params_) {
        if (paramBase + ps.size > gradParams.size())
            throw std::runtime_error("GPUSGD gradient count does not match registered parameters");

        const auto &first = gradParams[paramBase];
        ValidateGradientGroup("GPUSGD", gradParams, paramBase, ps.size, first);

        CombinedSlot slot;
        slot.size = ps.size;
        slot.base = totalSize;
        slot.sampleCount = first.sampleCount;
        slot.gradOffset = first.gradOffset;
        slot.gradStride = first.gradStride;
        slot.weightHandle = ps.weightHandle;
        slot.gradHandle = first.gradHandle;
        slots.push_back(slot);

        totalSize += ps.size;
        paramBase += ps.size;
    }
    return slots;
}

std::string GPUSGD::CombinedSignature(const std::vector<CombinedSlot> &slots,
                                       size_t totalSize) const {
    std::string sig = std::to_string(totalSize);
    for (const auto &s : slots) {
        sig += std::format("|{}:{}:{}:{}:{}", s.base, s.size,
                           s.sampleCount, s.gradOffset, s.gradStride);
    }
    return sig;
}

std::string GPUSGD::BuildCombinedShader(const std::vector<CombinedSlot> &slots,
                                         size_t totalSize) {
    const size_t n = slots.size();
    std::string src = "#version 430\nlayout(local_size_x = 256) in;\n\n";
    for (size_t i = 0; i < n; i++) {
        src += std::format("layout(std430, binding = {}) buffer WeightBuf{} {{ float weight{}[]; }};\n",
                           i, i, i);
    }
    for (size_t i = 0; i < n; i++) {
        src += std::format("layout(std430, binding = {}) readonly buffer GradBuf{} {{ float grad{}[]; }};\n",
                           n + i, i, i);
    }
    src += std::format("layout(std430, binding = {}) buffer VelocityBuf {{ float velocity[]; }};\n", 2 * n);
    src += std::format("layout(std430, binding = {}) readonly buffer HyperBuf {{ float h[]; }};\n\n", 2 * n + 1);
    src += "void main() {\n";
    src += "    uint i = gl_GlobalInvocationID.x;\n";
    src += std::format("    if (i >= {}u) return;\n\n", totalSize);

    for (size_t si = 0; si < n; si++) {
        const auto &s = slots[si];
        src += std::format("    if (i >= {}u && i < {}u) {{\n",
                           s.base, s.base + s.size);
        src += std::format("        uint j = i - {}u;\n", s.base);
        src += "        float g = 0.0;\n";
        src += std::format("        for (uint sidx = 0u; sidx < {}u; ++sidx) {{\n",
                           s.sampleCount);
        src += std::format("            g += grad{}[sidx * {}u + ({}u + j)];\n",
                           si, s.gradStride, s.gradOffset);
        src += "        }\n";
        src += "        g *= h[4];\n";
        src += std::format("        if (h[2] > 0.0) g += 2.0 * h[2] * weight{}[j];\n", si);
        src += "        if (h[3] > 0.0) g = clamp(g, -h[3], h[3]);\n";
        src += "        velocity[i] = h[1] * velocity[i] + g;\n";
        src += std::format("        weight{}[j] -= h[0] * velocity[i];\n", si);
        src += "        return;\n";
        src += "    }\n";
    }

    src += "}\n";
    return src;
}

void GPUSGD::StepCombined(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
                           bool sync) {
    size_t totalSize = 0;
    auto slots = BuildCombinedSlots(gradParams, totalSize);
    if (totalSize == 0) return;

    EnsureCombinedPipeline(slots, totalSize);
    UploadCombinedHyperParams(slots.empty() ? 0 : slots[0].sampleCount);
    DispatchCombined(slots, totalSize, sync);
}

void GPUSGD::EnsureCombinedPipeline(const std::vector<CombinedSlot> &slots, size_t totalSize) {
    std::string sig = CombinedSignature(slots, totalSize);
    if (_combinedPipeline != Backend::INVALID_PIPELINE_HANDLE &&
        _combinedSignature == sig) return;

    Runtime::Context::GetInstance().MakeCurrent();
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPUSGD backend not available");

    if (_combinedPipeline != Backend::INVALID_PIPELINE_HANDLE)
        backend->DestroyPipeline(_combinedPipeline);
    if (_combinedShader != Backend::INVALID_SHADER_HANDLE)
        backend->DestroyShader(_combinedShader);

    if (!_flatVelocity || _flatVelocitySize != totalSize) {
        std::vector<float> zeros(totalSize, 0.0f);
        _flatVelocity = std::make_unique<Runtime::Buffer<float>>(zeros, Runtime::BufferMode::ReadWrite);
        _flatVelocitySize = totalSize;
    }

    Backend::ShaderDesc shaderDesc;
    shaderDesc.type = Backend::ShaderType::Compute;
    shaderDesc.sourceCode = BuildCombinedShader(slots, totalSize);
    _combinedShader = backend->CreateShader(shaderDesc);

    Backend::PipelineDesc pipelineDesc;
    pipelineDesc.computeShader = _combinedShader;
    pipelineDesc.workGroupSizeX = 256;
    const uint32_t bindingCount = static_cast<uint32_t>(slots.size() * 2 + 2);
    for (uint32_t binding = 0; binding < bindingCount; binding++) {
        Backend::ResourceLayoutEntry entry;
        entry.binding = binding;
        entry.type = Backend::BindingType::Buffer;
        entry.readOnly = (binding >= slots.size() && binding < slots.size() * 2)
            || binding == bindingCount - 1;
        pipelineDesc.resources.push_back(entry);
    }
    _combinedPipeline = backend->CreatePipeline(pipelineDesc);
    if (_combinedPipeline == Backend::INVALID_PIPELINE_HANDLE)
        throw std::runtime_error("GPUSGD failed to create combined update pipeline");

    _combinedSignature = std::move(sig);
}

void GPUSGD::UploadCombinedHyperParams(size_t sampleCount) {
    if (!_combinedHyper || _combinedHyper->GetHandle() == Backend::INVALID_BUFFER_HANDLE) {
        std::vector<float> init(5, 0.0f);
        _combinedHyper = std::make_unique<Runtime::Buffer<float>>(init, Runtime::BufferMode::Read);
    }
    std::vector<float> h = {
        lr_, momentum_, weightDecay_, gradClip_,
        sampleCount == 0 ? 0.0f : 1.0f / static_cast<float>(sampleCount)
    };
    _combinedHyper->Upload(h);
}

void GPUSGD::DispatchCombined(const std::vector<CombinedSlot> &slots, size_t totalSize, bool sync) {
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPUSGD backend not available");

    backend->BindPipeline(_combinedPipeline);
    std::vector<Backend::ResourceBinding> bindings;
    auto addBuffer = [&](uint32_t binding, Backend::BufferHandle handle, bool readOnly) {
        Backend::ResourceBinding rb;
        rb.binding = binding;
        rb.type = Backend::BindingType::Buffer;
        rb.buffer = handle;
        rb.readOnly = readOnly;
        bindings.push_back(rb);
    };
    const uint32_t n = static_cast<uint32_t>(slots.size());
    for (uint32_t i = 0; i < n; i++) addBuffer(i, slots[i].weightHandle, false);
    for (uint32_t i = 0; i < n; i++) addBuffer(n + i, slots[i].gradHandle, true);
    addBuffer(2 * n, _flatVelocity->GetHandle(), false);
    addBuffer(2 * n + 1, _combinedHyper->GetHandle(), true);

    backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
    backend->Dispatch(static_cast<uint32_t>((totalSize + 255) / 256), 1, 1);
    backend->MemoryBarrier(Backend::BarrierType::Buffer);
    if (sync) backend->Finish();
}

std::string GPUSGD::BuildShader(size_t paramSize, size_t sampleCount,
                                 int gradOffset, int gradStride) {
    return std::format(R"GLSL(#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer WeightBuf {{ float weight[]; }};
layout(std430, binding = 1) buffer VelocityBuf {{ float velocity[]; }};
layout(std430, binding = 2) readonly buffer GradBuf {{ float grad[]; }};
layout(std430, binding = 3) readonly buffer HyperBuf {{ float h[]; }};

void main() {{
    uint i = gl_GlobalInvocationID.x;
    if (i >= {0}u) return;

    float g = 0.0;
    for (uint s = 0u; s < {1}u; ++s) {{
        g += grad[s * {2}u + ({3}u + i)];
    }}
    g *= h[4];

    if (h[2] > 0.0) g += 2.0 * h[2] * weight[i];
    if (h[3] > 0.0) g = clamp(g, -h[3], h[3]);

    velocity[i] = h[1] * velocity[i] + g;
    weight[i] -= h[0] * velocity[i];
}}
)GLSL", paramSize, sampleCount, gradStride, gradOffset);
}

void GPUSGD::EnsurePipeline(ParamSlotGPU &ps, const AD::ADKernel1D::GradientParamInfo &first) {
    if (ps.pipeline != Backend::INVALID_PIPELINE_HANDLE &&
        ps.compiledSamples == first.sampleCount &&
        ps.compiledGradOffset == first.gradOffset &&
        ps.compiledGradStride == first.gradStride) return;

    Runtime::Context::GetInstance().MakeCurrent();
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPUSGD backend not available");

    if (ps.pipeline != Backend::INVALID_PIPELINE_HANDLE) backend->DestroyPipeline(ps.pipeline);
    if (ps.shader != Backend::INVALID_SHADER_HANDLE) backend->DestroyShader(ps.shader);

    Backend::ShaderDesc shaderDesc;
    shaderDesc.type = Backend::ShaderType::Compute;
    shaderDesc.sourceCode = BuildShader(ps.size, first.sampleCount, first.gradOffset, first.gradStride);
    ps.shader = backend->CreateShader(shaderDesc);

    Backend::PipelineDesc pipelineDesc;
    pipelineDesc.computeShader = ps.shader;
    pipelineDesc.workGroupSizeX = 256;
    for (uint32_t binding = 0; binding <= 3; binding++) {
        Backend::ResourceLayoutEntry entry;
        entry.binding = binding;
        entry.type = Backend::BindingType::Buffer;
        entry.readOnly = binding >= 2;
        pipelineDesc.resources.push_back(entry);
    }
    ps.pipeline = backend->CreatePipeline(pipelineDesc);
    if (ps.pipeline == Backend::INVALID_PIPELINE_HANDLE)
        throw std::runtime_error("GPUSGD failed to create update pipeline");

    ps.compiledSamples = first.sampleCount;
    ps.compiledGradOffset = first.gradOffset;
    ps.compiledGradStride = first.gradStride;
}

void GPUSGD::UploadHyperParams(ParamSlotGPU &ps, size_t sampleCount) {
    if (!ps.hyper || ps.hyper->GetHandle() == Backend::INVALID_BUFFER_HANDLE) {
        std::vector<float> init(5, 0.0f);
        ps.hyper = std::make_unique<Runtime::Buffer<float>>(init, Runtime::BufferMode::Read);
    }
    std::vector<float> h = {
        lr_, momentum_, weightDecay_, gradClip_,
        sampleCount == 0 ? 0.0f : 1.0f / static_cast<float>(sampleCount)
    };
    ps.hyper->Upload(h);
}

void GPUSGD::DispatchSlot(ParamSlotGPU &ps, Backend::BufferHandle gradHandle, bool sync) {
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPUSGD backend not available");

    backend->BindPipeline(ps.pipeline);
    std::vector<Backend::ResourceBinding> bindings;
    auto addBuffer = [&](uint32_t binding, Backend::BufferHandle handle, bool readOnly) {
        Backend::ResourceBinding rb;
        rb.binding = binding;
        rb.type = Backend::BindingType::Buffer;
        rb.buffer = handle;
        rb.readOnly = readOnly;
        bindings.push_back(rb);
    };
    addBuffer(0, ps.weightHandle, false);
    addBuffer(1, ps.velocity->GetHandle(), false);
    addBuffer(2, gradHandle, true);
    addBuffer(3, ps.hyper->GetHandle(), true);
    backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
    backend->Dispatch(static_cast<uint32_t>((ps.size + 255) / 256), 1, 1);
    backend->MemoryBarrier(Backend::BarrierType::Buffer);
    if (sync) backend->Finish();
}

void GPUSGD::ReleasePipelines() {
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) return;
    if (_combinedPipeline != Backend::INVALID_PIPELINE_HANDLE) {
        backend->DestroyPipeline(_combinedPipeline);
        _combinedPipeline = Backend::INVALID_PIPELINE_HANDLE;
    }
    if (_combinedShader != Backend::INVALID_SHADER_HANDLE) {
        backend->DestroyShader(_combinedShader);
        _combinedShader = Backend::INVALID_SHADER_HANDLE;
    }
    for (auto &ps : params_) {
        if (ps.pipeline != Backend::INVALID_PIPELINE_HANDLE) {
            backend->DestroyPipeline(ps.pipeline);
            ps.pipeline = Backend::INVALID_PIPELINE_HANDLE;
        }
        if (ps.shader != Backend::INVALID_SHADER_HANDLE) {
            backend->DestroyShader(ps.shader);
            ps.shader = Backend::INVALID_SHADER_HANDLE;
        }
    }
}

// =============================================================================
// GPURMSprop
// =============================================================================

void GPURMSprop::AddParameter(size_t size, Backend::BufferHandle weightHandle) {
    if (size == 0 || weightHandle == Backend::INVALID_BUFFER_HANDLE)
        throw std::invalid_argument("GPURMSprop::AddParameter requires a live GPU weight buffer");

    ParamSlotGPU ps;
    ps.size = size;
    ps.weightHandle = weightHandle;
    std::vector<float> zeros(size, 0.0f);
    ps.squareAvg = std::make_unique<Runtime::Buffer<float>>(zeros, Runtime::BufferMode::ReadWrite);
    params_.push_back(std::move(ps));
}

void GPURMSprop::Step(AD::ADKernel1D &kernel, bool sync) {
    step_++;
    auto gradParams = kernel.GradientParams();

    if (params_.size() * 2 + 2 <= Backend::MAX_BUFFER_BINDINGS) {
        StepCombined(gradParams, sync);
        return;
    }

    size_t paramBase = 0;
    for (auto &ps : params_) {
        if (paramBase + ps.size > gradParams.size())
            throw std::runtime_error("GPURMSprop gradient count does not match registered parameters");

        const auto &first = gradParams[paramBase];
        ValidateGradientGroup("GPURMSprop", gradParams, paramBase, ps.size, first);
        EnsurePipeline(ps, first);
        UploadHyperParams(ps, first.sampleCount);
        DispatchSlot(ps, first.gradHandle, sync);
        paramBase += ps.size;
    }
}

void GPURMSprop::ValidateGradientGroup(const char *name,
                                        const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
                                        size_t paramBase, size_t size,
                                        const AD::ADKernel1D::GradientParamInfo &first) {
    if (first.gradHandle == Backend::INVALID_BUFFER_HANDLE)
        throw std::runtime_error(std::string(name) + " called before ADKernel1D::Backward created gradient buffers");
    for (size_t i = 0; i < size; i++) {
        const auto &gp = gradParams[paramBase + i];
        if (gp.gradHandle != first.gradHandle ||
            gp.gradStride != first.gradStride ||
            gp.sampleCount != first.sampleCount ||
            gp.gradOffset != first.gradOffset + static_cast<int>(i)) {
            throw std::runtime_error(std::string(name) + " requires tensor parameters to map to one contiguous gradient group");
        }
    }
}

std::vector<GPURMSprop::CombinedSlot>
GPURMSprop::BuildCombinedSlots(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
                                size_t &totalSize) const {
    std::vector<CombinedSlot> slots;
    slots.reserve(params_.size());
    totalSize = 0;

    size_t paramBase = 0;
    for (const auto &ps : params_) {
        if (paramBase + ps.size > gradParams.size())
            throw std::runtime_error("GPURMSprop gradient count does not match registered parameters");

        const auto &first = gradParams[paramBase];
        ValidateGradientGroup("GPURMSprop", gradParams, paramBase, ps.size, first);

        CombinedSlot slot;
        slot.size = ps.size;
        slot.base = totalSize;
        slot.sampleCount = first.sampleCount;
        slot.gradOffset = first.gradOffset;
        slot.gradStride = first.gradStride;
        slot.weightHandle = ps.weightHandle;
        slot.gradHandle = first.gradHandle;
        slots.push_back(slot);

        totalSize += ps.size;
        paramBase += ps.size;
    }
    return slots;
}

std::string GPURMSprop::CombinedSignature(const std::vector<CombinedSlot> &slots,
                                           size_t totalSize) const {
    std::string sig = std::to_string(totalSize);
    for (const auto &s : slots) {
        sig += std::format("|{}:{}:{}:{}:{}", s.base, s.size,
                           s.sampleCount, s.gradOffset, s.gradStride);
    }
    return sig;
}

std::string GPURMSprop::BuildCombinedShader(const std::vector<CombinedSlot> &slots,
                                             size_t totalSize) {
    const size_t n = slots.size();
    std::string src = "#version 430\nlayout(local_size_x = 256) in;\n\n";
    for (size_t i = 0; i < n; i++) {
        src += std::format("layout(std430, binding = {}) buffer WeightBuf{} {{ float weight{}[]; }};\n",
                           i, i, i);
    }
    for (size_t i = 0; i < n; i++) {
        src += std::format("layout(std430, binding = {}) readonly buffer GradBuf{} {{ float grad{}[]; }};\n",
                           n + i, i, i);
    }
    src += std::format("layout(std430, binding = {}) buffer SquareAvgBuf {{ float squareAvg[]; }};\n", 2 * n);
    src += std::format("layout(std430, binding = {}) readonly buffer HyperBuf {{ float h[]; }};\n\n", 2 * n + 1);
    src += "void main() {\n";
    src += "    uint i = gl_GlobalInvocationID.x;\n";
    src += std::format("    if (i >= {}u) return;\n\n", totalSize);

    for (size_t si = 0; si < n; si++) {
        const auto &s = slots[si];
        src += std::format("    if (i >= {}u && i < {}u) {{\n",
                           s.base, s.base + s.size);
        src += std::format("        uint j = i - {}u;\n", s.base);
        src += "        float g = 0.0;\n";
        src += std::format("        for (uint sidx = 0u; sidx < {}u; ++sidx) {{\n",
                           s.sampleCount);
        src += std::format("            g += grad{}[sidx * {}u + ({}u + j)];\n",
                           si, s.gradStride, s.gradOffset);
        src += "        }\n";
        src += "        g *= h[5];\n";
        src += std::format("        if (h[3] > 0.0) g += 2.0 * h[3] * weight{}[j];\n", si);
        src += "        if (h[4] > 0.0) g = clamp(g, -h[4], h[4]);\n";
        src += "        squareAvg[i] = h[1] * squareAvg[i] + (1.0 - h[1]) * g * g;\n";
        src += std::format("        weight{}[j] -= h[0] * g / sqrt(squareAvg[i] + h[2]);\n", si);
        src += "        return;\n";
        src += "    }\n";
    }

    src += "}\n";
    return src;
}

void GPURMSprop::StepCombined(const std::vector<AD::ADKernel1D::GradientParamInfo> &gradParams,
                               bool sync) {
    size_t totalSize = 0;
    auto slots = BuildCombinedSlots(gradParams, totalSize);
    if (totalSize == 0) return;

    EnsureCombinedPipeline(slots, totalSize);
    UploadCombinedHyperParams(slots.empty() ? 0 : slots[0].sampleCount);
    DispatchCombined(slots, totalSize, sync);
}

void GPURMSprop::EnsureCombinedPipeline(const std::vector<CombinedSlot> &slots, size_t totalSize) {
    std::string sig = CombinedSignature(slots, totalSize);
    if (_combinedPipeline != Backend::INVALID_PIPELINE_HANDLE &&
        _combinedSignature == sig) return;

    Runtime::Context::GetInstance().MakeCurrent();
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPURMSprop backend not available");

    if (_combinedPipeline != Backend::INVALID_PIPELINE_HANDLE)
        backend->DestroyPipeline(_combinedPipeline);
    if (_combinedShader != Backend::INVALID_SHADER_HANDLE)
        backend->DestroyShader(_combinedShader);

    if (!_flatSquareAvg || _flatSquareAvgSize != totalSize) {
        std::vector<float> zeros(totalSize, 0.0f);
        _flatSquareAvg = std::make_unique<Runtime::Buffer<float>>(zeros, Runtime::BufferMode::ReadWrite);
        _flatSquareAvgSize = totalSize;
    }

    Backend::ShaderDesc shaderDesc;
    shaderDesc.type = Backend::ShaderType::Compute;
    shaderDesc.sourceCode = BuildCombinedShader(slots, totalSize);
    _combinedShader = backend->CreateShader(shaderDesc);

    Backend::PipelineDesc pipelineDesc;
    pipelineDesc.computeShader = _combinedShader;
    pipelineDesc.workGroupSizeX = 256;
    const uint32_t bindingCount = static_cast<uint32_t>(slots.size() * 2 + 2);
    for (uint32_t binding = 0; binding < bindingCount; binding++) {
        Backend::ResourceLayoutEntry entry;
        entry.binding = binding;
        entry.type = Backend::BindingType::Buffer;
        entry.readOnly = (binding >= slots.size() && binding < slots.size() * 2)
            || binding == bindingCount - 1;
        pipelineDesc.resources.push_back(entry);
    }
    _combinedPipeline = backend->CreatePipeline(pipelineDesc);
    if (_combinedPipeline == Backend::INVALID_PIPELINE_HANDLE)
        throw std::runtime_error("GPURMSprop failed to create combined update pipeline");

    _combinedSignature = std::move(sig);
}

void GPURMSprop::UploadCombinedHyperParams(size_t sampleCount) {
    if (!_combinedHyper || _combinedHyper->GetHandle() == Backend::INVALID_BUFFER_HANDLE) {
        std::vector<float> init(6, 0.0f);
        _combinedHyper = std::make_unique<Runtime::Buffer<float>>(init, Runtime::BufferMode::Read);
    }
    std::vector<float> h = {
        lr_, beta_, eps_, weightDecay_, gradClip_,
        sampleCount == 0 ? 0.0f : 1.0f / static_cast<float>(sampleCount)
    };
    _combinedHyper->Upload(h);
}

void GPURMSprop::DispatchCombined(const std::vector<CombinedSlot> &slots, size_t totalSize, bool sync) {
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPURMSprop backend not available");

    backend->BindPipeline(_combinedPipeline);
    std::vector<Backend::ResourceBinding> bindings;
    auto addBuffer = [&](uint32_t binding, Backend::BufferHandle handle, bool readOnly) {
        Backend::ResourceBinding rb;
        rb.binding = binding;
        rb.type = Backend::BindingType::Buffer;
        rb.buffer = handle;
        rb.readOnly = readOnly;
        bindings.push_back(rb);
    };
    const uint32_t n = static_cast<uint32_t>(slots.size());
    for (uint32_t i = 0; i < n; i++) addBuffer(i, slots[i].weightHandle, false);
    for (uint32_t i = 0; i < n; i++) addBuffer(n + i, slots[i].gradHandle, true);
    addBuffer(2 * n, _flatSquareAvg->GetHandle(), false);
    addBuffer(2 * n + 1, _combinedHyper->GetHandle(), true);

    backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
    backend->Dispatch(static_cast<uint32_t>((totalSize + 255) / 256), 1, 1);
    backend->MemoryBarrier(Backend::BarrierType::Buffer);
    if (sync) backend->Finish();
}

std::string GPURMSprop::BuildShader(size_t paramSize, size_t sampleCount,
                                     int gradOffset, int gradStride) {
    return std::format(R"GLSL(#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer WeightBuf {{ float weight[]; }};
layout(std430, binding = 1) buffer SquareAvgBuf {{ float squareAvg[]; }};
layout(std430, binding = 2) readonly buffer GradBuf {{ float grad[]; }};
layout(std430, binding = 3) readonly buffer HyperBuf {{ float h[]; }};

void main() {{
    uint i = gl_GlobalInvocationID.x;
    if (i >= {0}u) return;

    float g = 0.0;
    for (uint s = 0u; s < {1}u; ++s) {{
        g += grad[s * {2}u + ({3}u + i)];
    }}
    g *= h[5];

    if (h[3] > 0.0) g += 2.0 * h[3] * weight[i];
    if (h[4] > 0.0) g = clamp(g, -h[4], h[4]);

    squareAvg[i] = h[1] * squareAvg[i] + (1.0 - h[1]) * g * g;
    weight[i] -= h[0] * g / sqrt(squareAvg[i] + h[2]);
}}
)GLSL", paramSize, sampleCount, gradStride, gradOffset);
}

void GPURMSprop::EnsurePipeline(ParamSlotGPU &ps, const AD::ADKernel1D::GradientParamInfo &first) {
    if (ps.pipeline != Backend::INVALID_PIPELINE_HANDLE &&
        ps.compiledSamples == first.sampleCount &&
        ps.compiledGradOffset == first.gradOffset &&
        ps.compiledGradStride == first.gradStride) return;

    Runtime::Context::GetInstance().MakeCurrent();
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPURMSprop backend not available");

    if (ps.pipeline != Backend::INVALID_PIPELINE_HANDLE) backend->DestroyPipeline(ps.pipeline);
    if (ps.shader != Backend::INVALID_SHADER_HANDLE) backend->DestroyShader(ps.shader);

    Backend::ShaderDesc shaderDesc;
    shaderDesc.type = Backend::ShaderType::Compute;
    shaderDesc.sourceCode = BuildShader(ps.size, first.sampleCount, first.gradOffset, first.gradStride);
    ps.shader = backend->CreateShader(shaderDesc);

    Backend::PipelineDesc pipelineDesc;
    pipelineDesc.computeShader = ps.shader;
    pipelineDesc.workGroupSizeX = 256;
    for (uint32_t binding = 0; binding <= 3; binding++) {
        Backend::ResourceLayoutEntry entry;
        entry.binding = binding;
        entry.type = Backend::BindingType::Buffer;
        entry.readOnly = binding >= 2;
        pipelineDesc.resources.push_back(entry);
    }
    ps.pipeline = backend->CreatePipeline(pipelineDesc);
    if (ps.pipeline == Backend::INVALID_PIPELINE_HANDLE)
        throw std::runtime_error("GPURMSprop failed to create update pipeline");

    ps.compiledSamples = first.sampleCount;
    ps.compiledGradOffset = first.gradOffset;
    ps.compiledGradStride = first.gradStride;
}

void GPURMSprop::UploadHyperParams(ParamSlotGPU &ps, size_t sampleCount) {
    if (!ps.hyper || ps.hyper->GetHandle() == Backend::INVALID_BUFFER_HANDLE) {
        std::vector<float> init(6, 0.0f);
        ps.hyper = std::make_unique<Runtime::Buffer<float>>(init, Runtime::BufferMode::Read);
    }
    std::vector<float> h = {
        lr_, beta_, eps_, weightDecay_, gradClip_,
        sampleCount == 0 ? 0.0f : 1.0f / static_cast<float>(sampleCount)
    };
    ps.hyper->Upload(h);
}

void GPURMSprop::DispatchSlot(ParamSlotGPU &ps, Backend::BufferHandle gradHandle, bool sync) {
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) throw std::runtime_error("GPURMSprop backend not available");

    backend->BindPipeline(ps.pipeline);
    std::vector<Backend::ResourceBinding> bindings;
    auto addBuffer = [&](uint32_t binding, Backend::BufferHandle handle, bool readOnly) {
        Backend::ResourceBinding rb;
        rb.binding = binding;
        rb.type = Backend::BindingType::Buffer;
        rb.buffer = handle;
        rb.readOnly = readOnly;
        bindings.push_back(rb);
    };
    addBuffer(0, ps.weightHandle, false);
    addBuffer(1, ps.squareAvg->GetHandle(), false);
    addBuffer(2, gradHandle, true);
    addBuffer(3, ps.hyper->GetHandle(), true);
    backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
    backend->Dispatch(static_cast<uint32_t>((ps.size + 255) / 256), 1, 1);
    backend->MemoryBarrier(Backend::BarrierType::Buffer);
    if (sync) backend->Finish();
}

void GPURMSprop::ReleasePipelines() {
    auto *backend = Runtime::Context::GetBackend();
    if (!backend) return;
    if (_combinedPipeline != Backend::INVALID_PIPELINE_HANDLE) {
        backend->DestroyPipeline(_combinedPipeline);
        _combinedPipeline = Backend::INVALID_PIPELINE_HANDLE;
    }
    if (_combinedShader != Backend::INVALID_SHADER_HANDLE) {
        backend->DestroyShader(_combinedShader);
        _combinedShader = Backend::INVALID_SHADER_HANDLE;
    }
    for (auto &ps : params_) {
        if (ps.pipeline != Backend::INVALID_PIPELINE_HANDLE) {
            backend->DestroyPipeline(ps.pipeline);
            ps.pipeline = Backend::INVALID_PIPELINE_HANDLE;
        }
        if (ps.shader != Backend::INVALID_SHADER_HANDLE) {
            backend->DestroyShader(ps.shader);
            ps.shader = Backend::INVALID_SHADER_HANDLE;
        }
    }
}

// =============================================================================
// SGD
// =============================================================================

void SGD::Step(AD::ADKernel1D &kernel) {
    if (!gpu_) {
        gpu_ = std::make_unique<GPUSGD>(lr_, momentum_);
        for (const auto &ps : params_) {
            if (!ps.buffer)
                throw std::runtime_error("SGD::Step now requires GPU-backed parameters; use AddTensor or pass a Buffer");
            gpu_->AddParameter(ps.size, ps.buffer->GetHandle());
        }
    }
    gpu_->SetWeightDecay(weightDecay_);
    gpu_->SetGradClip(gradClip_);
    gpu_->Step(kernel, false);
    step_ = gpu_->GetStep();
}

// =============================================================================
// RMSprop
// =============================================================================

void RMSprop::Step(AD::ADKernel1D &kernel) {
    if (!gpu_) {
        gpu_ = std::make_unique<GPURMSprop>(lr_, beta_, eps_);
        for (const auto &ps : params_) {
            if (!ps.buffer)
                throw std::runtime_error("RMSprop::Step now requires GPU-backed parameters; use AddTensor or pass a Buffer");
            gpu_->AddParameter(ps.size, ps.buffer->GetHandle());
        }
    }
    gpu_->SetWeightDecay(weightDecay_);
    gpu_->SetGradClip(gradClip_);
    gpu_->Step(kernel, false);
    step_ = gpu_->GetStep();
}

} // namespace GPU::NN
