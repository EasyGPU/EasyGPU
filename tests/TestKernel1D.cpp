/**
 * TestKernel1D.cpp:
 *      @Author         :   Margoo(qiuzhengyu@siggraph.org)
 *      @Date           :   2/13/2026
 */

#include <Kernel/Kernel.h>
#include <Runtime/Buffer.h>
#include <IR/Value/Var.h>

#include <iostream>

int main() {
    using namespace GPU;

    std::vector<float> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> output;

    Runtime::Buffer<float> buffer(input.size());
    Runtime::Buffer<float> inputBuffer(input);

    Kernel::Kernel1D kernel(
            [&](IR::Value::Var<int> &Id) {
                auto boundBuffer = buffer.Bind();
                auto boundInputBuffer = inputBuffer.Bind();

                boundBuffer[Id] = boundInputBuffer[Id] + 1.f;
            },
            input.size());
    kernel.Dispatch(1, true);

    buffer.Download(output);
    for (auto &item : output) {
        std::cout << item << " ";
    }

    return 0;
}