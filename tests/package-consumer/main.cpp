#include <GPU.h>

#include <string>

int main() {
	GPU::Kernel::InspectorKernel1D kernel(
		[](GPU::IR::Value::Var<int> &id) { GPU::IR::Value::Var<int> doubled = id * 2; });

	std::string error;
	return kernel.Validate(error) ? 0 : 1;
}
