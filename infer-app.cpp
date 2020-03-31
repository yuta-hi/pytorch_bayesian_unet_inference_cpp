#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>


void print_tensor(const at::Tensor &x) {
	std::cout << x << std::endl;
}


int main(int argc, const char* argv[]) {

	if (argc != 2) {
		std::cerr << "usage: infer-app <path-to-model>" << std::endl;
		return -1;
	}

	const auto path = argv[1];
	std::cout << path << std::endl;

	const auto b = 1, ch = 1, w = 40, h = 60;

	torch::manual_seed(0);
	torch::NoGradGuard no_grad;

	assert(torch::cuda::is_available());
	const auto device = torch::kCUDA;

	try {
		auto model = torch::jit::load(path);
		assert(model != nullptr);
		model.eval();
		model.to(device);

		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(torch::ones({ b, ch, w, h }).to(device)); // NOTE: dummy data

		auto outputs = model.forward(inputs);

		auto n_outputs = 1;
		if (outputs.isTuple()) n_outputs = outputs.toTuple()->elements().size();

		assert(n_outputs >= 2);

		auto label = outputs.toTuple()->elements()[0].toTensor();
		auto uncertainty = outputs.toTuple()->elements()[1].toTensor();

		std::cout << "label:" << std::endl;
		print_tensor(label);
		std::cout << "uncertainty:" << std::endl;
		print_tensor(uncertainty);

	}
	catch (std::runtime_error& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}
	catch (const c10::Error& e)
	{
		std::cerr << e.msg() << std::endl;
		return -1;
	}

	return 0;
}
