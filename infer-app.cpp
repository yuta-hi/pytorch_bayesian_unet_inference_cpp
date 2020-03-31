#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <itkImage.h>
#include <itkImageFileReader.h>


void print_tensor(const at::Tensor &x) {
	std::cout << x << std::endl;
}


namespace cvt {

	template<typename T>
	at::ScalarType scalar_type() {
		return at::typeMetaToScalarType(caffe2::TypeMeta::Make<T>());
	}

	template<typename T, unsigned int N>
	std::vector<T> itk_to_vector(const itk::Image<T, N> &src) {
		const auto numel = src.GetLargestPossibleRegion().GetNumberOfPixels();
		std::vector<T> dst(numel);
		std::memcpy(&dst[0], src.GetBufferPointer(), numel * sizeof(T));
		return dst;
	}

	template<typename T, unsigned int N>
	torch::Tensor itk_to_tensor(const itk::Image<T, N> &src) {
		const auto numel = src.GetLargestPossibleRegion().GetNumberOfPixels();
		const auto size  = src.GetLargestPossibleRegion().GetSize();

		std::vector<int64_t> size_int64_t;
		for (const auto &s : size) size_int64_t.push_back(static_cast<int64_t>(s));
		const auto dtype = cvt::scalar_type<T>();

		torch::Tensor dst = torch::empty(size_int64_t, dtype);
		std::memcpy(dst.data_ptr(), src.GetBufferPointer(), numel * sizeof(T));
		return dst;
	}
}


template<typename PixelType, unsigned int N>
torch::Tensor read_image(const std::string path){

	using ImageType = itk::Image<PixelType, N>;
	using ReaderType = itk::ImageFileReader<ImageType>;

	ImageType::ConstPointer image;
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(path);
	try {
		reader->Update();
		image = reader->GetOutput();
	}
	catch (itk::ExceptionObject &err)
	{
		std::cerr << err << std::endl;
		exit(-1);
	}

	return cvt::itk_to_tensor<PixelType, N>(*image);
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
