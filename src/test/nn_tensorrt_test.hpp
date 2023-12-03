#include <iostream>

#include "cshogi.h"
#include "nn_tensorrt.hpp"


int NNTensorRT_test0() {
	std::cout << "info: NNTensorRT_test0() start" << std::endl;

	const int max_batch_size = 512;
	std::cout << "info: init start" << std::endl;
	//NNTensorRT nn("x64/Debug/model_2_3000.onnx", max_batch_size);
	NNTensorRT nn("model_2_3000.onnx", max_batch_size);

	std::cout << "info: alloc start" << std::endl;

	float* input = nullptr;
	float* y_policy = nullptr;
	float* y_value = nullptr;
	cudaHostAlloc((void**)&input, sizeof(float) * N_FEATURE_WHC * max_batch_size, cudaHostAllocPortable);
	cudaHostAlloc((void**)&y_policy, sizeof(float) * N_LABEL_SIZE * max_batch_size, cudaHostAllocPortable);
	cudaHostAlloc((void**)&y_value, sizeof(float) * 1 * max_batch_size, cudaHostAllocPortable);

	y_policy[0] = 5;
	y_value[0] = 8;

	__Board board;
	std::cout << "info: set start" << std::endl;
	board.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
	std::cout << "info: make_input_features start" << std::endl;
	board.make_input_features(input);
	std::cout << "info: forward start" << std::endl;
	nn.forward(1, input, y_policy, y_value);
	std::cout << "info: forward done" << std::endl;

	auto pretty_print_as_3d = [](const float* ptr, const int shape0, const int shape1, const int shape2) {
		const int size = shape0 * shape1 * shape2;
		for (int i = 0; i < size; ++i) {
			std::cout << ptr[i] << " ";
			if ((i + 1) % shape2 == 0) {
				std::cout << std::endl;
			}
			if ((i + 1) % (shape1 * shape2) == 0) {
				std::cout << std::endl << std::endl;
			}
		}
		};

	std::cout << "info: pretty print start" << std::endl;
	pretty_print_as_3d(y_policy, N_LABEL_CHANNEL, 9, 9);

	std::cout << "info: value = " << std::endl;
	std::cout << *y_value << std::endl;

	std::cout << "info: " << typeid(y_policy).name() << std::endl;
	std::cout << "info: " << typeid(y_value).name() << std::endl;

	std::cout << "info: NNTensorRT_test0() done" << std::endl;

	return 1;
}