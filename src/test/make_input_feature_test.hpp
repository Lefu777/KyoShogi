#pragma once

#include "cshogi.h"

//typedef float(*)[N_FEATURE_WHC] my_feature_t;
//typedef float Dtype;


void mif_test0() {
	const int bs = 512;
	__Board board;
	board.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");

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

	// 予め要素数を指定しておかないと、
	// make_input_features() の時に、存在しないオブジェクトに対する参照を渡してしまうことに。
	std::vector<float[N_FEATURE_WHC]> _eval_queue(bs);

	auto& qt = _eval_queue[0];
	board.make_input_features(qt);

	pretty_print_as_3d(qt, N_FEATURE_CHANNEL, 9, 9);
}