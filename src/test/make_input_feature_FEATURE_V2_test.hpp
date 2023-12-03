#pragma once
#include <memory>
#include <cuda_runtime.h>

#include "config.hpp"
#include "cshogi.h"
#include "unpack_test.cuh"
#include "unpack_feature_V2.cuh"

#if defined(FEATURE_V2)
void make_input_features_FATURE_V2_test() {
	__Board board;
	std::cout << board.toSFEN() << std::endl;
	board.pos.print(std::cout);

	float* feature1 = new float[N_FEATURE1_WHC];
	float* feature2 = new float[N_FEATURE2_WHC];
	board.make_input_features(feature1, feature2);

	std::cout << "info: N_FEATURE1_CHANNEL = " << N_FEATURE1_CHANNEL << std::endl;
	std::cout << "info: N_FEATURE2_CHANNEL = " << N_FEATURE2_CHANNEL << std::endl;
	std::cout << "========== feature1 ==========" << std::endl;
	for (int i = 0; i < N_FEATURE1_WHC; ++i) {
		if (i % N_FEATURE_WIDTH == 0) {
			std::cout << "\n";
		}
		if (i % N_FEATURE_WH == 0) {
			std::cout << "\n";
			std::cout << "channel = " << (i / N_FEATURE_WH) + 1 << std::endl;
		}

		std::cout << *(feature1 + i) << " ";
	}

	std::cout << "========== feature2 ==========" << std::endl;
	for (int i = 0; i < N_FEATURE2_WHC; ++i) {
		if (i % N_FEATURE_WIDTH == 0) {
			std::cout << "\n";
		}
		if (i % N_FEATURE_WH == 0) {
			std::cout << "\n";
			std::cout << "channel = " << (i / N_FEATURE_WH) + 1 << std::endl;
		}

		std::cout << *(feature2 + i) << " ";
	}

	delete[] feature1;
	delete[] feature2;
}

// TODO
//     : 1.packed_feature2 の展開のテストを実装
//       2.unpack をGPU で実装。
//       4. unpack した結果をcpu に再度転送して、普通に作った入力特徴量(非packed)
//          と一致するか確認する。
void make_input_features_FATURE_V2_test2() {
	__Board board;
	board.set("1n3pk1l/1sr2sg2/3pB3p/1pP2BP2/2pS5/4KPs1P/1P4N2/5G3/+l+p2P3L w 2GN3Prnl3p 126");
	std::cout << board.toSFEN() << std::endl;
	board.pos.print(std::cout);

	std::unique_ptr<float[]> feature1 = std::make_unique<float[]>(N_FEATURE1);
	std::unique_ptr<float[]> feature2 = std::make_unique<float[]>(N_FEATURE2);
	std::unique_ptr<float[]> feature1_by_set = std::make_unique<float[]>(N_FEATURE1);
	std::unique_ptr<float[]> feature2_by_set = std::make_unique<float[]>(N_FEATURE2);
	std::unique_ptr<char[]> packed_feature1 = std::make_unique<char[]>(N_PACKED_FEATURE1_CHAR);
	std::unique_ptr<char[]> packed_feature2 = std::make_unique<char[]>(N_PACKED_FEATURE2_CHAR);
	board.make_input_features<float, false>(feature1.get(), feature2.get());
	board.make_input_features<float, true>(feature1_by_set.get(), feature2_by_set.get());
	board.make_input_features(packed_feature1.get(), packed_feature2.get());

	bool is_ok = true;
	//// feature vs feature_by_set
	std::cout << "feature vs feature_by_set" << std::endl;
	for (int i = 0; i < N_FEATURE1; ++i) {
		if (feature1[i] != feature1_by_set[i]) {
			std::cout << "feature1, not match: i = " << i << std::endl;
			is_ok = false;
		}
	}
	for (int i = 0; i < N_FEATURE2; ++i) {
		if (feature2[i] != feature2_by_set[i]) {
			std::cout << "feature2, not match: i = " << i << std::endl;
			is_ok = false;
		}
	}

	//// feature vs packed_feature
	std::cout << "feature vs packed_feature" << std::endl;
	for (int i = 0; i < N_FEATURE1; ++i) {
		// ((packed_feature1[i >> 3] >> (i & 7)) & 1
		//     : i >> 3 はi/8 と等価である。 (何番目のchar に格納されているか)
		//     : i & 7  はi%8 と等価である。 (char の中で何番目のbit か)
		//     : (packed_feature1[i >> 3] >> (i & 7)) によって、インデックスi に対応した
		//       bit がLSB に来る。
		//     : 最後は、&1 としてやって、そのLSB に来たbit を取り出してやるだけ。
		//     : (ごにょごにょやってたらdlshogi の実装と同様の式になったので一安心。)
		if (feature1[i] != ((packed_feature1[i >> 3] >> (i & 7)) & 1 )) {
			std::cout
				<< "feature1, not match: i = " << i << ",f[i]=" << feature1[i]
				<< ", p[i]=" << ((packed_feature1[i >> 3] >> (i & 7)) & 1) << std::endl;
			is_ok = false;
		}
	}
	for (int i = 0; i < N_FEATURE2; ++i) {
		const int channel_idx = i / N_FEATURE_WH;
		if (feature2[i] != ((packed_feature2[channel_idx >> 3] >> (channel_idx & 7)) & 1)) {
			std::cout << "feature2, not match: i = " << i << std::endl;
			is_ok = false;
		}
	}

	if (is_ok) {
		std::cout << "test was successful :)" << std::endl;
	}
	else {
		std::cout << "test failed." << std::endl;
	}
}



// TODO
//     : 1.packed_feature2 の展開のテストを実装
//       2.unpack をGPU で実装。
//       4. unpack した結果をcpu に再度転送して、普通に作った入力特徴量(非packed)
//          と一致するか確認する。
void make_input_features_FATURE_V2_test3() {
	//// sfens
	std::vector<std::string> sfens = {
		"1n3pk1l/1sr2sg2/3pB3p/1pP2BP2/2pS5/4KPs1P/1P4N2/5G3/+l+p2P3L w 2GN3Prnl3p 126",
		"ln1R1+B3/1kg+L2P1+r/1s2S2p1/g2pp4/pP4G2/1SpP4P/P3bP1P1/1KS6/LN1GP2+p1 w P2nl3p 130",
		"l2g2s1l/k3p4/1GPp3r1/p2G1+Bn1p/1SpnKB3/P2nP1P2/3P1P2P/1+p3G3/L4S3 w RSNL5P 164",
	};
	
	//// alloc
	const int batch_size = sfens.size();
	std::unique_ptr<float[]> feature1 = std::make_unique<float[]>(batch_size * N_FEATURE1);
	std::unique_ptr<float[]> feature2 = std::make_unique<float[]>(batch_size * N_FEATURE2);
	std::unique_ptr<char[]> packed_feature1 = std::make_unique<char[]>(batch_size * N_PACKED_FEATURE1_CHAR);
	std::unique_ptr<char[]> packed_feature2 = std::make_unique<char[]>(batch_size * N_PACKED_FEATURE2_CHAR);
	std::unique_ptr<float[]> unpacked_feature1 = std::make_unique<float[]>(batch_size * N_FEATURE1);
	std::unique_ptr<float[]> unpacked_feature2 = std::make_unique<float[]>(batch_size * N_FEATURE2);

	char* _input1_device;
	char* _input2_device;
	float* _unpacked_input1_device;
	float* _unpacked_input2_device;
	cudaMalloc((void**)&(_input1_device), SIZEOF_FEATURE1 * batch_size);
	cudaMalloc((void**)&(_input2_device), SIZEOF_FEATURE2 * batch_size);
	cudaMalloc((void**)&(_unpacked_input1_device), sizeof(float) * N_FEATURE1_WHC * batch_size);
	cudaMalloc((void**)&(_unpacked_input2_device), sizeof(float) * N_FEATURE2_WHC * batch_size);

	//// make input feature
	__Board board;
	for (int i = 0; i < batch_size; ++i) {
		const auto& sfen = sfens[i];
		board.set(sfen);
		board.pos.print(std::cout);
		board.make_input_features(
			feature1.get() + i * N_FEATURE1, feature2.get() + i * N_FEATURE2
		);
		board.make_input_features(
			packed_feature1.get() + i * N_PACKED_FEATURE1_CHAR, packed_feature2.get() + i * N_PACKED_FEATURE2_CHAR
		);
	}

	//// unpack
	// GPU でunpack した奴をhost に戻して来る。
	cudaMemcpyAsync(
		_input1_device, packed_feature1.get(), SIZEOF_FEATURE1 * batch_size,
		cudaMemcpyHostToDevice, cudaStreamPerThread
	);
	cudaMemcpyAsync(
		_input2_device, packed_feature2.get(), SIZEOF_FEATURE2 * batch_size,
		cudaMemcpyHostToDevice, cudaStreamPerThread
	);

	//unpack_feature1_test(_unpacked_input1_device, _input1_device, batch_size, cudaStreamPerThread);
	//unpack_feature2_test(_unpacked_input2_device, _input2_device, batch_size, cudaStreamPerThread);
	unpack_feature1(_unpacked_input1_device, _input1_device, batch_size, cudaStreamPerThread);
	unpack_feature2(_unpacked_input2_device, _input2_device, batch_size, cudaStreamPerThread);

	cudaMemcpyAsync(
		unpacked_feature1.get(), _unpacked_input1_device, sizeof(float) * N_FEATURE1_WHC * batch_size,
		cudaMemcpyDeviceToHost, cudaStreamPerThread
	);
	cudaMemcpyAsync(
		unpacked_feature2.get(), _unpacked_input2_device, sizeof(float) * N_FEATURE2_WHC * batch_size,
		cudaMemcpyDeviceToHost, cudaStreamPerThread
	);

    //// check
	bool is_ok = true;
	std::cout << "==================== feature1 ====================" << std::endl;
	for (int i = 0; i < N_FEATURE1 * batch_size; ++i) {
		if (feature1[i] != unpacked_feature1[i]) {
			std::cout
				<< "feature1, not match: i = " << i << "," << i % N_FEATURE1
				<< ", feature1[i]=" << feature1[i] << ", unpacked_feature1[i]=" << unpacked_feature1[i]
				<< std::endl;
			is_ok = false;
		}
	}
	std::cout << "==================== feature2 ====================" << std::endl;
		for (int i = 0; i < N_FEATURE2 * batch_size; ++i) {
		if (feature2[i] != unpacked_feature2[i]) {
			std::cout
				<< "feature2, not match: i = " << i << "," << i % N_FEATURE2
				<< ", feature2[i]=" << feature2[i] << ", unpacked_feature2[i]=" << unpacked_feature2[i]
				<< std::endl;
			is_ok = false;
		}
	}


	std::cout << "==================== result ====================" << std::endl;
		if (is_ok) {
		std::cout << "test was successful :)" << std::endl;
	}
	else {
		std::cout << "test failed." << std::endl;
	}

	cudaFree(_input1_device);
	cudaFree(_input2_device);
	cudaFree(_unpacked_input1_device);
	cudaFree(_unpacked_input2_device);
}

#endif