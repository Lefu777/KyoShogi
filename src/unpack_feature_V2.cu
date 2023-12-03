#include "unpack_feature_V2.cuh"

__global__ void _unpack_feature1(float* dst, char* src) {
	// NOTE
	//     : blockIdx.x  : そのbatch の番号
	//     : blockDim.x  : N_FEATURE2_CHANNEL
	//     : threadIdx.x : このbatch で何channel 目か
	//     : src は1batch 当たり、N_PACKED_FEATURE1_CHAR 個の配列の要素が必要。
	//       今のbatch_size では、
	const int& channel_idx = blockIdx.x * blockDim.x + threadIdx.x;               // 自分が担当するchannel
	const int& dst_offset = channel_idx * N_FEATURE_WH;
	const int& src_offset_bit_sofar = N_PACKED_FEATURE1_CHAR * blockIdx.x * 8;    // これまでのbatch のbit数
	const int& src_offset_bit_crrt = threadIdx.x * N_FEATURE_WH;                  // このbatch でのこれまでのbit数
	const int& src_offset_bit = src_offset_bit_sofar + src_offset_bit_crrt;
#pragma unroll
	for (int i = 0; i < N_FEATURE_WH; ++i) {
		const int& src_bit = src_offset_bit + i;    // 何bit 目か。
		dst[dst_offset + i] = (src[src_bit >> 3] >> (src_bit & 7)) & 1;
	}
}

__global__ void _unpack_feature2(float* dst, char* src) {
	// NOTE
	//     : blockIdx.x  : そのbatch の番号
	//     : blockDim.x  : N_FEATURE2_CHANNEL
	//     : threadIdx.x : 何channel 目か
	const int& channel_idx = blockIdx.x * blockDim.x + threadIdx.x;               // 自分が担当するchannel
	const int& src_offset = channel_idx * N_FEATURE_WH;
	const int& src_offset_bit_sofar = N_PACKED_FEATURE2_CHAR * blockIdx.x * 8;    // これまでのbatch のbit数
	const int& src_offset_bit_crrt = threadIdx.x;                                 // このbatch でのこれまでのbit数
	const int& src_offset_bit = src_offset_bit_sofar + src_offset_bit_crrt;
	const auto& v = (src[src_offset_bit >> 3] >> (src_offset_bit & 7)) & 1;
#pragma unroll
	for (int i = 0; i < N_FEATURE_WH; ++i) {
		dst[src_offset + i] = v;
	}
}

void unpack_feature1(float* dst, char* src, const int batch_size, cudaStream_t stream) {
	// NOTE
	//     : https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration
	//       0 は、この呼び出しに対して、block ごとに動的に確保される共有メモリのbyte 数。
	_unpack_feature1<<<batch_size, N_FEATURE1_CHANNEL, 0, stream >>>(dst, src);
}

void unpack_feature2(float* dst, char* src, const int batch_size, cudaStream_t stream) {
	// NOTE
	//     : https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration
	//       0 は、この呼び出しに対して、block ごとに動的に確保される共有メモリのbyte 数。
	_unpack_feature2<<<batch_size, N_FEATURE2_CHANNEL, 0, stream >>>(dst, src);
}