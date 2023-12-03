#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "cshogi.h"
#include "types.hpp"
// #include "buffers.hpp"
// #include "common.hpp"

#include "util.hpp"

#include "unpack_feature_V2.cuh"


// HACK: using 系使いたくないけど、一旦これで。。。
using namespace nvonnxparser;
using namespace nvinfer1;

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

class NNTensorRT {
private:
    const int _max_batch_size;
#if defined(FEATURE_V2)
    std::tuple<Dims, Dims> _input_dims;
#else
    Dims _input_dims;
#endif
    std::shared_ptr<nvinfer1::ICudaEngine> _engine;
    InferUniquePtr<nvinfer1::IExecutionContext> _context;


    // cpu(puct_player) から渡される特徴量をそっくりそのまま転送したもの。
    // USE_PACKED_FEATURE が有効なら、unpacked_feature_t に展開してやる。
    feature_t _input_device;
#if defined(USE_PACKED_FEATURE)
    unpacked_feature_t _unpacked_input_device;
#endif

    float* _output_p_device;
    float* _output_v_device;
    std::vector<void*> _input_bindings;

    //int _builder_optimization_level;
    bool _is_fp16;

public:
    void init(const std::string& onnx_file_name);
    NNTensorRT(const std::string& onnx_file_name, const int max_batch_size);
    NNTensorRT(const std::string& onnx_file_name, const int max_batch_size, const bool is_fp16);
    ~NNTensorRT();

    void set_is_fp16(const bool is_fp16) { _is_fp16 = is_fp16; }

    bool build_engine(const std::string& onnx_file_name);
    bool load_model(const std::string& onnx_file_name);
    bool forward(const int& batch_size, const feature_t input, float* output_p, float* output_v);
};