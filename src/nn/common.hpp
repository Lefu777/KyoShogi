#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

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

// なんか速くなる奴らしい。イマイチ分からん。
inline void enableDLA(IBuilder* builder, IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                << std::endl;
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback)
        {
            config->setFlag(BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(useDLACore);
    }
}

static auto StreamDeleter = [](cudaStream_t* pStream)
    {
        if (pStream)
        {
            cudaStreamDestroy(*pStream);
            delete pStream;
        }
    };

// なんすかこれ？？？？
inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}

// Dims は Dims32のこと
inline int64_t volume(const nvinfer1::Dims& d)
{
    // std::multiplies<int64_t>()は、掛け算を行う二項演算関数。
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}