#pragma once
#ifndef TENSORRT_BUFFERS_H
#define TENSORRT_BUFFERS_H

#include "NvInfer.h"
#include "common.hpp"
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

namespace Buffers
{

    //!
    //! \brief  The GenericBuffer class is a templated class for buffers.
    //!
    //! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
    //!          deallocation, querying of buffers on both the device and the host.
    //!          It can handle data of arbitrary types because it stores byte buffers.
    //!          The template parameters AllocFunc and FreeFunc are used for the
    //!          allocation and deallocation of the buffer.
    //!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
    //!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
    //!          size is the amount of memory in bytes to allocate.
    //!          The boolean indicates whether or not the memory allocation was successful.
    //!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
    //!          ptr is the allocated buffer address. It must work with nullptr input.
    //!
    template <typename AllocFunc, typename FreeFunc>
    class GenericBuffer
    {
    private:
        size_t mSize{ 0 }, mCapacity{ 0 };
        nvinfer1::DataType mType;
        void* mBuffer;
        // 以下はtemplate引数によって既に定義されている。
        AllocFunc allocFn;
        FreeFunc freeFn;

    public:
        //!
        //! これが唯一、デフォルトコンストラクタ(則ち、引数ナシ)になり得る
        //! \brief Construct an empty buffer.
        //!
        GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
            : mSize(0)
            , mCapacity(0)
            , mType(type)
            , mBuffer(nullptr)
        {
        }

        //!
        //! これも使われてる。
        //! \brief Construct a buffer with the specified allocation size in bytes.
        //!
        GenericBuffer(size_t size, nvinfer1::DataType type)
            : mSize(size)
            , mCapacity(size)
            , mType(type)
        {
            // (恐らく) &mBufferはvoid*へのポインタ。つまり、void**
            // this->nbBytes()はメンバ関数。
            std::cout << "info : size = " << size << std::endl;
            std::cout << "info : this->nbBytes() = " << this->nbBytes() << std::endl;
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                std::cout << "ERROR : std::bad_alloc." << std::endl;
                throw std::bad_alloc();
            }
        }

        GenericBuffer(GenericBuffer&& buf)
            : mSize(buf.mSize)
            , mCapacity(buf.mCapacity)
            , mType(buf.mType)
            , mBuffer(buf.mBuffer)
        {
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mType = nvinfer1::DataType::kFLOAT;
            buf.mBuffer = nullptr;
        }

        GenericBuffer& operator=(GenericBuffer&& buf)
        {
            if (this != &buf)
            {
                freeFn(mBuffer);
                mSize = buf.mSize;
                mCapacity = buf.mCapacity;
                mType = buf.mType;
                mBuffer = buf.mBuffer;
                // Reset buf.
                buf.mSize = 0;
                buf.mCapacity = 0;
                buf.mBuffer = nullptr;
            }
            return *this;
        }

        //!
        //! \brief Returns pointer to underlying array.
        //!
        void* data()
        {
            return mBuffer;
        }

        //!
        //! \brief Returns pointer to underlying array.
        //!
        const void* data() const
        {
            return mBuffer;
        }

        //!
        //! \brief Returns the size (in number of elements) of the buffer.
        //!
        size_t size() const
        {
            return mSize;
        }

        //!
        //! \brief Returns the size (in bytes) of the buffer.
        //!
        size_t nbBytes() const
        {
            return this->size() * getElementSize(mType);
        }

        //!
        //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
        //!
        void resize(size_t newSize)
        {
            mSize = newSize;
            if (mCapacity < newSize)
            {
                freeFn(mBuffer);
                if (!allocFn(&mBuffer, this->nbBytes()))
                {
                    throw std::bad_alloc{};
                }
                mCapacity = newSize;
            }
        }

        //!
        //! \brief Overload of resize that accepts Dims
        //!
        void resize(const nvinfer1::Dims& dims)
        {
            return this->resize(volume(dims));
        }

        ~GenericBuffer()
        {
            freeFn(mBuffer);
        }
    };


    // 以下4つは、メモリの確保と開放を行う(実質的な)関数
    class DeviceAllocator    // GPU上でメモリを確保
    {
    public:
        // 関数的呼び出し。
        // DeviceAllocator da;
        // da(ptr, size);
        // って感じだと思う。
        // pythonでいう __call__かな？？？
        bool operator()(void** ptr, size_t size) const
        {
            return cudaMalloc(ptr, size) == cudaSuccess;
        }
    };

    class DeviceFree    // GPU上のメモリを解放
    {
    public:
        void operator()(void* ptr) const
        {
            cudaFree(ptr);
        }
    };

    class HostAllocator    // CPU上でメモリを確保
    {
    public:
        bool operator()(void** ptr, size_t size) const
        {
            *ptr = malloc(size);
            return *ptr != nullptr;
        }
    };

    class HostFree    // CPU上のメモリを解放
    {
    public:
        void operator()(void* ptr) const
        {
            free(ptr);
        }
    };

    // typedef的な奴。違いは、調べろ。
    // <>内はtemplateの引数
    using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
    using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

    //!
    //! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
    //!
    class ManagedBuffer
    {
    public:
        DeviceBuffer deviceBuffer;
        HostBuffer hostBuffer;
    };

    //!
    //! GPU上(device)にinputのデータを送り、CPU上(host)にoutputのデータを送る。という処理を一括して担う。
    //! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
    //!
    //! \details This RAII class handles host and device buffer allocation and deallocation,
    //!          memcpy between host and device buffers to aid with inference,
    //!          and debugging dumps to validate inference. The BufferManager class is meant to be
    //!          used to simplify buffer management and any interactions between buffers and the engine.
    //!
    class BufferManager
    {
    public:
        static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

        //!
        //! \brief Create a BufferManager for handling buffer interactions with engine.
        //!
        BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batchSize = 0,
            const nvinfer1::IExecutionContext* context = nullptr)
            : mEngine(engine)
            , mBatchSize(batchSize)
        {
            // Full Dims implies no batch size.
            assert(engine->hasImplicitBatchDimension() || mBatchSize == 0);
            // Create host and device buffers
            // < point > mEngine->getNbBindings()
            //           これは、networkの入り口と出口の総数。(NbはNumberかな？)(Bindingは恐らく入り口と出口の、英語特有の総称と思われる。)
            //           今のリバーシ向けモデルは2。dlshogiは,input,policy,valueの3つ。これらにはそれぞれ、0,1 と 0,1,2という
            //           インデックスが割り当てられて、input,outputの次元やshape等が一括管理されてる。
            //
            // auto dims_input = mEngine->getBindingDimensions(0);    // 0 : inputのbindingIndex
            // auto dims_output = mEngine->getBindingDimensions(1);    // 1 : outputのbindingIndex
            // dims_input.nbDims で次元数が得られる。
            // dims_input.d で、shape の配列(int32_t*)が返され、2次元なら、shapeは、"dims_input.d[0] x dims_input.d[1]"で表される。
            // dims_input.MAX_DIMS は、恐らく、tensorRTの仕様上対応可能な最高次元数を返す。定数値は8 のはず。
            std::cout << "for().. " << std::endl;
            for (int i = 0; i < mEngine->getNbBindings(); i++)
            {
                std::cout << "info : i in for() = " << i << std::endl;
                // Dims32型とやらが返ってきた。
                auto dims = context ? context->getBindingDimensions(i) : mEngine->getBindingDimensions(i);
                // contextあり or mBatchSize == 0
                // sampleOnnxMNISTの場合、context == nullptrなので、
                // mBatchSize == 0なら 1
                // mBatchSize != 0なら mBatchSize
                size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);

                nvinfer1::DataType type = mEngine->getBindingDataType(i);
                // YET20220826 : なんすかこれ。
                int vecDim = mEngine->getBindingVectorizedDim(i);
                if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
                {
                    // assertで、エラーログにメッセージを残す方法。
                    assert(
                        vecDim == -1 &&
                        "assert : in the past experiment, this value was -1.(20220826)"
                    );
                    /*
                    int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
                    dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
                    vol *= scalarsPerVec;
                    */
                }
                // 手元の実験結果じゃ、負(本来の値 * -1)になるはず。(20220826)
                std::cout << "info : volume(dims) = " << volume(dims) << std::endl;
                vol *= volume(dims) * -1;
                // assert(volume(dims) > 0 && "vol(memory size) must be positive, so \"volume(dims)\" too. (because do : vol *= volume(dims))");

                // GenericBuffer 2つをpublicメンバ変数に持つ、class ManagedBuffer()をnew
                std::unique_ptr<ManagedBuffer> manBuf{ new ManagedBuffer() };
                // さっき新たに生成した ManagedBuffer 内の GenericBuffer を初期化する
                // (コンストラクタを使って、指定されたサイズ,typeのメモリをGPU,CPU上でそれぞれ確保する。)
                // @arg vol
                //     : サイズ
                // @arg type
                //     : 型
                // manBuf->deviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree> (vol, type);
                std::cout << "construct DeviceBuffer().. " << std::endl;
                std::cout << "info :  vol = " << vol << std::endl;
                manBuf->deviceBuffer = DeviceBuffer(vol, type);
                std::cout << "construct HostBuffer().. " << std::endl;
                manBuf->hostBuffer = HostBuffer(vol, type);

                // emplace_back(mBuffer);    // ※mBuffer は GenericBuffer のメンバ
                // emplace系の関数は、コンストラクタ引数の値を渡す。コンストラクタの実行はvectorの方に任される。
                // .data()は、
                // void*型の確保したメモリへのポインタ(？(ちょっとポインタとかまでは分からんけど、とにかく、確保したメモリ))
                // YET20220825 : これ何に使うの？？？
                //               The vector of device buffers needed for engine execution らしい。ほう。意味不明。
                //               GPU上のデータ。
                mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
                // これはコピーコンストラクタの引数。(コピーコンストラクタは、コンストラクタ引数ではなく、オブジェクト自身。正確には、const参照)
                // std::move()は、manBufへの所有権を(コピーコンストラクタ的に?)渡す。
                mManagedBuffers.emplace_back(std::move(manBuf));
            }
        }

        //!
        //! 推論の関数に渡す。
        //! \brief Returns a vector of device buffers that you can use directly as
        //!        bindings for the execute and enqueue methods of IExecutionContext.
        //!
        std::vector<void*>& getDeviceBindings()
        {
            return mDeviceBindings;
        }

        //!
        //! \brief Returns a vector of device buffers.
        //!
        const std::vector<void*>& getDeviceBindings() const
        {
            return mDeviceBindings;
        }

        //!
        //! \brief Returns the device buffer corresponding to tensorName.
        //!        Returns nullptr if no such tensor can be found.
        //!
        void* getDeviceBuffer(const std::string& tensorName) const
        {
            return getBuffer(false, tensorName);
        }

        //!
        //! \brief Returns the host buffer corresponding to tensorName.
        //!        Returns nullptr if no such tensor can be found.
        //!
        void* getHostBuffer(const std::string& tensorName) const
        {
            return getBuffer(true, tensorName);
        }

        //!
        //! \brief Returns the size of the host and device buffers that correspond to tensorName.
        //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
        //!
        size_t size(const std::string& tensorName) const
        {
            int index = mEngine->getBindingIndex(tensorName.c_str());
            if (index == -1)
                return kINVALID_SIZE_VALUE;
            return mManagedBuffers[index]->hostBuffer.nbBytes();
        }

        //!
        //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
        //!        rowCount parameter controls how many elements are on each line.
        //!        A rowCount of 1 means that there is only 1 element on each line.
        //!
        template <typename T>
        void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
        {
            assert(rowCount != 0);
            assert(bufSize % sizeof(T) == 0);
            T* typedBuf = static_cast<T*>(buf);
            size_t numItems = bufSize / sizeof(T);
            for (int i = 0; i < static_cast<int>(numItems); i++)
            {
                // Handle rowCount == 1 case
                if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                    os << typedBuf[i] << std::endl;
                else if (rowCount == 1)
                    os << typedBuf[i];
                // Handle rowCount > 1 case
                else if (i % rowCount == 0)
                    os << typedBuf[i];
                else if (i % rowCount == rowCount - 1)
                    os << " " << typedBuf[i] << std::endl;
                else
                    os << " " << typedBuf[i];
            }
        }

        //!
        //! \brief Copy the contents of input host buffers to input device buffers synchronously.
        //!
        void copyInputToDevice()
        {
            memcpyBuffers(true, false, false);
        }

        //!
        //! \brief Copy the contents of output device buffers to output host buffers synchronously.
        //!
        void copyOutputToHost()
        {
            memcpyBuffers(false, true, false);
        }

        //!
        //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
        //!
        void copyInputToDeviceAsync(const cudaStream_t& stream = 0)
        {
            memcpyBuffers(true, false, true, stream);
        }

        //!
        //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
        //!
        void copyOutputToHostAsync(const cudaStream_t& stream = 0)
        {
            memcpyBuffers(false, true, true, stream);
        }

        ~BufferManager() = default;

    private:
        void* getBuffer(const bool isHost, const std::string& tensorName) const
        {
            int index = mEngine->getBindingIndex(tensorName.c_str());
            if (index == -1)
                return nullptr;
            return (isHost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());
        }

        void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0)
        {
            for (int i = 0; i < mEngine->getNbBindings(); i++)
            {
                void* dstPtr
                    = deviceToHost ? mManagedBuffers[i]->hostBuffer.data() : mManagedBuffers[i]->deviceBuffer.data();
                const void* srcPtr
                    = deviceToHost ? mManagedBuffers[i]->deviceBuffer.data() : mManagedBuffers[i]->hostBuffer.data();
                const size_t byteSize = mManagedBuffers[i]->hostBuffer.nbBytes();
                const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
                if ((copyInput && mEngine->bindingIsInput(i)) || (!copyInput && !mEngine->bindingIsInput(i)))
                {
                    if (async) {
                        //CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
                        cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream);
                    }
                    else {
                        //CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
                        cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType);
                    }
                }
            }
        }

        std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              //!< The pointer to the engine
        int mBatchSize;                                              //!< The batch size for legacy networks, 0 otherwise.
        // CPU,GPU上のinput,outputのデータへのポインタ(2つ)の集まりであるmManagedBuffers へのスマートポインタ。
        std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers
        // GPU上のinput,output達のデータ(正確にはポインタか何か。)が入った
        std::vector<void*> mDeviceBindings;                          //!< The vector of device buffers needed for engine execution
    };

} // namespace samplesCommon

#endif // TENSORRT_BUFFERS_H
