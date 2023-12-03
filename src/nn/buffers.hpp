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
        // �ȉ���template�����ɂ���Ċ��ɒ�`����Ă���B
        AllocFunc allocFn;
        FreeFunc freeFn;

    public:
        //!
        //! ���ꂪ�B��A�f�t�H���g�R���X�g���N�^(�����A�����i�V)�ɂȂ蓾��
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
        //! ������g���Ă�B
        //! \brief Construct a buffer with the specified allocation size in bytes.
        //!
        GenericBuffer(size_t size, nvinfer1::DataType type)
            : mSize(size)
            , mCapacity(size)
            , mType(type)
        {
            // (���炭) &mBuffer��void*�ւ̃|�C���^�B�܂�Avoid**
            // this->nbBytes()�̓����o�֐��B
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


    // �ȉ�4�́A�������̊m�ۂƊJ�����s��(�����I��)�֐�
    class DeviceAllocator    // GPU��Ń��������m��
    {
    public:
        // �֐��I�Ăяo���B
        // DeviceAllocator da;
        // da(ptr, size);
        // ���Ċ������Ǝv���B
        // python�ł��� __call__���ȁH�H�H
        bool operator()(void** ptr, size_t size) const
        {
            return cudaMalloc(ptr, size) == cudaSuccess;
        }
    };

    class DeviceFree    // GPU��̃����������
    {
    public:
        void operator()(void* ptr) const
        {
            cudaFree(ptr);
        }
    };

    class HostAllocator    // CPU��Ń��������m��
    {
    public:
        bool operator()(void** ptr, size_t size) const
        {
            *ptr = malloc(size);
            return *ptr != nullptr;
        }
    };

    class HostFree    // CPU��̃����������
    {
    public:
        void operator()(void* ptr) const
        {
            free(ptr);
        }
    };

    // typedef�I�ȓz�B�Ⴂ�́A���ׂ�B
    // <>����template�̈���
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
    //! GPU��(device)��input�̃f�[�^�𑗂�ACPU��(host)��output�̃f�[�^�𑗂�B�Ƃ����������ꊇ���ĒS���B
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
            //           ����́Anetwork�̓�����Əo���̑����B(Nb��Number���ȁH)(Binding�͋��炭������Əo���́A�p����L�̑��̂Ǝv����B)
            //           ���̃��o�[�V�������f����2�Bdlshogi��,input,policy,value��3�B�����ɂ͂��ꂼ��A0,1 �� 0,1,2�Ƃ���
            //           �C���f�b�N�X�����蓖�Ă��āAinput,output�̎�����shape�����ꊇ�Ǘ�����Ă�B
            //
            // auto dims_input = mEngine->getBindingDimensions(0);    // 0 : input��bindingIndex
            // auto dims_output = mEngine->getBindingDimensions(1);    // 1 : output��bindingIndex
            // dims_input.nbDims �Ŏ�������������B
            // dims_input.d �ŁAshape �̔z��(int32_t*)���Ԃ���A2�����Ȃ�Ashape�́A"dims_input.d[0] x dims_input.d[1]"�ŕ\�����B
            // dims_input.MAX_DIMS �́A���炭�AtensorRT�̎d�l��Ή��\�ȍō���������Ԃ��B�萔�l��8 �̂͂��B
            std::cout << "for().. " << std::endl;
            for (int i = 0; i < mEngine->getNbBindings(); i++)
            {
                std::cout << "info : i in for() = " << i << std::endl;
                // Dims32�^�Ƃ�炪�Ԃ��Ă����B
                auto dims = context ? context->getBindingDimensions(i) : mEngine->getBindingDimensions(i);
                // context���� or mBatchSize == 0
                // sampleOnnxMNIST�̏ꍇ�Acontext == nullptr�Ȃ̂ŁA
                // mBatchSize == 0�Ȃ� 1
                // mBatchSize != 0�Ȃ� mBatchSize
                size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);

                nvinfer1::DataType type = mEngine->getBindingDataType(i);
                // YET20220826 : �Ȃ񂷂�����B
                int vecDim = mEngine->getBindingVectorizedDim(i);
                if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
                {
                    // assert�ŁA�G���[���O�Ƀ��b�Z�[�W���c�����@�B
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
                // �茳�̎������ʂ���A��(�{���̒l * -1)�ɂȂ�͂��B(20220826)
                std::cout << "info : volume(dims) = " << volume(dims) << std::endl;
                vol *= volume(dims) * -1;
                // assert(volume(dims) > 0 && "vol(memory size) must be positive, so \"volume(dims)\" too. (because do : vol *= volume(dims))");

                // GenericBuffer 2��public�����o�ϐ��Ɏ��Aclass ManagedBuffer()��new
                std::unique_ptr<ManagedBuffer> manBuf{ new ManagedBuffer() };
                // �������V���ɐ������� ManagedBuffer ���� GenericBuffer ������������
                // (�R���X�g���N�^���g���āA�w�肳�ꂽ�T�C�Y,type�̃�������GPU,CPU��ł��ꂼ��m�ۂ���B)
                // @arg vol
                //     : �T�C�Y
                // @arg type
                //     : �^
                // manBuf->deviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree> (vol, type);
                std::cout << "construct DeviceBuffer().. " << std::endl;
                std::cout << "info :  vol = " << vol << std::endl;
                manBuf->deviceBuffer = DeviceBuffer(vol, type);
                std::cout << "construct HostBuffer().. " << std::endl;
                manBuf->hostBuffer = HostBuffer(vol, type);

                // emplace_back(mBuffer);    // ��mBuffer �� GenericBuffer �̃����o
                // emplace�n�̊֐��́A�R���X�g���N�^�����̒l��n���B�R���X�g���N�^�̎��s��vector�̕��ɔC�����B
                // .data()�́A
                // void*�^�̊m�ۂ����������ւ̃|�C���^(�H(������ƃ|�C���^�Ƃ��܂ł͕�����񂯂ǁA�Ƃɂ����A�m�ۂ���������))
                // YET20220825 : ���ꉽ�Ɏg���́H�H�H
                //               The vector of device buffers needed for engine execution �炵���B�ق��B�Ӗ��s���B
                //               GPU��̃f�[�^�B
                mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
                // ����̓R�s�[�R���X�g���N�^�̈����B(�R�s�[�R���X�g���N�^�́A�R���X�g���N�^�����ł͂Ȃ��A�I�u�W�F�N�g���g�B���m�ɂ́Aconst�Q��)
                // std::move()�́AmanBuf�ւ̏��L����(�R�s�[�R���X�g���N�^�I��?)�n���B
                mManagedBuffers.emplace_back(std::move(manBuf));
            }
        }

        //!
        //! ���_�̊֐��ɓn���B
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
        // CPU,GPU���input,output�̃f�[�^�ւ̃|�C���^(2��)�̏W�܂�ł���mManagedBuffers �ւ̃X�}�[�g�|�C���^�B
        std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers
        // GPU���input,output�B�̃f�[�^(���m�ɂ̓|�C���^�������B)��������
        std::vector<void*> mDeviceBindings;                          //!< The vector of device buffers needed for engine execution
    };

} // namespace samplesCommon

#endif // TENSORRT_BUFFERS_H
