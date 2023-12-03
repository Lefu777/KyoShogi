﻿#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "cshogi.h"

void unpack_feature1(float* dst, char* src, const int batch_size, cudaStream_t stream);
void unpack_feature2(float* dst, char* src, const int batch_size, cudaStream_t stream);