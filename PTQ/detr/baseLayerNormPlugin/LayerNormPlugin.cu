/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#include "LayerNormPlugin.h"
#include <cmath>
#include <NvInferPlugin.h>
using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

__global__ void layerNormKernel(const float *pInput, float *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * 256 + threadIdx.x;

    __shared__ float temp[128];

    float value0 = pInput[index];
    float value1 = pInput[index + 128];

    temp[tx] = value0 + value1;
    __syncthreads();

    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }

    float mean = temp[0] / 256;
    __syncthreads();

    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean);
    __syncthreads();

    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float var = temp[0] / 256;

    pOutput[index]       = (value0 - mean) * rsqrtf(var + 6e-6);
    pOutput[index + 128] = (value1 - mean) * rsqrtf(var + 6e-6);
}

template<typename T>
__global__ void layerNormKernelV1(T *pInput, float *pGamma, float *pBeta, T *pOutput) 
{
    const int n = 768;
    const int tx = threadIdx.x, index = blockIdx.x * n + tx;
    T _x = pInput[index], _b = (T)pGamma[tx], _a = (T)pBeta[tx];

    __shared__ T mean_shared, var_shared;
    typedef cub::BlockReduce<T, n> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    T& ref0 = _x;
    T sum = BlockReduce(temp).Sum(ref0);

    if(tx == 0) 
    {
        mean_shared = sum / T(n);
    }

    __syncthreads();

    T moment = _x - mean_shared, moment2 = moment * moment;
    T& ref1 = moment2;
    T var = BlockReduce(temp).Sum(ref1);
    if(tx == 0) 
    {
        var_shared = var / T(n);
    }
    __syncthreads();

    pOutput[index] = (moment) * (T)rsqrtf(var_shared + 1e-6) * _b + _a;
}



// class LayerNormPluginV3
template<int VPT>
struct BytesToType;

template<>
struct BytesToType<2>
{
    using type = uint16_t;
};
template<>
struct BytesToType<4>
{
    using type = uint32_t;
};
template<>
struct BytesToType<8>
{
    using type = uint64_t;
};
template<>
struct BytesToType<16>
{
    using type = float4;
};

template<typename T>
using kvp = cub::KeyValuePair<T, T>;

template<typename T>
struct mySum
{
    __host__ __device__ __forceinline__ kvp<T> operator()(const kvp<T> &a, const kvp<T> &b) const
    {
        return kvp<T>(a.key + b.key, a.value + b.value);
    }
};


template<int Bytes>
__device__ inline void copy(const void *local, void *data)
{
    using T = typename BytesToType<Bytes>::type;

    const T *in  = static_cast<const T *>(local);
    T       *out = static_cast<T *>(data);
    *out         = *in;
}

template<typename T, int TPB, int VPT>
__global__ void layerNormKernelV3(const T *input, const T *gamma, const T *beta, T *output)
{
    const int   idx = blockIdx.x * 256 + VPT * threadIdx.x;
    T           localX[VPT], localGamma[VPT], localBeta[VPT];
    kvp<float>  localFloat2 = {0.f, 0.f};
    const float denominator = float(1) / float(256);

    copy<sizeof(T) * VPT>(&input[idx], localX);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp = denominator * (float)localX[it];
        localFloat2.key += tmp;
        localFloat2.value += tmp * (float)localX[it];
    }

    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);
    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);

    using BlockReduce = cub::BlockReduce<kvp<float>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float                             mu;     // mean
    __shared__ float                             rsigma; // 1 / std.dev.

    const kvp<float> sumKV = BlockReduce(temp).Reduce(localFloat2, mySum<float>());

    if (threadIdx.x == 0)
    {
        mu     = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + 1e-6);
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        localX[it] = ((float)localX[it] - mu) * rsigma * (float)localGamma[it] + (float)localBeta[it];
    }

    copy<sizeof(T) * VPT>(localX, &output[idx]);
}

template __global__ void layerNormKernelV3<float, 64, 4>(const float *, const float *, const float *, float *);
template __global__ void layerNormKernelV3<half, 32, 8>(const half *, const half *, const half *, half *);

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

    const float* input = static_cast<const float*>(inputs[0]);
    // const float* gamma = static_cast<const float*>(inputs[1]);
    // const float* beta = static_cast<const float*>(inputs[2]);
    float* output = static_cast<float*>(outputs[0]);
    // std::cout << "input type:" << inputDesc[0].type;
    layerNormKernelV1<float><<<nBlock, 768, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
    // layerNormKernel<<<nBlock, 256, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
    return 0;


    // WHERE_AM_I();
    // const int gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

    // if (inputDesc[0].type == DataType::kFLOAT)
    // {
    //     constexpr int VPT = 16 / sizeof(float);
    //     constexpr int TPB = 768 / VPT;
    //     (layerNormKernelV3<float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>((const float *)inputs[0], (const float *)inputs[1], (const float *)inputs[2], (float *)outputs[0]);
    // }
    // else
    // {
    //     constexpr int VPT = 16 / sizeof(half);
    //     constexpr int TPB = 768 / VPT;
    //     (layerNormKernelV3<half, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>((const half *)inputs[0], (const half *)inputs[1], (const half *)inputs[2], (half *)outputs[0]);
    // }
    // return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator); 

