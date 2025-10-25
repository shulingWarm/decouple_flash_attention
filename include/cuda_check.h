#pragma once
#include<iostream>
#include<cuda_runtime.h>

// cuda错误检查的函数，用于检查cudaFree, cudaMalloc等函数的返回值
inline void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cout<<cudaGetErrorString(result)<<std::endl;
    }
}