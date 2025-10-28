#include<iostream>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/pointer.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

// 指定Copy Atom的 CopyOperation
using CopyOpType = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
// 用来初始化cute::copy的copy_traits
using CopyAtomType = cute::Copy_Atom<CopyOpType, float>;

// 用于测试cute::copy的核函数
__global__ void copy_test(float* g_ptr) {
    // 新建一个用于复制的tensor
    // 原始的数据类型是: Tensor<SrcEngine, SrcLayout>
    // 创建一个空的tensor
    auto tensor = cute::make_tensor(make_gmem_ptr(g_ptr),Int<8>{});

    // 定义长度是8的shared memory
    __shared__ float s_ptr[8];
    // 定制shared memory tensor
    auto tensor_s = cute::make_tensor(make_smem_ptr(s_ptr),Int<8>{});

    // 新建一个复制策略的实体
    CopyAtomType copy_atom;

    // 调用cute::copy
    cute::copy(copy_atom, tensor, tensor_s);

    // 打印shared memory中的数据
    for (int i = 0; i < 8; ++i) {
        printf("s_ptr[%d] = %f\n", i, s_ptr[i]);
    }
}

int main() {
    // 准备用于测试的简单向量
    float h_ptr[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    // 复制到device上
    float* g_ptr;
    cudaMalloc(&g_ptr, sizeof(float) * 8);
    cudaMemcpy(g_ptr, h_ptr, sizeof(float) * 8, cudaMemcpyHostToDevice);
    //调用copy_test，只使用一个线程
    copy_test<<<1, 1>>>(g_ptr);
    cudaDeviceSynchronize();
    // 释放device内存
    cudaFree(g_ptr);
    return 0;
}