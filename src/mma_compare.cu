#include<iostream>
#include"cuda_check.h"
#include <cutlass/bfloat16.h>
#include"rand_generator.h"

typedef unsigned int u32;
using bf16 = cutlass::bfloat16_t;

// 复制shared memory的device函数
// 对于正常的矩阵相乘，A矩阵TILE_FIRST=false, B矩阵TILE_FIRST=true
// 这里面只考虑T的大小是2字节的情况
template<class T, u32 TILE_SIZE, u32 K_SIZE, u32 THREAD_COPY_SIZE, bool TILE_FIRST>
inline __device__ void copy_shared_time(T* dst_shared, T* thread_tile, T* global_mem, 
    u32 id_tile, u32 row_num, u32 col_num) {
    // global memory的起始位置
    T* global_mem_start;
    if constexpr (TILE_FIRST) {
        global_mem_start = global_mem + id_tile * K_SIZE * col_num;
    } else {
        global_mem_start = global_mem + id_tile * K_SIZE;
    }

    constexpr u32 ELEMENT_SIZE = sizeof(T);
    // 复制过程的循环次数
    constexpr u32 COPY_LOOP_NUM = THREAD_COPY_SIZE / (4/ELEMENT_SIZE);
    // 每次读取的element个数
    constexpr u32 ELEMENT_NUM_PER_LOAD = 4 / ELEMENT_SIZE;

    // 执行复制过程
    #pragma unroll
    for(u32 id_copy = 0; id_copy < COPY_LOOP_NUM; id_copy++) {
        // 计算global memory的读取位置
        u32 global_mem_offset;
    }
}

// 用于测试mma指令的核函数
template<class T, u32 M_TILE, u32 N_TILE, u32 K_TILE, u32 THREAD_NUM>
__global__ void mma_test_kernel(T* A, T* B, T* D) {
    // 初始化两组shared memory
    __shared__ T shared_a[M_TILE * K_TILE];
    __shared__ T shared_b[K_TILE * N_TILE];

    // 每个线程需要计算的数据量 64
    constexpr u32 DATA_NUM_PER_THREAD = M_TILE * N_TILE / THREAD_NUM;
    // 复制a切片时每个线程的复制量
    constexpr u32 A_SHARED_COPY_TILE_SIZE = M_TILE * K_TILE / THREAD_NUM;
    // 复制b切片时每个线程的复制量
    constexpr u32 B_SHARED_COPY_TILE_SIZE = K_TILE * N_TILE / THREAD_NUM;
    // 初始化每个线程负责计算的输出矩阵切片
    T thread_output_tile[DATA_NUM_PER_THREAD];
    T a_shared_copy_tile[A_SHARED_COPY_TILE_SIZE];
    T b_shared_copy_tile[B_SHARED_COPY_TILE_SIZE];

    
}

// 简单的mma核函数
// 这个简单实现的版本只测试性能，不考虑正确性
// 忽略将数据从global memory加载到shared 的过程
// IS_LARGE 表示是否使用较大切片的mma指令 
template<u32 K_LOOP_NUM, bool IS_LARGE>
__global__ void mma_test_kernel_simple(float* D) {
    // u32 a_tile[8];
    // u32 b_tile[4];
    // // 输出切片
    // float out_tile[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // // 根据block id计算偏移
    // float* block_output = D + blockIdx.x * 128;
    // // 循环K次
    // #pragma unroll
    // for(u32 id_k = 0; id_k < K_LOOP_NUM; id_k++) {
    //     // 随机初始化A矩阵
    //     #pragma unroll
    //     for(int i=0;i<8;++i) {
    //         a_tile[i] = id_k*i;
    //     }

    //     #pragma unroll
    //     for(int i=0;i<4;++i) {
    //         b_tile[i] = id_k*i;
    //     }

    //     // 调用mma指令
    //     if constexpr (IS_LARGE) {
    //         asm volatile(
    //             "mma.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
    //             "{%0,  %1,  %2,  %3},"
    //             "{%4,  %5, %6, %7, %8, %9, %10, %11},"
    //             "{%12, %13, %14, %15},"
    //             "{%16, %17, %18, %19};\n"
    //             : "=f"(out_tile[0]), "=f"(out_tile[1]), "=f"(out_tile[2]), "=f"(out_tile[3])
    //             :  "r"(a_tile[0]),  "r"(a_tile[1]),  "r"(a_tile[2]),  "r"(a_tile[3]), "r"(a_tile[4]), "r"(a_tile[5]), "r"(a_tile[6]), "r"(a_tile[7]),
    //                 "r"(b_tile[0]),  "r"(b_tile[1]), "r"(b_tile[2]), "r"(b_tile[3]),
    //                 "f"(out_tile[0]),  "f"(out_tile[1]),  "f"(out_tile[2]),  "f"(out_tile[3]));
    //     }
    // }
    // // 将结果写入到全局内存
    // #pragma unroll
    // for(int id_copy=0; id_copy < 8; id_copy++) {
    //     block_output[threadIdx.x + id_copy*32] = out_tile[id_copy];
    // }
}

int main() {
    constexpr u32 GROUP_NUM = 100;
    // 初始化一个16*8的device内存
    float* d_mem;
    cuda_check(cudaMalloc(&d_mem, 16*8*GROUP_NUM*sizeof(float)));

    // 调用核函数
    mma_test_kernel_simple<512, true><<<GROUP_NUM, 32>>>(d_mem);

    // 将device转换到cpu上
    float* h_mem = new float[16*8*GROUP_NUM];
    cuda_check(cudaMemcpy(h_mem, d_mem, 16*8*GROUP_NUM*sizeof(float), cudaMemcpyDeviceToHost));

    // 打印前10个结果
    for(int i=0;i<10;++i) {
        printf("%f ", h_mem[i]);
    }

    // 释放内存
    delete[] h_mem;
    cuda_check(cudaFree(d_mem));
    return 0;
}