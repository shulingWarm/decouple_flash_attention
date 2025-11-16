#include<iostream>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/pointer.hpp>
#include <cute/algorithm/copy.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include "bit_move_detector.h"

void printPaddedString(const std::string& inputStr, size_t targetLength) {
    // 检查输入字符串长度是否超过目标长度
    if (inputStr.length() > targetLength) {
        throw std::invalid_argument("错误：字符串长度超过指定字符数！");
    }
    
    // 创建结果字符串，初始化为输入字符串
    std::string result = inputStr;
    
    // 补足空格到目标长度
    result.append(targetLength - inputStr.length(), ' ');
    
    // 打印到控制台
    std::cout << result;
}

// 用于测试tiled_copy的核函数
__global__ void tiled_copy_test(float* src_data, float* dst_data) {
    // 原始数据直接按最朴素的方式来解释
    auto src_tensor = cute::make_tensor(cute::make_gmem_ptr(src_data),
        cute::make_shape(cute::Int<128>(), cute::Int<128>())
    );
    // 目标tensor使用展开维度
    auto dst_tensor = cute::make_tensor(cute::make_gmem_ptr(dst_data),
        cute::make_shape(cute::Int<128>(), cute::Int<128>())
    );
    // 当前线程块执行的tiled_copy方案
    cute::TiledCopy<
        cute::Copy_Atom<
            cute::AutoVectorizingCopyWithAssumedAlignment<32>,
            float
        >,
        cute::Layout<
            cute::tuple<
                cute::C<128>,
                cute::C<1>
            >,
            cute::tuple<
                cute::_1,
                cute::_1
            >
        >,
        cute::tuple<
            cute::_128,
            cute::_128
        >
    > tiled_copy;

    // 获取当前线程的tiled_copy切片
    auto thread_copy_slice = tiled_copy.get_thread_slice(2);
    // 根据切片计算src_tensor和dst_tensor的子tensor
    auto src_sub_tensor = thread_copy_slice.partition_S(src_tensor);
    auto dst_sub_tensor = thread_copy_slice.partition_D(dst_tensor);

    // 执行tiled_copy的复制
    cute::copy(tiled_copy, src_sub_tensor, dst_sub_tensor);
}

// 用于测试打印tiled_copy的函数
void print_tiled_copy_test() { 
    // tiled_copy的定义
    using TiledCopyType = cute::TiledCopy<
        cute::Copy_Atom<
            cute::AutoVectorizingCopyWithAssumedAlignment<17>,
            cutlass::bfloat16_t
        >,
        cute::Layout<
            cute::tuple<
                cute::tuple<
                    cute::C<3>,
                    cute::C<10>
                >,
                cute::C<7>
            >,
            cute::tuple<
                cute::tuple<
                    cute::C<37>,
                    cute::C<1>
                >,
                cute::C<31>
            >
        >,
        cute::tuple<
            cute::C<11>,
            cute::C<13>
        >
    >;

    // 原始数据的tv layout
    auto src_tv_layout = TiledCopyType::get_layoutS_TV();
    auto src_mn_layout = TiledCopyType::get_layoutS_MN();

}

// 直接用cpu数据测试tiled_copy
template<class T>
void test_tiled_copy_cpu() {
    // 开辟128*128的cpu内存
    T* cpu_ptr = (T*)malloc(128 * 128 * sizeof(T));
    // 从0开始按顺序填充数据
    for (int i = 0; i < 128 * 128; i++) {
        cpu_ptr[i] = i;
    }

    cute::ComposedLayout<
        cute::Swizzle<3, 3, 3>,
        cute::C<0>,
        cute::Layout<
            cute::tuple<
                cute::tuple<
                    cute::C<8>,
                    cute::C<16>
                >,
                cute::tuple<
                    cute::_64,
                    cute::_2
                >
            >,
            cute::tuple<
                cute::tuple<
                    cute::_64,
                    cute::_512
                >,
                cute::tuple<
                    cute::_1,
                    cute::C<8192>
                >
            >
        >
    > tensor_layout;

    cute::Layout<
        cute::tuple<
            cute::tuple<
                cute::C<8>,
                cute::C<16>
            >,
            cute::tuple<
                cute::_64,
                cute::_2
            >
        >,
        cute::tuple<
            cute::tuple<
                cute::_64,
                cute::_512
            >,
            cute::tuple<
                cute::_1,
                cute::C<8192>
            >
        >
    > inner_layout;

    // 把原始数据整理成tensor
    auto tensor = cute::make_tensor(cpu_ptr, tensor_layout);
    // // 打印tensor原本的layout
    // for(int i=0;i<128;++i) {
    //     for(int j=0;j<128;++j) {
    //         // 列有效位
    //         uint32_t valid_col = j % 8;
    //         // 块id
    //         uint32_t id_block = i/8;
    //         // 块内偏移量
    //         uint32_t offset_block = i % 8;
    //         // 执行swizzle异或
    //         uint32_t swizzled_row = valid_col ^ id_block;
    //         // 计算最终的列id
    //         uint32_t final_row = swizzled_row * 8 + offset_block;
    //         uint32_t curr_value = tensor(final_row*128 + j);
    //         uint32_t id_row = curr_value / 128;
    //         uint32_t id_col = curr_value % 128;
    //         printPaddedString("(" + std::to_string(id_row) + ", " + std::to_string(id_col) + ") ", 15);
    //     }
    //     std::cout<<std::endl;
    // }

    cute::TiledCopy<
        cute::Copy_Atom<
            cute::SM80_CP_ASYNC_CACHEGLOBAL<
                cutlass::uint128_t,
                cutlass::uint128_t
            >,
            cutlass::bfloat16_t
        >,
        cute::Layout<
            cute::tuple<
                cute::tuple<
                    cute::C<8>,
                    cute::C<16>
                >,
                cute::_8
            >,
            cute::tuple<
                cute::tuple<
                    cute::_128,
                    cute::_1
                >,
                cute::_16
            >
        >,
        cute::tuple<
            cute::C<16>,
            cute::C<64>
        >
    > tiled_copy;

    // 打印每个线程的情况
    for(int id_thread=0;id_thread<128;++id_thread) {
        std::cout<<"thread id: "<<id_thread<<std::endl;
        // 指定线程号的复制方案
        auto gmem_thr_copy_QKV = tiled_copy.get_thread_slice(id_thread);
        // 获取当前线程负责的切片
        auto thread_tile = gmem_thr_copy_QKV.partition_D(tensor);
        // 打印线程切片的内容
        for(int i=0;i<16;++i) {
            for(int j=0;j<8;++j) {
                std::cout<<thread_tile(i*8 + j)<<" ";
            }
            std::cout<<std::endl;
        }
    }
}

// 通过gpu的测试函数
void test_tiled_copy_gpu() {
    // 开辟128*128的cpu内存
    float* cpu_ptr = (float*)malloc(128 * 128 * sizeof(float));
    // 从0开始按顺序填充数据
    for (int i = 0; i < 128 * 128; i++) {
        cpu_ptr[i] = i;
    }
    // 把cpu数据转移到gpu上
    float* gpu_ptr;
    cudaMalloc(&gpu_ptr, 128 * 128 * sizeof(float));
    cudaMemcpy(gpu_ptr, cpu_ptr, 128 * 128 * sizeof(float), cudaMemcpyHostToDevice);
    // 开辟用于存储结果的gpu内存，大小同样是是128*128
    float* gpu_result;
    cudaMalloc(&gpu_result, 128 * 128 * sizeof(float));
    tiled_copy_test<<<1, 128>>>(gpu_ptr, gpu_result);
    // 把结果从gpu拷贝回cpu
    float* cpu_result = (float*)malloc(128 * 128 * sizeof(float));
    cudaMemcpy(cpu_result, gpu_result, 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost);
    // 打印前256个数据
    for (int i = 0; i < 1024; i++) {
        if(i % 128 == 0) {
            std::cout << std::endl;
        }
        std::cout << cpu_result[i] << " ";
    }
    std::cout << std::endl;
    // 释放内存
    cudaFree(gpu_ptr);
    cudaFree(gpu_result);
    free(cpu_ptr);
    free(cpu_result);
}

int main() {
    test_tiled_copy_cpu<uint16_t>();

    return 0;
}