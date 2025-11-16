#include<iostream>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/pointer.hpp>
#include <cute/algorithm/copy.hpp>

constexpr int CELL_SIZE = 32;
constexpr int TOTAL_SIZE = CELL_SIZE * CELL_SIZE * CELL_SIZE * CELL_SIZE;

// 执行数据重排的核函数
__global__ void reorganize_kernel(float* g_ptr, float* out_ptr) {
    // 创建一个32*32的共享内存
    __shared__ float s_ptr[CELL_SIZE * CELL_SIZE];
    // 定义共享内存tensor, 形状是(32,32)
    auto smem_tensor = cute::make_tensor(cute::make_smem_ptr(s_ptr), 
        cute::make_shape(cute::Int<CELL_SIZE>(), cute::Int<CELL_SIZE>()));
    // 定义全局内存tensor, 形状是(32,32,32,32)
    auto gmem_tensor = cute::make_tensor(cute::make_gmem_ptr(g_ptr), 
        cute::make_shape(cute::Int<CELL_SIZE>(), cute::Int<CELL_SIZE>(), 
            cute::Int<CELL_SIZE>(), cute::Int<CELL_SIZE>()));
    // 创建输出tensor，形状是(32,32,32,32)
    auto out_tensor = cute::make_tensor(cute::make_gmem_ptr(out_ptr), 
        cute::make_shape(cute::Int<CELL_SIZE>(), cute::Int<CELL_SIZE>(), 
            cute::Int<CELL_SIZE>(), cute::Int<CELL_SIZE>()));

    // 获取gmem_tensor的[bidx, bidy, tidx, :]的切片
    auto gmem_slice = gmem_tensor(cute::_, threadIdx.x, blockIdx.x, blockIdx.y);
    // 获取当前线程需要复制的shared_memory切片
    auto thread_copy_slice = smem_tensor(threadIdx.x, cute::_);
    // 执行复制
    cute::copy(gmem_slice, thread_copy_slice);
    // 调用同步
    __syncthreads();

    // 创建输出tensor的切片
    auto out_slice = out_tensor(blockIdx.x, blockIdx.y, threadIdx.x, cute::_);
    // 创建每个线程的输出切片
    auto thread_out_slice = smem_tensor(cute::_, threadIdx.x);
    // 执行复制
    cute::copy(thread_out_slice, out_slice);
}

// 打印layout的测试函数
void print_layout_test() {
    cute::Layout<
        cute::tuple<
            cute::C<16>,
            cute::C<64>
        >,
        cute::tuple<
            cute::_8,
            cute::_128
        >
    > temp_layout;

    // 调用layout的打印函数
    print_layout(temp_layout);
}

// 传入x,y,打印cpu_ptr[x,y,:,:]的结果
void print_slice(float* cpu_ptr, int x, int y) {
    std::cout << "Slice at [" << x << "," << y << ",:,:]:" << std::endl;
    for (int i = 0; i < CELL_SIZE; ++i) {
        for (int j = 0; j < CELL_SIZE; ++j) {
            int index = x * CELL_SIZE * CELL_SIZE * CELL_SIZE + 
                        y * CELL_SIZE * CELL_SIZE + 
                        i * CELL_SIZE + 
                        j;
            std::cout << cpu_ptr[index] << " ";
        }
        std::cout << std::endl;
    }
}

// 打印x,y的大粒度切片 cpu_ptr[:,:,x,y]
void print_topview_slice(float* cpu_ptr, int x, int y) {
    std::cout << "Topview slice at [:,:," << x << "," << y << "]:" << std::endl;
    for (int i = 0; i < CELL_SIZE; ++i) {
        for (int j = 0; j < CELL_SIZE; ++j) {
            int index = i * CELL_SIZE * CELL_SIZE * CELL_SIZE + 
                        j * CELL_SIZE * CELL_SIZE + 
                        x * CELL_SIZE + 
                        y;
            std::cout << cpu_ptr[index] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {

    print_layout_test();
    return 0;
    
    // 新建一个32*32*32*32的CPU内存, 类型是float
    float* cpu_ptr = new float[TOTAL_SIZE];
    // 随便初始化一下
    for (int i = 0; i < TOTAL_SIZE; ++i) {
        cpu_ptr[i] = static_cast<float>(i);
    }
    print_slice(cpu_ptr, 0, 0);

    // 新建一个32*32*32*32的GPU内存, 类型是float
    float* gpu_ptr;
    cudaMalloc(&gpu_ptr, sizeof(float) * TOTAL_SIZE);
    // 复制CPU内存到GPU内存
    cudaMemcpy(gpu_ptr, cpu_ptr, sizeof(float) * TOTAL_SIZE, cudaMemcpyHostToDevice);

    // 初始化相同大小的输出内存
    float* out_ptr;
    cudaMalloc(&out_ptr, sizeof(float) * TOTAL_SIZE);

    // 调用数据重排的核函数，block大小是(32,1,1)，grid大小是(32,32,1)
    reorganize_kernel<<<dim3(CELL_SIZE, CELL_SIZE, 1), dim3(CELL_SIZE, 1, 1)>>>(
        gpu_ptr, out_ptr);

    // 把输出结果复制到cpu_ptr上
    cudaMemcpy(cpu_ptr, out_ptr, sizeof(float) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
    // 打印[0,0,:,:]的结果
    print_slice(cpu_ptr, 0, 0);
    print_topview_slice(cpu_ptr, 0, 0);

    // 释放内存
    cudaFree(gpu_ptr);
    cudaFree(out_ptr);
    delete[] cpu_ptr;
    return 0;
    
}