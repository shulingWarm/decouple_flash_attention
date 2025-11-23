#include"rand_generator.h"
#include"cuda_check.h"
#include <cutlass/bfloat16.h>
#include<cmath>

using bf16 = cutlass::bfloat16_t;
using u32 = unsigned int;

static constexpr int M = 16;
static constexpr int N = 8;
static constexpr int K = 16;

// 用于随机初始化CPU矩阵的函数
template<class T>
void init_matrix(T *matrix, int rows, int cols, UniformRandomGenerator& generator) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (T)generator.generate(-0.5, 0.5);
            std::cout<<(float)matrix[i * cols + j]<<" ";
        }
        std::cout<<std::endl;
    }

    std::cout<<"-------------------------"<<std::endl;
}

// 测试ptx指令的核函数
// A和B实际是bf16类型，只是用float类型表示这样可以一次取两个数据
__global__ void test_ptx_kernel(u32 *A, bf16 *B, float *C, float *dst) {
    // 准备每个线程读取A的4个数据
    u32 a_fragment[4];
    // 每个线程要读取的B数据
    u32 b_fragment[2];
    // 准备c矩阵和d矩阵的切片
    float c_fragment[4];
    float d_fragment[4] = {0,0,0,0};
    // abcd矩阵的排布参考: 
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
    // 当前线程在块内的偏移量
    int in_block_row = threadIdx.x/4;
    int in_block_col = threadIdx.x%4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int block_row = i%2;
        int block_col = i/2;

        int element_row = block_row*8 + in_block_row;
        int element_col = block_col*4 + in_block_col;

        a_fragment[i] = A[element_row*8 + element_col];
    }

    in_block_row = threadIdx.x%4;
    in_block_col = threadIdx.x/4;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        // 当前写入位置的核心指针
        bf16* bf16_ptr = (bf16*)(&b_fragment[i]);
        int element_row = i*8 + in_block_row*2;
        bf16_ptr[0] = B[element_row*8 + in_block_col];
        bf16_ptr[1] = B[(element_row+1)*8 + in_block_col];
    }

    in_block_col = (threadIdx.x%4)*2;
    in_block_row = threadIdx.x/4;

    // 赋值CD矩阵的切片
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int element_row = in_block_row + i*8;
        int offset = element_row*8 + in_block_col;
        c_fragment[i*2] = C[offset];
        c_fragment[i*2+1] = C[offset+1];
    }

    // 执行gemm运算
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d_fragment[0]), "=f"(d_fragment[1]), "=f"(d_fragment[2]), "=f"(d_fragment[3])
      :  "r"(a_fragment[0]),  "r"(a_fragment[1]),  "r"(a_fragment[2]),  "r"(a_fragment[3]),
         "r"(b_fragment[0]),  "r"(b_fragment[1]),
         "f"(c_fragment[0]),  "f"(c_fragment[1]),  "f"(c_fragment[2]),  "f"(c_fragment[3]));

    // 将数据保存到dst矩阵
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int element_row = in_block_row + i*8;
        int offset = element_row*8 + in_block_col;
        dst[offset] = d_fragment[i*2];
        dst[offset+1] = d_fragment[i*2+1];
    }
}

// CPU的矩阵相乘函数，用于验证计算结果
void cpu_gemm(bf16 *A, bf16 *B, float *C, float* D, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            D[i * N + j] = sum + C[i * N + j];
        }
    }
}

// 计算两个矩阵的误差
void print_diff(float *cpu_result, float *gpu_result, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float diff = std::abs(cpu_result[i * N + j] - static_cast<float>(gpu_result[i * N + j]));
            printf("%.5g ", diff);
        }
        printf("\n");
    }
}

int main() {
    // 构造随机数生成器
    UniformRandomGenerator generator;

    // 创建CPU矩阵 A, B, C
    bf16 *A = new bf16[M * K];
    bf16 *B = new bf16[K * N];
    float *C = new float[M * N];

    // 用于存储结果的dst矩阵
    float *cpu_dst = new float[M * N];

    // 随机初始化三个矩阵
    init_matrix(A, M, K, generator);
    init_matrix(B, K, N, generator);
    init_matrix(C, M, N, generator);

    // 计算cpu的参考答案 
    cpu_gemm(A, B, C, cpu_dst, M, N, K);

    // 把A,B,C从CPU拷贝到GPU malloc的时候加上错误检查
    // 创建bf16的数据类型

    bf16 *d_A, *d_B;
    float *d_C;
    cuda_check(cudaMalloc(&d_A, M * K * sizeof(bf16)));
    cuda_check(cudaMalloc(&d_B, K * N * sizeof(bf16)));
    cuda_check(cudaMalloc(&d_C, M * N * sizeof(float)));
    cuda_check(cudaMemcpy(d_A, A, M * K * sizeof(bf16), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_B, B, K * N * sizeof(bf16), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 初始化gpu的存储结果的指针
    float *d_dst;
    cuda_check(cudaMalloc(&d_dst, M * N * sizeof(float)));
    // 调用核函数
    test_ptx_kernel<<<1, 32>>>((u32*)d_A, d_B, d_C, d_dst);
    // 把gpu的计算结果转到cpu上
    float cpu_result[M * N];
    cuda_check(cudaMemcpy(cpu_result, d_dst, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(d_A));
    cuda_check(cudaFree(d_B));
    cuda_check(cudaFree(d_C));
    cuda_check(cudaFree(d_dst));

    // 打印两个矩阵的误差
    print_diff(cpu_dst, cpu_result, M, N);
}