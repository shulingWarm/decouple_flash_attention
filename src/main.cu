#include<iostream>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <iostream>
#include"attention.h"
#include"run_fwd_template.h"
#include <cutlass/bfloat16.h>
#include "flash.h"
#include"simple_tensor.h"
#include<cmath>
#include"profile/time_profile.h"

void init_params(flash::Flash_fwd_params& params,
    SimpleTensor& q,
    SimpleTensor& k,
    SimpleTensor& v,
    SimpleTensor& o,
    SimpleTensor& softmax_lse // softmax的中间计算结果
) {
    // 把参数内容指定成空的
    params = {};
    // 设置bf16
    params.is_bf16 = true;
    // 设置q,k,v的指针
    params.q_ptr = q.cuda_ptr;
    params.k_ptr = k.cuda_ptr;
    params.v_ptr = v.cuda_ptr;
    
    params.q_row_stride = q.get_stride(-3);
    params.k_row_stride = k.get_stride(-3);
    params.v_row_stride = v.get_stride(-3);
    params.q_head_stride = q.get_stride(-2);
    params.k_head_stride = k.get_stride(-2);
    params.v_head_stride = v.get_stride(-2);
    params.o_ptr = o.cuda_ptr;
    params.o_row_stride = o.get_stride(-3);
    params.o_head_stride = o.get_stride(-2);

    // 默认 cu_seqlens_q_d 是空指针的情况
    params.q_batch_stride = q.get_stride(0);
    params.k_batch_stride = k.get_stride(0);
    params.v_batch_stride = v.get_stride(0);
    params.o_batch_stride = o.get_stride(0);

    // 参考flash attention里面原本的实现，直接写死成1
    params.num_splits = 1;
    // 其他没有用到的参数初始化成0
    params.knew_batch_stride = 0;
    params.knew_head_stride = 0;
    params.knew_row_stride = 0;
    params.vnew_batch_stride = 0;
    params.vnew_head_stride = 0;
    params.vnew_row_stride = 0;
    params.block_table_batch_stride = 0;

    // 这三个东西初始化成空指针
    // cu_seqlens_q cu_seqlens_k seqused_k
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.seqused_k = nullptr;

    // 返回softmax的时候才会用到
    params.p_ptr = nullptr;

    // 存储softmax的中间结果用的
    params.softmax_lse_ptr = softmax_lse.cuda_ptr;

    // Set the dimensions.
    params.b = q.shape[0];
    // q里面的head个数
    params.h = q.shape[2];
    // k里面的head个数
    params.h_k = k.shape[2];
    params.h_h_k_ratio = q.shape[2]/k.shape[2];
    // 记录qk的序列长度
    params.seqlen_q = q.shape[1];
    params.seqlen_k = k.shape[1];
    // 对128向上取整
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    params.seqlen_q_rounded = round_multiple(params.seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(params.seqlen_k, 128);
    // 记录head_dim
    params.d = q.shape[3];
    params.d_rounded = round_multiple(params.d, params.d <= 128 ? 32 : 64);

    // 只考虑softcap是零的情况
    // Remove potential NaN
    params.softcap = 0.0;
    params.scale_softmax = 0.08;
    params.scale_softmax_log2 = params.scale_softmax * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    // 前向推理的情况下这就是false
    params.is_causal = false;

    // 左右的窗口大小都是-1
    // if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    // if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = -1;
    params.window_size_right = -1;

    params.is_seqlens_k_cumulative = true;

    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
}

void print_params(Flash_fwd_params &params) { 
    // 打印除指针之外的参数内容
    printf("params: \n");
    printf("seqlen_q = %d\n", params.seqlen_q);
    printf("seqlen_k = %d\n", params.seqlen_k);
    printf("is_bf16 = %d\n", params.is_bf16);
    printf("is_causal = %d\n", params.is_causal);
    printf("window_size_left = %d\n", params.window_size_left);
    printf("window_size_right = %d\n", params.window_size_right);
    printf("p_dropout = %f\n", params.p_dropout);
    printf("softcap = %f\n", params.softcap);
    printf("scale_softmax = %f\n", params.scale_softmax);
    printf("scale_softmax_log2 = %f\n", params.scale_softmax_log2);
    printf("rp_dropout = %f\n", params.rp_dropout);
    printf("scale_softmax_rp_dropout = %f\n", params.scale_softmax_rp_dropout);
    printf("num_splits = %d\n", params.num_splits);
    // 打印 o_batch_stride o_row_stride o_head_stride
    printf("o_batch_stride = %ld\n", params.o_batch_stride);
    printf("o_head_stride = %ld\n", params.o_head_stride);
    printf("o_row_stride = %ld\n", params.o_row_stride);
    // 打印 knew_batch_stride knew_head_stride knew_row_stride 
    printf("knew_batch_stride = %ld\n", params.knew_batch_stride);
    printf("knew_head_stride = %ld\n", params.knew_head_stride);
    printf("knew_row_stride = %ld\n", params.knew_row_stride);
    printf("vnew_batch_stride = %ld\n", params.vnew_batch_stride);
    printf("vnew_head_stride = %ld\n", params.vnew_head_stride);
    printf("vnew_row_stride = %ld\n", params.vnew_row_stride);
    printf("q_batch_stride = %ld\n", params.q_batch_stride);
    printf("q_head_stride = %ld\n", params.q_head_stride);
    printf("q_row_stride = %ld\n", params.q_row_stride);
    printf("k_batch_stride = %ld\n", params.k_batch_stride);
    printf("k_head_stride = %ld\n", params.k_head_stride);
    printf("k_row_stride = %ld\n", params.k_row_stride);
    printf("v_batch_stride = %ld\n", params.v_batch_stride);
    printf("v_head_stride = %ld\n", params.v_head_stride);
    printf("v_row_stride = %ld\n", params.v_row_stride);
    printf("block_table_batch_stride = %ld\n", params.block_table_batch_stride);
    // 打印 h h_k
    printf("h = %d\n", params.h);
    printf("h_k = %d\n", params.h_k);
    printf("h_h_k_ratio = %d\n", params.h_h_k_ratio);

    // 打印所有的指针
    printf("o_ptr = %p\n", params.o_ptr);
    printf("oaccum_ptr = %p\n", params.oaccum_ptr);
    printf("p_ptr = %p\n", params.p_ptr);
    printf("softmax_lse_ptr = %p\n", params.softmax_lse_ptr);
    printf("softmax_lseaccum_ptr = %p\n", params.softmax_lseaccum_ptr);
    printf("cu_seqlens_q = %p\n", params.cu_seqlens_q);
    printf("cu_seqlens_k = %p\n", params.cu_seqlens_k);
    printf("leftpad_k = %p\n", params.leftpad_k);
    printf("seqused_k = %p\n", params.seqused_k);
    printf("blockmask = %p\n", params.blockmask);
    printf("knew_ptr = %p\n", params.knew_ptr);
    printf("vnew_ptr = %p\n", params.vnew_ptr);
    printf("rotary_cos_ptr = %p\n", params.rotary_cos_ptr);
    printf("rotary_sin_ptr = %p\n", params.rotary_sin_ptr);
    printf("cache_batch_idx = %p\n", params.cache_batch_idx);
    printf("block_table = %p\n", params.block_table);
    printf("rng_state = %p\n", params.rng_state);
    printf("alibi_slopes_ptr = %p\n", params.alibi_slopes_ptr);
}

int main() {
    SimpleTensor q,k,v,o;
    // 分别读取qkv三个tensor
    load_tensor_as_simple("/mnt/data/temp/q.bin", SimpleTensor::DataType::BF16, q);
    load_tensor_as_simple("/mnt/data/temp/k.bin", SimpleTensor::DataType::BF16, k);
    load_tensor_as_simple("/mnt/data/temp/v.bin", SimpleTensor::DataType::BF16, v);
    o.reset_shape(q.shape, q.data_type);
    q.to_cuda();
    k.to_cuda();
    v.to_cuda();
    o.to_cuda();
    // softmax的中间结果
    SimpleTensor softmax_lse;
    softmax_lse.reset_shape({q.shape[0], q.shape[1], q.shape[2]}, SimpleTensor::DataType::FP32);
    softmax_lse.to_cuda();
    flash::Flash_fwd_params params;
    init_params(params,q,k,v,o,softmax_lse);

    run_mha_fwd_hdim128<cutlass::bfloat16_t, false>(params, nullptr);
    // 把output转换到cpu上
    o.to_cpu();

    // 执行1000次统计用时
    auto main_process = [&]() {
        run_mha_fwd_hdim128<cutlass::bfloat16_t, false>(params, nullptr);   
    };
    // Warm up
    time_profile(main_process, 10);
    // 正式计时
    auto total_time = time_profile(main_process, 1000); 

    // 打印耗时
    printf("Total time: %u ms\n", total_time);

    // 打印output的前10个元素
    for (int i = 0; i < 10; i++) {
        std::cout << ((float*)o.data_ptr)[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}