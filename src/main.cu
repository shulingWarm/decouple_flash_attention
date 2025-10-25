#include<iostream>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <iostream>
#include"attention.h"
#include"run_fwd_template.h"
#include <cutlass/bfloat16.h>
#include "flash.h"
#include"simple_tensor.h"
#include<cmath>

void init_params(flash::Flash_fwd_params& params,
    SimpleTensor& q,
    SimpleTensor& k,
    SimpleTensor& v,
    SimpleTensor& o,
    SimpleTensor& softmax_lse // softmax的中间计算结果
) {
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
    params.o_ptr = o.data_ptr;
    params.o_row_stride = o.get_stride(-3);
    params.o_head_stride = o.get_stride(-2);

    // 默认 cu_seqlens_q_d 是空指针的情况
    params.q_batch_stride = q.get_stride(0);
    params.k_batch_stride = k.get_stride(0);
    params.v_batch_stride = v.get_stride(0);
    params.o_batch_stride = o.get_stride(0);

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
    // 打印output的前10个元素
    for (int i = 0; i < 10; i++) {
        std::cout << ((float*)o.data_ptr)[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}