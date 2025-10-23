#pragma once

#include "attention.h"
#include "kernel_traits.h"
#include "flash_fwd_launch_template.h"

using namespace flash;

template<typename T, bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    if constexpr(!Is_causal) {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, 
            false, // Is_dropout
            Is_causal>(params, stream);
    } else {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, 
            false, // Is_dropout
            Is_causal>(params, stream);
    }
}