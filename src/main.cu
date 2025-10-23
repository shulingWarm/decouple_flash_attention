#include<iostream>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include"attention.h"
#include"run_fwd_template.h"
#include <cutlass/bfloat16.h>
#include "flash.h"

int main() {
    flash::Flash_fwd_params params;
    run_mha_fwd_hdim128<cutlass::bfloat16_t, false>(params, nullptr);
    return 0;
}