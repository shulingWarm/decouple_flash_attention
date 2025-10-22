cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(DecoupleFlashAttn LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 检测并设置CUDA架构
set(CUDA_ARCH "sm_89")  # Ampere架构 (RTX 30系列/A100等)

# 主机编译器优化 (GCC/Clang)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -funroll-loops")

# CUDA专用优化标志
list(APPEND CUDA_OPTIONS
    # 架构相关
    --generate-code=arch=compute_89,code=${CUDA_ARCH}
    
    # 数学优化
    --use_fast_math
    --ftz=true          # 刷新非正规浮点数到零
    --prec-div=false    # 降低除法精度
    --prec-sqrt=false   # 降低平方根精度
    
    # 性能优化
    --fmad=true         # 启用浮点乘加
    -Xptxas -O3        # PTX优化级别
    -Xptxas --disable-warnings
    --restrict          # 指针限制优化
    
    # 向量化与并行
    -Xptxas -dlcm=ca   # 缓存控制
    --maxrregcount=0    # 不限制寄存器使用
    
    # 其他优化
    --default-stream per-thread
    --extra-device-vectorization
)

# 调试信息（发布模式保留行号）
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND CUDA_OPTIONS -lineinfo)
endif()

# 设置NVCC选项
string(REPLACE ";" " " CUDA_OPTIONS_STR "${CUDA_OPTIONS}")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} ${CUDA_OPTIONS_STR}")

find_package(Torch PATHS "/mnt/data/usrApp/anaconda3/lib/python3.12/site-packages/torch/share/cmake/Torch" REQUIRED)

# 添加可执行文件
add_executable(attention_main 
    src/main.cu
)

# 链接CUDA库
target_link_libraries(attention_main PRIVATE
    cudart
    cublas
    cufft
    ${TORCH_LIBRARIES}  # PyTorch核心库
)

target_include_directories(attention_main PRIVATE
    include/
    ${TORCH_INCLUDE_DIRS}  # PyTorch头文件
    "/mnt/data/code/flash-attention/csrc/cutlass/include"
)