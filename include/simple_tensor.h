#pragma once
#include<iostream>
#include<vector>
#include<fstream>

#include <cstdint>
#include <cuda_runtime.h> // 包含CUDA运行时头文件
#include "cuda_check.h" // 包含CUDA错误检查头文件

class SimpleTensor {
public:
    enum DataType {
        INT32,
        FP16,
        BF16,
        FP32
    };

    static uint32_t get_elem_size(DataType _data_type) {
        if(_data_type == DataType::INT32 || _data_type == DataType::FP32) {
            return 4;
        }
        return 2;
    }

    std::vector<uint32_t> shape;
    DataType data_type;
    
    char* data_ptr = nullptr;
    char* cuda_ptr = nullptr; // 新增：CUDA设备指针

    ~SimpleTensor() {
        if (data_ptr != nullptr) {
            delete[] data_ptr;
        }
        if (cuda_ptr != nullptr) {
            cuda_check(cudaFree(cuda_ptr)); // 释放CUDA内存
        }
    }

    uint32_t get_elem_num() const {
        if(shape.size() == 0)
            return 0;
        uint32_t total_elem_num = 1;
        for(auto each_dim : shape)
            total_elem_num *= each_dim;
        return total_elem_num;
    }

    void reset_shape(std::vector<uint32_t> shape, DataType data_type) {
        this->shape = shape;
        this->data_type = data_type;
        uint32_t total_elem_num = get_elem_num();
        if(data_ptr != nullptr) {
            delete[] data_ptr; 
        }
        if (cuda_ptr != nullptr) {
            cuda_check(cudaFree(cuda_ptr));
            cuda_ptr = nullptr;
        }
        data_ptr = new char[total_elem_num * get_elem_size(this->data_type)];
    }

    // 打印tensor的shape
    void print_shape() const {
        std::cout << "Shape: ";
        for(auto each_dim : shape)
            std::cout << each_dim << " ";
        std::cout << std::endl;
    }

    // 获取指定维度的步长（以元素个数为单位）
    // dim可以为负数，例如-1表示最后一个维度
    uint32_t get_stride(int id_dim) const {
        if (shape.empty()) {
            return 0; // 空tensor返回0
        }
        
        int ndim = static_cast<int>(shape.size());
        if (id_dim < 0) {
            id_dim = ndim + id_dim; // 将负数索引转换为正数
        }
        
        if (id_dim < 0 || id_dim >= ndim) {
            return 0; // 无效索引返回0
        }
        
        uint32_t stride = 1;
        for (int i = id_dim + 1; i < ndim; i++) {
            stride *= shape[i];
        }
        return stride;
    }

    // Copy data from CUDA device memory to host memory
    void to_cpu() {
        if (cuda_ptr == nullptr) {
            return; // No CUDA memory to copy from
        }
        
        uint32_t total_elem_num = get_elem_num();
        if (total_elem_num == 0) {
            return; // Empty tensor
        }
        
        size_t total_bytes = static_cast<size_t>(total_elem_num) * get_elem_size(data_type);
        // Copy data from device to host
        cuda_check(cudaMemcpy(data_ptr, cuda_ptr, total_bytes, cudaMemcpyDeviceToHost));
    }

    // 将数据复制到CUDA设备内存
    void to_cuda() {
        if (cuda_ptr != nullptr) {
            cuda_check(cudaFree(cuda_ptr)); // 如果已有CUDA内存，先释放
        }
        
        uint32_t total_elem_num = get_elem_num();
        if (total_elem_num == 0) {
            cuda_ptr = nullptr;
            return; // 空tensor无需复制
        }
        
        size_t total_bytes = static_cast<size_t>(total_elem_num) * get_elem_size(data_type);
        cuda_check(cudaMalloc(&cuda_ptr, total_bytes)); // 分配CUDA内存
        
        // 将数据从主机复制到设备
        cuda_check(cudaMemcpy(cuda_ptr, data_ptr, total_bytes, cudaMemcpyHostToDevice));
    }

    // 获取CUDA设备指针
    char* get_cuda_ptr() const {
        return cuda_ptr;
    }
};

template<class T>
T read_data(std::fstream& file_handle) {
    T ret;
    file_handle.read((char*)&ret, sizeof(T));
    return ret;
}

// 从文件中读取tensor的函数
void load_tensor_as_simple(std::string filePath, SimpleTensor::DataType data_type, 
    SimpleTensor& tensor) {
    // 读取文件
    std::fstream file_handle(filePath, std::ios::in|std::ios::binary);
    int dim_num = read_data<int>(file_handle);
    // 读取每个维度
    std::vector<uint32_t> dims(dim_num);
    for(uint32_t id_dim=0;id_dim<dim_num;++id_dim) {
        dims[id_dim] = read_data<int>(file_handle);
    }

    // 构造tensor
    tensor.reset_shape(dims, data_type);
    // 读取完整数据
    file_handle.read((char*)tensor.data_ptr, 
        tensor.get_elem_num()*SimpleTensor::get_elem_size(data_type));
}