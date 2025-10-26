#pragma once
#include <chrono>
#include<iostream>
#include <functional>

// 把一个std::function函数重复测试多次统计用时
uint32_t time_profile(std::function<void()> func, int repeat) {
    // 调用cuda同步
    cudaDeviceSynchronize();
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
        func();
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    // 返回以秒为ms的时间
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}