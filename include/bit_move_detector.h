#include <cstdint>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <random>
#include <bitset>
#include <map>

class BitMoveDetector {
private:
    using BlackBoxFunc = std::function<uint32_t(uint32_t)>;
    
    BlackBoxFunc black_box_;
    uint32_t valid_bits_;
    std::vector<int> permutation_;
    bool permutation_detected_;
    std::vector<bool> valid_mappings_; // 记录哪些比特映射是有效的
    std::vector<std::pair<uint32_t, uint32_t>> conflict_cases_; // 记录冲突的测试用例
    
    // 计算一个数的二进制中1的个数
    int popcount(uint32_t x) const {
        int count = 0;
        while (x) {
            count += (x & 1);
            x >>= 1;
        }
        return count;
    }
    
    // 找到最低位的1的位置（0-based）
    int find_lowest_set_bit(uint32_t x) const {
        if (x == 0) return -1;
        int pos = 0;
        while ((x & 1) == 0) {
            x >>= 1;
            pos++;
        }
        return pos;
    }
    
    // 检测比特位移动规律
    void detect_permutation() {
        permutation_.assign(valid_bits_, -1);
        valid_mappings_.assign(valid_bits_, false);
        conflict_cases_.clear();
        
        // 测试每个比特位
        for (int i = 0; i < valid_bits_; i++) {
            uint32_t test_input = 1u << i;
            uint32_t output = black_box_(test_input);
            
            // 检查输出是否在有效范围内
            if (output >= (1u << valid_bits_)) {
                std::cout << "警告: 比特位 " << i << " 的输出超出有效范围: 0x" 
                          << std::hex << output << std::endl;
                continue;
            }
            
            // 检查输出中1的个数
            int output_popcount = popcount(output);
            if (output_popcount != 1) {
                conflict_cases_.push_back({test_input, output});
                std::cout << "警告: 比特位 " << i << " 的映射产生多个比特位输出: 输入=0x" 
                          << std::hex << test_input << ", 输出=0x" << output << std::endl;
                continue;
            }
            
            // 找到输出比特位的位置
            int output_bit = find_lowest_set_bit(output);
            if (output_bit < 0 || output_bit >= valid_bits_) {
                conflict_cases_.push_back({test_input, output});
                std::cout << "警告: 比特位 " << i << " 映射到无效位置: " << output_bit << std::endl;
                continue;
            }
            
            // 检查是否已经映射过
            bool conflict = false;
            for (int j = 0; j < i; j++) {
                if (permutation_[j] == output_bit) {
                    conflict = true;
                    conflict_cases_.push_back({test_input, output});
                    conflict_cases_.push_back({1u << j, black_box_(1u << j)});
                    std::cout << "警告: 比特位 " << i << " 和 " << j 
                              << " 都映射到输出比特位 " << output_bit << std::endl;
                    break;
                }
            }
            
            if (!conflict) {
                permutation_[i] = output_bit;
                valid_mappings_[i] = true;
            }
        }
        
        // 检查是否有未映射的比特位
        std::vector<bool> output_covered(valid_bits_, false);
        for (int i = 0; i < valid_bits_; i++) {
            if (permutation_[i] != -1) {
                output_covered[permutation_[i]] = true;
            }
        }
        
        for (int i = 0; i < valid_bits_; i++) {
            if (!output_covered[i]) {
                std::cout << "警告: 输出比特位 " << i << " 没有被任何输入比特位映射" << std::endl;
            }
        }
        
        permutation_detected_ = true;
    }

public:
    BitMoveDetector(BlackBoxFunc black_box, uint32_t valid_bits) 
        : black_box_(black_box), valid_bits_(valid_bits), permutation_detected_(false) {
        if (valid_bits > 32) {
            throw std::invalid_argument("有效比特位数不能超过32");
        }
    }
    
    // 静态函数：快速检测并打印变换规则
    static void quick_detect(BlackBoxFunc black_box, uint32_t valid_bits) {
        BitMoveDetector detector(black_box, valid_bits);
        detector.detect();
        
        std::cout << "\n=== 比特移动规律检测结果 ===" << std::endl;
        std::cout << "有效比特位数: " << valid_bits << std::endl;
        detector.print_analysis();
        
        std::cout << "=============================" << std::endl;
    }
    
    // 检测比特移动规律
    void detect() {
        detect_permutation();
    }
    
    // 获取检测到的比特移动规律
    const std::vector<int>& get_permutation() const {
        if (!permutation_detected_) {
            throw std::runtime_error("尚未检测比特移动规律");
        }
        return permutation_;
    }
    
    // 获取有效的比特映射
    const std::vector<bool>& get_valid_mappings() const {
        if (!permutation_detected_) {
            throw std::runtime_error("尚未检测比特移动规律");
        }
        return valid_mappings_;
    }
    
    // 获取冲突的测试用例
    const std::vector<std::pair<uint32_t, uint32_t>>& get_conflict_cases() const {
        return conflict_cases_;
    }
    
    // 使用检测到的规律对输入进行变换（只对有效的比特位）
    uint32_t transform(uint32_t input) const {
        if (!permutation_detected_) {
            throw std::runtime_error("尚未检测比特移动规律");
        }
        
        uint32_t result = 0;
        uint32_t mask = (1u << valid_bits_) - 1;
        input &= mask;  // 只保留有效比特位
        
        for (int i = 0; i < valid_bits_; i++) {
            if (valid_mappings_[i] && (input & (1u << i))) {
                result |= (1u << permutation_[i]);
            }
        }
        
        return result;
    }
    
    // 验证检测结果的正确性（只对有效的比特位）
    bool verify(uint32_t test_cases = 100) const {
        if (!permutation_detected_) {
            return false;
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, (1u << valid_bits_) - 1);
        
        bool all_passed = true;
        for (uint32_t i = 0; i < test_cases; i++) {
            uint32_t input = dis(gen);
            uint32_t expected = black_box_(input);
            uint32_t actual = transform(input);
            
            if (expected != actual) {
                std::cout << "验证失败: 输入=0x" << std::hex << input 
                          << " (二进制: " << std::bitset<32>(input).to_string().substr(32-valid_bits_) << ")"
                          << ", 期望=0x" << expected 
                          << " (二进制: " << std::bitset<32>(expected).to_string().substr(32-valid_bits_) << ")"
                          << ", 实际=0x" << actual
                          << " (二进制: " << std::bitset<32>(actual).to_string().substr(32-valid_bits_) << ")"
                          << std::endl;
                all_passed = false;
            }
        }
        
        return all_passed;
    }
    
    // 打印比特移动规律分析
    void print_analysis() const {
        if (!permutation_detected_) {
            std::cout << "尚未检测到比特移动规律" << std::endl;
            return;
        }
        
        // 打印有效的比特移动规律
        std::cout << "\n有效的比特移动规律（输入位 -> 输出位）：" << std::endl;
        bool has_valid = false;
        for (int i = 0; i < valid_bits_; i++) {
            if (valid_mappings_[i]) {
                std::cout << "比特位 " << i << " -> " << permutation_[i] << std::endl;
                has_valid = true;
            }
        }
        
        if (!has_valid) {
            std::cout << "没有找到有效的比特移动规律" << std::endl;
        }
        
        // 打印无效的比特位映射
        std::cout << "\n无效的比特位映射：" << std::endl;
        bool has_invalid = false;
        for (int i = 0; i < valid_bits_; i++) {
            if (!valid_mappings_[i]) {
                uint32_t test_input = 1u << i;
                uint32_t output = black_box_(test_input);
                std::cout << "比特位 " << i << ": 输入=0x" << std::hex << test_input 
                          << " -> 输出=0x" << output 
                          << " (二进制: " << std::bitset<32>(output).to_string().substr(32-valid_bits_) << ")"
                          << std::endl;
                has_invalid = true;
            }
        }
        
        if (!has_invalid) {
            std::cout << "所有比特位映射都是有效的" << std::endl;
        }
        
        // 打印冲突的测试用例
        if (!conflict_cases_.empty()) {
            std::cout << "\n冲突的测试用例：" << std::endl;
            for (const auto& conflict : conflict_cases_) {
                std::cout << "输入=0x" << std::hex << conflict.first 
                          << " (二进制: " << std::bitset<32>(conflict.first).to_string().substr(32-valid_bits_) << ")"
                          << " -> 输出=0x" << conflict.second
                          << " (二进制: " << std::bitset<32>(conflict.second).to_string().substr(32-valid_bits_) << ")"
                          << std::endl;
            }
        }
        
        // 验证结果
        std::cout << "\n验证结果（仅对有效比特位）:" << std::endl;
        bool verified = verify(20); // 使用20个测试用例验证
        std::cout << "验证结果: " << (verified ? "✓ 通过" : "✗ 失败") << std::endl;
        
        // 打印几个示例
        std::cout << "\n示例测试:" << std::endl;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, (1u << valid_bits_) - 1);
        
        for (int i = 0; i < 3; i++) {
            uint32_t input = dis(gen);
            uint32_t expected = black_box_(input);
            uint32_t actual = transform(input);
            std::cout << "输入: 0x" << std::hex << input 
                      << " (二进制: " << std::bitset<32>(input).to_string().substr(32-valid_bits_) << ")"
                      << " -> 输出: 0x" << std::hex << actual
                      << " (二进制: " << std::bitset<32>(actual).to_string().substr(32-valid_bits_) << ")"
                      << " 期望: 0x" << expected
                      << " (二进制: " << std::bitset<32>(expected).to_string().substr(32-valid_bits_) << ")"
                      << " " << (expected == actual ? "✓" : "✗") << std::endl;
        }
    }
};