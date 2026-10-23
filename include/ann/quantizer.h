#pragma once
#ifndef ANN_QUANTIZER_H
#define ANN_QUANTIZER_H

#include <cstddef>
#include <cstdint>

namespace ann {

// 沿用总设计的量化抽象；具体 SQ/PQ 实现留在后续功能块。
class Quantizer {
public:
    virtual ~Quantizer() = default;
    virtual void Train(size_t n, const float* vecs) = 0;
    virtual size_t CodeSize() const = 0;
    virtual void Encode(const float* vec, uint8_t* code) const = 0;
    virtual void Decode(const uint8_t* code, float* vec) const = 0;

    // LUT 由调用方独占，查询不得在量化器成员中保存临时状态。
    virtual void BuildLUT(const float* query, float* lut) const = 0;
    virtual size_t LutSize() const = 0;
    virtual float DistanceFromLUT(const float* lut, const uint8_t* code) const = 0;
    virtual size_t MemoryUsage() const = 0;
    virtual const char* Name() const = 0;
};

}  // namespace ann
#endif  // ANN_QUANTIZER_H
