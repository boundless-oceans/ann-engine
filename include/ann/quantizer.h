#pragma once
#ifndef ANN_QUANTIZER_H
#define ANN_QUANTIZER_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ann {

// 沿用总设计的量化抽象；距离查表当前按平方 L2 解释。
class Quantizer {
public:
    virtual ~Quantizer() = default;
    // 原始向量维度；与CodeSize/LutSize一样由构造确定，训练不改变形状。
    virtual size_t Dim() const = 0;
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

// 每维独立均匀量化；输入/输出缓冲不得重叠，容量由调用方保证。
class ScalarQuantizer final : public Quantizer {
public:
    explicit ScalarQuantizer(size_t dim);
    size_t Dim() const override { return dim_; }
    // 有限的连续 n×dim 矩阵，n>0；成功后才替换已有模型。
    // Train 不得与同实例的任何其他操作并发。
    void Train(size_t n, const float* vecs) override;
    size_t CodeSize() const override { return dim_; }
    // 最近整数，半格向较大编码取整；超出训练范围时截断。
    void Encode(const float* vec, uint8_t* code) const override;
    void Decode(const uint8_t* code, float* vec) const override;
    // 查询不量化；LUT 布局 dim×256，lut[d*256+c]=(q[d]-Decode(c)[d])²。
    // 常量维的所有编码都解码为训练常量；LUT 仍为该维保留256项。
    void BuildLUT(const float* query, float* lut) const override;
    size_t LutSize() const override { return dim_ * 256; }
    float DistanceFromLUT(const float* lut, const uint8_t* code) const override;
    // 对象和模型数组申请容量，不含调用方编码/LUT或分配器内部开销。
    size_t MemoryUsage() const override;
    const char* Name() const override { return "SQ8"; }

private:
    void RequireTrained() const;

    size_t dim_;
    // 保存端点而非 float 步长，避免极小非零范围除以255后下溢为0。
    std::vector<float> mins_, maxs_;
    bool trained_ = false;
};

// 等长连续子空间的乘积量化；每段存一个字节，不做小于8位的位打包。
// 输入/输出缓冲不得重叠，容量由调用方保证；训练不得与同实例其他操作并发。
class ProductQuantizer final : public Quantizer {
public:
    ProductQuantizer(size_t dim, size_t m, size_t ksub = 256, uint64_t seed = 42);
    size_t Dim() const override { return dim_; }
    // n>=ksub；各子空间固定25轮KMeans，成功后才替换全部码本。
    void Train(size_t n, const float* vecs) override;
    size_t CodeSize() const override { return m_; }
    // 每段选择最近L2中心，等距取较小码字ID。
    void Encode(const float* vec, uint8_t* code) const override;
    void Decode(const uint8_t* code, float* vec) const override;
    // lut[s*ksub+c]=原始查询子空间到码字c的平方L2；失败时不写半张表。
    void BuildLUT(const float* query, float* lut) const override;
    size_t LutSize() const override { return m_ * ksub_; }
    float DistanceFromLUT(const float* lut, const uint8_t* code) const override;
    size_t MemoryUsage() const override;
    const char* Name() const override { return "PQ"; }

private:
    void RequireTrained() const;
    void CheckCodes(const uint8_t* code) const;

    size_t dim_, m_, dsub_, ksub_;
    uint64_t seed_;
    bool trained_ = false;
    // 连续布局[s][c][d]，共m×ksub×dsub=ksub×dim个float。
    std::vector<float> codebooks_;
};

}  // namespace ann
#endif  // ANN_QUANTIZER_H
