#include "ann/distance.h"

//实现函数
namespace ann {
//欧氏距离平方：Σ (aᵢ − bᵢ)²
dist_t L2SqrScalar(const float* a, const float* b, size_t dim){
    //定义欧式距离平方
    dist_t l2_sqr = 0.0f;
    for(size_t i = 0; i < dim; ++i){
        l2_sqr += (a[i]-b[i])*(a[i]-b[i]);
    }
    return l2_sqr;
}

//内积的相反数：− Σ (aᵢ × bᵢ)
dist_t InnerProductScalar(const float* a, const float* b, size_t dim){
    //定义内积
    dist_t inner_product = 0.0f;
    for(size_t i = 0; i < dim; ++i){
        inner_product += a[i]*b[i];
    }
    return -inner_product;
}

//按度量返回对应的函数指针
DistFn SelectDistFn(Metric metric){
    //metric有三种枚举值，根据不同的枚举值返回对应的函数指针
    if(metric == Metric::L2) return L2SqrScalar;
    else if(metric == Metric::InnerProduct) return InnerProductScalar;
    //归一化未实现，暂时回退
    else return InnerProductScalar;
}

} //namespace ann