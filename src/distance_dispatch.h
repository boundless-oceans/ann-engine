// 私有纯选择函数：供实现与测试复用，不属于 include/ann 的公共接口。
#pragma once
#ifndef ANN_SRC_DISTANCE_DISPATCH_H
#define ANN_SRC_DISTANCE_DISPATCH_H

#include "ann/distance.h"

namespace ann::detail {

// 只根据给定特性返回地址，不执行目标函数；模拟特性只用于测试选择策略。
DistFn SelectDistFnForFeatures(Metric metric, const CpuFeatures& features);

}  // namespace ann::detail

#endif  // ANN_SRC_DISTANCE_DISPATCH_H
