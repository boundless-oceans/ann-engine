// 距离函数的单元测试。
//
// 测试策略：先用手算得出来的小例子（2~3 维），因为出错时能一眼看出是哪一步不对。
// 直接拿大数据集跑召回率，只能告诉你"不对"，不能告诉你"哪里不对"。

#include <gtest/gtest.h>
#include <vector>

#include "ann/distance.h"

namespace {

// =============================================================================
//  用例的命名规则
//
//      TEST(套件名, 用例名)
//           ↑        ↑
//           分组      这一条测什么
//
//  - 完整名字是「套件名.用例名」，就是 --gtest_filter 用的字符串
//      ann_tests --gtest_filter='Distance.*'       整个套件
//      ann_tests --gtest_filter='Distance.L2*'     套件里 L2 开头的
//
//  - ⚠️ 两个名字里都不能有下划线 _，所以一律用驼峰：
//      TEST(L2_Sqr, Basic)   // ❌ 编译报错
//      TEST(L2Sqr,  Basic)   // ✅
//    原因：GTest 内部会拼出类名 L2Sqr_Basic_Test，允许你用 _ 就可能和
//    用户自己的类名撞车，所以它直接禁止了。
//
//  - 套件名填被测的模块名（Distance），用例名描述场景（L2HandComputed）
// =============================================================================

// -----------------------------------------------------------------------------
//  【示例用例】L2 平方距离：手算验证
//
//  这是留给你照抄形状的样板，后面 8 个用例按同样的结构写。
//
//  为什么选 2 维：手算得出来。
//      a = (0, 0)
//      b = (3, 4)
//      L2² = (0-3)² + (0-4)² = 9 + 16 = 25
//
//  如果这条失败了，怎么反推是哪一步错了：
//      得到 5          → 忘了平方（算成了开方后的距离）
//      得到 7          → 把 |aᵢ-bᵢ| 直接相加，而不是平方相加
//      得到 50         → 多乘了一个 2
//      得到 0          → 循环根本没进去（比如循环条件写成了 i > dim）
//
//  一个用例测出"错"很容易，难的是测出"错在哪"。手算用例的价值就在这里。
// -----------------------------------------------------------------------------
TEST(Distance, L2HandComputed) {
    // ① 准备输入
    //    数据很小，用定长数组就够，不必上 std::vector。
    //    const 不是为了性能，是为了让编译器帮你确认"测试没有修改输入"。
    const float a[2] = {0.0f, 0.0f};
    const float b[2] = {3.0f, 4.0f};

    // ② 调用被测函数。最后一个参数是维度，也就是数组长度。
    const ann::dist_t actual = ann::L2SqrScalar(a, b, 2);

    // ③ 断言
    //
    //    ⚠️ 这里绝对不能用 EXPECT_EQ！
    //       浮点运算天然有精度误差，L2 还要累加 dim 次，误差会累积，
    //       算出来的值几乎不可能精确等于手算的 25。
    //
    //    EXPECT_NEAR(实际值, 期望值, 容差)
    //        - 容差直接写成 1e-5f 这样的字面量，别人一眼知道你在比什么
    //        - 失败时 GTest 会把"差了多少"打出来，方便判断是算法错还是误差
    //
    //    （EXPECT_FLOAT_EQ 也能用，它按固定的 ULP 容差比较；
    //      手算用例推荐 EXPECT_NEAR，因为容差是显式的。）
    EXPECT_NEAR(actual, 25.0f, 1e-5f);

    // ④ 一条用例里可以放多个断言
    //    EXPECT_* 失败后【继续执行】，所以一次运行能看到这一条里所有的失败项。
    //    （ASSERT_* 则会立刻中止本条用例。）
    //
    //    这里顺便验证对称性：交换 a、b 结果应当不变。
    EXPECT_NEAR(ann::L2SqrScalar(b, a, 2), actual, 1e-5f);
}

// =============================================================================
//  以下 8 个用例。
//
//  每个都先想清楚三件事：
//    1. 这个用例在测什么性质？（公式？边界？接口契约？）
//    2. 手算的期望值是多少？
//    3. 如果实现写错了，哪个用例会失败？—— 想不出这一条的用例价值不大
// =============================================================================

// 用例 1：L2 —— 两个向量完全相同
//   输入 a = b = (1, 2, 3)，期望 0.0f
//   测：边界。距离最小是 0，不能为负，也不能因为浮点误差变成 -1e-8
TEST(Distance, L2ExactlySame){
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float b[3] = {1.0f, 2.0f, 3.0f};
    const ann::dist_t actual = ann::L2SqrScalar(a, b, 3);

    EXPECT_NEAR(actual, 0.0f, 1e-5f);
    EXPECT_NEAR(ann::L2SqrScalar(b, a, 3), actual, 1e-5f);
}


// 用例 2：L2 —— dim = 1
//   输入 a = (5)，b = (2)，期望 9.0f
//   测：最小维度。循环的 off-by-one 最容易在这里暴露
TEST(Distance, L2MinestDim){
    const float a[1] = {5.0f};
    const float b[1] = {2.0f};
    const ann::dist_t actual = ann::L2SqrScalar(a, b, 1);

    EXPECT_NEAR(actual, 9.0f, 1e-5f);
    EXPECT_NEAR(ann::L2SqrScalar(b, a, 1), actual, 1e-5f);
}

// 用例 3：L2 —— 较大的 dim
//   自己选一组数，比如 a 全是 1、b 全是 2，128 维
//   期望值可以算出来（每维贡献 1，共 128）
//   测：累加型实现有没有漏项 / 多算项
TEST(Distance, L2LargeDim){
    const std::vector<float> a(127, 1.0f);
    const std::vector<float> b(127, 2.0f);
    const ann::dist_t actual = ann::L2SqrScalar(a.data(), b.data(), 127);

    EXPECT_NEAR(actual, 127.0f, 1e-5f);
    EXPECT_NEAR(ann::L2SqrScalar(b.data(), a.data(), 127), actual, 1e-5f);
}

// 用例 4：内积 —— 手算，专门验证「返回相反数」
//   输入 a = (1, 2, 3)，b = (4, 5, 6)
//   内积 = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
//   期望返回值 = -32.0f        ★ 关键在于这个负号
//
//   这个用例的唯一使命：如果有人把 return -inner_product; 改回 return inner_product;
//   它会失败（得到 32）。
TEST(Distance, InnerProductHandCompute){
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float b[3] = {4.0f, 5.0f, 6.0f};
    const ann::dist_t actual = ann::InnerProductScalar(a, b, 3);

    EXPECT_NEAR(actual, -32.0f, 1e-5f);
    EXPECT_NEAR(ann::InnerProductScalar(b, a, 3), actual, 1e-5f);
}

// 用例 5：SelectDistFn(Metric::L2) —— 派发对不对
//   拿到函数指针，通过指针调用，结果应当与直接调 L2SqrScalar 一致
//   形状：
//       const ann::DistFn fn = ann::SelectDistFn(ann::Metric::L2);
//       EXPECT_NEAR(fn(a, b, 2), ann::L2SqrScalar(a, b, 2), 1e-5f);
//   注意这里 fn 是【变量】，调用时和普通函数写法一样
TEST(Distance, SelectDistFnL2){
    const float a[2] = {3.0f, 4.0f};
    const float b[2] = {1.0f, 3.0f};
    const ann::dist_t actual = ann::SelectDistFn(ann::Metric::L2)(a, b, 2);

    EXPECT_NEAR(actual, 5.0f, 1e-5f);
    // 通过函数指针调用得到的结果，应当与【直接调用】一致。
    // （不能拿 SelectDistFn 和自己比 —— 那样无论实现怎么写都会通过。）
    EXPECT_NEAR(ann::L2SqrScalar(a, b, 2), actual, 1e-5f);
}

// 用例 6：SelectDistFn(Metric::InnerProduct) —— 同上，对比 InnerProductScalar
TEST(Distance, SelectDistFnInnerProduct){
    const float a[2] = {2.0f, 6.0f};
    const float b[2] = {3.0f, 1.0f};
    const ann::dist_t actual = ann::SelectDistFn(ann::Metric::InnerProduct)(a, b, 2);
    
    EXPECT_NEAR(actual, -12.0f, 1e-5f);
    EXPECT_NEAR(ann::SelectDistFn(ann::Metric::InnerProduct)(b, a, 2), actual, 1e-5f);
}

// 用例 7：SelectDistFn(Metric::Cosine) —— 当前是临时回退到内积
//   断言它的结果 == InnerProductScalar 的结果
//   并且在注释里写明：等真正实现余弦归一化后，这个用例要改
TEST(Distance, SelectDistFnCosine){
    const float a[3] = {2.0f, 3.0f, 5.0f};
    const float b[3] = {3.0f, 6.0f, 4.0f};
    const ann::dist_t actual = ann::SelectDistFn(ann::Metric::Cosine)(a, b, 3);

    //TODO:未实现归一化，实现后要改
    EXPECT_NEAR(actual, -44.0f, 1e-5f);
    EXPECT_NEAR(ann::SelectDistFn(ann::Metric::Cosine)(b, a, 3), actual, 1e-5f);
}

// 用例 8（★ 最重要）：验证「值越小越相似」这个排序约定
//
//   这个用例测的不是算术，而是【所有索引都将依赖的接口契约】。
//
//   数据：
//       q              = (1.0f,  0.0f)    // 查询
//       same_dir_short = (1.0f,  0.0f)    // 短，和 q 同方向
//       same_dir_long  = (10.0f, 0.1f)    // 长，几乎和 q 同方向
//
//   用 L2 函数：
//       dist(q, same_dir_short) < dist(q, same_dir_long)
//       L2 不管向量长度，几乎重合的那个更近
//
//   用内积函数：
//       dist(q, same_dir_long) < dist(q, same_dir_short)
//       内积奖励长度，所以长向量更"近"；又因为返回的是相反数，
//       "更近"对应返回值"更小"—— 方向就统一了
//
//   为什么它最有价值：
//       用例 4 只能抓到"负号忘写了"这一种错。
//       但如果有人用别的方式破坏了「越小越相似」的语义
//       （比如在别处又加了一次负号，或者让某个度量返回了原始内积），
//       只有这条能抓到。
//       它是给 FlatIndex 的保险 —— FlatIndex 假定"按距离升序取前 k 个"永远正确。
TEST(Distance, LessSimilar){
    const float q[2] = {1.0f, 0.0f};
    const float same_dir_short[2] = {1.0f, 0.0f};
    const float same_dir_long[2] = {10.0f, 0.1f};

    // EXPECT_LT(a, b) 断言的是 a < b —— 比较符号由宏名表达，
    // 所以两个操作数用【逗号】分隔，不要自己再写一个 < 进去。
    EXPECT_LT(ann::L2SqrScalar(q, same_dir_short, 2),
              ann::L2SqrScalar(q, same_dir_long, 2));

    EXPECT_LT(ann::InnerProductScalar(q, same_dir_long, 2),
              ann::InnerProductScalar(q, same_dir_short, 2));
}


}  // namespace
