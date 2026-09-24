# Softplus 多项式指令优化

2026-09-23，GB10，CuTe SM120 路径。结论：直接近似 `log(1+exp(-abs(s)))` 在数值上可行，但本轮 GPU 实现更慢。实际测到的小幅收益来自降低原有 `log1p` 多项式阶数；训练收益没有稳定到足以启用新的默认配置。

## 实现

`flash_attn_4/softplus.py` 增加以下编译期选项，必须在进程启动、import/编译 kernel 前设置：

- `FA4_SOFTPLUS_DIRECT=1`：七阶多项式直接拟合 `g(a)=log(1+exp(-a))`，核心区间 `[0,4]`。
- `FA4_SOFTPLUS_DIRECT=2`：分别拟合 `[0,4]`、`[4,8]`，扩大快速路径覆盖。
- `FA4_SOFTPLUS_PACKED=1`：使用成对 FP32 FMA 求值，可用于原 log 多项式和 DIRECT=1。仅在 GB10 上验证硬件支持。
- `FA4_SOFTPLUS_LOG_DEGREE=3/4/5`：原稳定公式中的 `log1p(y)≈yP(y)` 的 P 阶数，默认仍为 5。
- `FA4_SOFTPLUS_EXACT=1` 保持优先级，禁用多项式 Softplus 路径。

直接近似使用 warp 一致分支：每个线程检查当前 score fragment 的范围，warp 内全部通过才执行多项式；否则恢复原稳定指数公式。Masked `-inf` 不触发 fallback，输出和梯度显式归零。区间之外不裁剪分数、不截断有限负分数的梯度。

拟合同时约束端点值、端点导数，在密集网格上联合优化函数值和导数误差。核心区间 backward 用 `P'` 构造 sigmoid：负分数 `-P'`，正分数 `1+P'`；fallback 使用原 sigmoid。因此核心多项式与其导数匹配，边界值和导数在拟合精度内连续。范围判定依赖同 warp 的其他值，近似路径切换仍可能产生小的舍入/近似差异。

原稳定公式的低阶方案继续用原 sigmoid 计算梯度，保持原实现“近似函数值、计算目标 Softplus 的解析导数”的方式；只降低 log 多项式的阶数。

这些选项只作用于 CuTe score map，不改变 Triton decode、fixed-KV 等后端。

## 数值误差

以下是 FP64 密集网格检查结果，不是全实数区间的严格误差证明，也不等于最终 attention 误差：

| 近似 | 函数值最大相对误差 | 导数最大相对误差 |
|---|---:|---:|
| direct `[0,4]` 七阶 | 0.01535% | 0.04746% |
| direct `[4,8]` 七阶 | 0.01373% | 0.04190% |
| `log1p(y)/y` 三阶 | 0.06382% | sigmoid 保持原计算 |
| `log1p(y)/y` 四阶 | 0.00935% | sigmoid 保持原计算 |

GPU 检查使用 BF16/FP16 实际输入：单 KV 暴露逐点函数值/梯度，包含 `±4`、`±8` 邻域、`±32`、`±80`；另测 causal、窗口、非连续输入、放大随机输入以触发 fallback，以及 torch.compile/autograd。正常 attention 相对最大范数误差阈值为 2.5%；逐点检查另用逐元素相对/绝对误差，FP16 次正规数保留绝对容差，不能声称对任意微小梯度都具有固定相对精度。

拟合可复现：`python trains/fit_softplus_direct.py --output /tmp/coeffs.json`。纯 CPU 系数检查在 `tests/test_softplus_polynomial.py`。

## 性能

比较同样随机种子、匹配形状的独立进程。每项为三次 CUDA Graph 测量的中位数，每次约 30ms，包含 attention 准备/转换等；不含投影、优化器和完整模型。Train 为 forward+backward。倍率是原五阶版时间 / 候选时间，>1 更快。

| 方案 | 形状数 | Prefill 几何平均倍率 | Train 几何平均倍率 |
|---|---:|---:|---:|
| direct `[0,4]` | 4 | 0.765x | 0.790x |
| direct `[0,8]` 双区间 | 4 | 0.644x | 0.689x |
| direct `[0,4]` + 成对 FMA | 4 | 0.806x | 0.784x |
| 原稳定公式 + 成对 FMA | 15 | 0.984x | 1.000x |
| 原稳定公式 + 三阶 log 多项式 | 15 | 1.020x | 1.006x |
| 原稳定公式 + 四阶 log 多项式 | 15 | 1.007x | 1.000x |

三阶方案的 D64 prefill 子集平均 1.032x；整个 15 形状 prefill 范围 0.996–1.057x。Train 范围 0.959–1.034x，平均收益较小且存在回退。四阶和成对 FMA 同样未带来稳定的训练收益。跨进程、小百分比差异存在时钟/运行噪声，不把这些结果当作同进程严格交错的因果测量。

直接近似引入 fragment 范围归约、warp vote、分支、fallback 代码，以及训练时另一条导数多项式链。省掉特殊函数并不意味着总指令或关键路径更短。尚未用动态 profiler 分解各项开销，所以这里只把它们列为可能原因，不把回退归因于未经测量的 spill 或某条指令。

默认保留五阶原路径。对愿意接受稍大函数近似误差的 prefill，可显式测试：

```bash
FA4_SOFTPLUS_LOG_DEGREE=3 python your_program.py
```

不建议默认开启 `FA4_SOFTPLUS_DIRECT`。没有把这次实验描述成已经解决 prefill/train 全面领先 Softmax。

## 复现测量

```bash
# 0=原版，1=direct4，2=direct8，3=packed log，4=packed direct4，5=log3，6=log4
python trains/bench_softplus_direct.py --mode 0 --output /tmp/original.json
python trains/bench_softplus_direct.py --mode 5 --output /tmp/log3.json
python trains/bench_softplus_direct.py --mode 1 --quick --output /tmp/direct4.json
python trains/summarize_softplus_direct.py --baseline /tmp/original.json --output /tmp/summary.json /tmp/log3.json /tmp/direct4.json
```

所有候选测量开始前都会执行 GPU 数值和 autograd 检查；`--check-only` 可仅运行检查。原始数据为 `trains/softplus_direct_mode*_*.json`，汇总为 `softplus_direct_summary.json`；JSON 保留各次测量和对应 Softmax 控制。

最终验证：`python -m unittest discover -s tests -p 'test_softplus_*.py'` 的 **30 项测试全部通过**（186.3 秒）。原版及 6 个候选均通过独立进程 GPU 数值/梯度检查。`git diff --check` 和新增 Python 文件语法检查通过。测量覆盖原版与 3 个较轻候选的 15 形状，以及 3 个直接近似候选的 4 形状初筛；没有为明显回退的直接近似继续扩大全套性能测试。
