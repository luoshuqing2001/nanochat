# Softplus 原生 schedule 第二轮实现

2026-09-23，GB10 / SM121（SM120 kernel），BF16 为性能测试主类型。

## 实现与默认策略

- Backward：early-dV 完成后复用 P 的共享内存存放 dS；新 barrier 保护 P 的最后读取。
- `bwd_schedule="inline"`：在寄存器中从 query 行号推导 `n^-alpha`，移除内部两份行统计缓冲的分配、传输及共享内存。继承接口的占位参数仍在。
- `bwd_schedule="scaled"`：把 `n^-alpha` 提到 dO，和原有 dQ 清零融合，消除逐 score 的两次缩放；dO 仍以输入 dtype 存储，因此舍入不同，FP16 小梯度可能额外下溢。
- `poly_estrin=True`：相同多项式系数使用 Estrin 求值，改变依赖链而不降低阶数。
- `warp_overlap=True`：不同 warp 交错执行下一片 QK 与当前片非线性/PV，保留为实验方案。
- `cute_stream_waves=2`：同一 CuTe MMA/prefetch 主循环上的持久化 Stream-K。完整 query tile 直接写 O，仅跨 worker 的边界写私有 FP32 partial，最后合并。
- `stream_tail=True`：仅拆末尾 query tiles。现有 `stream_waves` 仍指 Triton 版本。

默认只为 SM120、BF16、D64、等头数、等长 Q/K、T>=2048 的 CuTe backward 启用共享 P/dS。D128 保留上一轮 early-dV 策略。其余新选项显式开启，支持 autograd 和 torch.compile。Decode 的默认 dispatch 不变。

```python
from flash_attn_4.softplus_api import softplus_attn_fa4_func

out = softplus_attn_fa4_func(q, k, v)  # 保守默认
out = softplus_attn_fa4_func(q, k, v, bwd_schedule="inline",
                            cute_stream_waves=2)  # 显式实验
out = softplus_attn_fa4_func(q, k, v, bwd_schedule="previous")  # 上一轮 backward
```

## 候选消融

15 个形状，覆盖 B1/B8、H3/5/6/7、T1792/2048/2560/6144/8192、D64/128、global/window。每项取 3 次 CUDA graph 测量中位数，每次约 30ms；包含准备、清零、转换、partial 合并。比值为上一版 / 候选，越大越快。

| 候选 | 几何平均加速 | 最差 | 最好 |
|---|---:|---:|---:|
| backward 共享 P/dS | 1.037x | 0.984x | 1.141x |
| backward 预缩放 dO | 0.970x | 0.888x | 1.139x |
| backward inline scale + 共享 P/dS | 1.025x | 0.948x | 1.156x |
| forward Estrin | 0.988x | 0.966x | 1.014x |
| forward warp 交错 | 0.737x | 0.670x | 0.939x |
| forward CuTe Stream-K | 0.704x | 0.254x | 1.180x |
| forward 仅拆尾部 | 0.941x | 0.606x | 0.999x |

Stream-K 在 B1/H6/T2048/global 的 D64、D128 分别从 0.0722→0.0612ms、0.1220→0.1085ms；这是相对相同 tile 的 CuTe forward，并非所有生产 dispatch 的统一提升。大 batch/长序列存在严重回退，因此不默认开启。

原始数据：`softplus_native_bwd_full.json`、`softplus_native_fwd_full.json`。候选数据采集早于公共 API 接入；最终综合评估另存，避免混淆。

## 为什么数学简化仍未保证更快

Softplus 消除了 online max/sum 与重缩放，但 QK/PV 两次矩阵乘法和 O/dQ 累加仍在。分拆收益取决于空闲 SM 和尾部工作量；原本已有大量 tile 时，再引入持久化循环和边界合并未必合算。

预缩放 dO 把逐元素算术换成一次完整 dO 读写及精度损失；实测多数形状的净收益为负。Estrin 增加中间值，warp 交错增加 score fragment 存活范围及分支；它们缩短依赖链的潜在收益没有在此 GB10 实现上兑现。具体慢因需结合寄存器和 spill 资源数据，不能仅凭算法复杂度断言。

FP16 极小权重的 P/dS 量化误差在原路径中也存在。针对强负分数/小梯度，BF16 与 FP32 参考比较，FP16 额外检查共享/inline 相对原路径不增加误差；这不代表 FP16 对极小梯度达到同等 FP32 相对精度。

## 最终三阶段评估

最终默认路径，58 个主集案例 + 16 个额外 strided/边界案例。Softmax 对每个案例取 FA4、同 tile 调优 FA4、PyTorch SDPA 中最快者。表中是 `Softmax 时间 / Softplus 时间` 的几何平均；>1 表示 Softplus 更快。Train 仅 attention forward+backward，不含投影、完整模型和优化器。

| 测试集 / 执行模式 | Prefill | Decode | Train |
|---|---:|---:|---:|
| 主集 CUDA Graph | 0.987x | 1.499x | 0.972x |
| 额外集 CUDA Graph | 0.927x | 1.408x | 0.936x |
| 主集 eager | 0.792x | 0.625x | 0.662x |
| 额外集 eager | 0.521x | 0.582x | 0.489x |

主集分别 23/15/20 案例，胜出 3/12/7；额外集分别 6/4/6 案例，胜出 0/4/0。Decode 未改动，倍率与上一轮基本一致。Eager 包含 Python/dispatch 开销，只有 10 次调用的平均，不是 CUDA Graph 中位数；短任务噪声较大，不能据此声称端到端加速。

与同进程 `bwd_schedule="previous"` 配对：20 个主集训练案例平均 1.024x，新默认实际命中的 7 个 D64 案例平均 **1.064x**，范围 1.015–1.100x。额外集命中的 B2/H5/T3584/D64 strided 案例为 **1.071x**。额外集沿用上一轮的 schedule-holdout 清单；其中三个 D128 形状的连续布局已出现在本轮原生消融，故不把整个额外集声称为完全未见的 holdout。D64/3584 未参与本轮消融。

这轮完成了提出的候选实现、验证与筛选，但没有达到“prefill/train 全面超过 Softmax”。默认收益主要来自 D64 backward 的资源缩减；调度自由度本身不保证实际更快。

数据：`softplus_native_general_gb10.json`、`softplus_native_holdout_gb10.json`、`softplus_native_summary.json`。综合数据包含源码 SHA256、环境信息和各次测量值。

## 复现

```bash
python trains/bench_softplus_native.py --phase bwd --output /tmp/native_bwd.json
python trains/bench_softplus_native.py --phase fwd --output /tmp/native_fwd.json
python trains/bench_softplus_native.py --phase bwd --quick --resources-only --output /tmp/native_bwd_resources.json
python trains/bench_softplus_native.py --phase fwd --quick --resources-only --output /tmp/native_fwd_resources.json
python trains/bench_softplus_general.py --native-control --eager --output /tmp/native_general.json
python trains/bench_softplus_general.py --schedule-holdout --native-control --eager --output /tmp/native_holdout.json
python trains/summarize_softplus_native.py /tmp/native_general.json /tmp/native_holdout.json --output /tmp/native_summary.json
python -m unittest discover -s tests -p 'test_softplus_*.py'
```

仅在 GB10 上实测，不推断到 Hopper/其他 GPU；实验 forward 的 SM80 支持未经本轮硬件验证。
## Backward 资源证据

编译后 CUDA function attributes + CUDA occupancy API，B1/H6/T2048/global：

| D / 方案 | 共享内存 | 寄存器/线程 | local bytes/线程 | 静态最大驻留 blocks/SM |
|---|---:|---:|---:|---:|
| D64 previous | 57 KiB | 239 | 0 | 1 |
| D64 shared | 49 KiB | 255 | 24 | 2 |
| D64 inline | 48 KiB | 255 | 0 | 2 |
| D128 previous（early-dV） | 97 KiB | 255 | 72 | 1 |
| D128 shared | 89 KiB | 255 | 72 | 1 |
| D128 inline | 88 KiB | 255 | 144 | 1 |

D64 跨过了两 block 驻留的共享内存阈值，D128 没有；D128 inline 还增加 local memory。这解释了为何同样的缓冲缩减主要帮助 D64。这里是静态资源上限，不能等同于实测 active warps 或把 local bytes 直接当作动态 spill 流量。共享内存字节由被测固定 tile 布局计算；寄存器/local memory 来自编译后属性。

Forward 的全局 D128：previous/Estrin/warp-overlap/Stream-K 分别使用 168/212/251/214 寄存器/线程，local bytes 都为 0，静态驻留上限都为 2 blocks/SM。**这组证据不支持把 forward 回退归因于 spill 或静态 occupancy 下降。** 交错执行增加了 fragment 指令/分支，持久化增加 work-table、tile 边界同步及 partial epilogue；各项动态占比仍需进一步 profiling 才能分解。不能把 backward 的资源结论直接套到 forward。

资源原始记录：`softplus_native_bwd_resources.json`、`softplus_native_fwd_resources.json`，附对应源码 hash。

## 最终验证

5 组共 **28 项测试全部通过**：native 6、schedule 7、general 5、optimization 5、fixed-KV 5。覆盖前向/梯度参考、BF16/FP16、强负分数与小梯度、非整齐/非等长序列、窗口、非连续布局、分块覆盖、autograd/torch.compile、自动 dispatch、Softmax 控制和 KV-cache append。最终综合评估的源码 hash 与当前内核逐项匹配，`git diff --check` 通过。
