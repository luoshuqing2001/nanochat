# Softplus Attention：通用优化与跨形状评估

日期：2026-09-22。设备：NVIDIA GB10，48 SM，每 block 99 KiB shared memory；PyTorch 2.14.0+cu130。结论仅适用于实测设备、实现和形状。

## 最终结果

已完成通用策略优化，但实测没有达到全面胜过 Softmax。74 个 case 全部运行完成：prefill 29、decode 19、train 26。主矩阵旧版有一个正确性失败，旧版加速比汇总排除该项。

下表为 CUDA Graph speedup，**大于 1 表示新版 Softplus 更快**。旧版比较来自不同轮次，尤其 decode 需结合后面的同进程配对消融解读。

| 阶段 | 主矩阵项数 | 相对旧 Softplus | 相对最快 Softmax | Holdout 项数 | Holdout 相对最快 Softmax |
|---|---:|---:|---:|---:|---:|
| prefill | 23 | 1.38× | 0.98× | 6 | 0.98× |
| decode | 15 | 1.15× | 1.51× | 4 | 1.55× |
| train | 20 | 1.25× | 0.94× | 6 | 0.96× |

全部 74 项合并后，对最快 Softmax 的几何平均 speedup 为 prefill **0.98×**、decode **1.52×**、train **0.95×**。按 ±3% 持平区间，胜/平/负分别为 prefill **3/11/15**、decode **12/5/2**、train **4/5/17**。所以平均值不能替代逐 case 结果，也不能称为全面领先。

当前项目基线（推理 SDPA、训练默认 FA4）的合并 speedup 为 prefill 3.63×、decode 1.74×、train 1.14×。prefill 的较大比值主要受 SDPA 局部/矩形 mask 路径拖累；对同样优化的 FA4，优势大幅缩小，故不以这个数字宣传 Softplus 的数学优势。

**Eager 尚未全面改善。** 合并矩阵相对当前项目基线的 eager speedup：decode 0.72×、train 0.79×。这些是较粗略的 10 次平均，但明确说明 Graph 中的 GPU 加速不能直接当作未编译 Python 调用的加速。模型端要分别测 compile/CUDA Graph、KV append、projection 和整个 step 后才能下结论。

代表性结果（ms；BF16；完整 causal 除特别标注）：

| 场景 | Softplus | 最快 Softmax | 结论 |
|---|---:|---:|---|
| Prefill B1 H6 T2048 D128 | 0.1217 | 0.1134 | 0.93× |
| Prefill B1 H1 T4096 D128 | 0.0879 | 0.1124 | 1.28× |
| Prefill B1 H6 Q113 K2053 D128 packed | 0.0306 | 0.0454 | 1.48× |
| Decode B1 H6 KV4096 D128 | 0.0102 | 0.0226 | 2.21× |
| Decode B1 H6 KV65536 D128 | 0.8702 | 0.8773 | 1.01× |
| Train B1 H6 T2048 D128 | 0.5286 | 0.4794 | 0.91× |
| Train B1 H6 T8192 D128 | 6.2589 | 5.2799 | 0.84× |
| Train B8 H12 T2048 D128 W511 packed | 4.7356 | 5.4258 | 1.15× |

不是每项旧版比较都改善：例如 H1/T4096/D128 train 为约 0.346 → 0.364ms，约 5% 回退；新的稳定 sigmoid/通用调度没有完全追平旧特调。H1/T16384 的第一轮回退（约 4.48ms）则通过移除不划算的 backward 切分恢复至 4.02ms（旧版约 4.04ms）。矩形 Q512/K4096 prefill 从第一轮约 0.0621ms 改到 0.0510ms，但仍慢于 tuned Softmax 0.0468ms。

15 项回归测试通过（内部覆盖多组形状/dtype/窗口/alpha），包括 FP32 数值/梯度参考、packed QKV、decode cache append、边界长度、torch.compile fullgraph，以及 tuned Softmax 对照的梯度验证。撤回 expm1 候选后又重新运行 5 项 general 测试，全部通过。

## 同进程 decode 配对消融

同一进程、同一 QKV、相同 Graph 方法交替测优化前后（B1/H6，BF16，完整 KV），结果如下。大于 1 表示新版更快。

| KV 长度 | D64 新版相对旧版 | D128 新版相对旧版 |
|---|---:|---:|
| 128 | 1.32× | 1.39× |
| 4096 | 1.26× | 1.15× |
| 16384 | 1.01× | 1.01× |
| 65536 | 1.03× | 0.99× |

因此短/中 KV 的收益较明确；长 KV 基本持平。跨轮次主矩阵中的部分长 decode“回退”不能归因于本次代码变化：同进程比较时，两版长 KV 耗时接近。该补测没有覆盖所有 batch/layout，不能据此排除其他形状的回退。

以主矩阵 B1/H6/KV65536/D128 为例，K/V payload 合计约 201 MB，Softplus 约 0.870ms、SDPA 约 0.877ms；按 payload 算有效读取速率约 231 GB/s。这里没有测实际 DRAM transaction 或峰值带宽，所以仅支持两者读 K/V 成本接近的解释，不宣称已证明达到硬件带宽极限。

## 评估口径

Softplus 定义为 `O_i = n_i^(-alpha) sum_j softplus(q_i k_j / sqrt(D)) v_j`；性能矩阵使用 alpha=1。它与 Softmax 是不同的数学函数，不能互相作为数值正确性参考，也没有评估训练收敛或模型质量。

- 主矩阵 58 个 case：23 prefill、15 decode、20 train；另设 16 个未参与调优的 holdout：6 prefill、4 decode、6 train。
- B=1/2/4/8、H=1/2/3/4/6/8/12、D=64/128；主矩阵 BF16 与少量 FP16，连续及 packed QKV；完整 causal、局部窗口、矩形 prefill、非整齐长度、decode KV 至 65536。不是这些维度的笛卡尔积，不涵盖 GQA、varlen、FP8 或其他 GPU。
- train 是 attention forward + `autograd.grad`，不是完整模型/optimizer step。prefill/decode 不含 KV append、projection 和冷编译。
- 主结果是三个交替顺序 CUDA Graph 测量的中位数，每次约 30ms；包含 output/workspace 初始化、归约、cast 等所有 launch。附带 eager 10 次平均（包括 Python 调度，仅作辅助，波动比 Graph 大）。
- 对照：项目 FA4 Softmax 默认参数、项目 SDPA helper，以及使用相同 tile/pipeline 优化的 FA4 Softmax。最后一个对照防止把通用 kernel 调优收益误认为 Softplus 数学形式的收益；其 forward/gradient 已与 FP32 SDPA 对照验证。
- “当前项目基线”指训练 FA4、推理 SDPA；“最快 Softmax”是每个 case 中三个 Softmax 实测结果的最小值，不是穷尽所有可能实现的理论最优。
- SDPA 局部/矩形 prefill 按项目 helper 构造 mask，计入成本；FA4 使用隐式 mask。因此单独胜过该 SDPA 路径不能证明 score map 更快。
- 跨 case 汇总为不加权几何平均 speedup（对照时间 / Softplus 时间），不能换算为实际模型吞吐。±3% 作为实用持平区间，不是统计置信区间。

## 保留的通用优化

1. SM120 forward 使用 M64；D64 用 N128，D128 用 N64。减小 query tile 的 shared memory 压力，增加驻留/并行机会。其他架构仍走原有 tile 策略。
2. SM120 backward 使用 Q 双缓冲、dO 单缓冲：D128 不再等待每次 Q load，也避免两者都双缓冲超过 99 KiB shared memory。支持的显式调优参数也提供给 Softmax 对照。
3. backward 的 FP32 dQ 清零与两个 row-scale buffer 初始化合并为一个 kernel；alpha=0/1 特化，其他 alpha 保留。
4. 自动切分依据有效 tile 数、可见窗口、batch×heads 和 SM 数。短窗口、已有足够 block 的长 query、已有前缀且负载较均匀的矩形 query 避免额外 atomic workspace。低 head 并行度的固定 KV chunk 用连续公式选择，移除原先 H1/特定长度和 B1H6T2048 的查表。
5. decode 保留独立均匀 KV tasks。小 workspace、低 head 并行度使用 partial output + fused sum/cast，两次 launch；长 KV/高 head 并行度用 FP32 atomic，避免大型 partial workspace。单 chunk 直接写最终输出。
6. Triton score map 使用已有 CuTe 的稳定多项式近似，复用 exp 求 sigmoid；CuTe backward 同样共享 exp 求 sigmoid，避免负区间消减误差。不是降低输入 dtype；多项式仍是近似，不宣称 bitwise 等同精确 softplus。
7. 修复 packed QKV 的 Triton backward：输入 stride 与连续梯度 buffer 地址分开。旧主矩阵 `B2H4T1536D128 W255` 曾发生非法内存访问，before 文件保留该失败，不编造耗时；新实现通过梯度检查。

这里仍然会有 GPU thread block（CTA）。独立 KV 任务不需要跨 block 协作同步，但 CUDA/Triton kernel 仍以 block 执行；“无需跨 CTA 同步”和“没有 CTA”不同。

## 反思与被否决的尝试

**独立可加不等于所有规模都更快。** Softplus 去掉了 row max/sum 和在线重标定，但稳定 score map 仍需 exp，加上 log 或多项式；反向还要 sigmoid。Softmax 的 reduction 通常在片上完成，不等于一次额外显存遍历。QK/PV 及梯度矩阵乘法、读写 QKV 的主要成本也没有消失。Softmax 也可以拆 KV，只是要合并 partial max/sum/output；Softplus 的合并更简单，并不代表其他成本同时下降。

**负载均衡有成本。** 均匀切 KV 可以减少 causal 三角形的尾部不均衡，但过多 tasks 会重复读 Q、写 FP32 partial/atomic buffer，再清零、归约和转换。可见窗口短或本来已有很多 blocks 时，这些成本大于调度收益。保留显式 fixed 模式，自动模式按工作量选择。

**bs=1 并不单独决定并行度。** H×query tiles 也提供并行度；decode 的 Tq=1 才尤其依赖 KV splitting。很短任务中多一个 launch/Python dispatch 就可能抵消 GPU 算子收益。长 decode 两者都必须读 K/V，消除 Softmax reduction 后速度仍可能接近。

**训练不能只看 forward。** dQ/dK/dV 的 MMA、FP32 atomic workspace、重算 scores 和转换成本通常更大；通用 Q 双缓冲也能加速 Softmax。全局训练应与 SDPA 和同样调优的 FA4 比较，不能仅挑默认 FA4 较弱的参数。

实际尝试后未启用：

- 更大的 Triton backward tile：部分配置超过 shared memory 上限；更改 CuTe M/N 的部分候选出现断言或 dV 误差，均未采用。
- 所有 decode 都用 partial reduction：短 KV 有利，长 KV/多 head 增加 workspace 流量；改为按工作量选择。
- 所有低 head 长序列都做 backward splitting：新流水线下部分长序列已经有足够任务，额外 dK/dV atomic 反而回退；删除该例外。
- CuTe 稳定 expm1 形式（小值用三阶展开）：通过正确性测试，但最终矩阵发现 D64 训练显著回退（例如 T8192 完整 causal 约 3.06ms → 4.14ms），因此撤回，保留共享 exp + reciprocal 方案。附加指令/寄存器和调度压力是可能原因，未取得硬件计数器证实；不能仅凭减少某个表达式的活跃范围预测实际性能。
- 显式 CuTe FMA 多项式：单点 T8192D64 prefill 实测约 0.822ms，而先前约 0.789ms；没有保留。该单点仅用于否决候选，不作跨形状结论。

Nsight Compute 硬件计数器访问失败（ERR_NVGPUCTRPERM），没有修改驱动权限。对带宽/寄存器瓶颈的解释是基于数据量、kernel 结构和 timing 的推断，不是 profiler 测量结论。

## 复现与原始记录

```sh
python -m unittest discover -s tests -p test_softplus_general.py
python -m unittest discover -s tests -p test_softplus_optimization.py
python -m unittest discover -s tests -p test_softplus_fixed_kv.py
python trains/bench_softplus_general.py --eager --output trains/softplus_general_after_gb10.json
python trains/bench_softplus_general.py --eager --holdout --output trains/softplus_general_holdout_gb10.json
python trains/summarize_softplus_general.py
# 可选：用优化前 softplus_decode.py 做同进程配对消融
python trains/bench_softplus_decode_before_after.py --baseline /path/to/old/softplus_decode.py --output trains/softplus_general_decode_paired_gb10.json
```

- `softplus_general_before_gb10.json`：此次优化前，58 项（1 个旧实现失败），仅默认 FA4/SDPA 对照。
- `softplus_general_iteration1_gb10.json`：第一轮实现的完整矩阵，保留迭代证据。
- `softplus_general_expm1_rejected_gb10.json`：稳定 expm1 候选的中止矩阵，因训练回退未采用。
- `softplus_general_after_gb10.json`、`softplus_general_holdout_gb10.json`：最终实现；含源码 SHA256、每轮样本、eager 耗时。
- `softplus_general_decode_paired_gb10.json`：同进程旧版/新版 decode 对照，内嵌旧版源码及 SHA256，便于重建 baseline 文件。
- `softplus_general_summary.csv`：逐 case 比值；`softplus_general_summary.json`：分阶段汇总。
- `softplus_general_{fwd,bwd,cute_bwd,decode}_tuning.json`：候选调优记录，包括无效配置；不全是生产路径。

