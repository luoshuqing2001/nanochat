# Softplus 混合归约实验

GB10，2026-09-23。保留默认五阶 Softplus 多项式，不改变数学定义。

## 实现

`softplus_attn_fa4` 和 `softplus_attn_fa4_func` 新增显式选项：

```python
# 完整 query tile 在寄存器/块内完成累加并直接写 O。
# 只有跨 CTA 拆分的 tile 使用一个共享 FP32 atomic 累加槽。
out = softplus_attn_fa4_func(q, k, v, cute_stream_waves=2,
                            stream_atomic=True)

# 去除持久化 worker 循环，每个 CTA 只处理一个工作片段。
# 按 KV 工作量预算只拆长 tile，短 tile 仍然直接写 O。
out = softplus_attn_fa4_func(q, k, v, cute_stream_waves=4,
                            stream_tiles=True, stream_atomic=True)
```

`stream_atomic=False` 保留私有 FP32 partial 槽作配对控制；`stream_tiles`
不兼容 `stream_tail`。两种归约使用相同的 QK/Softplus/PV 主循环和调度表。
atomic 每个片段只在最终 epilogue 写一次，不在每个 KV tile 都写回。

atomic workspace 只覆盖被拆分的 query tiles，清零它们后启动 attention，
再用 finish kernel 缩放、转换并写回这些 tiles。完整 tiles 不清零、不做 atomic。
因此相较私有槽，它减少 partial 容量和最终读取，但仍额外需要清零；
尚未实现最后一个 CTA 直接完成输出的跨 CTA 发布协议。
所有零填充和 finish kernel 均计入测量。使用 FP32 atomic，存在浮点累加顺序非确定性。

`stream_tiles` 的预算为每个 batch/head 的有效 KV tile 迭代总量除以目标 workers，
向上取整。一个 query tile 超过预算时拆成尽量等长的片段，按片段长度递减排列。
不拼接短 query tiles，编译期消除持久化循环及其每段末尾 barrier。

## 评测口径

- `bench_softplus_hybrid.py`：20 个形状 × prefill/train；B1/B2/B8、H1/H3/H5/H6、
  T769/1792/2048/3584/4096/6144/8192、D64/128、全局/窗口和连续/strided。
- 每例配对默认路径、persistent private/atomic、tiles 2/4/8 waves private/atomic，
  以及同 tile 调优 Softmax FA4。固定候选逐个报告，不把逐形状择优冒充通用加速。
- CUDA Graph 每个方案交替顺序测量 3 × 30ms，取中位数；eager 是 10 次调用均值，
  Python/dispatch 噪声较大，单独保留原始值。
- `bench_softplus_hybrid_model.py`：实际 nanochat GPT 12 层，H6，V32768，同权重、
  同输入，包含投影、RoPE、归一化、MLP、词表头；train 包含 loss 和完整 backward。
  不含优化器、数据加载、分布式通信，也不包含 prefill 写 KV-cache。
  全局层使用被测调度，窗口层保持默认。CUDA event 计时已捕获 graph 的 replay，每组 10 次、共 5 组，交替顺序取中位数。
  Softmax 作为不同数学算子的性能控制，不要求与 Softplus 输出相等。

## Profiling 限制

已尝试 Nsight Compute，驱动返回 `ERR_NVGPUCTRPERM`，原始记录在
`softplus_hybrid_ncu_default.csv`。本轮没有硬件计数器数据，不能把时间差确定归因于
某类 stall、spill 或 memory throughput。无需更改驱动权限即可复现配对计时。

## 复现

```bash
python trains/bench_softplus_hybrid.py --quick --output trains/softplus_hybrid_quick.json
python trains/bench_softplus_hybrid.py --tiles --output trains/softplus_hybrid_tiles_full.json
python trains/bench_softplus_hybrid_model.py --output trains/softplus_hybrid_model.json
python trains/bench_softplus_hybrid_model.py --reverse --output trains/softplus_hybrid_model_reverse.json
python -m unittest discover -s tests -p 'test_softplus_*.py'
```

## 算子结果

20 个形状，以下是 `原默认耗时 / 候选耗时` 的几何平均，>1 表示更快。

| 固定方案 | 全部 prefill | 全部 train | 仅全局 prefill（10例） | 仅全局 train（10例） |
|---|---:|---:|---:|---:|
| persistent 2 waves，private | 0.771x | 0.920x | 0.860x | 0.952x |
| persistent 2 waves，atomic | 0.739x | 0.903x | 0.844x | 0.943x |
| 单片段 2 waves，private | 0.949x | 0.991x | 1.034x | 1.025x |
| 单片段 2 waves，atomic | 0.926x | 0.986x | 1.026x | 1.023x |
| 单片段 4 waves，private | 0.883x | 0.972x | — | — |
| 单片段 4 waves，atomic | 0.838x | 0.946x | — | — |
| 单片段 8 waves，private | 0.796x | 0.935x | — | — |
| 单片段 8 waves，atomic | 0.754x | 0.915x | — | — |

新 atomic 没有稳定胜过相同调度的 private partial。大 batch 不适合强行压缩到少量
persistent workers；短窗口的拆分工作过细时，归约开销明显超过收益。

**实际是否拆分很重要**：单片段 2 waves 在 10 个全局形状中的 8 个没有拆分，
收益主要来自长任务优先顺序，不能归功于 atomic。真正拆分的 B1/H1/T4096/D64
相比原默认，private 的 prefill/train 为 1.175x/1.195x，atomic 为 1.100x/1.175x；
同组 D128 的 prefill 则回退。原始 JSON 的 `plans` 给出 CTA 数、拆分 query tile 数和
FP32 workspace 大小。这里的原默认包含现有自动 dispatch，不是强制无拆分 CuTe。

同一批数据中，原默认 Softplus 对调优 Softmax FA4 为 prefill 0.991x、train 0.998x。
不能用几个获益案例声称 Softplus 已全面超过 FA4。

原始记录：`softplus_hybrid_quick.json`（第一轮 persistent/tail）、
`softplus_hybrid_tiles_quick.json`、`softplus_hybrid_tiles_full.json`、
`softplus_hybrid_summary.json`。快测和全测分开保存，不合并成独立样本。

## 模型结果与反转顺序复测

实际 GPT 12 层，四种配置：B1/H6，T2048/D128/L、T2048/D128/SSSL、
T2048/D64/L、T4096/D128/L。窗口层保持原默认；每个候选使用相同权重与 token。
第二个进程反转 graph 构建顺序，两轮都交替 replay 顺序。
下表仍是原默认 / 候选的几何平均。

| 方案 | 首轮前向 | 首轮 F+B | 反转构建顺序前向 | 反转构建顺序 F+B |
|---|---:|---:|---:|---:|
| persistent private | 1.009x | 1.013x | 0.993x | 0.992x |
| persistent atomic | 1.014x | 1.017x | 0.994x | 0.992x |
| 单片段 private | 1.022x | 1.022x | 0.993x | 0.988x |
| 单片段 atomic | 1.021x | 1.023x | 0.985x | 0.982x |

这些模型形状的单片段 2 waves **没有拆分 query tile**，atomic 标志不会产生实际
atomic 输出更新。它们的结果不能用来证明 atomic 归约带来模型加速。

首轮约 2% 的收益在反向构建顺序下没有保留，不把两轮简单合并后的小正数称为稳定收益。
Softmax 控制的相对表现也随顺序显著变化，说明完整模型的结果存在顺序/分配相关混杂；
具体是缓存、地址布局、频率还是其他因素，本轮没有硬件计数器证据来进一步区分。
本轮**没有证明稳定端到端加速，也没有证明全面超过调优 FA4**。

初次模型计时尝试用 `do_bench_cudagraph(graph.replay)` 嵌套捕获，被当前 runtime 拒绝。
最终两份模型 JSON 均使用 CUDA event 直接计时 replay，失败尝试没有混入统计。
PyTorch 对部分 backward capture 发出了 AccumulateGrad stream mismatch 警告；
最终 capture/replay 成功，单独的算子/梯度/torch.compile 回归负责验证数值正确性。

原始数据：`softplus_hybrid_model.json`、`softplus_hybrid_model_reverse.json`。

## 默认策略

**默认 dispatch 和 Softplus 五阶计算保持不变。** 两个新选项保留为可复现的实验接口，
已有 backward 实现不变。只有单独的前向调度发生变化，train 评估的是它对 F+B 的净影响。
没有把窗口退化、atomic 开销或 graph 顺序敏感的小收益隐藏进默认策略。

这次实验支持继续研究“完整 tile 直接写回、只归约边界/长 tile”的方向，但不支持
“atomic 一定比 private reduction 更快”。下一步若继续做 atomic，值得验证将
少量 split tile 的最终输出合并到最后完成的 CTA 中，省掉独立 finish kernel；
那需要新的全局内存发布/完成协议，不能直接省略同步。本轮未实现该协议。

## 最终验证

完整 Softplus 回归 **33 项全部通过**（214.441 秒），涵盖 BF16/FP16、前向和梯度、
窗口/非整齐/非等长序列、strided 输入、autograd、torch.compile、CUDA Graph 重放、
CPU 调度覆盖及 atomic 槽映射。`git diff --check` 通过。

源码与数据 SHA256、最终测试记录见 `softplus_hybrid_final_manifest.json`。
算子计时后仅修改内核文件中的说明性注释/docstring；最终测试使用最终源码。
实验只在 GB10 实测，未验证其他 GPU 的性能。
