# Softplus D128 backward：8-warp 调度

在 GB10 上实现并启用一个保持现有 Softplus/梯度计算公式不变的 backward 配置。
改变的是线程分工和流水线；没有降低多项式精度，也没有采用数学消融。

## 最终配置

- 同一个 64x64 Q/KV tile，从 128 threads / 4 warps 改为 256 threads / 8 warps。
- 同一个 KV owner 仍持有完整 dK/dV 累加器，只是分摊给更多线程。没有增加 KV 切分、dK/dV 全局归约或 dQ 原子操作次数。
- 保留 early-dV；Q/dO pipeline stages 从 2/1 改为 1/1。
- score/dP、dK/dV、dQ 的 MMA warp 布局从 4x1 变为 4x2。
- 为跨列 warp 使用坐标 mask：原 R2P 快速 mask 假设线程负责的列连续，不能直接套到 4x2 布局。
- dQ 和 GQA dK/dV 后处理必须匹配主 kernel 的线程数与 accumulator 布局。

调优编译结果：原 D128 early-dV 为 255 registers/thread、72 local bytes/thread；8-warp 单阶段为 226 registers/thread、0 local bytes/thread。
线程块的总寄存器需求并未降低，但每线程的寄存器需求降到上限以内，且同一个驻留 CTA 有更多活跃 warp。
这支持寄存器/执行调度解释；没有使用硬件 stall counters 来精确分摊收益。

## 自动选择与显式选择

`softplus_attn_fa4_func` 的自动路径仅在以下条件选择新配置：
SM120、BF16、Dq=Dv=128、Q/K/V head 数相同、方形全 causal 序列、T>=2048、CuTe backward、early-dV 已启用且 backward query chunk 为 0。

保留已有小网格 split 调度和 D64 路径。明确设置 `early_dv=False` 可保留原自动路径；`bwd_schedule="previous"` 可用于完整旧配置对照。
显式 `bwd_schedule="warp8"` 选择新 backward；`head_lpt=True` 独立控制 forward。

```python
out = softplus_attn_fa4_func(q, k, v)  # 符合条件时自动使用
out = softplus_attn_fa4_func(q, k, v, bwd_schedule="warp8")
```

低层 `_flash_attn_bwd(..., sm120_bwd_num_threads=256, sm120_bwd_tile=(64,64,1,1))`
也支持 Softmax 对照，但没有修改 Softmax 的默认 dispatch。

## Backward 独立调优结果

B=1、H=6、D128、BF16、causal，单位 ms。每组均保持 KV-owner 调度一致。

|T|Softplus 原配置|Softplus 8-warp|Softmax 原配置|Softmax 同样优化|
|---|---:|---:|---:|---:|
|32K|69.206|58.367|59.716|55.848|
|64K|271.948|228.425|237.534|219.008|

Softplus backward 耗时降低约 16%；相对同样优化后的 Softmax 仍慢约 4%–5%。

## 公开接口前向 + 反向

使用相同 head-local LPT forward。Softplus 通过公开 autograd 接口调用；Softmax 对照包含 forward、LSE、backward 与预/后处理。
这是 attention 算子前向+反向，不是整个模型训练 step。GPU 计时使用单个存活 CUDA graph、20ms warmup、60ms 目标窗口、三轮交替顺序中位数。

|B/T/H/D|Softplus 之前|Softplus 之后|耗时降低|Softmax 128线程|Softmax 256线程|
|---|---:|---:|---:|---:|---:|
|1 / 2K / 1 / 128|0.141|0.141|约 0%|0.170|0.161|
|1 / 2K / 6 / 128|0.513|0.462|10.0%|0.488|0.464|
|8 / 2K / 12 / 128|7.428|6.738|9.3%|7.164|6.906|
|1 / 8K / 6 / 128|5.954|5.341|10.3%|5.450|5.154|
|1 / 32K / 6 / 128|89.079|77.081|13.5%|80.551|75.195|
|1 / 64K / 6 / 128|354.040|309.973|12.4%|314.212|293.611|
|1 / 8K / 6 / 64|2.756|2.753|约 0%|2.838|2.561|

反转形状与候选顺序后，D128 32K 为 90.061→77.839 ms，64K 为 353.292→310.879 ms，收益仍约 12%–14%。
同样优化后的 Softmax 对照分别为 74.503 和 292.939 ms。
2K/H6 的优势幅度有波动（反向顺序约 7%）；2K/H1 与 D64 默认路径未改变，小幅差异属于测量波动。

Prefill 实现未改变：正向顺序的 D128 32K 为 Softplus 19.653 / Softmax 19.501 ms，64K 为 78.236 / 76.424 ms。
本轮没有改变或重新评估 decode。

## 未采用的候选与发现

1. 延迟 sigmoid：保存带符号 exponential，在 dS 阶段计算 reciprocal/选择。D128 local bytes 从 72 增到 80，8K 从约 4.64 变成 4.68 ms，未采用。
2. 窄 KV tile：N32 的 D128 寄存器降至 234、local bytes 为 0，但 8K 从约 4.66 增至 6.03 ms；N16 更慢。额外 Q/dO 读取和 dQ 原子累加使其没有泛化收益。低层显式调优参数保留，自动策略不使用。
3. 窄 tile 必须同步缩小 dK/dV MMA 的 M 方向 warp 数；否则线程会覆盖 tile 外的行，造成 dV 错误。已修正并增加独立梯度检查。
4. 256 threads 不能只改 launch 大小：必须修正跨列 warp mask 和梯度后处理布局。错误候选均未参与有效性能比较。

## 验证与范围

`tests/test_softplus_warp8.py` 覆盖：独立 FP32 reference、BF16/FP16、非整块与非方形序列、局部 mask、零可见 KV 行、不同 alpha、GQA backward、极端分数、torch.compile/default autograd、Softmax 同配置控制、窄 tile 梯度。

GQA backward 用独立 reference 输出作为占位输入：原有 packed-GQA forward 在该测试形状下已有 reference 误差，本轮没有修改它，也没有把 GQA 纳入自动启用范围。
性能测试中，新旧 Softplus 输出完全一致；梯度最大绝对差除以 reference 最大绝对值，在反向顺序测试中最高约 8.83e-5。

现有全套 Softplus 回归的最终结果记录在 `softplus_warp8_summary.json`。

## 复现与原始数据

- `tune_softplus_lifetime.py`：延迟 sigmoid、窄 tile、128/256 threads、stage/shared-PdS 调优。
- `softplus_lifetime_threads_fixed.json`、`softplus_lifetime_threads_long.json`：有效线程数调优与编译资源。
- `softplus_lifetime_narrow_fixed.json`：修正 MMA 布局后的窄 tile。
- `bench_softplus_warp8.py`、`softplus_warp8_final.json`、`softplus_warp8_reverse.json`，及 metadata 中的代码 hash。
- 初始 `*_tiles.json`、`*_threads.json` 含被正确性检查拒绝的配置，不能用于性能结论。

下一步若继续优化，应在当前 8-warp 配置上重新测 scalar math 与多列 warp mask 成本；不能把旧 4-warp 的瓶颈结论原样套用。
