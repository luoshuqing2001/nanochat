# Softplus / Softmax 对照诊断

本次只新增诊断脚本和结果，没有修改生产 attention 数学或默认 dispatch。
设备为 GB10，执行仓库的 SM120 CuTe / mma.sync 路径；结论不能直接外推到其他 GPU 的 FA4 实现。

## 对照方法

B=1、H=6、BF16、causal。Forward 使用相同 head-local LPT，D64 tile=64x128，D128 tile=64x64。
Backward 使用相同 unsplit KV-owner 调度，tile=64x64、Q/dO stages=2/1。
测量使用单个存活计算图、预热、交替顺序、三轮中位数；各轮样本保存在 JSON。
sp_early 提前计算 dV；sp_shared 进一步复用 P/dS 共享内存。
这些是匹配调度的算子比较，并非每个形状分别取所有 Softmax 配置的最优值，也不是全模型训练时间。

|形状|阶段|Softplus ms|Softmax ms|说明|
|---|---|---:|---:|---|
|32K, D64|prefill|11.077|9.959|baseline|
|32K, D64|backward|28.577|32.281|sp_shared|
|64K, D128|prefill|79.429|77.923|baseline_repeat|
|64K, D128|backward|271.719|236.855|sp_early, baseline_repeat|

## 主要发现

### Forward：省去行归约，但逐元素映射更贵

当前 Softplus 为 max(s,0) + y*P5(y)，y=exp2(-abs(s)*log2(e))。
仍需逐元素 exponential，加上多项式依赖链。Softmax 的 online row max/sum/rescale 确实被省掉，但它们的节省不保证覆盖多项式开销。

D64 forward 无 LSE 对照二进制的静态指令计数：

|指令|Softplus|Softmax|
|---|---:|---:|
|HMMA|512|512|
|FFMA|1536|262|
|EX2|256|262|
|SHFL|0|20|

这是展开后二进制的静态计数，不是动态执行次数或周期比例。它说明 tensor-core 工作相同，而新增标量计算与减少的归约存在权衡。

### Backward：Softplus 值与导数不同，增加计算和中间状态

Softmax 用 P 同时计算 dV 和 dS=P*(dP-D)。Softplus 用 softplus(s) 计算 dV，但 dS 需要 sigmoid(s)，另乘当前定义的行长度缩放。
当前实现共用 exponential，但 sigmoid 仍需 reciprocal/选择，并在中间 fragment 中保存。

|D128 backward|寄存器/线程|local bytes/线程|静态 FFMA|静态 EX2|静态 RCP|
|---|---:|---:|---:|---:|---:|
|Softplus early-dV|255|72|192|52|32|
|Softmax|255|32|32|32|0|

SASS 确实存在 LDL/STL 指令，寄存器压力与 local-memory 访问不是仅由源码推测。
这些数据不能单独量化 spill 的耗时，或证明 SFU/带宽已饱和。

64K/D128 profiler 分解（每次调用平均 ms，独立于稳定计时，不能逐项拼接上表）：

|阶段|Softplus|Softmax|
|---|---:|---:|
|预处理|1.033|1.865|
|backward 主 kernel|272.267|234.152|
|后处理|1.375|1.377|

差距集中在主 kernel；预处理节省不足以抵消。Softplus 仍需预处理来清零 dQ、准备行长度信息。

### 共享内存：必须跨过驻留门槛

CUDA driver occupancy 查询，128 threads/CTA，实际 cubin 与配置对应的动态共享内存：

|配置|动态共享内存/CTA|资源限定 CTA/SM 上限|
|---|---:|---:|
|D64, P/dS 分开|57 KiB|1|
|D64, P/dS 复用|49 KiB|2|
|D128, P/dS 分开|97 KiB|1|
|D128, P/dS 复用|89 KiB|1|

这是资源限定上限，不是运行时 achieved occupancy。
D64 backward 从 early-dV 的 33.864 ms 改善到共享版 28.577 ms，超过 Softmax 32.281 ms。
D128/64K 共享版 275.414 ms 对 early-dV 271.719 ms，没有收益；节省 8 KiB 未增加驻留数量，仍受高寄存器压力影响。

## 数学消融：仅用于诊断

以下两个消融都不是可部署的正确 Softplus attention：no_log 去掉 log 修正，no_sigmoid 把非 mask 位置导数设为 1。

|64K/D128|Softplus backward ms|同次 Softmax ms|
|---|---:|---:|
|完整 baseline_repeat|271.719|236.855|
|去掉 log 修正 no_log_repeat|256.528|237.834|
|去掉 sigmoid no_sigmoid|256.860|236.700|

每个消融相对完整实现改善约 5.5–5.6%，但仍比各自 Softmax 对照慢约 8%。
消融同时改变指令、活跃变量和编译器调度，不能把改善相加，也不能把残差全部归因于 spill。
Forward 去掉 log 修正后为 76.977 ms，对照 Softmax 78.338 ms；完整版本为 79.429/77.923 ms。

降低到 P3 在 32K/D128 上使 forward 接近持平（19.483/19.445 ms），但 backward 仍为 67.832/60.305 ms；其寄存器仍为 255、local bytes 仍为 72。降低指令数未消除资源瓶颈。

## 异常数据与边界

初次 no_log 部分配置出现双方同时约 2.3 倍减速；log3 的 64K backward 也有很大的轮间漂移。
这些结果不用于跨进程绝对耗时归因。Softmax 对照的对应 cubin hash 一致，但没有历史硬件计数器证明减速原因。
随后重跑 64K no_log 和完整 baseline，控制组分别为 237.834 和 236.855 ms，支持上述诊断。
本次未重新测 decode，也没有声称完整训练端到端收益；重点是长序列 prefill/backward 差距。

## 后续优先级

1. D128 backward 的活跃区间：调整 P、dV、dP、sigmoid、dS 的生成/消费顺序，以减少 local-memory 访问；延后 sigmoid 也有保存 score 或重算的成本，必须实测。
2. 调整 tile/pipeline，使资源用量真正跨过寄存器或共享内存门槛；不能仅以节省字节数评价收益。
3. 继续研究同时计算 Softplus 与 sigmoid 的低成本近似，并分别验证前向、梯度误差；单纯降低 log 多项式次数不足以解决 D128 backward。
4. 长序列已有充分 query/head 并行时，进一步 KV splitting 的收益需要抵消输出归约、读写和重复加载成本，不能从可加性推出必然提速。

## 复现材料

- `diagnose_softplus_bottleneck.py`：匹配调度计时、进程内数学消融、资源/SASS、可选 profiler。
- `diagnose_softplus_occupancy.py`：对 retained cubin 查询驻留上限；需先生成 baseline 的 `/tmp/softplus_bottleneck_cubins`。
- `softplus_bottleneck_baseline.json`、`softplus_bottleneck_baseline_repeat.json`。
- `softplus_bottleneck_no_log_repeat.json`、`softplus_bottleneck_no_sigmoid.json`、`softplus_bottleneck_log3.json`。
- `softplus_bottleneck_occupancy.json`，以及各测量文件的 `.metadata.json`。
