# 8-warp 后续优化：mask、计算顺序与 MMA 分工

本轮保持 Softplus 多项式次数与梯度公式不变，测试了内部 tile 跳过 mask、延迟 sigmoid、Estrin 求值、提前 FP32 行缩放，以及 8 种 MMA warp 分工。
结论：没有取得上一轮那样的明显加速。保留一个小幅改善长序列的显式候选，默认配置不变。

## 可用实现

```python
out = softplus_attn_fa4_func(q, k, v, bwd_schedule="warp8_dkv")
```

要求 SM120、Dq=Dk=Dv=128。该模式使用 256 threads、64x64 tile、Q/dO stages=1/1、early-dV。
score/dP 与 dQ 的 warp 布局仍为 4x2；dK/dV 从 4x2 改为 2x4。
同一个 KV owner 仍持有相同输出范围，没有新增全局归约或改变计算公式。
编译资源从原布局的 226 registers/thread 变为 224，两者 local memory 均为 0。

低层接口新增 `sm120_bwd_warp_layout=(MSdP, NdKV, MdQ)`，用于显式调优。
三个分量支持 2 或 4，对应三个 MMA 的行方向 warp 数；总 warp 数仍为 8。
该选项限定 256 threads、64x64 tile、D128。cache key 已包含三个 MMA 布局，梯度后处理沿用对应布局。

自动选择仍使用上一轮 `warp8` 的 `(4,4,4)`；新候选为 `(4,2,4)`。
`(4,4,2)` 也完成测试，但没有作为另一套自动策略加入。

## 算子前向 + 反向

GB10，BF16，full causal，D128；双方使用相同 head-local forward。
表中旧版指上一轮已经优化的 **8-warp Softplus**，不是更早的 4-warp 版本。
所有时间为整个 attention 前向+反向 GPU 时间，不是全模型训练时间。
五轮交替顺序取中位数；另用独立进程反转形状和候选顺序复测。

|B/T/H|旧布局 ms|新 dK/dV 布局 ms|耗时变化|反向顺序旧→新 ms|
|---|---:|---:|---:|---:|
|1 / 2K / 6|0.4689|0.4657|-0.68%|0.4779→0.4737，存在大幅轮间波动|
|8 / 2K / 12|6.7008|6.6394|-0.92%|异常，不用于收益判断|
|1 / 8K / 6|5.2987|5.3306|+0.60%|5.3645→5.3013|
|1 / 32K / 6|79.3478|78.2774|-1.35%|79.1121→78.0795|
|1 / 64K / 6|311.7300|310.9281|-0.26%|311.2054→305.6848|

32K 两轮改善均约 1.3%；64K 改善在 0.3%–1.8% 之间，8K 的方向不一致。
这支持保留为显式候选，但不足以宣称普适改善或切换默认策略。

同样采用新 dK/dV 布局的 Softmax 在正向顺序 32K/64K 为 74.9209/293.0793 ms，仍快于 Softplus。
本次对比没有宣称 Softplus 已反超同样优化后的 Softmax。

反向顺序 B8/T2K/H12 的两个 Softplus 候选同时升到约 14.5 ms，Softmax 未同比例升高；
末尾 B1/T2K 的原 Softplus 前两轮约 1.05 ms，后三轮恢复约 0.47 ms。
没有历史硬件计数器确认原因，故不把这些形状作为稳定收益证据。原始样本全部保留。

## 未采用的优化

`tune_softplus_mask.py` 在独立进程中替换方法，生产数学与默认 mask 保持原样。

|64K/H6/D128 backward 候选|Softplus ms|同次 Softmax ms|
|---|---:|---:|
|8-warp baseline|229.954|213.970|
|内部 tile 跳过 mask|229.901|215.789|
|延迟 sigmoid|230.824|216.407|
|跳过 mask + 延迟 sigmoid|233.520|214.140|
|Estrin 多项式求值|229.684|217.921|
|提前将行缩放乘到 sigmoid|228.711|217.207|

这些单独进程间的微小差异不足以证明收益，Softmax 控制也存在波动。
mask skip 在 32K 上为 59.309 ms，对照 baseline 57.952 ms；没有泛化改善。
提前行缩放还改变浮点乘法顺序，可能影响极小值的下溢行为；没有将它作为可部署优化接入。
上述候选仅做有限值检查与探索性计时，不等同于通过完整正确性验证。

MMA 布局搜索对 Softplus 和 Softmax 都测试了 `(4/2,4/2,4/2)` 的全部 8 种组合，并在计时前对旧布局检查梯度。
Backward 单项中 `(4,2,4)` 在 32K 为 59.058→56.812 ms、64K 为 229.940→225.449 ms；
完整前向+反向收益明显更小，因此最终结论以前向+反向复测为准。

## 正确性与复现

- `tests/test_softplus_warp_layout.py`：全部 8 种布局的独立 FP32 梯度 reference，BF16/FP16、GQA backward、局部 mask、非整块非方形序列、非连续输入，以及公开接口的 torch.compile 与 split backward。
- 同时回归 `tests/test_softplus_warp8.py`，覆盖之前的默认路径、极端分数与 Softmax 控制。
- 两轮完整算子测量中 forward 输出完全一致，梯度差异保存在每行 `errors`。
- `tune_softplus_mask.py` 与 `softplus_mask_*.json`：mask/数学调度探索。
- `tune_softplus_warp_layout.py` 与 `softplus_warp_layout.json`、`softplus_warp_layout_long.json`：8 种布局搜索。
- `bench_softplus_warp_layout.py` 与 `softplus_warp_layout_final.json`、`softplus_warp_layout_reverse.json`：前向+反向配对比较和反向顺序复测。
- 最终验证结果与文件 hash 记录在 `softplus_warp_layout_summary.json`。

本轮没有修改 prefill/decode kernel，也没有把新增显式选项自动应用到模型默认路径。
