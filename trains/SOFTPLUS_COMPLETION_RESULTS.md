# Softplus 全局调度与最后完成者归约

2026-09-23，GB10 / SM120 kernel。保持原 Softplus 五阶计算。

## 实现

```python
from flash_attn_4.softplus_api import softplus_attn_fa4_func

# 全局排序和按成本模型选择性拆分，私有 partial + 独立 finish。
out = softplus_attn_fa4_func(q, k, v, cute_stream_waves=2,
                            stream_tiles=True, stream_global=True)

# 同一调度，但由最后完成的 CTA 合并并写出。
out = softplus_attn_fa4_func(q, k, v, cute_stream_waves=2,
                            stream_tiles=True, stream_global=True,
                            stream_complete=True)

# 独立的 D64 backward 候选；不是默认策略。
out = softplus_attn_fa4_func(q, k, v, bwd_schedule="kv_group2")
```

推理接口 `softplus_attn_fa4` 也支持 `stream_global`、`stream_complete`。
`stream_global` 要求 `stream_tiles=True` 和正的 `cute_stream_waves`；
`stream_complete` 要求 CuTe stream 方案且不能与输出数据 `stream_atomic` 并用。

### 全局调度

旧 grid 的 x 轴先遍历一个 head 的所有任务；新 grid 拉平 batch/head/query，
优先遍历同等工作量的所有 batch/head，再进入下一段任务。每个 CTA 只执行一个
片段，没有 persistent worker 内部循环。它是静态全局任务排序，不是动态工作窃取。

CPU planner 用虚拟执行槽模拟 LPT makespan，并比较“完整 query tile”和“拆分一组
等长度长 tile”。只有预测 makespan 改善超过 1% 才接受，最多 4 个片段/块，
最小片段长度约 4 个 KV tile。尝试次数有界，计划按形状缓存。
KV 迭代为成本单位，CTA setup 和 split epilogue 各计 2 单位；这些是待校准的
启发式，不是硬件计数器测量。虚拟槽不代表真实 SM 绑定。
窗口 attention 保持完整 tile，只做排序，不拆 KV。

### 最后完成者协议

1. 每个 CTA 把未缩放的 FP32 accumulator 写入自己的 private slot。
2. 所有线程执行 GPU fence 和 CTA barrier，发布各自写入。
3. CTA leader 对该 query tile 的计数器执行 GPU-scope `acq_rel atomic_add`。
4. 最后完成者通过共享内存标志和 barrier 通知本 CTA 所有线程。
5. 该 CTA 读取全部 partial，做固定槽顺序求和、count scale、dtype 转换和输出写回。

不自旋、不等待尚未调度的 CTA；没有逐元素输出 atomic。完成计数器是每调用私有，
不能缓存为跨调用共享的可变状态。每次调用的清零包含在计时/graph replay 内；
无拆分时不启动计数器清零 kernel。独立 finish kernel 被省略，但有拆分时仍有一个
小计数器清零操作，不能把这称为无条件少一次 kernel launch。

### Backward KV 分组

`kv_group2` 使用 SM120 D64 的 `(M,N,Q_stages,dO_stages)=(64,128,1,1)`，
启用原有 early-dV + P/dS 共享。dK/dV 由一个 KV owner 累加，dQ 在 128 个 KV
位置内先通过 MMA 归约，再 atomic 写出。相对 N64，满块区域 dQ 更新次数约减半，
但 dK/dV 累加器和 score tile 更大，资源占用/并行度可能恶化。

初试 M32/N128 出现大幅 dQ 误差，已撤下该布局，不在可选配置里。最终版本仅允许
D64/M64/N128；D128 保留原 backward。模式要求不拆 query owner 范围，默认
`bwd_m_chunk="auto"` 在此模式解析为 0。Softmax 也能显式使用同样的合法 D64
backward tile，benchmark 中列为 `softmax_wide`，避免只给 Softplus 调 tile。

## 评测方法

- 算子：18 个形状 × prefill/train，共 36 案例。B1/B2/B8、H1/H3/H5/H6、
  D64/128、T769～8192、全局/窗口、连续/strided。
- 每例配对原默认、persistent private/complete、global private/complete，
  D64 额外配对 KV 分组以及组合方案。Softmax 使用调优 FA4，D64 train 增加 wide 控制。
- 每方案 3 次 CUDA Graph 计时、每次约 30ms、交替顺序，中位数。包含清零、转换、
  独立或融合归约。CPU 计划构建和 JIT 在预热完成，不属于稳定态 graph 时间。
- JSON 的 `plans` 记录实际拆分数量，未拆分的案例不应被计作 atomic/归约收益。
- 模型：真实 12 层 GPT，词表 32768，同参数和输入；前向包含投影、MLP、词表头，
  train 包含 loss+backward。不包含优化器、数据加载、分布式通信、KV-cache 写入。
  仅全局层采用候选，窗口层保持默认。
- 模型计时改为**任意时刻只保留一个 CUDA Graph**，统一 capture stream，loss detach
  后返回，避免保留 AccumulateGrad 节点。每个 trial 重新串行 capture，交替构建顺序，
  先 replay 5 次，随后用 CUDA event 测 10 次；共 4 个 trial。

## 复现

```bash
python trains/bench_softplus_completion.py --quick --output trains/softplus_completion_quick.json
python trains/bench_softplus_completion.py --output trains/softplus_completion_full.json
python trains/bench_softplus_completion_model.py --output trains/softplus_completion_model.json
python trains/bench_softplus_completion_model.py --reverse --output trains/softplus_completion_model_reverse.json
python -m unittest discover -s tests -p 'test_softplus_*.py'
```


## Results and follow-up

The full 36-case attention-only experiment is recorded in
`softplus_completion_full.json`; matched Softmax scheduling controls are in
`softplus_completion_controls.json`. Relative to original default, global-private
geomean speedups were 0.996x prefill / 0.9995x train; global-complete achieved
0.9895x / 1.0030x. D64 `kv_group2` training achieved only 0.6972x over its nine cases.
These candidates therefore remain opt-in. Last-completer correctness does not
establish a general performance benefit.

At B1/H6/T2048/D64 full causal, the matched-control experiment measured prefill
0.07209 ms default Softplus, 0.05513 ms global-private, 0.05384 ms global-complete,
0.06600 ms tuned Softmax and 0.04749 ms globally reordered Softmax. Much of the
scheduling benefit also applies to Softmax; comparison against only its original
ordering would overstate a Softplus-specific advantage.

A subsequent implementation adds a fixed **token** cap with a direct-write bypass
for short tiles. See `SOFTPLUS_KV_CAP_RESULTS.md` for the API and the 256/512/1024/2048
experiments. Unlike the heuristic planner above, the cap is a hard upper bound and
may split windowed tasks. `stream_global=True` with a cap selects head interleaving
without invoking the heuristic split planner.

The model benchmark script is available but has not been run for this revision;
these numbers are not whole-model throughput claims.
