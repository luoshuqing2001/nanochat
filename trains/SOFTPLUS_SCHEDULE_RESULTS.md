# Softplus schedule 实施与评估

2026-09-22，NVIDIA GB10（SM120，48 SM）。对应 [设计](SOFTPLUS_SCHEDULE_DESIGN.md)。本轮没有改变 score 的数学定义、近似多项式或输入/输出精度。

## 已实现

- **early-dV backward**：先消费 `P` 做 `dV`，再生成 `dP`；保留 sigmoid 给 `dS`。编译 key 包含开关。仍沿用 KV-owned backward 和原有 dQ 归约；不是已经删除全部 Softmax 统计量缓冲的原生 backward。
- **CuTe fragment forward**：大 KV shared-memory tile 内分别计算 N32/N64 的 QK、Softplus、PV；加载粒度与计算粒度独立。支持保留 Q 寄存器、V 预取重叠、非边界 tile 的 mask 快路径。
- **persistent Stream-K forward**：每个 batch/head 内，把 query-major KV tile 工作流等分；完整 query tile 直接写 O，仅 worker 边界写私有 FP32 partial，再由 finish kernel 合并。没有输出清零、输出 atomic 或 block 间自旋。
- 以上显式选项接入 `softplus_attn_fa4` / `softplus_attn_fa4_func`，训练 custom op 的 fake、autograd、compile 路径同步更新。

Stream-K 的 CPU plan 有完整覆盖与唯一输出所有权检查。每 head 的 P 个 worker 至多拆 P−1 个 query tile、产生 2(P−1) 个 partial。它只限制合并量，不代表计算或访存成本也降为 O(P)。例如 B1/H6/T2048/D64、M64/N64、2 waves：每 head 16 workers，13 个 split tiles、26 个 slots，FP32 workspace 为 2,555,904 bytes；T8192 时为 2,949,120 bytes。plan 在 host 构建并缓存，计时不包括首次 plan 构建、上传和编译。

## 使用

```python
from flash_attn_4.softplus_api import softplus_attn_fa4_func

# 仅改变 backward 的生命周期
out = softplus_attn_fa4_func(q, k, v, early_dv=True)

# persistent forward + early-dV backward
out = softplus_attn_fa4_func(q, k, v, stream_waves=2, early_dv=True)

# CuTe 小 score fragment；加载 tile 保持原来的大小
out = softplus_attn_fa4_func(q, k, v, fragment_n=32, q_in_regs=True)
```

这些 forward 选项互斥于原有显式 split / balanced_chunk。Stream-K 限制为 CUDA BF16/FP16、相同 Q/K/V head 数和 D64/D128、feature 连续、0<Tq<=Tk。CuTe fragment 不支持自定义 mask、varlen、GQA 或 paged KV。early-dV 的运行验证限于 SM120。

## 对照与测量方法

- `softplus_schedule_bwd_v2.json`：17 个形状的 raw backward 前后配对，包含初始化、归约、cast。
- `softplus_schedule_fragment_pipeline_quick.json`：4 个代表形状，8 种 fragment/Q-register 组合。
- `softplus_schedule_stream_full.json`：17 个形状，7 种 worker/tile/stage 组合。此文件的 previous 是 **unsplit CuTe**，不是自动选择后的生产 baseline。
- `softplus_schedule_general_gb10.json`：58 个 prefill/decode/train case；同次测试生产自动调度、显式候选、默认 FA4、同样 tile/pipeline 优化后的 FA4、PyTorch SDPA。
- 训练时间是 attention forward + backward；不包括完整模型、投影、KV append、optimizer。
- Graph 时间为 3 次测量的中位数，每次 30 ms；正反顺序交替。Eager 单独记录，包含 Python/autograd/launch 开销。
- 不把所有形状各自的最快候选拼成一个不存在的自动调度器；不拿与 Softmax 的数值差异当正确性判据。

`softplus_schedule_bwd_quick.json` 已明确标为无效：初版实验开关没有赋给 kernel 实例，实际比较了两次旧 kernel。修复后重新跑了 v2，旧数据不参与结论。

## 初步消融结果

early-dV 在 D128 全局 backward 上测到小幅收益：B1/H6/T8192 为 4.791→4.655 ms，B8/H6/T2048 为 3.075→2.955 ms。D64 与窗口形状多数接近噪声范围，不能推广为统一收益。

Stream-K 在 B1/H6/T2048/D64 全局 prefill 的重复实验中约 71.9→66.7 µs。但长序列、多 batch、D128 和窗口多数退化。D128 改为 1 stage 可改善其自身表现，但仍普遍不及 CuTe；不能仅根据三角工作量更均匀就启用。

Fragment 的 Q-register/预取修订追回了部分损失，却没有跨形状稳定胜出。D64/N128-load/N64-fragment/Q-reg 在代表全局形状约 72.2 µs，原路径约 71.7 µs。小片段引入的 operand 重读、更多 MMA/转换指令和同步，可能抵消较小 score buffer 的收益；这是结合代码和计时的解释，不能冒充 Nsight 运行时计数器结论。

## 默认选择与全面结果

默认仅开启 early-dV 的已测收益范围：SM120、BF16、D128、Tq=Tk>=2048、全局 causal、CuTe backward。FP16、D64、窗口与短序列保留旧顺序。`early_dv=False` 可回退；Stream-K 与 fragment 均保持显式实验选项。

主矩阵 58 cases，加上新的 16-case holdout（与上一轮 general holdout 不同），全部完成。以下是 `最快 Softmax / Softplus` 的几何平均，>1 表示 Softplus 快。Softmax 取同次测量中的 SDPA / FA4 / tuned FA4 最快值。训练列按照上述固定默认规则，选取同次消融中对应的 early-dV 或旧路径；不是逐 case 挑最小时间。最终 API 的单独前后配对见 `softplus_schedule_policy_gb10.json`。

| 数据集 / 模式 | Prefill | Decode | Train (F+B) |
|---|---:|---:|---:|
| 主矩阵，CUDA Graph | 0.981× | 1.503× | 0.950× |
| 新 holdout，CUDA Graph | 0.927× | 1.398× | 0.917× |
| 主矩阵，eager | 0.820× | 0.608× | 0.697× |
| 新 holdout，eager | 0.678× | 0.590× | 0.536× |

主矩阵 prefill/decode/train 分别为 23/15/20 cases，holdout 为 6/4/6。Eager 含 host/autograd 开销且波动明显，不应由 Graph speedup 推断 eager 或整模型收益。decode 本轮未改 kernel，表中是回归测量，不是本轮新增加速。

主矩阵所有 train 的 early-dV 强制开启消融仅有 1.004× 几何平均收益；其 FP16 case 为 0.966×。固定默认规则对应的主矩阵 train 改善约 0.47%，holdout 约 0.29%，收益很小。代表性 BF16 D128 全局训练：B1/H6/T8192 为 6.291→6.088 ms，B8/H6/T2048 约 1.034×；新的 B1/H5/T6144、B2/H7/T2560 分别约 1.011×、1.006×。这些数据不能支持“全面胜过 Softmax”。

最终默认 API 的独立配对复测中，5 个触发新规则的形状均更快，收益为 0.6%–1.8%（前后向总时间）；其余走同一旧 kernel 的形状仍有计时波动，不能归因为代码加速或退化。

显式 Stream-K 在生产自动 baseline 上的主矩阵 prefill 几何平均仅 0.714×；fragment 为 0.868×。它们提供了可继续迭代的实现，但当前不适合替换生产默认。

## 编译资源证据

资源通过保留 CUBIN 后的 CUDA driver 属性查询获得。`local_bytes` 是每线程 local memory footprint，不是运行时 spill traffic；静态资源上限也不是实际运行 occupancy。CUDA 返回的 static SMEM 为 0，因为这些 CuTe kernel 用的是动态 shared memory。

| 全局 BF16 backward，M64/N64 | registers/thread 旧→early-dV | local bytes/thread 旧→early-dV |
|---|---:|---:|
| D64 | 239→254 | 0→0 |
| D128 | 255→255 | 120→72 |

D128 降低的是 local footprint，寄存器数仍卡在 255；D64 的寄存器反而增加。因此“少一个活跃 tile”不能等同于“每线程一定少 32 个寄存器”。按 SharedStorage 布局，Q2/dO1 backward 的动态 SMEM 为 D64 58,368 bytes、D128 99,328 bytes；CUDA occupancy API 以这些动态 SMEM 和实际 block size 计算得到两者均为 1 block/SM（见 policy 资源文件）；early-dV 不会改变这个静态上限。

D64 forward 从 N128 整 tile 改为 N128-load/N32-fragment，寄存器 168→144，但未获得计时收益；D128 对应大加载实验反而从 N64 基线的 168 上升到 225。小 score fragment 只减少部分临时状态，布局、operand 和预取状态也影响最终资源。

Stream-K 的 D128/M64/N64 从 2 stages 改为 1 stage，shared memory 由 57,344 降至 40,960 bytes；registers 从 168 增至 206，Triton 报告两者均无 spills。这与其 1-stage 更快的实测一致，但仍没有追上 CuTe。D64 的 2-stage 版本 shared memory 仅 24,576 bytes，不能把 D128 的 stage 选择照搬到 D64。

原始属性记录见 `softplus_schedule_bwd_resources.json.resources.json` 和 `softplus_schedule_fwd_resources.json.resources.json`。没有本轮 Nsight 实际利用率/带宽计数器，不能把资源推导当成硬件利用率测量。

## 尚未解决的瓶颈

1. Softplus 的可加性消除了 Softmax 在线归约依赖，却不会减少 QK/PV 的主矩阵乘法，也不保证非线性指令更少。
2. GB10 使用 mma.sync 路径；较小 fragment 没有自动获得异步 MMA/非线性重叠。若要继续，应针对编译后寄存器/共享内存和指令流水重写，而不是继续缩小 tile。
3. 现有 Stream-K 原型用 Triton，CuTe baseline 的 operand 复用和流水更成熟。跨 backend 对比不能把差距全部归因于调度；更有价值的下一步是把边界合并策略接到同一 CuTe 主循环，保留 tile 和流水再做单变量实验。
4. early-dV 仅做生命周期重排，伪 LSE/dPsum buffer 及相应加载仍存在；删除它们属于下一项独立实现，不能把本轮称为完整原生 backward。
5. 本轮未实现双 fragment accumulator、跨 warp producer-consumer 或 backward Stream-K。设计把它们列为前几项有资源证据后再做的后续方案。

## 复现

```bash
python -m unittest discover -s tests -p 'test_softplus_schedule.py'
python trains/bench_softplus_schedule.py --phase bwd --output /tmp/bwd.json
python trains/bench_softplus_schedule.py --phase fwd --quick --output /tmp/fragment.json
python trains/bench_softplus_schedule.py --phase stream --output /tmp/stream.json
python trains/bench_softplus_schedule.py --phase policy --resources --output /tmp/policy.json
python trains/bench_softplus_general.py --schedules --eager --output /tmp/general.json
python trains/bench_softplus_general.py --schedules --schedule-holdout --eager --output /tmp/holdout.json
```

最新脚本中 `softplus` 是最终自动规则；`softplus_previous` 明确禁用 early-dV，方便新的配对复测。已保存的 general/holdout 是启用默认规则前的消融运行，故其中 `softplus` 表示上一版默认，`softplus_early_dv` 表示候选；文件保留原始数据及源码 hash，没有改写标签。`softplus_schedule_summary.json` 的 selected 项由文中固定规则选取。

## 正确性与回归

22 个测试方法全部通过：schedule 7、general 5、optimization 5、fixed-KV 5。覆盖 BF16/FP16、alpha=0/.5/1、packed QKV、矩形和边界长度、窗口、三项梯度、强负 scores、CPU plan 覆盖/partial 所有权以及 `torch.compile(fullgraph=True)`。最终 equal-head 默认规则另做了自动路径 compile 复核。`git diff --check` 通过。

最终源码 hash 见 `softplus_schedule_final_manifest.json`。policy 测量后仅补充了默认规则的 equal-head 限制，已计时的相同 head 数路径不变。资源保留文件写入临时目录，仓库只保留 JSON 属性报告。
