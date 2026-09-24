# Softplus 专用 schedule：下一轮设计

实施更新：三项首版实现、正确性与跨形状实测见 [SOFTPLUS_SCHEDULE_RESULTS.md](SOFTPLUS_SCHEDULE_RESULTS.md)。原设计保留如下。

2026-09-22。本文是结合源码和论文的设计推导，**不是新 kernel 的性能实测**。上一轮 GB10 结果仍以 SOFTPLUS_GENERAL.md 为准。

## 判断

上一轮只探索了 Softplus 优化空间的一部分。当前 forward 继承 `FlashAttentionForwardSm80.compute_one_n_block`，依次做完整 QK tile、score map、PV；SM120 类继承 SM80 主循环。Backward 也主要通过两个 hook 替换 Softmax 数学表达式。因此，尚不能用此前结果判断“Softplus 专用主循环”的性能上限。

数学形式（省略 batch/head）：

- `S_ij = beta Q_i K_j^T`，`beta = D^(-1/2)`；`c_i = n_i^(-alpha)`。
- `O_i = c_i sum_j phi(S_ij) V_j`，`phi=softplus`。
- 对任意互不重叠的 KV 分组，`O_i = sum_g O_i^(g)`。分组可以不等长，数学结果不依赖完成顺序；浮点累加顺序会改变舍入结果。
- 每个 score 必须完成整个 D 维内积后才能应用 phi。独立性发生在不同 `(i,j)` 之间，不允许把 phi 分配到 D 维的部分内积上。

更有价值的自由度：score fragment 完成即可被 PV 消费，不必等待同一行的其他 fragment，也不会要求修改此前积累的 O。Softmax 也能细分 tile，只是需要更新在线统计量、重标定或合并 partial statistics；不存在“Softmax 不能并行”的绝对界限。

## 从 FlashAttention 学什么

| 工作 | 关键启发 | Softplus 设计取舍 |
|---|---|---|
| FA1 | IO-aware tiling，避免物化 N×N 矩阵 | 继续融合 QK、非线性和 PV |
| FA2 | 减少非 MMA 工作；warp 按 Q 分工，减少通信 | 不因支持 atomic 就把所有层次都切成 KV 并行 |
| FA3 | 跨迭代流水，交错 MMA 与非线性，控制额外 buffer | score fragment 尽早消费，增加独立指令，同时限制活跃数据 |
| FA4 | 算法与流水协同；关注非 MMA 资源；LPT 与 L2 locality | 删除 Softmax 专有依赖，调度仍要兼顾缓存、寄存器和 SMEM |
| Stream-K | 按总 inner-loop 工作量分配 workers | 只在 worker 区间边界拆输出，避免每行都碎片化 |

论文来源：[FA1](https://arxiv.org/abs/2205.14135)、[FA2](https://tridao.me/publications/flash2/flash2.pdf)、[FA3](https://arxiv.org/html/2407.08608v1)、[FA4](https://arxiv.org/html/2603.05451v1)、[Stream-K](https://arxiv.org/abs/2301.03598)。下面的方案是针对本仓库的推导，不能当作论文已经验证的 Softplus 优化。

GB10 的 SM120 不能直接套用 B200 的 TMEM/tcgen05/2-CTA MMA 流水。当前源码用 mma.sync；NVIDIA CUTLASS 也明确区分了 SM120/121 和数据中心 Blackwell 的 tcgen05 能力。[官方实现说明](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py)

## A：Forward 按 score fragment 流式计算

把三种粒度解耦：global→shared 的 KV 预取块、QK/PV 的计算 fragment、跨 worker 的工作区间。

起始候选：M=64，KV 预取 N_load=128，非线性/PV 的 fragment N_frag=32 或 64；保持相同输入精度与 score map。

```text
预取较大的 K/V tile 到 shared memory
对其中每个 KV fragment：
    S_frag = Q @ K_frag^T       # 完成整个 D 维内积
    P_frag = softplus(S_frag)
    O_acc += P_frag @ V_frag
    释放 S_frag / P_frag
最终 O = c * O_acc
```

第一版用单 accumulator，先测试降低 live state 的收益，再尝试两套 fragment ping-pong。多个 O accumulator 只在实测证明依赖链更贵时引入，因为会增加寄存器和末尾归约。

理想 FP32 score buffer 从 `4 M N_load` bytes 变为 `4 M N_frag`：M64/N128→N32 时为 32 KiB→8 KiB（这里只数一个 S buffer）。整个 kernel 的寄存器需求不会同比下降；O、Q/K operands 等仍占资源。

这不是简单把原 kernel 的 N 改小：预取仍用大块，布局/生命周期要专门重写，防止增加 global transactions。SM120 上首先依靠 cp.async、warp 交错和指令调度；不假设单 warp 内存在 B200 式异步 MMA。在有 TMEM 的硬件上，可以进一步用独立 MMA/elementwise producer-consumer 流水。

风险：更小 fragment 增加 Q/operand 的 shared/register 重读、循环/同步及 MMA 发射成本；减少 buffer 的收益可能被抵消。分别对比 N_load=N_frag、小 fragment/大预取，以及双 fragment 三种实现，不能仅凭 occupancy 预测收益。

## B：按总工作量划分的 persistent schedule

现有固定 KV chunk 均衡的是单个 task 的长度，但会把很多 output tile 拆成多个 task，引入大量合并。替代方案是将所有有效 `(query tile, KV tile)` 迭代按 query-major 排成工作流。

令 query tile m 需要 `w_m` 次真正执行的 KV MMA 迭代，`W=sum_m w_m`。选择 P 个逻辑 persistent workers，每个获得 `[floor(pW/P), floor((p+1)W/P))`。

每个 worker 处理它的连续区间：

- 完整覆盖的 query tile：Q/O 留在片上，最终直接写 O。
- 在区间首/尾被截断的 query tile：写独立 FP32 partial slot。
- 一个小的 finish kernel 只合并被截断的 query tile，并融合 c scaling/cast。

不采用“先普通 store 初始化 O，其他 block 同时 atomic_add O”，因为没有跨 block 顺序保证。第一版的 private slots 无需清零且无自旋/跨 block 等待；如果测 atomic 版本，必须预先清零相应 compact workspace。

关键上界：P 个连续区间只有 P−1 个内部切口；每个切口最多切开一个 query tile。因此 split output tile 数最多 P−1，split fragments 总数最多 2(P−1)（一个 tile 可跨多个切口）。这个结论只描述该顺序下的输出合并，不意味着额外输入读写也只有 O(P)，也不自动适用于 backward 的所有梯度。

P 由实际可驻留 blocks、SM 数和工作量选取，先测试 1/2/4 个逻辑 worker 每 SM；驻留不是由 grid 大小保证。按 batch/head 分组保留 L2 locality，避免为了全局平均而跨大量互不相关的 KV。窗口很短且任务已充足时保留整 query tile；Tq=1 时退化为现有 decode 的 split-KV/partial reduction，优势主要在 prefill 的三角形与尾部。

可进一步只对最后不足完整 wave 的工作使用这种切分，其余走普通整 tile 路径。先比较同一 kernel 内两类任务，避免为小问题引入两个主 kernel 的额外 launch。

Causal 对角 tile 即使只一半元素有效，MMA 通常仍算完整 tile；`w_m` 应按实际执行的 MMA 而不是有效 score 数估计。对 irregular/window 边界可加实测权重。

## C：Backward 原生主循环，优先缩短中间量生命周期

令 `G=dO`，一个 score tile 的计算为：

```text
S = beta Q K^T
P = c * softplus(S)
U = sigmoid(S)
E = G V^T
R = c * E * U
partial dV = P^T G
partial dQ = beta R K
partial dK = beta R^T Q
```

当前 inherited mainloop 先形成 P/U，再算 E，随后才消费 P 计算 dV，结构上可能同时保留 P、U、E 三个 FP32 tile。建议先做如下合法重排：

```text
S → P,U → dV += P^T G → 释放 P
E = G V^T
R = c * E * U            → 释放 U，原地复用 E
dK += beta R^T Q
partial dQ = beta R K    → 释放 R
```

这样在这一段从同时保留三份 MN 中间量，目标降为两份。64×64 的一个 FP32 buffer 为 16 KiB，按 128 threads 均摊是 32 个 32-bit registers/thread；实际节省多少必须看编译后的 register/spill 报告，而非仅看 Python 对象数。更早的 dV 也可能延后 E 或破坏预取，因此要与原顺序做受控对比。

进一步的原生版本直接由 query index 推导 c，取消两份伪 LSE/dPsum 的 global/shared buffer 及对应加载路径；原生 backward 接口也不需要保存 O/LSE。保留 dQ 初始化直到它的输出归属策略也改变，不能声称删除统计量就消除了所有 preprocess。

先保留 KV-owned 调度：dK/dV 累加后一次写，dQ 保持现有归约方式。不要一开始给 dQ、dK、dV 都加全局 atomic。之后再对 KV-owned 的 query 扫描应用方案 B，减少额外拆分 dK/dV 的次数。其 O(P) 边界上界只限制新增的 dK/dV 合并，dQ 的多源归约仍存在。

Softmax backward 在已知 LSE 和 row-dot 后也能按 tile 计算，不能将 tile 独立性完全算作 Softplus 独占。这里有个具体差别：Softmax 的 P 同时供 dV 和 dS 使用；Softplus 的 P 用于 dV、U 用于 dS。early-dV 可以消除当前 Softplus 适配引入的第三份活跃中间量，但并不自动胜过本来只需要 P/E 的 Softmax。Softplus 额外收益在于不需要那两个统计量，且 c 可由索引求得。公平比较应把可移植的重排也用于 Softmax。

## 优先级与验收

1. C 的 early-dV 生命周期重排：直接针对目前最弱的全局 train，先不改任务分配，便于定位收益。
2. A 的大预取/小 fragment forward：针对普通多头 prefill 的 score-map 与寄存器成本。
3. B 的 Stream-K 式工作流：针对低 head、causal 尾部、矩形 prefill；decode 保留专用路径。
4. 跨 warp producer-consumer 或更多 accumulator：前面三项的 register/SMEM/吞吐证据支持后再做。

每项先做单变量消融，再组合。记录 kernel 数、registers/thread、spill、SMEM/block、compiled occupancy 限制、workspace bytes 与实测时间。Nsight 计数器仍可能受权限限制，静态资源报告不应冒充运行时利用率。

数值：BF16/FP16、alpha 0/.5/1、强负 scores、mask 边界、packed QKV、forward/三项梯度。任何改变 score 近似、dtype 或舍入位置的版本单列，不能算成纯 schedule 收益。

性能：重跑主矩阵与新的独立 holdout；同次测默认/同样优化后的 Softmax；Graph 与 eager 分开，包含所有归约/清零/cast。先证明减少临界路径或中间数据搬运，再谈全面领先。目前没有证据承诺这些候选的 speedup。
