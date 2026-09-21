# Run summary: d12_hybridswa_muon_20260920_202806

## Model

- layers (n_layer): 12
- model dim (n_embd): 768, heads: 6 (kv: 6)
- sequence_len: 2048, window_pattern: `SSSL` (hybrid SWA)
- vocab_size: 32768

## Parameters

- wte: 25,165,824
- value_embeds: 150,994,944
- lm_head: 25,165,824
- transformer_matrices: 84,935,088
- scalars: 50
- total: 286,261,730

## Setup


## Training horizon

- grad accum steps: 16
- iterations: 2,520
- total training tokens: 1,321,205,760
- tokens : scaling params: 12
- estimated training FLOPs: 1,003,715,000,000,000,000

## Results

- final train loss (EMA): 2.78544
- final val bpb: 0.843776
- best val bpb: 0.843776
- min val bpb (trainer): 0.843776

## Throughput

- GPU: NVIDIA GB10
- compute dtype: torch.bfloat16
- median step time (ms): 10,404.6
- median tokens/sec: 50389
- peak memory (MiB): 15,910.6
- total training time (min): 437.38

Raw per-step metrics: `metrics.jsonl`. Full stdout: `train.log`.
