# Vendored tokenizer

The tokenizer is a *derived* artifact that nanochat keeps outside the repo, in
`$NANOCHAT_BASE_DIR/tokenizer`. That is fine on one machine and a problem across
machines: a re-trained tokenizer is not the same tokenizer, so bits-per-byte numbers
stop comparing and the token budget shifts. It is 545 KB, so it lives in git instead.

    vocab size : 32,768
    trained on : ClimbMix (karpathy/climbmix-400b-shuffle), the same corpus this repo trains on
    trained    : 2026-09-18, with scripts/tok_train.py
    used by    : every run in logs/, including d12_hybridswa_muon_20260920_202806 (val bpb 0.8438)

    tokenizer.pkl   412,126 bytes  sha256 ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2
    token_bytes.pt  132,649 bytes  sha256 009ef93d20dd4684497c19c4b7fed29278b53f20022d2dc39264e27b9eefa2a8

## Install it on a new machine

    mkdir -p "${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/tokenizer"
    cp trains/tokenizer/tokenizer.pkl trains/tokenizer/token_bytes.pt \
       "${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/tokenizer/"

`INSTALL=1 bash trains/setup_env.sh` does this for you.

`tokenizer.pkl` is a pickled `tiktoken.Encoding` (protocol 4), so it loads on any
Python 3.4+ with tiktoken installed, on any architecture. `token_bytes.pt` is a torch
tensor of per-token byte lengths, used to convert loss into bits per byte.

## Re-training instead

`python -m scripts.tok_train` produces a *different* tokenizer unless the corpus and
parameters match exactly. Only do it if you are starting a new line of experiments
that will not be compared against the existing runs.
