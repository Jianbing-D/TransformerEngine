# Forward and Backward Algorithms

## Forward Algorithm (Numerically Stable Online Softmax)

### Phase 1: FwdMainLoop cuteDSL kernel
For each split `s` of vocab:
1. **GEMM**: `logits[m, s*V:(s+1)*V] = hidden[m, :] @ weight[s*V:(s+1)*V, :].T`
2. **Online Softmax Epilogue** (in TMEM→register pipeline):
   - Load logit tile from TMEM
   - Update running max: `_max_old = _max; _max = fmax(_max, logit)`
   - Update accumulate: `_accu = exp(_max_old - _max) * _accu + exp(logit - _max)`
   - If `label` falls in this vocab range: accumulate `logit_at_label`
3. **Write** `_max[token, split]`, `_accu[token, split]`, `_logprobs[token]` to GMEM

### Phase 2: Triton Epilogue (`forward_dp_epilogue` for DP, `forward_tp_epilogue` + `forward_tp_epilogue_update_logprobs` for TP)
- Reduce `_max[token, :]` and `_accu[token, :]` across all splits using streaming max-accumulate
- Convert `accumulate` to LSE: `LSE = log(global_accu) + global_max`
- Compute `logprobs[token] = logit_at_label - LSE`
- Apply `ignore_index` mask: set logprob to 0 for ignored tokens
- Apply reduction (sum/mean using `num_valid_tokens`)

### TP Mode Epilogue
In TP mode, each rank only holds a shard of the vocab. The reductions happen across ranks:
1. `all_reduce(_max, MAX)` — global max across vocab shards
2. On a dedicated CUDA stream: `all_reduce(_logprobs, SUM)` — sum logits at label positions (only one rank contributes a non-zero value per token)
3. After all_reduce: `forward_tp_epilogue` converts per-split accu using globally-reduced max
4. `all_reduce(accumulate, SUM)` — sum the exp-accumulators
5. `forward_tp_epilogue_update_logprobs` finishes the LSE computation

---

## Backward Algorithm

### Current Implementation: `kDlogitsSplitN`
For each vocab split `s` (sequential loop):
1. **`BwdPartialDlogits` kernel** computes `d_logits_partial[num_tokens, vocab_per_split]`:
   ```
   softmax[m, v] = exp(logits[m, s*V+v] - LSE[m])
   d_logits[m, v] = dlogprobs[m] * (softmax[m, v] - one_hot(label[m] == s*V+v))
   ```
   - Uses same GEMM structure as forward (hidden × weight.T)
   - `accu` (which is LSE after forward epilogue) is loaded to compute softmax
   - `dlogprobs` is scaled by `1/num_valid_tokens` if reduction == mean

2. **`cublas.addmm`** accumulates `d_hidden`:
   ```
   d_hidden += d_logits_partial @ weight[s*V:(s+1)*V, :]   (FP32 accumulation)
   ```
   - `beta=0` for first split, `beta=1` for subsequent splits

3. **`torch.matmul`** computes `d_weight`:
   ```
   d_weight[s*V:(s+1)*V, :] = d_logits_partial.T @ hidden
   ```

### Future Implementation: `kFused` (`BwdDHiddenDWeight`)
- Status: **WIP, not yet called in the entry point**
- Fuses steps 1+2+3 into a single kernel pass
- Uses persistent scheduler (`StaticPersistentScheduler`) for load-balancing across SMs
- More complex: 16-warp CTA layout with softmax warps, epilog warps, load/mma/store warps
- Uses TMEM for: logits accumulator, d_hidden accumulator, d_weight accumulator, p (probability) buffer
- Uses TMA reduce (ADD) for writing d_hidden and d_weight back to GMEM

---

## Mathematical Foundation

**Forward pass** — computing the NLL loss from hidden states $\mathbf{h} \in \mathbb{R}^{T \times D}$ and weight matrix $\mathbf{W} \in \mathbb{R}^{V \times D}$, with integer labels $y \in \mathbb{Z}^T$:

$$z_{t,v} = \sum_{d} h_{t,d} \cdot W_{v,d} \qquad \text{(logits via GEMM, shape } T \times V \text{)}$$

$$m_t = \max_v z_{t,v} \qquad \text{(running max for numerical stability)}$$

$$A_t = \sum_v \exp(z_{t,v} - m_t) \qquad \text{(shifted partition function)}$$

$$\mathrm{LSE}_t = \log A_t + m_t \qquad \text{(log-sum-exp)}$$

$$\mathrm{NLL}_t = \mathrm{LSE}_t - z_{t,\, y_t} \qquad \text{(cross-entropy loss)}$$

**Backward pass** — via chain rule:

$$\frac{\partial \mathrm{NLL}_t}{\partial z_{t,v}} = \mathrm{softmax}(z_t)_v - \mathbf{1}_{[v = y_t]} = \exp(z_{t,v} - \mathrm{LSE}_t) - \mathbf{1}_{[v = y_t]}$$

$$\frac{\partial L}{\partial z_{t,v}} = \frac{\partial L}{\partial \mathrm{NLL}_t} \cdot \Bigl(\exp(z_{t,v} - \mathrm{LSE}_t) - \mathbf{1}_{[v = y_t]}\Bigr)$$

$$\frac{\partial L}{\partial h_t} = \sum_v \frac{\partial L}{\partial z_{t,v}} \cdot W_v \qquad (d_{\text{hidden}},\ \text{shape}\ T \times D)$$

$$\frac{\partial L}{\partial W_v} = \sum_t \frac{\partial L}{\partial z_{t,v}} \cdot h_t \qquad (d_{\text{weight}},\ \text{shape}\ V \times D)$$

**Key property**: $\partial L / \partial z_{t,v}$ depends on $z_{t,v}$ (the logits), which were not saved. They must be recomputed from $\mathbf{h}$ and $\mathbf{W}$. This is the source of the 3× backward FLOPs vs forward.
