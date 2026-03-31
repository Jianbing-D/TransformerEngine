# CuTeDSL Epilogue Fusion Patterns

## Epilogue Fusion Configuration (EFC) Framework

EFC provides a declarative way to define custom GEMM epilogues. The framework handles
TMEM->RMEM->SMEM->GMEM data movement; you only write the element-wise computation.

### EFC contract
```python
def my_epilogue(efc_config, C, bias, alpha, beta):
    # 1. Access accumulator
    acc = efc_config.accum()

    # 2. Load auxiliary tensors (ALL loads before ANY stores)
    c_val = C.load()
    bias_val = bias.remap_modes[0, :, :].load()  # broadcast along M

    # 3. Compute
    result = alpha * acc + beta * c_val + bias_val

    # 4. Store outputs
    C.store(result)

    # 5. Optional: apply activation
    return efc_config.relu(result)
```

### Key rules
- All `.load()` calls must precede `.store()` calls
- `efc_config.accum()` gives the MMA accumulator
- Broadcasting via `.remap_modes[dim_mapping]`:
  - `C.remap_modes[:, 0, 1].load()` — broadcast NxL along M
  - `X.remap_modes[0, :, 1].load()` — broadcast MxL along N
  - `scalar.remap_modes[:, :, :].load()` — 0-d tensor broadcast to all dims
  - `Y.remap_modes[1, 0, 2].store()` — transposed output

### Available activations
```python
efc_config.relu(x)
efc_config.leaky_relu(x)
efc_config.tanh(x)
efc_config.sigmoid(x)
efc_config.silu(x)
efc_config.hardswish(x)
efc_config.gelu(x)
# Identity (passthrough) also available
```

### Using EFC with DenseGemmEFC
```python
from blackwell.epilogue.common_dense_gemm_efc import DenseGemmEFC

kernel = DenseGemmEFC(
    epilogue_fn=my_epilogue,
    M=M, N=N, K=K, L=L,
    a_dtype=cutlass.Float16, b_dtype=cutlass.Float16,
    c_dtype=cutlass.Float16, d_dtype=cutlass.Float16,
    acc_dtype=cutlass.Float32,
)
kernel.compile()
kernel(a, b, c, bias, alpha, beta)
```

## Manual epilogue patterns (without EFC)

### Standard TMEM -> GMEM epilogue
```python
# 1. TMEM -> RMEM (tcgen05.Ld)
tmem_copy = tcgen05.make_tmem_copy(tcgen05.Ld32x32bOp(Repetition.x64), acc_tensor)
cute.copy(tmem_copy, tCtAcc, tTR_rAcc)

# 2. Type convert (e.g., fp32 -> fp16)
converted = tTR_rAcc.load().to(cutlass.Float16)
frag.store(converted)

# 3. Epilogue sub-tiling for ILP
epi_tiles = cute.zipped_divide(frag, epi_tiler)
for i in range(num_epi_subtiles):
    cute.autovec_copy(epi_tiles[..., i], gmem_out[..., i])
```

### Epilogue with TMA store (warp-specialized)
```python
# RMEM -> SMEM
tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom, tiled_mma)
cute.copy(tiled_copy_r2s, retiled_acc, smem_epi[..., stage])
cute.arch.fence_view_async_shared()

# SMEM -> GMEM via TMA store
store_pipe.producer_commit(state)
# TMA warp handles actual store
cpasync.copy(tma_store_atom, smem_epi[..., stage], gmem_out)
```

### Epilogue with fused activation (lambda)
```python
# Pass activation as Constexpr lambda to kernel
kernel = DenseGemmKernel(
    epilogue_op=lambda x: cute.where(x > 0, x, cute.full_like(x, 0))  # ReLU
)
```

## Dual output epilogue
```python
def dual_epilogue(efc_config, D, Aux, alpha, beta, C):
    acc = efc_config.accum()
    c_val = C.load()
    aux_val = alpha * acc + beta * c_val
    Aux.store(aux_val)          # pre-activation output
    D.store(efc_config.gelu(aux_val))  # post-activation output
```

## Read-modify-write epilogue
```python
def rmw_epilogue(efc_config, Y, x_factor):
    acc = efc_config.accum()
    # Transposed read-modify-write: read Y, add, write back
    y_val = Y.remap_modes[1, 0, 2].load()  # transposed read
    Y.remap_modes[1, 0, 2].store(y_val + acc * x_factor)
```
