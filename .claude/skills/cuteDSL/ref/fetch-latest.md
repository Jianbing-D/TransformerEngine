# Fetching Latest CuTeDSL APIs and Examples

The CuTeDSL API evolves across CUTLASS releases (v4.0 -> v4.1 -> v4.2 -> v4.3 -> v4.4+).
The static reference files in this skill may become outdated. Use this procedure to pull
fresh information when needed.

## When to fetch

- User mentions a CuTeDSL API or pattern you don't recognize
- User says "this API changed" or "this doesn't work anymore"
- User is targeting a new GPU architecture not covered in ref/arch-features.md
- User asks about experimental features (`cute.experimental`)
- You're unsure if a pattern is still current

## How to fetch from public GitHub

### 1. Browse the latest examples directory
```
WebFetch: https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL
```
This gives the directory listing. Look for new subdirectories or files.

### 2. Read a specific example (use raw URL)
```
WebFetch: https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/<arch>/<filename>.py
```
Examples:
- `ampere/elementwise_add.py`
- `hopper/dense_gemm.py`
- `blackwell/dense_gemm.py`
- `blackwell/rmsnorm.py`

### 3. Browse the source code
```
WebFetch: https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL/cutlass/cute
```

### 4. Read a specific source file
```
WebFetch: https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/cute/<module>.py
```
Key files:
- `__init__.py` — all exported symbols
- `algorithm.py` — gemm(), copy() signatures
- `atom.py` — MMA/Copy atom APIs
- `tensor.py` — Tensor/TensorSSA APIs
- `core.py` — Layout APIs
- `math.py` — math operations

### 5. Check recent commits for API changes
```
WebFetch: https://api.github.com/repos/NVIDIA/cutlass/commits?path=python/CuTeDSL&per_page=10
```

### 6. Check the changelog / release notes
```
WebFetch: https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/CHANGES.md
```
Or:
```
WebFetch: https://github.com/NVIDIA/cutlass/releases
```

## How to fetch from internal GitLab (if accessible)

The internal repo at `gitlab-master.nvidia.com/dlarch-fastkernels/dynamic-kernel-generator`
contains additional examples at `cutlass_ir/compiler/python/examples/`.

### Prerequisites
- Requires NVIDIA SSO authentication
- Set `GITLAB_TOKEN` environment variable, or
- Clone the repo locally and read files directly

### If repo is cloned locally
Ask the user for the local clone path. The examples live at `<clone_path>/cutlass_ir/compiler/python/examples/`.
```
Glob: <clone_path>/cutlass_ir/compiler/python/examples/**/*.py
```

Key internal-only directories to check for new patterns:
```
<clone_path>/cutlass_ir/compiler/python/examples/rubin/           # SM107 Rubin examples
<clone_path>/cutlass_ir/compiler/python/examples/internal/feynman/ # SM140 Feynman (pre-release)
<clone_path>/cutlass_ir/compiler/python/examples/internal/blackwell/ # Internal BW patterns
<clone_path>/cutlass_ir/compiler/python/examples/experimental/     # Experimental APIs
<clone_path>/cutlass_ir/compiler/python/examples/block_api/        # Block API (higher-level)
<clone_path>/cutlass_ir/compiler/python/examples/blackwell/epilogue/ # EFC framework
<clone_path>/cutlass_ir/compiler/python/examples/blackwell/tutorial_gemm/ # Step-by-step tutorials
<clone_path>/cutlass_ir/compiler/python/examples/blackwell/tutorial_tma/  # TMA tutorials
```

Check git history for recent changes:
```
Bash: cd <clone_path> && git log --oneline -20 -- cutlass_ir/compiler/python/examples/
```

### If token is available
```
WebFetch: https://gitlab-master.nvidia.com/api/v4/projects/dlarch-fastkernels%2Fdynamic-kernel-generator/repository/tree?path=cutlass_ir/compiler/python/examples&ref=master
Headers: PRIVATE-TOKEN: <token>
```

## After fetching

If you discover API changes or new patterns:
1. Update the relevant ref/*.md file in this skill
2. If it's a new architecture or major feature, create a new ref file
3. Inform the user of what changed
