"""
Pytorch Wrapper for cuBLAS operation.
"""
import nvmath
from nvmath.bindings import cublas

import torch
from typing import Optional
import ctypes

def get_cublas_dtype(
    tensor: torch.Tensor,
) -> nvmath.CudaDataType:
    dtype = tensor.dtype
    if dtype == torch.float32:
        return nvmath.CudaDataType.CUDA_R_32F
    elif dtype == torch.float16:
        return nvmath.CudaDataType.CUDA_R_16F
    elif dtype == torch.bfloat16:
        return nvmath.CudaDataType.CUDA_R_16BF
    elif dtype == torch.float64:
        return nvmath.CudaDataType.CUDA_R_64F
    elif dtype == torch.int8:
        return nvmath.CudaDataType.CUDA_R_8I
    else:
        raise ValueError(f"Unsupported tensor data type: {dtype}")

def get_compute_type(
    a_type: nvmath.CudaDataType,
    c_type: nvmath.CudaDataType,
) -> cublas.ComputeType:
    if ((a_type == nvmath.CudaDataType.CUDA_R_16F or a_type == nvmath.CudaDataType.CUDA_R_16BF) and c_type == nvmath.CudaDataType.CUDA_R_32F):
        return cublas.ComputeType.COMPUTE_32F
    if (a_type == nvmath.CudaDataType.CUDA_R_16F or a_type == nvmath.CudaDataType.CUDA_R_16BF):
        return cublas.ComputeType.COMPUTE_32F
    if (a_type == nvmath.CudaDataType.CUDA_R_32F):
        return cublas.ComputeType.COMPUTE_32F
    if (a_type == nvmath.CudaDataType.CUDA_R_64F):
        return cublas.ComputeType.COMPUTE_64F
    return cublas.ComputeType.COMPUTE_32F

def cublas_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C_opt: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    assert A.is_cuda, "A must be a CUDA tensor"
    assert B.is_cuda, "B must be a CUDA tensor"
    assert A.dim() == 2, "A must be 2D"
    assert B.dim() == 2, "B must be 2D"

    if not isinstance(alpha, float):
        alpha = float(alpha)
    if not isinstance(beta, float):
        beta = float(beta)

    M = A.size(1) if trans_a else A.size(0)
    K_a = A.size(0) if trans_a else A.size(1)
    K_b = B.size(1) if trans_b else B.size(0)
    N = B.size(0) if trans_b else B.size(1)
    assert K_a == K_b, "Incompatible matrix dimensions for GEMM"
    K = K_a

    out_dtype = out_dtype if out_dtype is not None else A.dtype

    C = None
    if C_opt is not None:
        C = C_opt
        assert C.is_cuda, "C must be a CUDA tensor"
        assert C.size(0) == M and C.size(1) == N, "C has incompatible dimensions"
        # If beta is 0, we can ignore C's values
        if beta != 0.0 and C.dtype != out_dtype:
            C = C.to(out_dtype)
    else:
        C = torch.empty((M, N), device=A.device, dtype=out_dtype)

    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    a_type = get_cublas_dtype(A)
    b_type = get_cublas_dtype(B)
    c_type = get_cublas_dtype(C)

    handle = torch.cuda.current_blas_handle()

    ori_mode = cublas.get_pointer_mode(handle)
    cublas.set_pointer_mode(handle, cublas.PointerMode.HOST)
    
    compute_type = get_compute_type(a_type, c_type)

    lda = A.stride(0)
    ldb = B.stride(0)
    ldc = C.stride(0)


    op_a = cublas.Operation.T if trans_a else cublas.Operation.N
    op_b = cublas.Operation.T if trans_b else cublas.Operation.N

    alpha_ptr, beta_ptr = None, None
    if compute_type == cublas.ComputeType.COMPUTE_64F:
        _alpha = ctypes.c_double(alpha)
        _beta = ctypes.c_double(beta)
        alpha_ptr = ctypes.addressof(_alpha)
        beta_ptr = ctypes.addressof(_beta)
    else:
        _alpha = ctypes.c_float(alpha)
        _beta = ctypes.c_float(beta)
        alpha_ptr = ctypes.addressof(_alpha)
        beta_ptr = ctypes.addressof(_beta)

    cublas.gemm_ex(
        handle,
        op_b,
        op_a,
        N, 
        M,
        K,
        alpha_ptr,
        B.data_ptr(),
        b_type,
        K if trans_b else N,
        A.data_ptr(),
        a_type,
        M if trans_a else K,
        beta_ptr,
        C.data_ptr(),
        c_type,
        N,
        compute_type,
        cublas.GemmAlgo.DEFAULT_TENSOR_OP
    )
    cublas.set_pointer_mode(handle, ori_mode)
    return C


def addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Performs: out = beta * input + alpha * (mat1 @ mat2)
    
    Equivalent to torch.addmm but using cuBLAS directly.
    
    Args:
        input: Bias tensor [M, N]
        mat1: First matrix [M, K]
        mat2: Second matrix [K, N]
        alpha: Scalar for matrix product
        beta: Scalar for input
        out_dtype: Output data type (optional)
    
    Returns:
        Output tensor [M, N]
    
    Example:
        >>> bias = torch.zeros(1024, 2048, device='cuda')
        >>> A = torch.randn(1024, 512, device='cuda')
        >>> B = torch.randn(512, 2048, device='cuda')
        >>> C = addmm(bias, A, B, alpha=1.0, beta=0.0)
    """
    return cublas_gemm(mat1, mat2, out, alpha, beta, False, False, out_dtype)