import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute import typing as cutlass_typing
from functools import partial
from typing import Tuple


@dsl_user_op
def fma(
    a: cutlass.Float32,
    b: cutlass.Float32,
    c: cutlass.Float32,
    *, loc=None, ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass_typing.Float32.mlir_type,
            [
                cutlass_typing.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass_typing.Float32(b).ir_value(loc=loc, ip=ip),
                cutlass_typing.Float32(c).ir_value(loc=loc, ip=ip),
            ],
            f"""{{\n\t
            fma.rn.ftz.f32 $0, $1, $2, $3;\n\t
            \n\t}}""",
            "=f, f, f, f",
            loc=loc,
            ip=ip,
        )
    )
