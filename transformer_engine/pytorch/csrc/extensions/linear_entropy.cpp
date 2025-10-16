#include "../extensions.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

void fused_linear_cross_entropy_fwd_mainloop(
    at::Tensor hidden, 
    at::Tensor weight, 
    at::Tensor labels,
    int32_t ignore_index) {
    NVTE_CHECK(hidden.dim() == 2, "hidden must be a 2D tensor");
    NVTE_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    NVTE_CHECK(labels.dim() == 1, "labels must be a 1D tensor");

    NVTE_SCOPED_GIL_RELEASE({
        nvte_linear_cross_entropy_fwd_mainloop();
    });
}

} // namespace transformer_engine::pytorch