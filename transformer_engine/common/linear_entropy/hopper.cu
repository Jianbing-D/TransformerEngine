#include "kernels.h"

namespace transformer_engine {
namespace linear_entropy {

namespace fwd {

template void linear_entropy_fwd_mainloop<MainloopTraits<ArchType::Hopper, cutlass::half_t, float>>();
template void linear_entropy_fwd_mainloop<MainloopTraits<ArchType::Hopper, cutlass::bfloat16_t, float>>();

} // namespace fwd

} // namespace linear_entropy
} // namespace transformer_engine