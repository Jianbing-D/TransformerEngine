#ifndef TRANSFORMER_ENGINE_LINEAR_ENTROPY_KERNELS_H
#define TRANSFORMER_ENGINE_LINEAR_ENTROPY_KERNELS_H

#include "traits.h"

#include <cutlass/cluster_launch.hpp>

namespace transformer_engine {
namespace linear_entropy {
namespace fwd {

namespace {

template <typename traits>
__global__ void linear_entropy_fwd_mainloop_ampere() {}

template <typename traits>
__global__ void linear_entropy_fwd_mainloop_hopper() {}

template <typename traits>
__global__ void linear_entropy_fwd_mainloop_blackwell() {}

} // anonymous namespace

template <typename traits, 
          std::enable_if_t<traits::kArch == ArchType::Ampere, int> = 0>
void linear_entropy_fwd_mainloop() {
    printf("Ampere\n");
}

template <typename traits, 
          std::enable_if_t<traits::kArch == ArchType::Hopper, int> = 0>
void linear_entropy_fwd_mainloop() {
    printf("Hopper\n");
}

template <typename traits, 
          std::enable_if_t<traits::kArch == ArchType::Blackwell, int> = 0>
void linear_entropy_fwd_mainloop() {
    printf("Blackwell\n");

    dim3 block(traits::kThreads);
    dim3 grid(2, 1, 1);
    dim3 cluster(2, 1, 1);
    int32_t shared_mem_size = 0;

    auto kernel = linear_entropy_fwd_mainloop_blackwell<traits>;
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));

    cutlass::ClusterLaunchParams params = {grid, block, cluster, shared_mem_size};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, static_cast<void const*>(&kernel));
    CUTE_CHECK_LAST();
}

extern template void linear_entropy_fwd_mainloop<MainloopTraits<ArchType::Ampere, cutlass::half_t, float>>();
extern template void linear_entropy_fwd_mainloop<MainloopTraits<ArchType::Hopper, cutlass::half_t, float>>();
extern template void linear_entropy_fwd_mainloop<MainloopTraits<ArchType::Blackwell, cutlass::half_t, float>>();

extern template void linear_entropy_fwd_mainloop<MainloopTraits<ArchType::Ampere, cutlass::bfloat16_t, float>>();
extern template void linear_entropy_fwd_mainloop<MainloopTraits<ArchType::Hopper, cutlass::bfloat16_t, float>>();
extern template void linear_entropy_fwd_mainloop<MainloopTraits<ArchType::Blackwell, cutlass::bfloat16_t, float>>();

} // namespace fwd
} // namespace linear_entropy
} // namespace transformer_engine

#endif // TRANSFORMER_ENGINE_LINEAR_ENTROPY_KERNELS_H