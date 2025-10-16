// It fuses linear and cross entropy.

#include "kernels.h"

#include <transformer_engine/linear_entropy.h>
#include "../common.h"

namespace transformer_engine {
namespace linear_entropy {

#define ARCH_SWITCH(arch, Arch, ...)                            \
    do {                                                        \
        switch (arch) {                                         \
            case ArchType::Blackwell: {                         \
                constexpr ArchType Arch = ArchType::Blackwell;  \
                { __VA_ARGS__ }                                 \
                break;                                          \
            }                                                   \
            case ArchType::Hopper: {                            \
                constexpr ArchType Arch = ArchType::Hopper;     \
                { __VA_ARGS__ }                                 \
                break;                                          \
            }                                                   \
            case ArchType::Ampere: {                            \
                constexpr ArchType Arch = ArchType::Ampere;     \
                { __VA_ARGS__ }                                 \
                break;                                          \
            }                                                   \
        }                                                       \
    } while (false)

} // namespace linear_entropy
} // namespace transformer_engine

void nvte_linear_cross_entropy_fwd_mainloop() {
    NVTE_API_CALL(nvte_linear_cross_entropy_fwd_mainloop);

    using transformer_engine::linear_entropy::ArchType;
    namespace fwd = transformer_engine::linear_entropy::fwd;
    
    static ArchType arch = []() {
        int major = 0;
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));

        switch (major) {
            case 10:
                return ArchType::Blackwell;
            case 9:
                return ArchType::Hopper;
            case 8:
                return ArchType::Ampere;
            default: {
                CUTE_LOG("Unsupported architecture: %d", major);
                exit(1);
            }
        }
    }();

    ARCH_SWITCH(arch, Arch, 
        using traits = fwd::MainloopTraits<Arch, cutlass::half_t, float>;
        fwd::linear_entropy_fwd_mainloop<traits>();
    );
}    

