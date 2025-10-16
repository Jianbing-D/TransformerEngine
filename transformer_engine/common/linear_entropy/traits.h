#ifndef TRANSFORMER_ENGINE_COMMON_LINEAR_ENTROPY_TRAITS_H
#define TRANSFORMER_ENGINE_COMMON_LINEAR_ENTROPY_TRAITS_H

#include <cstdint>

#include <cutlass/numeric_types.h>              // CUTLASS numeric types
#include <cute/tensor.hpp>                      // CuTe tensor implementation
#include <cute/arch/cluster_sm90.hpp>           // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp>   // Compile time in constants such as _1, _256 etc.
#include <cute/algorithm/cooperative_copy.hpp>  // Auto vectorized copy operation
#include <cute/arch/tmem_allocator_sm100.hpp>   // TMEM allocator for SM100
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

namespace transformer_engine {
namespace linear_entropy {

enum class ArchType : uint32_t {
    Ampere = 0,
    Hopper,
    Blackwell,
    NUM = 3
};


namespace fwd {

template <ArchType arch,
          typename InT,
          typename OutT>
struct MainloopTraits {
    static constexpr ArchType kArch = arch;

    static_assert(cute::sizeof_bits_v<InT> == 16, "InT must be 16 bits");

    static constexpr bool kUseTMA = (arch != ArchType::Ampere);

    // MMA instruction
    // D in REG, A in REG, B in REG, 16x8x16
    using Ampere_MMA_INST = std::conditional_t<
        std::is_same_v<InT, cutlass::bfloat16_t>,
        cute::SM80_16x8x16_F32BF16BF16F32_TN,
        cute::SM80_16x8x16_F32F16F16F32_TN>;
    // D in REG, A in SMEM, B in SMEM, 64x256x16
    using Hopper_MMA_INST = std::conditional_t<
        std::is_same_v<InT, cutlass::bfloat16_t>,
        cute::SM90_64x256x16_F32BF16BF16_SS<cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>,
        cute::SM90_64x256x16_F32F16F16_SS<cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>>;
    // D in TMEM, A in SMEM, B in SMEM, 256x256x16
    using Blackwell_MMA_INST = cute::SM100_MMA_F16BF16_2x1SM_SS<InT, InT, float, 256, 256, cute::UMMA::Major::K, cute::UMMA::Major::K>;

    using MMA_INST = std::conditional_t<
        arch == ArchType::Ampere,
        Ampere_MMA_INST,
        std::conditional_t<
            arch == ArchType::Hopper,
            Hopper_MMA_INST,
            Blackwell_MMA_INST>>;

    using MMA_ATOM = cute::MMA_Atom<MMA_INST>;
    using MMA_ATOM_TRAITS = cute::MMA_Traits<MMA_INST>;

    // tile Shape, MxNxK, as each element is 16 bits, so 128B swizzle will be 256 elements
    using Tiler = cute::Tile<cute::_256, cute::_256, cute::_256>;

    // form 64x8x16 as an Atom, 4x32x16 repeats
    using Ampere_ThreadLayout = cute::Layout<cute::Shape<cute::_4, cute::_1, cute::_1>>;
    // 64x256x16 as an Atom, 4x1x16 repeats
    using Hopper_ThreadLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
    // 256x256x16 as an Atom, 1x1x16 repeats
    using Blackwell_ThreadLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
    using ThreadLayout = std::conditional_t<
        arch == ArchType::Ampere,
        Ampere_ThreadLayout,
        std::conditional_t<
            arch == ArchType::Hopper,
            Hopper_ThreadLayout,
            Blackwell_ThreadLayout>>;

    static constexpr int32_t _kMMAThreads = cute::get<0>(cute::shape(typename MMA_ATOM_TRAITS::ThrID{}));
    static constexpr int32_t kMMAThreads = arch == ArchType::Blackwell ? 32 : _kMMAThreads;
    static constexpr int32_t kThreads = kMMAThreads
                                        * cute::get<0>(cute::shape(ThreadLayout{})) 
                                        * cute::get<1>(cute::shape(ThreadLayout{}))
                                        * cute::get<2>(cute::shape(ThreadLayout{}));

    using TiledMMA = decltype(cute::make_tiled_mma(
        MMA_ATOM{},
        ThreadLayout{},
        Tiler{}
    ));
};

} // namespace fwd


} // namespace linear_entropy
} // namespace transformer_engine

#endif