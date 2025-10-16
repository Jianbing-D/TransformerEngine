#ifndef TRANSFORMER_ENGINE_COMMON_LINEAR_CROSS_ENTROPY_H
#define TRANSFORMER_ENGINE_COMMON_LINEAR_CROSS_ENTROPY_H

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

void nvte_linear_cross_entropy_fwd_mainloop();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // TRANSFORMER_ENGINE_COMMON_LINEAR_CROSS_ENTROPY_H