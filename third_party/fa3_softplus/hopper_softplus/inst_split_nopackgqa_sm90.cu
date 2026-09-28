// Split-KV forward without PackGQA, for the softplus builds (their split epilogue reduces from
// registers in the non-packed layout; stock FA3 always packs when splitting on SM90).
#include "flash_fwd_launch_template.h"

template void run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, /*Split=*/true, /*PagedKVNonTMA=*/false, /*Has_softcap=*/false, /*PackGQA=*/false>(Flash_fwd_params &params, cudaStream_t stream);
