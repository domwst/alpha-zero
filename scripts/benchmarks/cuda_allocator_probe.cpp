// Optional BATCH-001 probe. Load only in separate profiling runs, after libtorch.
// Compile against the exact libtorch/CUDA headers used by the Rust executable.
#include <c10/cuda/CUDACachingAllocator.h>
#include <cstdio>
extern "C" int alz_dump_cuda_allocator_stats(const char* path) {
  try {
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    auto file = std::fopen(path, "w");
    if (!file) return 1;
    std::fprintf(file, "{\"schema_version\":1,\"allocator\":\"native\"");
#define FIELD(name) std::fprintf(file, ",\"" #name "\":{\"current\":%lld,\"peak\":%lld,\"allocated\":%lld,\"freed\":%lld}", (long long)stats.name[0].current, (long long)stats.name[0].peak, (long long)stats.name[0].allocated, (long long)stats.name[0].freed)
    FIELD(allocated_bytes); FIELD(reserved_bytes); FIELD(inactive_split_bytes);
    FIELD(allocation); FIELD(segment);
#undef FIELD
    std::fprintf(file, ",\"allocation_retries\":%lld,\"ooms\":%lld,\"device_allocations\":%lld,\"device_frees\":%lld,\"synchronize_all_streams\":%lld}\n", (long long)stats.num_alloc_retries, (long long)stats.num_ooms, (long long)stats.num_device_alloc, (long long)stats.num_device_free, (long long)stats.num_sync_all_streams);
    return std::fclose(file);
  } catch (...) { return 2; }
}
