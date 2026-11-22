# Feature files

Behavior specs in Gherkin. They're the spec for the test suite in `tests/`.
`tests/features.rs` runs them with the `cucumber` crate, and the steps are in `tests/steps/`.

## Where a behavior is tested

One home per behavior, so the two don't drift apart:

- **Driver behavior** (`cudash::driver`) lives here. `tests/driver.rs` keeps only cudarc's own examples and what no
  scenario
  covers.
- **Images** (`CudaImage`) are in `tests/images.rs`, not here, because reading an image back needs hand-written Vulkan.
- **NVIDIA interop** has no step definitions, so its Rust tests (`tests/interop_*.rs`) are the checks. A scenario for it
  is tagged `@untested` until the Rust tests cover it, and isn't written when they already do.

## Files

| File                       | Covers                                                     | Needs                   |
|----------------------------|------------------------------------------------------------|-------------------------|
| `driver_memory.feature`    | Allocation and handles in `cudash::driver`                 | Any Vulkan 1.3 GPU      |
| `driver_transfers.feature` | Host, device and view copies in `cudash::driver`           | Any Vulkan 1.3 GPU      |
| `driver_kernels.feature`   | SPIR-V launches, arguments, limits and block sizes         | Any Vulkan 1.3 GPU      |
| `driver_streams.feature`   | Ordering, fork, synchronize, threads                       | Any Vulkan 1.3 GPU      |
| `driver_graphs.feature`    | Stream capture, graph replay, refused calls                | Any Vulkan 1.3 GPU      |
| `driver_extensions.feature` | Optional shader extensions: kernels using them, `MissingFeature` for capabilities the GPU lacks | Any Vulkan 1.3 GPU; some scenarios skip by GPU feature |
| `driver_context.feature`   | Device choice and description                              | Any Vulkan 1.3 GPU      |
| `interop_memory.feature`   | No CUDA: `ctx.as_cuda()`, `stream.as_cuda()`, lending give errors | Any Vulkan 1.3 GPU      |
| `interop_sharing.feature`  | Turn-taking and semaphore timeouts. Nothing runs these yet | NVIDIA with CUDA        |
| `interop_lend.feature`     | `AsCuda` lends and paired streams. No steps yet            | NVIDIA with CUDA        |
| `slang_kernels.feature`    | One Slang kernel on both backends                          | NVIDIA for the PTX side |

## Tags

| Tag              | Meaning                                                                              |
|------------------|--------------------------------------------------------------------------------------|
| `@vulkan`        | Runs on any Vulkan 1.3 GPU, including AMD and Intel                                  |
| `@nvidia`        | Needs an NVIDIA GPU with CUDA. Skips elsewhere unless `CUDASH_REQUIRE_NVIDIA` is set |
| `@non-nvidia`    | Needs a Vulkan GPU from another vendor. Skips on an NVIDIA-only machine              |
| `@cudarc`        | Needs the `cudarc` feature, which is on by default                                   |
| `@untested`      | Behaviour the code has, but no test checks yet                                       |
| `@planned`       | Designed in `plan/api.md`, not implemented                                           |
| `@open-question` | Depends on something the project hasn't settled. See `notes/`                        |
| `@gpu-has-<x>`, `@gpu-lacks-<x>` | Skips unless the first GPU has (or lacks) a shader feature. Names are in `gpu_has` in `tests/steps/mod.rs` |
| `@known-gap`     | Known to fail today. The scenario states the intended behaviour                      |
