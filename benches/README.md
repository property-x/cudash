# Bench plan

What we want to measure next, why, and what each bench needs before it can run.
Each bench `bench_name.rs` has a report `bench_name.md`.

## Conventions

- **Harness:** plain `std::time`, with the helpers in `benches/common/mod.rs`. No dependencies.
- **Statistic:** median of 7 batches after a warm-up. Anything that measures a single run says so.
- **Synchronization:** every timed operation ends with a synchronize inside the timed batch, unless the
  case is named "async".
- **Device:** driver benches run on any Vulkan 1.3 GPU, with no cudarc feature. Benches that need CUDA
  go behind `required-features = ["cudarc"]`, as `memory` does.
- **Output:** one report per bench, named after it, with the setup, results, observations, limits and
  a reproduce command, as in `launch.md`.
- **Frame:** 2560 x 2160 `f32`, one element per pixel, 22,118,400 bytes, as in `transfer.md`.

## Existing benches

| Bench            | What it measures                                                                                                                                                                     | Report              |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `launch`         | Kernel launch overhead with an empty kernel: async, with a synchronize, one and 4096 workgroups, and the cold first launch                                                           | `launch.md`         |
| `transfer`       | `alloc_zeros` and every copy direction at 4 KiB, one frame and four frames, plus the host-to-device paths (`Vec`, pinned, registered) idle and behind queued work, and one-off costs | `transfer.md`       |
| `transfer_sweep` | Host-to-device copy time and GB/s against size, 4 KiB to a frame, for `memcpy_htod`, `memcpy_htod_reused` and `memcpy_htod_unstaged`. Writes a CSV and an SVG plot                   | `transfer_sweep.md` |
| `transfer_kinds` | The same sizes for a plain `Vec`, a pinned slice and a registered `Vec`, 200 timed copies per point, with the spread. Writes a CSV and an SVG plot                                   | `transfer_kinds.md` |

## Plan

| Priority | Bench            | Question it answers                                                    | Notes                                                                                                      |
|----------|------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| 1        | `frame`          | Does a whole frame fit the 16.7 ms budget?                             | Needs a frame kernel.                                                                                      |
| 2        | `transfer_sweep` | Where do fixed per-call costs stop dominating?                         | Host to device done (`transfer_sweep.md`). Device to host and device to device not yet. `Cargo.toml` entry |
| 3        | `batching`       | How much do many small submissions cost against one?                   | An API to batch launches. Not built.                                                                       |
| 4        | `kernel_args`    | What do arguments and new block sizes cost?                            | /                                                                                                          |
| 5        | `streams`        | Does work on several streams scale, and what do `fork` and waits cost? | Nothing, but results depend on the shared queue                                                            |
| 6        | `alloc_churn`    | Is a suballocator worth building?                                      | /                                                                                                          |
| 7        | `sync`           | What does `synchronize` cost when idle, and when there's pending work? | /                                                                                                          |
| 8        | `modules`        | What do `load_module` and `load_function` cost?                        | /                                                                                                          |
| 9        | `cudarc_parity`  | How does cudash compare with cudarc on the same operations?            | Needs Nvidia machine                                                                                       |
| 10       | `interop`        | What do `lend` and the first import into CUDA cost?                    | Needs Nvidia machine, and the interop API (`lend`) is not built yet                                        |

