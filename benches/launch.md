# Kernel launch report

> **AI-generated summary** of the benchmark results. The numbers come from real runs, but the observations are
> interpretation and can be wrong. Check them against the CSV or a fresh run before relying on them.

Measured with `benches/launch.rs`.

## Setup

|        |                                                                                             |
|--------|---------------------------------------------------------------------------------------------|
| GPU    | AMD Steam Machine (RADV NAVI33), via Vulkan                                                 |
| Driver | cudash `driver` module (Vulkan 1.3, no CUDA)                                                |
| Kernel | `double_all` (from `tests/kernels/double_all.slang`) with `n = 0`, so the body does nothing |
| Timing | Median of 7 batches per case, after a warm-up                                               |
| Build  | `cargo bench`, release profile, inside the `ubuntu` distrobox                               |

The kernel body is empty, so the times are almost all launch cost: building arguments, recording a command
buffer, submitting it, and later reclaiming it.

## Results

Times are per launch. "Before" is the first report (2026-11-07). "Now" is four runs on 2026-11-08, same machine, with
the range in brackets. `src/` changed in between, and no commit exists to name.

| Case                                                           | Before   | Now (median of 4 runs)     |
|----------------------------------------------------------------|----------|----------------------------|
| First launch, cold (single sample, includes pipeline creation) | 7.40 ms  | **0.24 ms** (0.17-1.21 ms) |
| Async, 1 workgroup                                             | 32.53 µs | 38.3 µs (32.9-43.8)        |
| Launch and sync, 1 workgroup                                   | 52.61 µs | 60.3 µs (54.5-67.2)        |
| Async, 4096 workgroups                                         | 37.96 µs | 37.9 µs (36.7-38.2)        |
| Launch and sync, 4096 workgroups                               | 56.96 µs | 60.0 µs (58.0-62.6)        |

"Async" queues the launches back to back and synchronizes once at the end. Its time is the total divided by
the number of launches, so the final synchronize is spread across them.

## Observations

- **Each launch costs about 33 to 44 µs even when it isn't waited on.** The 1-workgroup case moved by a third between
  runs, so the earlier 33 µs was the low end, not a typical value. The 4096-workgroup case was steady at about 38 µs.
  Each launch allocates, records and submits its own command buffer. This report doesn't profile it.
- **Waiting on every launch adds about 20 µs.**
- **Dispatch size barely matters at this scale.**
- **The cold first launch is now far cheaper: 0.17 to 1.2 ms against 7.4 ms.** I don't know why. A change in `src/`
  (such as pipeline creation or caching) is the likely cause, but I haven't checked it. It is one sample per run and
  varied 7x between runs, so don't quote a single figure.

## Limits of this report

- One machine, one GPU, one driver (RADV). Other GPUs or drivers may differ a lot.
- Four runs. Treat differences under about 20% as noise.
- The cold first-launch figure is a single sample.
- The kernel does no work. A real kernel adds its own execution time on top of these numbers.

## Reproduce

```text
cargo bench --bench launch
```
