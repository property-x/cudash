# batching

Bursts of N launches through cudash itself (not raw `ash`): a normal stream, a batching stream (`GpuStream::set_batching(true)`,
Cargo feature `experimental`) and one captured graph. Asked by `main`; target is real-time bursts of 4 to 32 launches at the lowest
latency. Kernel `double_all`, n = 0, one workgroup, one buffer argument. Every burst ends with `stream.synchronize()`.

## Setup

- **AMD:** Steam Machine, RADV NAVI33, 2026-11-23, working tree on `9ff8019` (uncommitted, includes the batching change in `src/`). 5 runs.
- **NVIDIA:** GTX 1650 Ti at .62, driver 615.78.08, same date and tree. 5 runs.
- `benches/batching.rs`: per run, 300 timed bursts per N after 20 warm-up bursts; `inline`, `batched` and `graph` are interleaved burst by burst.
  A run reports the median burst. The tables show the median of the 5 run medians with the range of run medians in brackets.
  - `inline`: `launch_builder().arg(&mut buf).arg(&n).launch()` N times, then `synchronize()`.
  - `batched`: the same on a stream with `set_batching(true)`; `synchronize()` flushes.
  - `graph`: N launches captured once per N (`begin_capture` / `end_capture`), then `graph.launch()` + `synchronize()` per burst.
  - **Burst** = clock from the first call to the end of `synchronize()`. **CPU** = time inside the N `launch()` calls (graph: inside `graph.launch()`).
- Dependency check: 16 launches on a 1-element buffer holding 1.0, each doubling the previous output (n = 1), then `clone_dtoh`; 50 bursts per
  stream. Expected 65536.

## Burst time, µs (per launch µs in parentheses), median [range of run medians]

| N | inline AMD | batched AMD | graph AMD | inline NVIDIA | batched NVIDIA | graph NVIDIA |
|---|---|---|---|---|---|---|
| 1 | 57 [54-58] (56.7) | 56 [54-58] (56.2) | 53 [51-62] (53.3) | 44 [30-48] (44.4) | 43 [29-47] (43.1) | 43 [27-47] (42.8) |
| 2 | 88 [84-130] (44.0) | 60 [56-99] (30.2) | 58 [50-86] (28.9) | 42 [40-43] (21.1) | 37 [33-39] (18.3) | 31 [30-34] (15.6) |
| 4 | 153 [151-190] (38.3) | 68 [67-101] (16.9) | 57 [56-78] (14.2) | 52 [51-54] (13.0) | 43 [43-46] (10.8) | 38 [38-42] (9.5) |
| 8 | 280 [271-289] (35.0) | 83 [79-85] (10.4) | 69 [60-72] (8.6) | 95 [93-120] (11.9) | 61 [60-71] (7.6) | 52 [52-60] (6.5) |
| 16 | 541 [535-552] (33.8) | 115 [111-125] (7.2) | 84 [76-85] (5.3) | 171 [168-197] (10.7) | 96 [95-103] (6.0) | 80 [80-89] (5.0) |
| 32 | 1083 [1053-1092] (33.8) | 218 [202-221] (6.8) | 142 [139-143] (4.4) | 329 [328-337] (10.3) | 165 [164-169] (5.2) | 136 [136-139] (4.3) |

## CPU time inside the launch calls, µs (whole burst, not per launch)

| N | inline AMD | batched AMD | graph AMD | inline NVIDIA | batched NVIDIA | graph NVIDIA |
|---|---|---|---|---|---|---|
| 1 | 19.0 | 3.6 | 15.8 | 5.6 | 2.6 | 3.1 |
| 2 | 38.6 | 6.5 | 16.0 | 10.2 | 4.7 | 3.4 |
| 4 | 73.3 | 11.8 | 16.0 | 20.2 | 8.8 | 4.2 |
| 8 | 136.5 | 21.1 | 16.2 | 40.0 | 16.8 | 6.0 |
| 16 | 257.7 | 40.1 | 17.2 | 79.2 | 31.6 | 9.2 |
| 32 | 559.8 | 82.9 | 23.1 | 161.4 | 62.1 | 15.9 |

Run ranges are within 10 % except as shown in the first table (AMD N = 2-4 have one slow run each, NVIDIA N = 1 is bimodal, 27-30 or 44-48 µs
between runs, as in `submit_cost.md`; cause unknown).

## Dependency chain

16 launches, each reading the previous output, 50 bursts: **correct (65536) on inline and batched, on both GPUs, in all 5 runs each.**
Note the check ends with `clone_dtoh`, so it also covers the flush on a device-to-host copy, not only `synchronize()`.

## Against the raw `ash` numbers

Total burst µs, N = 4 / 8 / 16 / 32, AMD then NVIDIA.

| | AMD | NVIDIA |
|---|---|---|
| raw `inline` / `timeline wait` (`submit_cost.md`) | 139 / 263 / 506 / 1011 | 52 / 87 / 144 / 282 |
| cudash `inline` | 153 / 280 / 541 / 1083 | 52 / 95 / 171 / 329 |
| raw `batched`, one submit (`submit_cost.md`) | 46 / 48 / 54 / 64 | 33 / 40 / 51 / 77 |
| raw `adaptive` (`submit_strategy.md`, N x per-launch) | 88 / 97 / 104 / 163 | 43 / 60 / 64 / 86 |
| cudash `batched` | 68 / 83 / 115 / 218 | 43 / 61 / 96 / 165 |

- cudash `inline` adds 4-8 % over raw on AMD and up to 17 % on NVIDIA at N = 32 (the bookkeeping and labels of `launch_cost.md`).
- cudash `batched` is 2.1-5 x faster than `inline` on AMD for N >= 4 and 1.2-2 x on NVIDIA. Against raw one-submit batching it adds
  about 22 / 35 / 61 / 154 µs on AMD and 10 / 21 / 45 / 88 µs on NVIDIA (N = 4 / 8 / 16 / 32). That excess grows faster than N, and
  the CPU table shows part of it: the N `launch()` calls themselves take 2.5-3 µs each on AMD and 1.9-2.0 on NVIDIA when batched,
  against 0.2-1.2 µs raw. The remaining time is flush, GPU execution and the wait. **The split of that excess is not measured**
  (no profile of the flush path); it is not known whether it is recording cost, the `Pending` entry with N keeps, or GPU time of N barriers.

## Verdict for 4-32 launches

- **Graph is the fastest** where the same N operations repeat: 57-142 µs on AMD, 38-136 µs on NVIDIA, and the CPU cost is nearly flat
  (16-23 µs on AMD, 4-16 µs on NVIDIA). It needs the burst to be the same every time.
- **Batched** is within 1.2-1.5 x of the graph for N = 4-16 (AMD 68 / 83 / 115 against 57 / 69 / 84; NVIDIA 43 / 61 / 96 against 38 / 52 / 80),
  and 1.5 x at N = 32, with no capture. It takes most of the win over `inline`: AMD 153 -> 68 at N = 4, 541 -> 115 at N = 16;
  NVIDIA 171 -> 96 at N = 16. On NVIDIA the win is small at N = 4 (52 -> 43 µs).
- **Inline** is only competitive at N = 1-2 on NVIDIA. On AMD, anything from N = 2 up gains from batching or a graph.
- Batching does not help N = 1 (no change, as expected).
- Cost of batching to watch: the batched burst is still 1.5-2.5 x the raw one-submit batch at N = 16-32, so cudash's path leaves room.
  Per launch the CPU cost grows linearly (2.5 µs per launch on AMD); at N = 32 that is 83 µs of the 218.

## Limits

- One trivial kernel, one buffer, one stream; no other work in flight and no memory copies in the burst.
- Latency here is wall time to the end of `synchronize()`, with the GPU idle before each burst. Not a frame.
- AMD drifts by about 10 µs between sessions (`launch_cost.md`); all variants were interleaved so they share it, but compare absolute
  numbers against other reports only with that in mind.
- `adaptive` rows are from `submit_strategy.md` (N x per-launch figure of its table b), a different harness.

## Reproduce

```text
cargo bench --features experimental --bench batching
```

`Cargo.toml` entry:

```toml
[[bench]]
name = "batching"
harness = false
required-features = ["experimental"]
```
