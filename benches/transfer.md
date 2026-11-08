# Memory transfer report

> **AI-generated summary** of the benchmark results. The numbers come from real runs, but the observations are
> interpretation and can be wrong. Check them against the CSV or a fresh run before relying on them.

Measured with `benches/transfer.rs`.

## Setup

|            |                                                               |
|------------|---------------------------------------------------------------|
| GPU        | AMD Steam Machine (RADV NAVI33), via Vulkan                   |
| Driver     | cudash `driver` module (Vulkan 1.3, no CUDA)                  |
| GPU link   | PCIe 4.0 x8 (16 GT/s), about 14 GB/s usable in practice       |
| CPU        | AMD Custom CPU 1772                                           |
| Frame      | 2560 x 2160, `f32`, one element per pixel                     |
| Frame size | 5,529,600 elements = 22,118,400 bytes (about 22.1 MB)         |
| Timing     | Median of 7 batches per case, after a warm-up                 |
| Build      | `cargo bench`, release profile, inside the `ubuntu` distrobox |

Every transfer is followed by a synchronize inside the timed batch, so async work is counted in full.
Allocating operations also include the drop of what they allocated.

## Time to transfer one frame

Run 1 of 3, 2026-11-07, same machine (RADV NAVI33, Mesa 26.0.8, no commits). Spread over the three runs in brackets.
"Before" is the first report, taken before `memcpy_htod` got staging reuse.

| Transfer                                          | Before  | Now                     | Throughput now |
|---------------------------------------------------|---------|-------------------------|----------------|
| Host to device, `clone_htod`                      | 5.95 ms | **3.15 ms** (3.09-3.15) | 7.02 GB/s      |
| Host to device, `memcpy_htod` (existing buffer)   | 5.86 ms | **3.23 ms** (3.23-3.28) | 6.85 GB/s      |
| Device to host, `clone_dtoh`                      | 5.74 ms | **5.69 ms** (5.65-5.87) | 3.89 GB/s      |
| Device to host, `memcpy_dtoh` (existing buffer)   | 8.38 ms | **6.92 ms** (6.92-7.15) | 3.19 GB/s      |
| Device to device, `clone_dtod`                    | 0.34 ms | **0.31 ms**             | 70.9 GB/s      |
| Device to device, `memcpy_dtod` (existing buffer) | 0.24 ms | **0.24 ms**             | 92.0 GB/s      |

These host-to-device rows are `Vec` sources, so they take the reused-staging path. Pinned and registered sources are
below. Allocating a frame buffer (`alloc_zeros`) takes 0.21 ms (0.21-0.27).

## All results

Medians per operation, run 1 of 3. Throughput is bytes moved per second.

| Operation              | 4 KiB     | 1 frame (2560x2160) | 4 frames  |
|------------------------|-----------|---------------------|-----------|
| alloc_zeros            | 99.35 µs  | 208.23 µs           | 1.038 ms  |
| clone_htod             | 101.19 µs | 3.150 ms            | 12.739 ms |
| memcpy_htod (existing) | 54.96 µs  | 3.227 ms            | 12.702 ms |
| clone_dtoh             | 107.47 µs | 5.688 ms            | 41.028 ms |
| memcpy_dtoh (existing) | 107.32 µs | 6.924 ms            | 32.099 ms |
| clone_dtod             | 100.09 µs | 311.73 µs           | 980.13 µs |
| memcpy_dtod (existing) | 55.96 µs  | 240.35 µs           | 780.76 µs |

Before the change (first report): `clone_htod` 131.56 µs / 5.950 ms / 21.605 ms, `memcpy_htod` 100.41 µs / 5.863 ms /
21.864 ms. The other rows are within noise of the first report except `memcpy_dtoh` at a frame (see below). The 4-frame
rows were run once per run and moved a lot for allocation (0.57-1.04 ms), so don't read them closely. At 4 KiB,
throughput is under 0.1 GB/s in every row, so the times are fixed per-call cost.

## Observations

- **Host to device is no longer the slowest path.** A `Vec` upload takes about 3.2 ms, down from 5.9 ms, because
  staging buffers are reused (see the host paths section). Pinned and registered sources take 1.66 ms.
- **Device to host is now the slowest.** `memcpy_dtoh` still allocates a staging buffer and copies out of it: 6.9 ms at
  a
  frame (3.2 GB/s), against 0.24 ms device to device. It was not changed. Nothing in this report says which stage
  dominates; the staging-buffer todo's step 1 (time each stage) still applies.
- **`memcpy_dtoh` into an existing buffer is slower than `clone_dtoh`** at one frame: 6.92-7.15 ms against 5.65-5.87 ms
  over three runs, so this is real and not noise (the first report guessed noise). At four frames it reverses (32.1 ms
  against 41.0 ms) in the one run measured. Cause unknown.

## Host to device paths: five rows, now

Measured 2026-11-07 on AMD Steam Machine (RADV NAVI33), Mesa 26.0.8, no commits yet, three full runs of
`cargo bench --bench transfer`. Cells are run 1 with the range over the three runs in brackets. Same existing
destination buffer for all paths. `call` is the time until the call returns; `total` adds a synchronize. Each cell is
the median of 500 (4 KiB) or 50 (frame) calls after 10 warm-up calls, each followed by a synchronize. "Queued" means six
4-frame `memcpy_dtod` (about 4.4 ms, measured alone including submission) were submitted before each timed call, outside
the timed part. The pinned source was filled once, and the registered `Vec` was registered once, before timing.

| Path                                  | Size  | Stream | Call                    | Total                    |
|---------------------------------------|-------|--------|-------------------------|--------------------------|
| `memcpy_htod`, `Vec` (reused staging) | 4 KiB | idle   | 18.0 µs (17.6-18.0)     | 51.7 µs (51.7-52.0)      |
| `memcpy_htod`, `PinnedHostSlice`      | 4 KiB | idle   | 12.0 µs (12.0-15.6)     | **46.8 µs** (46.8-50.5)  |
| `memcpy_htod`, registered `Vec`       | 4 KiB | idle   | 16.4 µs (15.6-16.5)     | 53.0 µs (51.2-53.9)      |
| `memcpy_htod_reused`                  | 4 KiB | idle   | 12.2 µs (12.2-18.0)     | 47.7 µs (47.7-56.1)      |
| `memcpy_htod_unstaged`                | 4 KiB | idle   | 96.1 µs (95.6-97.2)     | 98.2 µs (97.8-99.5)      |
| `memcpy_htod`, `Vec`                  | 4 KiB | queued | 34.7 µs (25.3-34.7)     | 4.309 ms (4.304-4.311)   |
| `memcpy_htod`, `PinnedHostSlice`      | 4 KiB | queued | 24.5 µs (22.8-24.5)     | 4.311 ms (4.299-4.311)   |
| `memcpy_htod`, registered `Vec`       | 4 KiB | queued | 25.0 µs (22.8-25.0)     | 4.305 ms (4.305-4.310)   |
| `memcpy_htod_reused`                  | 4 KiB | queued | 33.8 µs (25.1-33.8)     | 4.310 ms (4.309-4.310)   |
| `memcpy_htod_unstaged`                | 4 KiB | queued | 4.316 ms (blocks)       | 4.319 ms (4.308-4.319)   |
| `memcpy_htod`, `Vec`                  | frame | idle   | 1.610 ms (1.43-1.61)    | 3.374 ms (3.25-3.37)     |
| `memcpy_htod`, `PinnedHostSlice`      | frame | idle   | **57.6 µs** (57.6-66.9) | **1.660 ms** (1.66-1.67) |
| `memcpy_htod`, registered `Vec`       | frame | idle   | 54.4 µs (54.4-65.0)     | 1.656 ms (1.656-1.675)   |
| `memcpy_htod_reused`                  | frame | idle   | 1.552 ms (1.39-1.55)    | 3.341 ms (3.21-3.34)     |
| `memcpy_htod_unstaged`                | frame | idle   | 2.223 ms (2.16-2.64)    | 2.225 ms (2.16-2.65)     |
| `memcpy_htod`, `Vec`                  | frame | queued | 1.936 ms (1.78-1.94)    | 5.912 ms (5.88-5.91)     |
| `memcpy_htod`, `PinnedHostSlice`      | frame | queued | **61.1 µs** (61.1-69.7) | 5.881 ms (5.82-5.88)     |
| `memcpy_htod`, registered `Vec`       | frame | queued | 66.1 µs (66.1-75.8)     | 5.873 ms (5.81-5.87)     |
| `memcpy_htod_reused`                  | frame | queued | 1.927 ms (1.83-1.93)    | 5.940 ms (5.75-5.94)     |
| `memcpy_htod_unstaged`                | frame | queued | 5.909 ms (blocks)       | 5.914 ms (5.82-5.91)     |

**Which rows are before the change:** none of the table above. The pre-change `memcpy_htod` rows (new staging buffer
on every call: 82 µs idle 4 KiB, 5.87 ms idle frame, 6.25 ms queued frame total) are in the "Before" subsection below.
`memcpy_htod_reused` and `memcpy_htod_unstaged` are the same code as before; their new numbers match the old ones
(reused idle frame total 3.33 ms before, 3.34 now).

Spread: idle frame totals move 3-4% for the staged paths, under 1% for pinned and registered, and unstaged had one
outlier run (2.65 ms against 2.16 and 2.22). The staged frame call moves 12% (1.43 to 1.61 ms). Queued frame totals
are within 1-3% of each other in every run. Queued 4 KiB calls are 23-35 µs and the first run was the highest. At 4 KiB
idle, differences between pinned, registered, reused and `Vec` (47 to 56 µs total) are inside the spread.

### One-off costs, one frame (22.1 MB)

Median of 30 (50 for the refills), three runs, range in brackets.

| Operation                                                    | Time                    |
|--------------------------------------------------------------|-------------------------|
| `host_register` + `host_unregister`                          | 307 µs (243-307)        |
| `alloc_pinned` + drop, memory not touched                    | 599 µs (571-599)        |
| `alloc_pinned`, first fill, drop                             | 2.83 ms (2.80-2.83)     |
| `vec![1.5; frame]` + drop (reference)                        | 656 µs (611-656)        |
| Refill a `Vec` with `copy_from_slice`                        | 2.53 ms (2.52-2.53)     |
| Refill a pinned slice with `as_mut_slice()?.copy_from_slice` | **1.17 ms** (1.14-1.17) |
| `as_mut_slice()?` alone, nothing running                     | 0.03 µs                 |
| `as_mut_slice()?` right after a copy from it                 | 1.69 ms (1.69-1.70)     |

- **Register costs 0.3 ms for a frame**, about a sixth of one pinned/registered upload (1.66 ms). It pays for itself
  after one upload against reused staging (saves 1.7 ms each) and has no API surprise. `alloc_pinned` is 0.6 ms, but
  that excludes the first touch: the first fill adds about 2.2 ms (page faults), paid once.
- **Filling pinned memory is not slower than filling a `Vec`.** Refilling an already-touched pinned slice took 1.17 ms
  against 2.53 ms for the `Vec`. I don't know why the `Vec` is slower; both were warm and the source was a separate
  22 MB `Vec`. Do not take it as "pinned is faster to write", only that it is not slower. The `as_mut_slice` call itself
  costs nothing when no
  copy is using the memory.
- **The cost to know about is the wait.** `as_mut_slice` right after an upload from the same slice waits for the copy,
  1.69 ms here, which is the whole GPU copy. A loop that writes into one pinned slice and uploads it, one frame
  at a time, gets no overlap. Use two slices and alternate to overlap the fill of one with the upload of the other.
  That wait was not tested with two slices.

### Conclusion

| Path                   | Idle frame                             | Behind queued work                    | Needs                                                                                  |
|------------------------|----------------------------------------|---------------------------------------|----------------------------------------------------------------------------------------|
| `Vec`, reused staging  | 3.37 ms; call blocks 1.6 ms (CPU copy) | call 1.9 ms, total same as the others | nothing; works for any memory including `static`                                       |
| `PinnedHostSlice`      | **1.66 ms**, call 58 µs                | call 61 µs                            | `unsafe alloc_pinned`, data written into it                                            |
| Registered `Vec`       | **1.66 ms**, call 54 µs                | call 66 µs                            | `unsafe host_register`, 0.3 ms once; no writes during copies; unregister before drop   |
| `memcpy_htod_reused`   | 3.34 ms                                | 1.93 ms                               | same as `Vec` path now: redundant                                                      |
| `memcpy_htod_unstaged` | 2.22 ms, blocks                        | 5.91 ms, blocks                       | pinned and registered beat it at a frame; at 4 KiB it is 2x slower than all the others |

- Pinned and registered are equal within the spread (1.656 and 1.660 ms) and halve an idle frame upload against the
  `Vec` path (3.37 to 1.66 ms). They return in about 60 µs whatever the stream is doing, against 1.6 to 1.9 ms of CPU
  copying for the staging path. The CPU is free for the whole upload.
- The 1.66 ms is the GPU reading host memory over the link: 22.1 MB at 13 GB/s, close to the 14 GB/s estimate for PCIe
  4.0 x8.
  Not much more can be gained there.
- Behind queued work all five paths finish within 1-3% of each other (5.8-5.9 ms): the GPU is the limit. The paths only
  differ in what the CPU does meanwhile, except `memcpy_htod_unstaged`, which blocks for all of it.
- At 4 KiB nothing matters: 47 to 56 µs everywhere, except `memcpy_htod_unstaged` (98 µs, about 90 µs of set-up).
- `memcpy_htod_reused` is now the same as plain `memcpy_htod` (3.34 against 3.37 ms, call 1.55 against 1.61 ms) and
  can go. `memcpy_htod_unstaged` is slower than both new direct paths and blocks, so it has no case left either.
- Staged `Vec` uploads and frame loops: when the data is written once per frame, writing straight into a pinned slice
  skips the 1.5 ms CPU copy, but only if filling it doesn't wait for the previous upload (see above).

One GPU, one driver. NVIDIA and Intel are unmeasured, and `host_register` support there is untested.

### Before: `memcpy_htod` with a new staging buffer per call

**These rows are from before the change.** Plain `memcpy_htod` there is the old path: a new staging buffer on every
call. Kept as the baseline.

Measured on AMD Steam Machine (RADV NAVI33), Mesa 26.0.8, no commits yet. Three full runs of
`cargo bench --bench transfer`, rows below are run 1 with the spread over the three in brackets. Same `Vec<f32>`
source and the same existing destination buffer for all paths. `call` is the time until the call returns; `total`
adds a synchronize. Each cell is the median of 500 (4 KiB) or 50 (frame) calls, after 10 warm-up calls, each
followed by a synchronize. "Queued" means six 4-frame `memcpy_dtod` (about 4.4 ms of GPU time, measured alone) were
submitted before each timed call, outside the timed part.

| Path                   | Size  | Stream | Call                  | Total                    |
|------------------------|-------|--------|-----------------------|--------------------------|
| `memcpy_htod`          | 4 KiB | idle   | 32.0 µs (30.3-32.0)   | 82.3 µs (77.5-82.3)      |
| `memcpy_htod_reused`   | 4 KiB | idle   | 14.5 µs (14.5-15.6)   | **47.1 µs** (47.1-48.6)  |
| `memcpy_htod_unstaged` | 4 KiB | idle   | 95.1 µs (87.8-95.1)   | 97.4 µs (90.0-97.4)      |
| `memcpy_htod`          | 4 KiB | queued | 55.7 µs               | 4.311 ms                 |
| `memcpy_htod_reused`   | 4 KiB | queued | 23.9 µs               | 4.313 ms                 |
| `memcpy_htod_unstaged` | 4 KiB | queued | **4.309 ms** (blocks) | 4.313 ms                 |
| `memcpy_htod`          | frame | idle   | 3.828 ms (3.77-3.83)  | 5.871 ms (5.84-5.87)     |
| `memcpy_htod_reused`   | frame | idle   | 1.510 ms (1.38-1.56)  | 3.326 ms (3.22-3.33)     |
| `memcpy_htod_unstaged` | frame | idle   | 2.264 ms (2.20-2.26)  | **2.267 ms** (2.21-2.27) |
| `memcpy_htod`          | frame | queued | 3.642 ms (3.63-3.66)  | 6.246 ms (6.25-6.27)     |
| `memcpy_htod_reused`   | frame | queued | 1.969 ms (1.86-1.97)  | 6.072 ms (6.07-6.11)     |
| `memcpy_htod_unstaged` | frame | queued | **6.052 ms** (blocks) | 6.056 ms (6.05-6.08)     |

Queued 4 KiB rows differ between runs by under 0.02 ms in total. The existing rows rerun in the same session
matched the earlier ones (frame `memcpy_htod` 5.86-5.89 ms), so the old and new numbers are comparable.

Run-to-run spread: totals move by 1-3% at a frame and 5% at 4 KiB. The reused frame call moved 12% (1.38 to
1.56 ms). Gaps below those are not conclusions.

What it shows:

- **Reused staging answers the page-fault question.** The staged frame call takes 3.8 ms, the reused one 1.4-1.6 ms. The
  CPU copy into a warm buffer is about 1.5 ms (about 15 GB/s), so roughly 2.3 ms of the staged call is allocating and
  faulting in a new buffer, not copying. The earlier 3.00 ms "CPU copy" was inflated by that. This bench does
  not split allocation from page faults, so how the 2.3 ms divides is not measured.
- **Idle frame: import wins, reused is second.** Import totals 2.27 ms, reused 3.33 ms, staged 5.87 ms. Import
  beats reused by about 1.05 ms (30%), well outside the spread. Reused beats staged by 2.5 ms. The remaining
  import advantage is the CPU copy that reused still does.
- **Idle 4 KiB: reused wins.** 47 µs against 82 µs staged and 97 µs import. Import's set-up (page lookup, memory
  import, buffer creation) is about 90 µs per call here and doesn't pay off at this size.
- **Queued, import blocks for the whole queue.** The call returns after 4.3 ms (4 KiB) or 6.05 ms (frame),
  against 0.02-0.06 ms and 2-3.6 ms for the non-blocking paths. The CPU can do nothing else for that time,
  where the staged and reused paths return and let it go on.
- **Queued totals are nearly equal at a frame** (6.25, 6.07, 6.06 ms): the CPU copy for the staged and reused paths
  overlaps the busy GPU, and the GPU copy of about 1.7-1.9 ms comes after the queued work in all three. Import's
  total is 0.02 to 0.2 ms better, near the spread. The "queued work alone" figure (4.4 ms) includes submitting
  the six copies, so it isn't the pure GPU time and the totals above can sit slightly below it.
- **Reuse happened, by inference.** There was a synchronize before every timed call, so the previous copy had
  finished; the reused call is 15 µs against 32 µs for a new allocation at 4 KiB, and 1.5 against 3.8 ms at a frame.
  No allocation counter was read, so this is not directly verified.

The conclusion drawn from these numbers (reused staging, no import) led to the current design, above.

## Limits of this report

- One machine, one GPU, one driver (RADV). Other GPUs or drivers may differ a lot.
- Three runs of the whole bench for the host path and one-off tables; the top two tables are run 1 with ranges for the
  frame rows. Allocation and 4 KiB numbers moved noticeably between runs, so treat differences under about 20% as noise
  there.
- Only `f32` was measured. Other element types go through the same code, but their numbers aren't here.
- The `memcpy_dtod` rows measure device copies already in place, while the `clone_dtod` rows include
  allocation.

## Reproduce

```text
cargo bench --bench transfer
```
