# Host-to-device copy sweep

> **AI-generated summary** of the benchmark results. The numbers come from real runs, but the observations are
> interpretation and can be wrong. Check them against the CSV or a fresh run before relying on them.

Measured with `benches/transfer_sweep.rs`. Data: `transfer_sweep.csv`. Plot: `transfer_sweep.svg` (made by
`plot_sweep.py`).

## Setup

|        |                                                                                                                                            |
|--------|--------------------------------------------------------------------------------------------------------------------------------------------|
| GPU    | AMD Steam Machine (RADV NAVI33), Mesa 26.0.8, kernel 7.2.7-valve2                                                                          |
| Date   | 2026-11-07, no commit (nothing is committed yet)                                                                                           |
| Paths  | `memcpy_htod`, `memcpy_htod_reused`, `memcpy_htod_unstaged` (the last two are experimental)                                                |
| Sizes  | 4 KiB to 16 MiB in powers of two, plus 640x540, 1280x1080 and 2560x2160 `f32` (22,118,400 bytes)                                           |
| Timing | Median of 7 batches per size and path, after a warm-up. Each copy is followed by a synchronize, on an idle stream, into an existing buffer |

## Update, 2026-11-08

`memcpy_htod` in `src/` now reuses its staging buffer, so the tables below, measured on 2026-11-07, are out of date for
`memcpy_htod`. A re-run gave 3.20 ms at 2560x2160 against 5.83 ms below, the same as `memcpy_htod_reused` (3.20 ms)
within
noise. `memcpy_htod_unstaged` is unchanged (2.25 ms). `transfer_sweep.csv` and the plot are the re-run. For `pinned` and
`registered` sources see `transfer_kinds.md`.

## Results (2026-11-07, before the change)

Copy plus synchronize, second of two runs.

| Size                | `memcpy_htod`       | `memcpy_htod_reused` | `memcpy_htod_unstaged` |
|---------------------|---------------------|----------------------|------------------------|
| 4 KiB               | 104.6 µs            | 57.2 µs              | 103.1 µs               |
| 64 KiB              | 104.8 µs            | 64.3 µs              | 107.5 µs               |
| 1 MiB               | 291.8 µs (3.6 GB/s) | 181.6 µs (5.8 GB/s)  | 209.9 µs (5.0 GB/s)    |
| 1280x1080 (5.5 MB)  | 1.33 ms (4.2 GB/s)  | 0.66 ms (8.4 GB/s)   | 0.67 ms (8.3 GB/s)     |
| 16 MiB              | 4.42 ms (3.8 GB/s)  | 2.25 ms (7.5 GB/s)   | 1.74 ms (9.7 GB/s)     |
| 2560x2160 (22.1 MB) | 5.83 ms (3.8 GB/s)  | 3.17 ms (7.0 GB/s)   | 2.26 ms (9.8 GB/s)     |

The 2560x2160 row in the first run was 5.77, 3.35 and 2.26 ms. The full table is in the CSV.

## Observations

- **Run to run, the median difference was 1.8%, the largest 7%.** The 7% cases were around 1 to 1.4 MB. Treat smaller
  differences as noise.
- **Below about 64 KiB the time is fixed cost.** `memcpy_htod` and `memcpy_htod_unstaged` sit at about 100 µs, and
  `memcpy_htod_reused` at 55 to 65 µs. Throughput there is only a few hundred MB/s.
- **At a frame, `memcpy_htod_unstaged` is 2.6x faster than `memcpy_htod` and `memcpy_htod_reused` is 1.8x faster.** At 1
  to 2 MiB, `memcpy_htod_reused` is the fastest. From about 5 MB (1280x1080) it ties with unstaged, and from 8 MiB
  unstaged wins.
- **`memcpy_htod_reused` drops from 8.4 GB/s at 5.5 MB to 7.0 GB/s at a frame.** The cause isn't measured. A guess is
  that the CPU copy into the staging buffer falls out of cache at large sizes, which unstaged avoids.
- **Unstaged is no faster than `memcpy_htod` below 128 KiB.** That matches an earlier one-off 4 KiB measurement.
- `memcpy_htod_unstaged` is the only path that reaches near the GPU copy rate (about 11.8 GB/s in an earlier one-off
  measurement).

## Limits

- Idle stream only. The queued-work case is in `transfer.rs` (`host_paths`).
- One machine, one GPU, one driver, and a unified-memory GPU, so PCIe cards may differ a lot.
- Data was `vec![1.5; n]`, imported from the heap. The unstaged path fails on `static` data and wasn't tried with it.
- The plot was generated but not looked at, because there's no SVG viewer here. Check `transfer_sweep.svg` by eye.
- The bench isn't registered in `Cargo.toml` yet. These runs used a copy of the repo with the
  entry added.

## Reproduce

```text
just bench transfer_sweep   # once the Cargo.toml entry exists
python3 benches/plot_sweep.py
```
