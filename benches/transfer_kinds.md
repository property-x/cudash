# Host-to-device copy by kind of host memory

> **AI-generated summary** of the benchmark results. The numbers come from real runs, but the observations are
> interpretation and can be wrong. Check them against the CSV or a fresh run before relying on them.

Measured with `benches/transfer_kinds.rs`. Data: `transfer_kinds.csv`. Plot: `transfer_kinds.svg` (from
`plot_sweep.py`).

## Setup

|              |                                                                                                  |
|--------------|--------------------------------------------------------------------------------------------------|
| GPU          | AMD Steam Machine (RADV NAVI33), Mesa 26.0.8, kernel 7.2.7-valve2                                |
| Date         | 2026-11-07, no commit (nothing is committed yet)                                                 |
| Operation    | `memcpy_htod` into an existing device buffer, then `synchronize`, on an idle stream              |
| `vec`        | `Vec<f32>`, copied into the stream's reused staging buffer first                                 |
| `pinned`     | `PinnedHostSlice<f32>` from `alloc_pinned`, read by the GPU directly                             |
| `registered` | `Vec<f32>` registered once with `host_register`, read by the GPU directly                        |
| Sizes        | 4 KiB to 16 MiB in powers of two, plus 640x540, 1280x1080 and 2560x2160 `f32` (22,118,400 bytes) |
| Timing       | 200 individually timed copies per point, after 10 warm-up copies. The table shows the median     |

## Results

Median per copy, from the first of two runs. The CSV also has the minimum and the 10th and 90th percentile of the 200
copies.

| Size                | `vec`               | `pinned`            | `registered`        |
|---------------------|---------------------|---------------------|---------------------|
| 4 KiB               | 50.0 µs (0.08 GB/s) | 48.9 µs (0.08 GB/s) | 50.8 µs (0.08 GB/s) |
| 64 KiB              | 60.4 µs (1.09 GB/s) | 58.1 µs (1.13 GB/s) | 52.6 µs (1.25 GB/s) |
| 256 KiB             | 84.0 µs (3.1 GB/s)  | 78.7 µs (3.3 GB/s)  | 76.0 µs (3.5 GB/s)  |
| 1 MiB               | 183 µs (5.7 GB/s)   | 132 µs (7.9 GB/s)   | 136 µs (7.7 GB/s)   |
| 640x540 (1.4 MB)    | 222 µs (6.2 GB/s)   | 168 µs (8.2 GB/s)   | 170 µs (8.1 GB/s)   |
| 1280x1080 (5.5 MB)  | 659 µs (8.4 GB/s)   | 483 µs (11.5 GB/s)  | 484 µs (11.4 GB/s)  |
| 16 MiB              | 2.19 ms (7.7 GB/s)  | 1.29 ms (13.0 GB/s) | 1.28 ms (13.2 GB/s) |
| 2560x2160 (22.1 MB) | 3.33 ms (6.6 GB/s)  | 1.67 ms (13.2 GB/s) | 1.67 ms (13.3 GB/s) |

All 16 sizes are in the CSV and the plot.

## Observations

- **`pinned` and `registered` are the same speed at every size**, within about 3% in the first run. A registered `Vec`
  costs nothing extra per copy once it's registered.
- **At a frame both are 2.0x faster than `vec`** (1.67 ms against 3.33 ms), and reach 13.2 GB/s against 6.6 GB/s.
- **Below about 64 KiB all three are the same**, at 50 to 60 µs. That's fixed cost, so the kind of memory doesn't matter
  there.
- **The gap opens from about 256 KiB** and is 1.4x at 1 MiB.
- **`vec` falls from 8.4 GB/s at 5.5 MB to 6.6 GB/s at a frame**, while the other two keep rising. The cause isn't
  measured. A guess is the CPU copy into the staging buffer leaving the cache.
- **Spread within a run is small at large sizes.** At a frame the 10th to 90th percentile is 3.06 to 3.55 ms for `vec`
  and 1.66 to 1.80 ms for `pinned`.
- **Compared with `transfer_sweep.md`, `vec` is faster than the `memcpy_htod` figure there** (3.3 ms against 5.8 ms at a
  frame). `memcpy_htod` in `src/` now reuses its staging buffer, so it matches the old `memcpy_htod_reused` row (3.2
  ms). `transfer_sweep.md` has since been re-run and updated.

## Variance

A second full run differed from the first by 12% at the median point, and by up to 38% (`pinned` at 16 KiB, a small copy
where 50 µs jitter is large). At 4 KiB all three read 60 to 64 µs (large sizes, such as 16 MiB, differed by under 4%) in
the second run against 49 to 51 µs in the first, so the whole machine got slower, not one kind. The ordering and the
ratios at a frame held (second run: 3.34, 1.67 and 1.67 ms). Differences between kinds under about 15% at small sizes
are noise.

## Limits

- Idle stream only, with one copy in flight.
- One machine, one GPU, one driver, and a unified-memory GPU. A PCIe card may differ a lot, and 13 GB/s here isn't a
  link rate.
- Registration happens once per size, outside the timing. Its cost isn't measured, and neither is `alloc_pinned`.
- The plot was generated and parses as valid SVG, but I couldn't look at it.
- The bench isn't registered in `Cargo.toml` yet. The runs used a copy of the repo with the entry
  added.

## Reproduce

```text
just bench transfer_kinds   # once the Cargo.toml entry exists
python3 benches/plot_sweep.py transfer_kinds
```
