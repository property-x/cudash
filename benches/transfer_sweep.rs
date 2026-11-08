//! Host-to-device copy time against size, for `memcpy_htod`, `memcpy_htod_reused` and
//! `memcpy_htod_unstaged`, from 4 KiB up to a 2560x2160 `f32` frame.
//!
//! Each timed copy is followed by a synchronize, on an otherwise idle stream. The destination
//! exists already, so allocation isn't counted. Besides the table on stdout, the bench writes
//! `benches/transfer_sweep.csv`, which `benches/plot_sweep.py` turns into an SVG.
//!
//! Runs on any Vulkan 1.3 GPU. Run with:
//!
//! ```text
//! cargo bench --bench transfer_sweep
//! python3 benches/plot_sweep.py
//! ```

#[path = "common/mod.rs"]
mod common;

use cudash::driver::*;

const ELEM_BYTES: usize = std::mem::size_of::<f32>();
const PATHS: [&str; 3] = ["memcpy_htod", "memcpy_htod_reused", "memcpy_htod_unstaged"];

/// (label, bytes): powers of two from 4 KiB to 16 MiB, then three image sizes of `f32`.
fn sizes() -> Vec<(String, usize)> {
    let mut v: Vec<(String, usize)> = (12..=24)
        .map(|p| (format!("{} KiB", (1usize << p) >> 10), 1usize << p))
        .collect();
    for (w, h) in [(640, 540), (1280, 1080), (2560, 2160)] {
        v.push((format!("{w}x{h}"), w * h * ELEM_BYTES));
    }
    v.sort_by_key(|&(_, b)| b);
    v
}

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let device = ctx.name()?;
    common::print_header(
        &device,
        &format!("Each row is the median of {} batches.", common::SAMPLES),
    );

    let mut csv = format!("# device: {device}\npath,bytes,label,ns\n");
    for (label, bytes) in sizes() {
        let elems = bytes / ELEM_BYTES;
        let host: Vec<f32> = vec![1.5; elems];
        let mut dev = stream.alloc_zeros::<f32>(elems)?;
        let iters = ((64 << 20) / bytes).clamp(20, 1000);
        for path in PATHS {
            let mut copy = || match path {
                "memcpy_htod" => stream.memcpy_htod(&host[..], &mut dev),
                "memcpy_htod_reused" => stream.memcpy_htod_reused(&host[..], &mut dev),
                _ => stream.memcpy_htod_unstaged(&host[..], &mut dev),
            };
            // Unstaged can refuse to import; skip the row rather than abort the sweep.
            if let Err(e) = copy() {
                eprintln!("{path} {label}: {e}");
                continue;
            }
            let ns = common::measure_batch(iters, |k| {
                for _ in 0..k {
                    copy().unwrap();
                    stream.synchronize().unwrap();
                }
            });
            common::print_row(path, &label, ns, Some(bytes));
            csv += &format!("{path},{bytes},{label},{ns:.0}\n");
        }
    }
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/transfer_sweep.csv");
    std::fs::write(out, csv).expect("write csv");
    Ok(())
}
