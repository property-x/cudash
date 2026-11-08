//! Host-to-device `memcpy_htod` from three kinds of host memory, at the sizes of
//! `transfer_sweep` (4 KiB up to a 2560x2160 `f32` frame):
//!
//! - `vec`: a plain `Vec<f32>`, copied through the stream's staging buffer.
//! - `pinned`: a `PinnedHostSlice<f32>` from `alloc_pinned`, which the GPU reads directly.
//! - `registered`: a `Vec<f32>` registered once with `host_register`, also read directly.
//!
//! Every copy is timed on its own, including a synchronize, on an idle stream into an existing
//! buffer. Each point is `RUNS` timed copies after a warm-up. `benches/transfer_kinds.csv` gets
//! the median, minimum, 10th and 90th percentile of those. Plot with `benches/plot_sweep.py`.
//!
//! ```text
//! cargo bench --bench transfer_kinds
//! python3 benches/plot_sweep.py transfer_kinds
//! ```

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use cudash::driver::*;

const ELEM_BYTES: usize = std::mem::size_of::<f32>();
/// Timed copies per point. At least 100 were asked for.
const RUNS: usize = 200;
const WARMUP: usize = 10;

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
        &format!("Each row is the median of {RUNS} individually timed copies."),
    );

    let mut csv = format!("# device: {device}\npath,bytes,label,ns,min,p10,p90\n");
    for (label, bytes) in sizes() {
        let elems = bytes / ELEM_BYTES;
        let vec: Vec<f32> = vec![1.5; elems];
        let registered: Vec<f32> = vec![1.5; elems];
        // SAFETY: `registered` isn't touched or moved until it's unregistered below.
        let reg = unsafe { ctx.host_register(&registered[..]) };
        let mut pinned = unsafe { ctx.alloc_pinned::<f32>(elems)? };
        pinned.as_mut_slice()?.fill(1.5);
        let mut dev = stream.alloc_zeros::<f32>(elems)?;

        for kind in ["vec", "pinned", "registered"] {
            if kind == "registered"
                && let Err(e) = &reg
            {
                eprintln!("registered {label}: {e}");
                continue;
            }
            let mut ns: Vec<f64> = Vec::with_capacity(RUNS);
            for i in 0..WARMUP + RUNS {
                let t0 = Instant::now();
                match kind {
                    "vec" => stream.memcpy_htod(&vec[..], &mut dev)?,
                    "pinned" => stream.memcpy_htod(&pinned, &mut dev)?,
                    _ => stream.memcpy_htod(&registered[..], &mut dev)?,
                }
                stream.synchronize()?;
                if i >= WARMUP {
                    ns.push(t0.elapsed().as_nanos() as f64);
                }
            }
            ns.sort_by(f64::total_cmp);
            let at = |q: f64| ns[((RUNS - 1) as f64 * q) as usize];
            let median = at(0.5);
            common::print_row(kind, &label, median, Some(bytes));
            csv += &format!(
                "{kind},{bytes},{label},{median:.0},{:.0},{:.0},{:.0}\n",
                ns[0],
                at(0.1),
                at(0.9)
            );
        }
        if reg.is_ok() {
            ctx.host_unregister(registered.as_ptr())?;
        }
    }
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/transfer_kinds.csv");
    std::fs::write(out, csv).expect("write csv");
    Ok(())
}
