//! Memory transfers on cudash's Vulkan driver, at a 2560x2160 frame.
//!
//! The element type is `f32`, one element per pixel. A frame is 2560 x 2160 = 5,529,600
//! elements, which is 22,118,400 bytes. The sizes are 4 KiB and four frames, for scale.
//!
//! Every operation is followed by a synchronize inside the timed batch, so copies that run
//! asynchronously are counted in full. Allocating operations also include the drop of what
//! they allocated.
//!
//! Runs on any Vulkan 1.3 GPU, NVIDIA or not. Run with:
//!
//! ```text
//! cargo bench --bench transfer
//! ```

#[path = "common/mod.rs"]
mod common;

use cudash::driver::*;

const WIDTH: usize = 2560;
const HEIGHT: usize = 2160;
const FRAME_ELEMS: usize = WIDTH * HEIGHT;
const ELEM_BYTES: usize = std::mem::size_of::<f32>();
const FRAME_BYTES: usize = FRAME_ELEMS * ELEM_BYTES;

/// (label, element count).
const SIZES: [(&str, usize); 3] = [
    ("4 KiB", (4 << 10) / ELEM_BYTES),
    ("1 frame 2560x2160", FRAME_ELEMS),
    ("4 frames", 4 * FRAME_ELEMS),
];

fn iters_for(bytes: usize) -> usize {
    match bytes {
        b if b <= 4 << 10 => 2_000,
        b if b <= FRAME_BYTES => 100,
        _ => 25,
    }
}

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    common::print_header(
        &ctx.name()?,
        &format!(
            "Frame: {WIDTH}x{HEIGHT} f32 = {FRAME_ELEMS} elements = {FRAME_BYTES} bytes. \
             Each row is the median of {} batches.",
            common::SAMPLES
        ),
    );

    for (label, elems) in SIZES {
        let bytes = elems * ELEM_BYTES;
        let iters = iters_for(bytes);
        let host: Vec<f32> = vec![1.5; elems];
        let mut host_out: Vec<f32> = vec![0.0; elems];
        let mut dev = stream.alloc_zeros::<f32>(elems)?;
        let mut dev_other = stream.alloc_zeros::<f32>(elems)?;
        let src = stream.clone_htod(&host[..])?;

        let ns = common::measure_batch(iters, |k| {
            for _ in 0..k {
                let b = stream.alloc_zeros::<f32>(elems).unwrap();
                stream.synchronize().unwrap();
                common::keep(b);
            }
        });
        common::print_row("alloc_zeros", label, ns, Some(bytes));

        let ns = common::measure_batch(iters, |k| {
            for _ in 0..k {
                let b = stream.clone_htod(&host[..]).unwrap();
                stream.synchronize().unwrap();
                common::keep(b);
            }
        });
        common::print_row("clone_htod", label, ns, Some(bytes));

        let ns = common::measure_batch(iters, |k| {
            for _ in 0..k {
                stream.memcpy_htod(&host[..], &mut dev).unwrap();
                stream.synchronize().unwrap();
            }
        });
        common::print_row("memcpy_htod (existing)", label, ns, Some(bytes));

        let ns = common::measure_batch(iters, |k| {
            for _ in 0..k {
                common::keep(stream.clone_dtoh(&src).unwrap());
            }
        });
        common::print_row("clone_dtoh", label, ns, Some(bytes));

        let ns = common::measure_batch(iters, |k| {
            for _ in 0..k {
                stream.memcpy_dtoh(&src, &mut host_out[..]).unwrap();
                stream.synchronize().unwrap();
            }
        });
        common::print_row("memcpy_dtoh (existing)", label, ns, Some(bytes));

        let ns = common::measure_batch(iters, |k| {
            for _ in 0..k {
                let b = stream.clone_dtod(&src).unwrap();
                stream.synchronize().unwrap();
                common::keep(b);
            }
        });
        common::print_row("clone_dtod", label, ns, Some(bytes));

        let ns = common::measure_batch(iters, |k| {
            for _ in 0..k {
                stream.memcpy_dtod(&src, &mut dev_other).unwrap();
                stream.synchronize().unwrap();
            }
        });
        common::print_row("memcpy_dtod (existing)", label, ns, Some(bytes));
    }

    host_paths(&stream)?;
    Ok(())
}

/// Elements in each queued `memcpy_dtod` and the copies queued before a timed call: about
/// 6 x 0.8 ms of GPU time, well over a frame's host copy.
const BUSY_ELEMS: usize = 4 * FRAME_ELEMS;
const BUSY_COPIES: usize = 6;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn fmt(ns: f64) -> String {
    if ns >= 1e6 {
        format!("{:.3} ms", ns / 1e6)
    } else {
        format!("{:.2} µs", ns / 1e3)
    }
}

/// Median over `iters` runs of `f`, in nanoseconds.
fn time_n(
    iters: usize,
    mut f: impl FnMut() -> Result<(), DriverError>,
) -> Result<f64, DriverError> {
    let mut t = Vec::new();
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        f()?;
        t.push(t0.elapsed().as_nanos() as f64);
    }
    Ok(median(&mut t))
}

/// The five host-to-device paths into one existing buffer, idle and behind queued work.
/// Per call it times the call itself and the call plus a synchronize, and reports the
/// median of all calls. Then the one-off costs: register, alloc_pinned, and filling.
fn host_paths(stream: &std::sync::Arc<CudaStream>) -> Result<(), DriverError> {
    use std::time::Instant;
    let ctx = stream.context().clone();
    let a = stream.alloc_zeros::<f32>(BUSY_ELEMS)?;
    let mut b = stream.alloc_zeros::<f32>(BUSY_ELEMS)?;
    println!();
    println!("host paths: call = time until the call returns, total = call plus synchronize");
    println!(
        "{:<28} {:<8} {:<8} {:>12} {:>12}",
        "path", "size", "stream", "call", "total"
    );
    for (label, elems, iters) in [
        ("4 KiB", (4 << 10) / ELEM_BYTES, 500),
        ("frame", FRAME_ELEMS, 50),
    ] {
        let host: Vec<f32> = vec![1.5; elems];
        let reg: Vec<f32> = vec![1.5; elems];
        unsafe { ctx.host_register(&reg[..])? };
        let mut pinned = unsafe { ctx.alloc_pinned::<f32>(elems)? };
        pinned.as_mut_slice()?.fill(1.5);
        let mut dev = stream.alloc_zeros::<f32>(elems)?;
        for busy in [false, true] {
            if busy {
                let t = time_n(iters, || {
                    for _ in 0..BUSY_COPIES {
                        stream.memcpy_dtod(&a, &mut b)?;
                    }
                    stream.synchronize()
                })?;
                common::print_row("queued work alone", label, t, None);
            }
            for path in [
                "htod (Vec, reused staging)",
                "htod (PinnedHostSlice)",
                "htod (registered Vec)",
                "htod_reused (Vec)",
                "htod_unstaged (Vec)",
            ] {
                let mut calls = Vec::new();
                let mut totals = Vec::new();
                for i in 0..iters + 10 {
                    if busy {
                        for _ in 0..BUSY_COPIES {
                            stream.memcpy_dtod(&a, &mut b)?;
                        }
                    }
                    let t0 = Instant::now();
                    match path {
                        "htod (Vec, reused staging)" => stream.memcpy_htod(&host[..], &mut dev)?,
                        "htod (PinnedHostSlice)" => stream.memcpy_htod(&pinned, &mut dev)?,
                        "htod (registered Vec)" => stream.memcpy_htod(&reg[..], &mut dev)?,
                        "htod_reused (Vec)" => stream.memcpy_htod_reused(&host[..], &mut dev)?,
                        _ => stream.memcpy_htod_unstaged(&host[..], &mut dev)?,
                    }
                    let t1 = t0.elapsed();
                    stream.synchronize()?;
                    if i >= 10 {
                        calls.push(t1.as_nanos() as f64);
                        totals.push(t0.elapsed().as_nanos() as f64);
                    }
                }
                println!(
                    "{:<28} {:<8} {:<8} {:>12} {:>12}",
                    path,
                    label,
                    if busy { "queued" } else { "idle" },
                    fmt(median(&mut calls)),
                    fmt(median(&mut totals))
                );
            }
        }
        ctx.host_unregister(reg.as_ptr())?;
    }

    println!();
    println!("one-off costs, one frame (median)");
    let reg: Vec<f32> = vec![1.5; FRAME_ELEMS];
    let t = time_n(30, || {
        unsafe { ctx.host_register(&reg[..])? };
        ctx.host_unregister(reg.as_ptr())
    })?;
    println!("{:<40} {:>12}", "host_register + host_unregister", fmt(t));
    let t = time_n(30, || {
        drop(unsafe { ctx.alloc_pinned::<f32>(FRAME_ELEMS)? });
        Ok(())
    })?;
    println!("{:<40} {:>12}", "alloc_pinned + drop (untouched)", fmt(t));
    let t = time_n(30, || {
        let mut p = unsafe { ctx.alloc_pinned::<f32>(FRAME_ELEMS)? };
        p.as_mut_slice()?.fill(1.5);
        Ok(())
    })?;
    println!("{:<40} {:>12}", "alloc_pinned + first fill + drop", fmt(t));
    let t = time_n(30, || {
        let v: Vec<f32> = vec![1.5; FRAME_ELEMS];
        common::keep(v);
        Ok(())
    })?;
    println!(
        "{:<40} {:>12}",
        "vec![1.5; frame] + drop (reference)",
        fmt(t)
    );

    // Refill of memory that is already mapped and touched.
    let src: Vec<f32> = vec![2.5; FRAME_ELEMS];
    let mut v: Vec<f32> = vec![0.0; FRAME_ELEMS];
    let mut p = unsafe { ctx.alloc_pinned::<f32>(FRAME_ELEMS)? };
    p.as_mut_slice()?.fill(0.0);
    let t = time_n(50, || {
        v.copy_from_slice(&src);
        common::keep(&v);
        Ok(())
    })?;
    println!("{:<40} {:>12}", "refill Vec (copy_from_slice)", fmt(t));
    let t = time_n(50, || {
        p.as_mut_slice()?.copy_from_slice(&src);
        Ok(())
    })?;
    println!(
        "{:<40} {:>12}",
        "refill pinned (idle, copy_from_slice)",
        fmt(t)
    );
    let t = time_n(50, || {
        p.as_mut_slice()?;
        Ok(())
    })?;
    println!("{:<40} {:>12}", "as_mut_slice alone (idle)", fmt(t));
    // After a copy from it: the accessor waits for that copy.
    let mut t = Vec::new();
    for _ in 0..50 {
        stream.memcpy_htod(&p, &mut dev_for(stream)?)?;
        let t0 = Instant::now();
        p.as_mut_slice()?;
        t.push(t0.elapsed().as_nanos() as f64);
    }
    println!(
        "{:<40} {:>12}",
        "as_mut_slice right after a copy",
        fmt(median(&mut t))
    );
    Ok(())
}

fn dev_for(stream: &std::sync::Arc<CudaStream>) -> Result<CudaSlice<f32>, DriverError> {
    stream.alloc_zeros::<f32>(FRAME_ELEMS)
}
