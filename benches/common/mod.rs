//! Timing helpers shared by the benches. Not a bench target itself.
#![allow(dead_code)]

use std::{hint::black_box, time::Instant};

/// Timed batches per case. The reported time is the median batch.
pub const SAMPLES: usize = 7;

/// Median nanoseconds per call, from `SAMPLES` batches of `iters` calls of `op`.
pub fn measure(iters: usize, mut op: impl FnMut()) -> f64 {
    measure_batch(iters, |k| {
        for _ in 0..k {
            op();
        }
    })
}

/// Median nanoseconds per item for a batch closure that does `k` items and then waits for
/// them. The batch closure is timed as a whole, so whatever it waits on is counted too.
pub fn measure_batch(iters: usize, mut batch: impl FnMut(usize)) -> f64 {
    // Warm up, so lazy setup (pipelines, first allocations) doesn't land in batch one.
    batch(iters.min(20));
    let mut per_item: Vec<f64> = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            batch(iters);
            start.elapsed().as_nanos() as f64 / iters as f64
        })
        .collect();
    per_item.sort_by(f64::total_cmp);
    per_item[SAMPLES / 2]
}

/// Keeps a value alive so the optimizer can't drop the work that made it.
pub fn keep<T>(value: T) {
    black_box(value);
}

pub fn print_header(device: &str, extra: &str) {
    println!("device: {device}");
    if !extra.is_empty() {
        println!("{extra}");
    }
    println!();
    println!(
        "{:<34} {:<16} {:>14} {:>12}",
        "operation", "case", "median/op", "throughput"
    );
}

/// One output row. `bytes` gives a throughput column when the operation moves data.
pub fn print_row(operation: &str, case: &str, ns: f64, bytes: Option<usize>) {
    let time = if ns >= 1_000_000.0 {
        format!("{:.3} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} µs", ns / 1_000.0)
    };
    let throughput = match bytes {
        // bytes / ns is GB/s.
        Some(b) => format!("{:.2} GB/s", b as f64 / ns),
        None => String::new(),
    };
    println!("{operation:<34} {case:<16} {time:>14} {throughput:>12}");
}
