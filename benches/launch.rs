//! Kernel launch overhead on cudash's Vulkan driver.
//!
//! The kernel is `tests/kernels/double_all.slang` with `n = 0`, so its body does nothing and
//! the timing is almost all launch cost: building the arguments, recording a command buffer,
//! submitting it, and later reclaiming it.
//!
//! Cases:
//! - `async`: N launches queued back to back, then one synchronize. Per launch = total / N.
//!   This is the cost a caller pays when it doesn't wait for the GPU.
//! - `+ sync`: launch then synchronize, every time. This is the round trip.
//! - 1 and 4096 workgroups, to see how the dispatch size changes the cost.
//! - `first launch (cold)`: the first launch of the function, which also creates its pipeline.
//!   It's a single sample, so it's rough.
//!
//! Runs on any Vulkan 1.3 GPU, NVIDIA or not. Run with:
//!
//! ```text
//! cargo bench --bench launch
//! ```

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use cudash::driver::*;

/// Compiled from `tests/kernels/double_all.slang` by `build.rs`.
const KERNEL: &str = concat!(env!("OUT_DIR"), "/double_all.spv");

/// Batch sizes. Async gets more launches because each one is cheap to queue.
const ASYNC_LAUNCHES: usize = 2_000;
const ROUND_TRIPS: usize = 500;

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Spirv::from_file(KERNEL))?;
    let f = module.load_function("double_all")?;

    // One element, and n = 0, so the kernel body never runs.
    let mut buf = stream.alloc_zeros::<f32>(1)?;
    let n = 0i32;

    let one = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let many = LaunchConfig {
        grid_dim: (4096, 1, 1),
        ..one
    };

    let mut launch = |cfg: LaunchConfig| -> Result<(), DriverError> {
        let mut args = stream.launch_builder(&f);
        args.arg(&mut buf).arg(&n);
        unsafe { args.launch(cfg) }
    };

    // The first launch builds the pipeline, so time it on its own, before the warm-up.
    let start = Instant::now();
    launch(one)?;
    stream.synchronize()?;
    let cold_ns = start.elapsed().as_nanos() as f64;

    common::print_header(
        &ctx.name()?,
        &format!(
            "Each row is the median of {} batches. A single synchronize ends every async batch,\n\
             so its cost is spread across the launches.",
            common::SAMPLES
        ),
    );

    common::print_row("first launch (cold)", "1 workgroup", cold_ns, None);

    let ns = common::measure_batch(ASYNC_LAUNCHES, |k| {
        for _ in 0..k {
            launch(one).unwrap();
        }
        stream.synchronize().unwrap();
    });
    common::print_row("launch, async", "1 workgroup", ns, None);

    let ns = common::measure_batch(ROUND_TRIPS, |k| {
        for _ in 0..k {
            launch(one).unwrap();
            stream.synchronize().unwrap();
        }
    });
    common::print_row("launch + sync", "1 workgroup", ns, None);

    let ns = common::measure_batch(ASYNC_LAUNCHES, |k| {
        for _ in 0..k {
            launch(many).unwrap();
        }
        stream.synchronize().unwrap();
    });
    common::print_row("launch, async", "4096 workgroups", ns, None);

    let ns = common::measure_batch(ROUND_TRIPS, |k| {
        for _ in 0..k {
            launch(many).unwrap();
            stream.synchronize().unwrap();
        }
    });
    common::print_row("launch + sync", "4096 workgroups", ns, None);

    Ok(())
}
