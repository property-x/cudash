//! CPU cost of one plain cudash launch: `launch_builder(&f).arg(..).launch(cfg)` in a loop on
//! one stream, as a user writes it. `double_all` with `n = 0`, one workgroup, one buffer arg.
//!
//! For each N: N launches back to back, one synchronize at the end. Reports
//! - `in launch`: time spent from before the first launch to after the last one, per launch
//!   (what the caller blocks on),
//! - `total`: the same plus the final synchronize, per launch,
//! - `launch+sync`: one launch followed by a synchronize, per call (N = 1 loops).
//!
//! Each case is `REPS` repetitions; prints median and min..max. Run with:
//!
//! ```text
//! cargo bench --bench launch_cost
//! ```
//!
//! The breakdown inside `launch()` needs a patched `src/`, see `benches/launch_cost.md`.

use std::time::Instant;

use cudash::driver::*;

const KERNEL: &str = concat!(env!("OUT_DIR"), "/double_all.spv");
const REPS: usize = 9;

fn stats(mut v: Vec<f64>) -> String {
    v.sort_by(f64::total_cmp);
    format!(
        "{:7.2}  ({:.2}..{:.2})",
        v[v.len() / 2],
        v[0],
        v[v.len() - 1]
    )
}

fn main() -> Result<(), DriverError> {
    let ctx = GpuContext::new(0)?;
    let stream = ctx.default_stream();
    let f = ctx
        .load_module(Spirv::from_file(KERNEL))?
        .load_function("double_all")?;
    let mut buf = stream.alloc_zeros::<f32>(1)?;
    let n = 0i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = || unsafe { stream.launch_builder(&f).arg(&mut buf).arg(&n).launch(cfg) };

    println!("device: {}", ctx.name()?);
    println!("per launch in µs, median (min..max) of {REPS} reps\n");
    println!("{:>6} {:>26} {:>26}", "N", "in launch", "total (with sync)");
    for count in [16usize, 128, 1024, 10_000] {
        let (mut inside, mut total) = (vec![], vec![]);
        for rep in 0..REPS + 2 {
            let start = Instant::now();
            for _ in 0..count {
                launch()?;
            }
            let launched = start.elapsed();
            stream.synchronize()?;
            let done = start.elapsed();
            if rep >= 2 {
                inside.push(launched.as_nanos() as f64 / count as f64 / 1e3);
                total.push(done.as_nanos() as f64 / count as f64 / 1e3);
            }
        }
        println!("{count:>6} {:>26} {:>26}", stats(inside), stats(total));
    }

    let mut round = vec![];
    for rep in 0..REPS + 2 {
        let start = Instant::now();
        for _ in 0..500 {
            launch()?;
            stream.synchronize()?;
        }
        if rep >= 2 {
            round.push(start.elapsed().as_nanos() as f64 / 500.0 / 1e3);
        }
    }
    println!("\nlaunch+sync per call: {}", stats(round));
    Ok(())
}
