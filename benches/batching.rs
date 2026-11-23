//! Bursts of N launches in cudash itself: a normal stream (`inline`), a batching stream (`batched`,
//! `set_batching(true)`) and one captured graph (`graph`). Kernel `double_all`, n = 0, one workgroup,
//! one buffer argument. Each burst ends with `stream.synchronize()`.
//!
//! Prints `R <variant> <N> <burst_us> <cpu_us>` lines (median over `REPS` bursts: whole burst
//! including the synchronize, and the time inside the N `launch()` calls), then a dependency-chain
//! check. Aggregated by hand into `batching.md`.
//!
//! ```text
//! cargo bench --features experimental --bench batching
//! ```

use std::time::Instant;

use cudash::driver::{sys::*, *};

const KERNEL: &str = concat!(env!("OUT_DIR"), "/double_all.spv");
const NS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const REPS: usize = 300;
const CFG: LaunchConfig = LaunchConfig {
    grid_dim: (1, 1, 1),
    block_dim: (1, 1, 1),
    shared_mem_bytes: 0,
};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), DriverError> {
    let ctx = GpuContext::new(0)?;
    println!("device: {}", ctx.name()?);
    let module = ctx.load_module(Spirv::from_file(KERNEL))?;
    let f = module.load_function("double_all")?;
    let n = 0i32;

    let inline = ctx.new_stream()?;
    let batched = ctx.new_stream()?;
    batched.set_batching(true)?;
    let graph_stream = ctx.new_stream()?;
    let mut bufs: Vec<_> = [&inline, &batched, &graph_stream]
        .iter()
        .map(|s| s.alloc_zeros::<f32>(1))
        .collect::<Result<_, _>>()?;

    // One graph per N, captured once.
    let mut graphs = Vec::new();
    for &count in &NS {
        graph_stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        for _ in 0..count {
            let mut args = graph_stream.launch_builder(&f);
            args.arg(&mut bufs[2]).arg(&n);
            unsafe { args.launch(CFG) }?;
        }
        graphs.push(
            graph_stream
                .end_capture(CUgraphInstantiate_flags(0))?
                .expect("capture invalidated"),
        );
    }

    // `burst[v]` runs one burst of `count` launches and returns (burst_us, cpu_us).
    let mut burst = |v: usize, i: usize, count: usize| -> Result<(f64, f64), DriverError> {
        let stream = [&inline, &batched, &graph_stream][v];
        let start = Instant::now();
        if v == 2 {
            graphs[i].launch()?;
        } else {
            for _ in 0..count {
                let mut args = stream.launch_builder(&f);
                args.arg(&mut bufs[v]).arg(&n);
                unsafe { args.launch(CFG) }?;
            }
        }
        let cpu = start.elapsed();
        stream.synchronize()?;
        Ok((start.elapsed().as_secs_f64() * 1e6, cpu.as_secs_f64() * 1e6))
    };

    for (i, &count) in NS.iter().enumerate() {
        let mut t = [(); 3].map(|_| (Vec::new(), Vec::new()));
        for rep in 0..REPS + 20 {
            // Interleaved, so drift hits all variants alike.
            for (v, samples) in t.iter_mut().enumerate() {
                let (total, cpu) = burst(v, i, count)?;
                if rep >= 20 {
                    samples.0.push(total);
                    samples.1.push(cpu);
                }
            }
        }
        for (v, name) in ["inline", "batched", "graph"].into_iter().enumerate() {
            let (total, cpu) = &mut t[v];
            println!("R {name} {count} {:.2} {:.2}", median(total), median(cpu));
        }
    }

    // Dependency chain: 16 launches on one 1-element buffer, each doubling the previous output.
    for (name, batching) in [("inline", false), ("batched", true)] {
        let s = ctx.new_stream()?;
        s.set_batching(batching)?;
        let mut x = s.clone_htod(&[1.0f32])?;
        let one = 1i32;
        let mut ok = true;
        for _ in 0..50 {
            for _ in 0..16 {
                let mut args = s.launch_builder(&f);
                args.arg(&mut x).arg(&one);
                unsafe { args.launch(CFG) }?;
            }
            ok &= s.clone_dtoh(&x)?[0] == 65536.0;
            s.memcpy_htod(&[1.0f32], &mut x)?;
        }
        println!(
            "CHAIN {name} 16 launches x 50 bursts, each reads the previous output: {}",
            if ok { "correct (65536)" } else { "WRONG" }
        );
    }
    Ok(())
}
