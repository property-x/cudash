//! Attacks on stream batching (`experimental` feature): threads, errors in a batch, `device_ptr`,
//! `memcpy_htod_reused`, host memory the GPU reads late. Every GPU wait runs under `within`, so a
//! missing flush fails instead of hanging. Scenarios per flush point are in
//! `features/driver_batching.feature`.
#![cfg(feature = "experimental")]

use std::{
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use cudash::driver::*;

const DOUBLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/double_all.spv"));

fn within<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || drop(tx.send(f())));
    rx.recv_timeout(Duration::from_secs(20))
        .unwrap_or_else(|_| panic!("{what} hung"))
}

fn batching(ctx: &Arc<GpuContext>) -> Arc<GpuStream> {
    let s = ctx.new_stream().unwrap();
    s.set_batching(true).unwrap();
    s
}

fn double_fn(ctx: &Arc<GpuContext>) -> GpuFunction {
    ctx.load_module(Spirv::from_binary(DOUBLE.to_vec()))
        .unwrap()
        .load_function("double_all")
        .unwrap()
}

/// Eight threads share one batching stream, each with its own buffer, mixing copies, launches and
/// reads (reads flush while the others are recording).
#[test]
fn threads_share_one_batching_stream() {
    within("threads", || {
        let ctx = GpuContext::new(0).unwrap();
        let stream = batching(&ctx);
        let f = double_fn(&ctx);
        thread::scope(|scope| {
            for t in 0..8u32 {
                let (stream, f) = (&stream, &f);
                scope.spawn(move || {
                    for round in 0..50u32 {
                        let mut x = stream.clone_htod(&[1.0f32 + t as f32; 64]).unwrap();
                        let n = 64i32;
                        let mut b = stream.launch_builder(f);
                        b.arg(&mut x).arg(&n);
                        unsafe { b.launch(LaunchConfig::for_num_elems(64)) }.unwrap();
                        if round % 7 == 0 {
                            stream.flush().unwrap();
                        }
                        let y = stream.clone_dtoh(&x).unwrap();
                        assert_eq!(y, [2.0 * (1.0 + t as f32); 64], "thread {t} round {round}");
                    }
                });
            }
        });
        ctx.synchronize().unwrap();
    });
}

/// Launch errors come back at once and leave the batch usable, work before and after still runs.
#[test]
fn launch_limit_in_the_middle_of_a_batch() {
    within("launch limit", || {
        let ctx = GpuContext::new(0).unwrap();
        let s = batching(&ctx);
        let f = double_fn(&ctx);
        let mut x = s.clone_htod(&[1.0f32; 8]).unwrap();
        let n = 8i32;
        let launch = |x: &mut GpuSlice<f32>, extra: usize, block: u32| {
            let pad = vec![0u8; extra];
            let mut b = s.launch_builder(&f);
            b.arg(x).arg(&n);
            for byte in &pad {
                b.arg(byte);
            }
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (8, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
        };
        launch(&mut x, 0, 1).unwrap();
        // 300 one-byte arguments overflow the push constants.
        assert!(matches!(
            launch(&mut x, 300, 1),
            Err(DriverError::LaunchLimit(_)) | Err(_)
        ));
        assert!(launch(&mut x, 0, u32::MAX).is_err(), "huge block");
        launch(&mut x, 0, 1).unwrap();
        assert_eq!(s.clone_dtoh(&x).unwrap(), [4.0f32; 8]);
    });
}

/// `device_ptr` marks the use at the stream's latest value, which may still be in the open batch.
/// Another stream that then reads must flush, and the SyncOnDrop must not wedge anything.
#[test]
fn device_ptr_on_a_batching_stream() {
    within("device_ptr", || {
        let ctx = GpuContext::new(0).unwrap();
        let s = batching(&ctx);
        let mut x = s.clone_htod(&[5u32; 16]).unwrap();
        s.memcpy_htod(&[6u32; 16], &mut x).unwrap();
        {
            let (addr, guard) = x.device_ptr(&s);
            assert_ne!(addr, 0);
            drop(guard);
        }
        {
            let (_, guard) = x.device_ptr_mut(&s);
            drop(guard);
        }
        let other = ctx.new_stream().unwrap();
        assert_eq!(other.clone_dtoh(&x).unwrap(), [6u32; 16]);
        s.synchronize().unwrap();
    });
}

/// `memcpy_htod_reused` reuses one staging buffer when the GPU is done with it. In a batch the GPU
/// isn't done, so two calls before a flush must not share the buffer.
#[test]
fn htod_reused_twice_before_a_flush() {
    within("htod_reused", || {
        let ctx = GpuContext::new(0).unwrap();
        let s = batching(&ctx);
        let mut a = s.alloc_zeros::<u32>(1024).unwrap();
        let mut b = s.alloc_zeros::<u32>(1024).unwrap();
        s.memcpy_htod_reused(&vec![1u32; 1024], &mut a).unwrap();
        s.memcpy_htod_reused(&vec![2u32; 1024], &mut b).unwrap();
        s.memcpy_htod_reused(&vec![3u32; 1024], &mut a).unwrap();
        assert_eq!(s.clone_dtoh(&b).unwrap(), [2u32; 1024]);
        assert_eq!(s.clone_dtoh(&a).unwrap(), [3u32; 1024]);
        for i in 0..600u32 {
            s.memcpy_htod_reused(&vec![i; 1024], &mut a).unwrap();
        }
        assert_eq!(s.clone_dtoh(&a).unwrap(), [599u32; 1024]);
    });
}

/// Pinned memory read by the GPU in a batch. Changing it through its accessor flushes first, so the
/// copy sees the old value. Changing it through a raw pointer taken earlier can't be detected, and
/// the copy sees whatever is there when the batch runs. Records, doesn't require, the second.
#[test]
fn pinned_host_memory_changed_before_the_flush() {
    within("pinned", || {
        let ctx = GpuContext::new(0).unwrap();
        let s = batching(&ctx);
        let mut pinned = unsafe { ctx.alloc_pinned::<u32>(64) }.unwrap();
        pinned.as_mut_slice().unwrap().fill(1);
        let raw = pinned.as_mut_ptr().unwrap();
        let mut d1 = s.alloc_zeros::<u32>(64).unwrap();
        s.memcpy_htod(&pinned, &mut d1).unwrap();
        // The accessor waits for the GPU's use of the buffer, which flushes.
        pinned.as_mut_slice().unwrap().fill(2);
        let mut d2 = s.alloc_zeros::<u32>(64).unwrap();
        s.memcpy_htod(&pinned, &mut d2).unwrap();
        // Raw pointer write, no flush.
        unsafe { std::slice::from_raw_parts_mut(raw, 64) }.fill(3);
        let (g1, g2) = (s.clone_dtoh(&d1).unwrap()[0], s.clone_dtoh(&d2).unwrap()[0]);
        assert_eq!(g1, 1, "accessor write must flush the earlier batch first");
        eprintln!(
            "pinned: raw write before flush seen as {g2} (3 = read late, 2 = read at queue time)"
        );
        assert!(g2 == 2 || g2 == 3);
    });
}

/// Registered host memory: the copy reads it when the GPU runs it, per CUDA's rules. With batching
/// that is at the flush, later than without. Records what happens.
#[test]
fn registered_host_memory_changed_before_the_flush() {
    within("registered", || {
        let ctx = GpuContext::new(0).unwrap();
        let s = batching(&ctx);
        let mut host = vec![1u32; 4096];
        if unsafe { ctx.host_register(&host) }.is_err() {
            eprintln!("host_register unsupported here");
            return;
        }
        let mut d = s.alloc_zeros::<u32>(4096).unwrap();
        s.memcpy_htod(&host, &mut d).unwrap();
        host.fill(2); // against the safety contract (must synchronize first); no way to detect
        let got = s.clone_dtoh(&d).unwrap()[0];
        eprintln!("registered: write before flush seen as {got} (2 = read at flush)");
        ctx.host_unregister(host.as_ptr()).unwrap();
    });
}

/// Thousands of buffers created and dropped on a batching stream: memory is only reclaimed after
/// flushes, which the 256-operation limit forces.
#[test]
fn many_short_lived_buffers() {
    within("many buffers", || {
        let ctx = GpuContext::new(0).unwrap();
        let s = batching(&ctx);
        for i in 0..5000u32 {
            let b = s.clone_htod(&[i; 1024]).unwrap();
            let c = s.clone_dtod(&b).unwrap();
            drop((b, c));
        }
        s.synchronize().unwrap();
    });
}

/// Dropping a batching stream while another thread waits on its buffer. FAILS today with DeviceLost:
/// `todo/batching-drop-race.md`.
#[test]
fn drop_stream_while_another_thread_reads() {
    within("drop vs read", || {
        let ctx = GpuContext::new(0).unwrap();
        let owner = ctx.new_stream().unwrap();
        for _ in 0..50 {
            let mut x = owner.alloc_zeros::<u32>(256).unwrap();
            let s = batching(&ctx);
            s.memcpy_htod(&[9u32; 256], &mut x).unwrap();
            thread::scope(|scope| {
                let r = scope.spawn(|| owner.clone_dtoh(&x).unwrap());
                drop(s);
                assert_eq!(r.join().unwrap(), [9u32; 256]);
            });
        }
    });
}
