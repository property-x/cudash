//! Launching kernels on shared buffers with real cudarc: scalar and pointer arguments, grid
//! sizes, views and ordering on one stream.
//! These need an NVIDIA GPU and print SKIPPED (or panic, with CUDASH_REQUIRE_NVIDIA set).

mod common;

use common::*;
use cudash::interop::cudarc::{
    driver::{LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

#[test]
fn float_scalar_argument() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let (ctx, interop) = (&gpu.cuda, &gpu.interop);
    let stream = interop.stream();
    let f = load_kernel(ctx, AXPY_CU, "axpy")?;

    let n = 1000usize;
    let x_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = buf_from(interop, &x_host)?;
    let mut y = stream.clone_htod(&vec![1.0f32; n])?;
    let a = 2.5f32;
    let n_arg = n as i32;

    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut y);
    builder.arg(&x);
    builder.arg(&a);
    builder.arg(&n_arg);
    unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }?;

    let want: Vec<f32> = x_host.iter().map(|x| 2.5 * x + 1.0).collect();
    assert_eq!(stream.clone_dtoh(&y)?, want);
    Ok(())
}

#[test]
fn i64_scalar_argument_with_negative_values() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let (ctx, interop) = (&gpu.cuda, &gpu.interop);
    let stream = interop.stream();
    let f = load_kernel(ctx, ADD_I64_CU, "add_i64")?;

    let mut data = buf_from(interop, &[-5i64, 0, 7])?;
    let k = -10i64;
    let n = 3i32;

    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut data);
    builder.arg(&k);
    builder.arg(&n);
    unsafe { builder.launch(LaunchConfig::for_num_elems(3)) }?;

    assert_eq!(stream.clone_dtoh(&data)?, [-15i64, -10, -3]);
    Ok(())
}

/// One thread per element across many blocks. Each element must hold its own index.
#[test]
fn launch_covers_a_million_elements() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let (ctx, interop) = (&gpu.cuda, &gpu.interop);
    let stream = interop.stream();
    let f = load_kernel(ctx, FILL_INDEX_CU, "fill_index")?;

    let n = 1_000_000usize;
    let mut out = buf::<u64>(interop, n)?;
    let n_arg = n as i32;

    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut out);
    builder.arg(&n_arg);
    unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }?;

    let got = stream.clone_dtoh(&out)?;
    let wrong = got
        .iter()
        .enumerate()
        .filter(|&(i, &v)| v != i as u64)
        .count();
    assert_eq!(
        wrong, 0,
        "{wrong} of {n} elements weren't written by their own thread"
    );
    Ok(())
}

/// An explicit grid and block size. The grid has 1024 threads for 1000 elements,
/// so the kernel's bounds check has to stop the extra 24.
#[test]
fn explicit_launch_config_with_bounds_check() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let (ctx, interop) = (&gpu.cuda, &gpu.interop);
    let stream = interop.stream();
    let f = load_kernel(ctx, FILL_INDEX_CU, "fill_index")?;

    let n = 1000usize;
    let mut out = buf::<u64>(interop, n)?;
    let n_arg = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (4, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut out);
    builder.arg(&n_arg);
    unsafe { builder.launch(cfg) }?;

    let want: Vec<u64> = (0..n as u64).collect();
    assert_eq!(stream.clone_dtoh(&out)?, want);
    Ok(())
}

/// A kernel gets a view of part of a buffer. Only the view's elements change.
#[test]
fn launch_on_a_mutable_view() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let (ctx, interop) = (&gpu.cuda, &gpu.interop);
    let stream = interop.stream();
    let f = load_kernel(ctx, DOUBLE_CU, "double_in_place")?;

    let mut buf = buf_from(interop, &[1.0f32, 2.0, 3.0, 4.0, 5.0])?;
    let n = 3i32;
    {
        let mut view = buf.slice_mut(1..4);
        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut view);
        builder.arg(&n);
        unsafe { builder.launch(LaunchConfig::for_num_elems(3)) }?;
    }

    assert_eq!(stream.clone_dtoh(&buf)?, [1.0, 4.0, 6.0, 8.0, 5.0]);
    Ok(())
}

/// Launches on one stream run in order, so three doublings give eight times the input.
#[test]
fn consecutive_launches_run_in_order() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let (ctx, interop) = (&gpu.cuda, &gpu.interop);
    let stream = interop.stream();
    let f = load_kernel(ctx, DOUBLE_CU, "double_in_place")?;

    let mut data = buf_from(interop, &[1.0f32, 2.0])?;
    let n = 2i32;
    for _ in 0..3 {
        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut data);
        builder.arg(&n);
        unsafe { builder.launch(LaunchConfig::for_num_elems(2)) }?;
    }

    assert_eq!(stream.clone_dtoh(&data)?, [8.0, 16.0]);
    Ok(())
}

#[test]
fn missing_kernel_name_is_an_error() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let ctx = &gpu.cuda;
    assert!(load_kernel(ctx, SIN_CU, "no_such_kernel").is_err());
    Ok(())
}

#[test]
fn invalid_cuda_source_fails_to_compile() {
    let Some(_gpu) = nvidia() else {
        return;
    };
    assert!(compile_ptx("this is not CUDA C").is_err());
}
