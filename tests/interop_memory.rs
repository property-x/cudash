//! Lending cudash slices to cudarc: what a lent slice looks like, the errors, ordering against
//! cudash work, and repeated lends.
//! These need an NVIDIA GPU and print SKIPPED (or panic, with CUDASH_REQUIRE_NVIDIA set).

mod common;

use std::sync::Arc;

use cudash::{
    driver::{self, CudaContext, Spirv},
    interop::{
        AsCuda, Error,
        cudarc::driver::{DeviceSlice, LaunchConfig, PushKernelArg},
    },
};

use common::*;

const DOUBLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/double_all.spv"));

#[test]
fn lent_slices_read_as_zeros_for_several_types() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let stream = &gpu.stream;

    let bytes = stream.alloc_zeros::<u8>(17)?;
    let lent = bytes.as_cuda()?;
    assert_eq!(lent.stream().clone_dtoh(&lent)?, vec![0u8; 17]);
    let wide = stream.alloc_zeros::<u64>(9)?;
    let lent = wide.as_cuda()?;
    assert_eq!(lent.stream().clone_dtoh(&lent)?, vec![0u64; 9]);
    let floats = stream.alloc_zeros::<f64>(5)?;
    let lent = floats.as_cuda()?;
    assert_eq!(lent.stream().clone_dtoh(&lent)?, vec![0.0f64; 5]);
    let signed = stream.alloc_zeros::<i32>(3)?;
    let lent = signed.as_cuda()?;
    assert_eq!(lent.stream().clone_dtoh(&lent)?, vec![0i32; 3]);
    Ok(())
}

#[test]
fn lending_an_empty_slice_is_an_error() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let empty = gpu.stream.alloc_zeros::<u32>(0)?;
    assert!(matches!(empty.as_cuda(), Err(Error::InvalidLength)));
    let mut empty = empty;
    assert!(matches!(empty.as_cuda_mut(), Err(Error::InvalidLength)));
    Ok(())
}

/// Lending on a CUDA stream from a different CUDA context. Two cudash contexts on one GPU
/// are separate Vulkan devices with the same UUID, so `as_cuda_with` accepts them: this
/// checks that and that the data is still readable. A stream on another physical GPU can
/// only be tested with two NVIDIA GPUs (`@untested`).
#[test]
fn lending_on_a_stream_of_a_second_cudash_context_works() -> cudash::interop::Result<()> {
    let Some(_gpu) = nvidia() else {
        return Ok(());
    };
    let other = CudaContext::new(0)?;
    let mut buf = other.default_stream().clone_htod(&[1u32, 2, 3, 4])?;
    let cuda = cudash::interop::cuda_context_for(&other)?.new_stream()?;
    assert_eq!(cuda.clone_dtoh(&buf.as_cuda_with(&cuda)?)?, [1, 2, 3, 4]);
    assert_eq!(
        cuda.clone_dtoh(&buf.as_cuda_mut_with(&cuda)?)?,
        [1, 2, 3, 4]
    );
    Ok(())
}

#[test]
fn a_lent_slice_has_the_slices_length_and_size() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let slice = gpu.stream.alloc_zeros::<u16>(10)?;
    let lent = slice.as_cuda()?;
    assert_eq!(lent.len(), 10);
    assert_eq!(lent.num_bytes(), 20);
    Ok(())
}

/// A stream the caller made works with `as_cuda_with`, and isn't the paired one.
#[test]
fn lending_on_a_caller_chosen_stream() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let mine = gpu.cuda.new_stream()?;
    let slice = gpu.stream.clone_htod(&[7u32])?;
    let lent = slice.as_cuda_with(&mine)?;
    assert!(Arc::ptr_eq(lent.stream(), &mine));
    assert_eq!(mine.clone_dtoh(&lent)?, [7]);
    drop(lent);
    let mut slice = slice;
    let lent = slice.as_cuda_mut_with(&mine)?;
    assert!(Arc::ptr_eq(lent.stream(), &mine));
    Ok(())
}

/// Lends from one cudash stream share a CUDA stream (even across slices), lends from
/// another cudash stream get a different one, and none is cudarc's null stream.
#[test]
fn paired_streams_are_one_to_one_and_not_the_default() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let other_stream = gpu.stream.fork()?;
    let a = gpu.stream.alloc_zeros::<u32>(4)?;
    let b = gpu.stream.alloc_zeros::<u32>(4)?;
    let c = other_stream.alloc_zeros::<u32>(4)?;
    let (la, lb, lc) = (a.as_cuda()?, b.as_cuda()?, c.as_cuda()?);
    let paired = la.stream().clone();
    assert!(Arc::ptr_eq(la.stream(), lb.stream()));
    assert!(!Arc::ptr_eq(la.stream(), lc.stream()));
    let default = gpu.cuda.default_stream();
    for lent in [&la, &lb, &lc] {
        assert!(!Arc::ptr_eq(lent.stream(), &default));
        assert!(!lent.stream().cu_stream().is_null());
    }
    drop((la, lb, lc));
    // Still the same stream after the lends ended.
    let again = a.as_cuda()?;
    assert!(Arc::ptr_eq(again.stream(), &paired));
    Ok(())
}

/// cudash work queued on the slice has to finish before CUDA reads it.
#[test]
fn lending_waits_for_queued_cudash_kernels() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let stream = &gpu.stream;
    let module = gpu.ctx.load_module(Spirv::from_binary(DOUBLE))?;
    let f = module.load_function("double_all")?;

    let n = 1 << 20;
    let mut data = stream.clone_htod(&vec![1.0f32; n])?;
    let n_arg = n as i32;
    for _ in 0..10 {
        let mut builder = stream.launch_builder(&f);
        driver::PushKernelArg::arg(&mut builder, &mut data);
        driver::PushKernelArg::arg(&mut builder, &n_arg);
        unsafe { builder.launch(driver::LaunchConfig::for_num_elems(n as u32)) }?;
    }

    let lent = data.as_cuda()?;
    let got = lent.stream().clone_dtoh(&lent)?;
    assert!(got.iter().all(|&v| v == 1024.0), "CUDA saw unfinished work");
    Ok(())
}

/// The reverse: CUDA work on the paired stream has to finish before the borrow ends, so
/// cudash sees it with no further synchronization.
#[test]
fn ending_a_lend_waits_for_queued_cuda_work() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let stream = &gpu.stream;
    let f = load_kernel(&gpu.cuda, FILL_INDEX_CU, "fill_index")?;
    let n = 1 << 22;

    let mut out = stream.alloc_zeros::<u64>(n)?;
    {
        let mut lent = out.as_cuda_mut()?;
        let n_arg = n as i32;
        let cuda = lent.stream().clone();
        let mut builder = cuda.launch_builder(&f);
        builder.arg(&mut lent).arg(&n_arg);
        unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }?;
    }
    let got = stream.clone_dtoh(&out)?;
    assert!(got.iter().enumerate().all(|(i, &v)| v == i as u64));
    Ok(())
}

/// Lending and dropping in a loop shouldn't fail or hang. This doesn't detect leaks; that
/// needs an external check (e.g. watching `nvidia-smi` or the fd count).
#[test]
fn lend_and_drop_many_times() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let stream = &gpu.stream;

    let slice = stream.clone_htod(&vec![3u8; 64 * 1024])?;
    for _ in 0..500 {
        drop(slice.as_cuda()?);
    }

    // Many live lends of different slices at once, each 64 KiB.
    let slices: Vec<_> = (0..16)
        .map(|_| stream.clone_htod(&vec![3u8; 64 * 1024]))
        .collect::<Result<_, _>>()?;
    let lent: Vec<_> = slices
        .iter()
        .map(|s| s.as_cuda())
        .collect::<Result<_, _>>()?;
    let last = lent.last().expect("16 slices were lent");
    assert_eq!(last.stream().clone_dtoh(last)?, vec![3u8; 64 * 1024]);
    Ok(())
}
