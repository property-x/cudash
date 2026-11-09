//! Lending cudash slices to cudarc: what a lent slice looks like, the errors, ordering against
//! cudash work, and repeated lends.
//! These need an NVIDIA GPU and print SKIPPED (or panic, with CUDASH_REQUIRE_NVIDIA set).

mod common;

use cudash::{
    driver::{self, CudaContext, Spirv},
    interop::{
        Error, Interop,
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
    let (interop, stream) = (&gpu.interop, &gpu.stream);
    let cuda = interop.stream();

    let bytes = stream.alloc_zeros::<u8>(17)?;
    assert_eq!(cuda.clone_dtoh(&interop.lend(&bytes)?)?, vec![0u8; 17]);
    let wide = stream.alloc_zeros::<u64>(9)?;
    assert_eq!(cuda.clone_dtoh(&interop.lend(&wide)?)?, vec![0u64; 9]);
    let floats = stream.alloc_zeros::<f64>(5)?;
    assert_eq!(cuda.clone_dtoh(&interop.lend(&floats)?)?, vec![0.0f64; 5]);
    let signed = stream.alloc_zeros::<i32>(3)?;
    assert_eq!(cuda.clone_dtoh(&interop.lend(&signed)?)?, vec![0i32; 3]);
    Ok(())
}

#[test]
fn lending_an_empty_slice_is_an_error() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let empty = gpu.stream.alloc_zeros::<u32>(0)?;
    assert!(matches!(
        gpu.interop.lend(&empty),
        Err(Error::InvalidLength)
    ));
    Ok(())
}

/// The slice came from another cudash context, so it is on another Vulkan device.
#[test]
fn lending_a_slice_from_another_context_is_an_error() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let other = CudaContext::new(0)?;
    let mut foreign = other.default_stream().alloc_zeros::<u32>(4)?;
    assert!(matches!(
        gpu.interop.lend(&foreign),
        Err(Error::DeviceMismatch)
    ));
    assert!(matches!(
        gpu.interop.lend_mut(&mut foreign),
        Err(Error::DeviceMismatch)
    ));
    Ok(())
}

#[test]
fn a_lent_slice_has_the_slices_length_and_size() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let slice = gpu.stream.alloc_zeros::<u16>(10)?;
    let lent = gpu.interop.lend(&slice)?;
    assert_eq!(lent.len(), 10);
    assert_eq!(lent.num_bytes(), 20);
    Ok(())
}

/// Two `Interop`s on one context lend the same slice independently.
#[test]
fn interop_can_be_created_twice_for_one_context() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let second = Interop::new(&gpu.ctx, gpu.cuda.default_stream())?;
    let slice = gpu.stream.clone_htod(&[7u32])?;
    assert_eq!(second.stream().clone_dtoh(&second.lend(&slice)?)?, [7]);
    Ok(())
}

/// cudash work queued on the slice has to finish before CUDA reads it.
#[test]
fn lending_waits_for_queued_cudash_kernels() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let (interop, stream) = (&gpu.interop, &gpu.stream);
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

    let lent = interop.lend(&data)?;
    let got = interop.stream().clone_dtoh(&lent)?;
    assert!(got.iter().all(|&v| v == 1024.0), "CUDA saw unfinished work");
    Ok(())
}

/// The reverse: CUDA work on the interop stream has to finish before the borrow ends, so
/// cudash sees it with no further synchronization.
#[test]
fn ending_a_lend_waits_for_queued_cuda_work() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let (interop, stream) = (&gpu.interop, &gpu.stream);
    let f = load_kernel(&gpu.cuda, FILL_INDEX_CU, "fill_index")?;
    let n = 1 << 22;

    let mut out = stream.alloc_zeros::<u64>(n)?;
    {
        let mut lent = interop.lend_mut(&mut out)?;
        let n_arg = n as i32;
        let mut builder = interop.stream().launch_builder(&f);
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
    let (interop, stream) = (&gpu.interop, &gpu.stream);

    let slice = stream.clone_htod(&vec![3u8; 64 * 1024])?;
    for _ in 0..500 {
        drop(interop.lend(&slice)?);
    }

    // Many live lends of different slices at once, each 64 KiB.
    let slices: Vec<_> = (0..16)
        .map(|_| stream.clone_htod(&vec![3u8; 64 * 1024]))
        .collect::<Result<_, _>>()?;
    let lent: Vec<_> = slices
        .iter()
        .map(|s| interop.lend(s))
        .collect::<Result<_, _>>()?;
    let last = lent.last().expect("16 slices were lent");
    assert_eq!(interop.stream().clone_dtoh(last)?, vec![3u8; 64 * 1024]);
    Ok(())
}
