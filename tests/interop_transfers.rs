//! Moving data between the host and lent cudash slices, and between lent slices (including
//! views), with real cudarc streams.
//! These need an NVIDIA GPU and print SKIPPED (or panic, with CUDASH_REQUIRE_NVIDIA set).

mod common;

use std::fmt::Debug;

use cudash::{
    driver,
    interop::{
        AsCuda,
        cudarc::driver::{DeviceRepr, ValidAsZeroBits},
    },
};

use common::*;

/// Both directions through a lend: cudash writes and cudarc reads, then cudarc writes and
/// cudash reads. Nothing may change on the way.
fn round_trip<T>(gpu: &Nvidia, data: &[T]) -> cudash::interop::Result<()>
where
    T: DeviceRepr + ValidAsZeroBits + driver::DeviceRepr + driver::ValidAsZeroBits,
    T: PartialEq + Debug + Clone,
{
    let stream = &gpu.stream;

    let written = stream.clone_htod(data)?;
    let lent = written.as_cuda()?;
    assert_eq!(lent.stream().clone_dtoh(&lent)?, data);
    drop(lent);

    let mut target = stream.alloc_zeros::<T>(data.len())?;
    {
        let mut lent = target.as_cuda_mut()?;
        lent.stream().clone().memcpy_htod(data, &mut lent)?;
    }
    assert_eq!(stream.clone_dtoh(&target)?, data);
    Ok(())
}

#[test]
fn round_trip_each_element_type() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };

    round_trip(&gpu, &[1u8, 255, 0])?;
    round_trip(&gpu, &[u16::MAX, 0, 7])?;
    round_trip(&gpu, &[u32::MAX, 0xC0DA_F00D])?;
    round_trip(&gpu, &[u64::MAX, 1 << 40])?;
    round_trip(&gpu, &[i32::MIN, -1, i32::MAX])?;
    round_trip(&gpu, &[i64::MIN, -42, i64::MAX])?;
    round_trip(&gpu, &[1.5f32, -2.25, f32::MAX])?;
    round_trip(&gpu, &[f64::MIN_POSITIVE, 1e300, -0.5])?;
    Ok(())
}

/// Writing into a view of a lent slice changes only the view's elements.
#[test]
fn host_to_device_into_a_view_leaves_neighbours_alone() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let stream = &gpu.stream;

    let mut buf = stream.alloc_zeros::<u32>(8)?;
    {
        let mut lent = buf.as_cuda_mut()?;
        let cuda = lent.stream().clone();
        let mut whole = lent.as_view_mut();
        cuda.memcpy_htod(&[1u32, 2, 3], &mut whole.slice_mut(2..5))?;
    }
    assert_eq!(stream.clone_dtoh(&buf)?, [0, 0, 1, 2, 3, 0, 0, 0]);
    Ok(())
}

#[test]
fn device_to_host_into_an_existing_slice() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let stream = &gpu.stream;

    let buf = stream.clone_htod(&[4u8, 5, 6])?;
    let lent = buf.as_cuda()?;
    let mut host = vec![0u8; 3];
    lent.stream().memcpy_dtoh(&lent, &mut host)?;
    assert_eq!(host, [4, 5, 6]);
    Ok(())
}

/// Copies one view into another lent slice, at a different offset.
#[test]
fn device_to_device_between_views() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let stream = &gpu.stream;

    let src = stream.clone_htod(&[10u32, 11, 12, 13])?;
    let mut dst = stream.alloc_zeros::<u32>(6)?;
    {
        let src = src.as_cuda()?;
        let mut dst = dst.as_cuda_mut()?;
        let cuda = dst.stream().clone();
        let mut whole = dst.as_view_mut();
        cuda.memcpy_dtod(&src.slice(1..3), &mut whole.slice_mut(3..5))?;
    }
    assert_eq!(stream.clone_dtoh(&dst)?, [0, 0, 0, 11, 12, 0]);
    Ok(())
}
