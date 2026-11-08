//! Moving data between the host and shared buffers, and between shared buffers (including
//! views), with real cudarc streams.
//! These need an NVIDIA GPU and print SKIPPED (or panic, with CUDASH_REQUIRE_NVIDIA set).

mod common;

use std::fmt::Debug;

use cudash::interop::{
    Interop,
    cudarc::driver::{DeviceRepr, ValidAsZeroBits},
};

use common::*;

/// Copies `data` to the device and back, and checks that nothing changed.
fn round_trip<T>(interop: &Interop, data: &[T]) -> cudash::interop::Result<()>
where
    T: DeviceRepr + ValidAsZeroBits + PartialEq + Debug + Clone,
{
    let b = buf_from(interop, data)?;
    assert_eq!(interop.stream().clone_dtoh(&b)?, data);
    Ok(())
}

#[test]
fn round_trip_each_element_type() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let interop = &gpu.interop;

    round_trip(interop, &[1u8, 255, 0])?;
    round_trip(interop, &[u16::MAX, 0, 7])?;
    round_trip(interop, &[u32::MAX, 0xC0DA_F00D])?;
    round_trip(interop, &[u64::MAX, 1 << 40])?;
    round_trip(interop, &[i32::MIN, -1, i32::MAX])?;
    round_trip(interop, &[i64::MIN, -42, i64::MAX])?;
    round_trip(interop, &[1.5f32, -2.25, f32::MAX])?;
    round_trip(interop, &[f64::MIN_POSITIVE, 1e300, -0.5])?;
    Ok(())
}

/// Writing into a view changes only the view's elements.
#[test]
fn host_to_device_into_a_view_leaves_neighbours_alone() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let interop = &gpu.interop;
    let stream = interop.stream();

    let mut buf = buf::<u32>(interop, 8)?;
    stream.memcpy_htod(&[1u32, 2, 3], &mut buf.slice_mut(2..5))?;
    assert_eq!(stream.clone_dtoh(&buf)?, [0, 0, 1, 2, 3, 0, 0, 0]);
    Ok(())
}

#[test]
fn device_to_host_into_an_existing_slice() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let interop = &gpu.interop;
    let stream = interop.stream();

    let buf = buf_from(interop, &[4u8, 5, 6])?;
    let mut host = vec![0u8; 3];
    stream.memcpy_dtoh(&buf, &mut host)?;
    assert_eq!(host, [4, 5, 6]);
    Ok(())
}

/// Copies one view into another buffer, at a different offset.
#[test]
fn device_to_device_between_views() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let interop = &gpu.interop;
    let stream = interop.stream();

    let src = buf_from(interop, &[10u32, 11, 12, 13])?;
    let mut dst = buf::<u32>(interop, 6)?;
    stream.memcpy_dtod(&src.slice(1..3), &mut dst.slice_mut(3..5))?;
    assert_eq!(stream.clone_dtoh(&dst)?, [0, 0, 0, 11, 12, 0]);
    Ok(())
}
