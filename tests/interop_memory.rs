//! Creating shared buffers: zeroed contents, sizes, Vulkan handles, usage, and repeated
//! create and drop.
//! These need an NVIDIA GPU and print SKIPPED (or panic, with CUDASH_REQUIRE_NVIDIA set).

mod common;

use ash::vk::{self, Handle};
use cudash::interop::{Error, cudarc::driver::DeviceSlice};

use common::*;

#[test]
fn alloc_zeros_is_zeroed_for_several_types() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let interop = &gpu.interop;
    let stream = interop.stream();

    let bytes = buf::<u8>(interop, 17)?;
    assert_eq!(stream.clone_dtoh(&bytes)?, vec![0u8; 17]);
    let wide = buf::<u64>(interop, 9)?;
    assert_eq!(stream.clone_dtoh(&wide)?, vec![0u64; 9]);
    let floats = buf::<f64>(interop, 5)?;
    assert_eq!(stream.clone_dtoh(&floats)?, vec![0.0f64; 5]);
    let signed = buf::<i32>(interop, 3)?;
    assert_eq!(stream.clone_dtoh(&signed)?, vec![0i32; 3]);
    Ok(())
}

#[test]
fn zero_length_allocations_are_rejected() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let interop = &gpu.interop;

    assert!(matches!(buf::<u32>(interop, 0), Err(Error::InvalidLength)));
    Ok(())
}

#[test]
fn len_and_byte_size_match_the_element_type() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let interop = &gpu.interop;

    let buf = buf::<u16>(interop, 10)?;
    assert_eq!(buf.len(), 10);
    assert_eq!(buf.num_bytes(), 20);
    assert!(!buf.is_empty());
    Ok(())
}

/// Each buffer gets its own Vulkan buffer and memory, and both handles are real.
#[test]
fn buffers_have_distinct_valid_vulkan_handles() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let interop = &gpu.interop;

    let a = buf::<u32>(interop, 8)?;
    let b = buf::<u32>(interop, 8)?;
    assert!(!a.vk_buffer().is_null());
    assert!(!a.vk_memory().is_null());
    assert_ne!(a.vk_buffer(), b.vk_buffer());
    assert_ne!(a.vk_memory(), b.vk_memory());
    Ok(())
}

/// Buffers made through the lower-level API take their usage and memory flags from the caller.
#[test]
fn interop_buffers_take_the_requested_usage() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let interop = &gpu.interop;

    let device = interop.create_buffer::<u32>(16, vk::BufferUsageFlags::TRANSFER_SRC)?;
    assert_eq!(device.len(), 16);

    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let staging =
        interop.create_buffer_with::<u32>(16, vk::BufferUsageFlags::TRANSFER_DST, host)?;
    assert_eq!(staging.len(), 16);
    Ok(())
}

/// Allocating and dropping in a loop shouldn't fail or hang. This doesn't detect
/// leaks; that needs an external check (e.g. watching `nvidia-smi` or the fd count).
#[test]
fn create_and_drop_many_buffers() -> cudash::interop::Result<()> {
    let Some(gpu) = nvidia() else {
        return Ok(());
    };
    let interop = &gpu.interop;
    let stream = interop.stream();

    for _ in 0..500 {
        let buf = buf::<u8>(interop, 64 * 1024)?;
        drop(buf);
    }

    // Many live buffers at once, each 64 KiB.
    let live: Vec<_> = (0..16)
        .map(|_| buf::<u8>(interop, 64 * 1024))
        .collect::<Result<Vec<_>, _>>()?;
    let last = live.last().expect("16 buffers were allocated");
    assert_eq!(stream.clone_dtoh(last)?.len(), 64 * 1024);
    Ok(())
}
