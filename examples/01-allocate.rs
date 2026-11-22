//! cudarc's `examples/01-allocate.rs` (MIT OR Apache-2.0) on Vulkan. Only the import changed.

use cudash::driver::{DriverError, GpuContext, GpuSlice};

fn main() -> Result<(), DriverError> {
    let ctx = GpuContext::new(0)?;
    let stream = ctx.default_stream();

    // unsafe initialization of unset memory
    let _: GpuSlice<f32> = unsafe { stream.alloc::<f32>(10) }?;

    // this will have memory initialized as 0
    let _: GpuSlice<f64> = stream.alloc_zeros::<f64>(10)?;

    // initialize with slices!
    let _: GpuSlice<usize> = stream.clone_htod(&[0; 10])?;
    let _: GpuSlice<u32> = stream.clone_htod(&[1, 2, 3])?;

    Ok(())
}
