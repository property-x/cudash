//! cudarc's `examples/13-copy-multi-gpu.rs` (MIT OR Apache-2.0) on Vulkan. A Vulkan buffer
//! belongs to one device, so the copy goes through the host instead of `clone_dtod`.

use cudash::driver::{DriverError, GpuContext, GpuSlice};

fn main() -> Result<(), DriverError> {
    let size = 10;

    let ctx1 = GpuContext::new(0)?;
    let stream1 = ctx1.default_stream();
    let a: GpuSlice<f64> = stream1.alloc_zeros::<f64>(size)?;

    let ctx2 = GpuContext::new(1)?;
    let stream2 = ctx2.default_stream();

    let b = stream2.clone_htod(&stream1.clone_dtoh(&a)?)?;

    assert_eq!(stream2.clone_dtoh(&b)?, stream1.clone_dtoh(&a)?);
    println!("Copied {} from {} to {}", size, ctx1.name()?, ctx2.name()?);

    Ok(())
}
