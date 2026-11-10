//! cudarc's `examples/10-function-attributes.rs` (MIT OR Apache-2.0) on Vulkan. Registers,
//! PTX versions and occupancy are CUDA's, so this prints the limits that bound a launch,
//! which belong to the device rather than the kernel.

use cudash::driver::{CudaContext, DriverError, LaunchConfig, Spirv};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;

    println!("Device: {}", ctx.name()?);
    println!();

    // Load the module with the sin_kernel
    let module = ctx.load_module(Spirv::from_file(concat!(
        env!("OUT_DIR"),
        "/sin_kernel.spv"
    )))?;
    let _sin_kernel = module.load_function("sin_kernel")?;

    let limits = unsafe {
        ctx.vk_instance()
            .get_physical_device_properties(ctx.vk_physical_device())
    }
    .limits;

    println!("=== Limits for any kernel ===");
    println!();
    println!(
        "  Shared memory:            {} bytes",
        limits.max_compute_shared_memory_size
    );
    println!(
        "  Argument bytes:           {}",
        limits.max_push_constants_size
    );
    println!(
        "  Max threads per block:    {}",
        limits.max_compute_work_group_invocations
    );
    println!();

    let cfg = LaunchConfig::for_num_elems(1);
    println!("=== Launch Configuration for 1 element ===");
    println!("  Block size:               {}", cfg.block_dim.0);
    println!("  Grid size:                {}", cfg.grid_dim.0);

    Ok(())
}
