//! cudarc's `examples/10-function-attributes.rs` (MIT OR Apache-2.0) on Vulkan. PTX versions and
//! occupancy are CUDA's. This prints the limits that bound a launch, which belong to the device,
//! and what the driver reports about the kernel (registers, spills), which depends on the driver.

use cudash::driver::{DriverError, GpuContext, LaunchConfig, Spirv, StatisticValue};

fn main() -> Result<(), DriverError> {
    let ctx = GpuContext::new(0)?;

    println!("Device: {}", ctx.name()?);
    println!();

    // Load the module with the sin_kernel
    let module = ctx.load_module(Spirv::from_file(concat!(
        env!("OUT_DIR"),
        "/sin_kernel.spv"
    )))?;
    let sin_kernel = module.load_function("sin_kernel")?;

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
    println!();

    println!("=== sin_kernel with that block size, as the driver reports it ===");
    match sin_kernel.statistics(cfg.block_dim) {
        Ok(statistics) => {
            for s in statistics {
                let value = match s.value {
                    StatisticValue::Bool(v) => v.to_string(),
                    StatisticValue::I64(v) => v.to_string(),
                    StatisticValue::U64(v) => v.to_string(),
                    StatisticValue::F64(v) => v.to_string(),
                };
                println!(
                    "  {:<25} {value:<8} {}",
                    format!("{}:", s.name),
                    s.description
                );
            }
        }
        Err(DriverError::MissingFeature(_)) => println!("  The driver reports none."),
        Err(e) => return Err(e),
    }

    Ok(())
}
