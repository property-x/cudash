//! cudarc's `examples/12-context-config.rs` (MIT OR Apache-2.0) on Vulkan. CUDA's limits and
//! cache settings have no equivalent, and Vulkan's limits are fixed, so this only reads them.

use ash::vk;
use cudash::driver::{CudaContext, DriverError};

fn main() -> Result<(), DriverError> {
    let dev = CudaContext::new(0)?;

    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    unsafe {
        dev.vk_instance()
            .get_physical_device_properties2(dev.vk_physical_device(), &mut properties)
    };
    let limits = properties.properties.limits;

    println!("=== Context Limits ===");
    println!("Device memory: {} MB", dev.total_mem()? / 1024 / 1024);
    println!(
        "Largest allocation: {} MB",
        maintenance3.max_memory_allocation_size / 1024 / 1024
    );
    println!("Allocation count: {}", limits.max_memory_allocation_count);
    println!("Argument bytes: {}", limits.max_push_constants_size);
    println!(
        "Grid size per launch: {:?}",
        limits.max_compute_work_group_count
    );

    Ok(())
}
