//! Prints what cudash sees on each GPU: the device, driver, PCI bus ID, memory, the extensions cudash
//! enabled, the optional ones this GPU lacks, and the cooperative matrix shapes. Paste the
//! output into an extension request.
//!
//! Run: `cargo run --example caps`

use cudash::{
    ash::{self, vk},
    driver::*,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for ordinal in 0..GpuContext::device_count()? as usize {
        let ctx = GpuContext::new(ordinal)?;
        let (instance, pd) = (ctx.vk_instance(), ctx.vk_physical_device());

        let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut v12 = vk::PhysicalDeviceVulkan12Properties::default();
        v11.p_next = &mut v12 as *mut _ as *mut _;
        let mut properties = vk::PhysicalDeviceProperties2 {
            p_next: &mut v11 as *mut _ as *mut _,
            ..Default::default()
        };
        unsafe { instance.get_physical_device_properties2(pd, &mut properties) };
        let api = properties.properties.api_version;
        let (free, total) = ctx.mem_get_info()?;

        println!("# GPU {ordinal}: {}", ctx.name()?);
        println!(
            "- driver: {:?} {:?}",
            v12.driver_name_as_c_str()?,
            v12.driver_info_as_c_str()?
        );
        println!(
            "- Vulkan {}.{}.{}, subgroup size {}",
            vk::api_version_major(api),
            vk::api_version_minor(api),
            vk::api_version_patch(api),
            v11.subgroup_size,
        );
        println!(
            "- PCI bus ID: {}",
            ctx.pci_bus_id().as_deref().unwrap_or("unknown")
        );
        println!("- memory: {} MiB free of {} MiB", free >> 20, total >> 20);

        let enabled = ctx.extensions();
        println!("- enabled extensions:");
        for name in enabled {
            println!("  - {}", name.to_string_lossy());
        }
        println!("- optional extensions this GPU lacks:");
        for name in OPTIONAL_EXTENSIONS
            .iter()
            .filter(|name| !enabled.contains(name))
        {
            println!("  - {}", name.to_string_lossy());
        }

        if enabled.contains(&ash::khr::cooperative_matrix::NAME) {
            println!("- cooperative matrix shapes (MxNxK, A B C result, scope, saturating):");
            for p in cooperative_matrix_shapes(instance, pd)? {
                println!(
                    "  - {}x{}x{}, {:?} {:?} {:?} {:?}, {:?}, {}",
                    p.m_size,
                    p.n_size,
                    p.k_size,
                    p.a_type,
                    p.b_type,
                    p.c_type,
                    p.result_type,
                    p.scope,
                    p.saturating_accumulation == vk::TRUE,
                );
            }
        }
        println!();
    }
    Ok(())
}

/// `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`, called through its function pointer,
/// since ash's wrapper differs between the release and the `ash_latest` git version.
fn cooperative_matrix_shapes(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
) -> Result<Vec<vk::CooperativeMatrixPropertiesKHR<'static>>, vk::Result> {
    let entry =
        unsafe { ash::Entry::load() }.map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
    #[cfg(not(feature = "ash_latest"))]
    let functions = ash::khr::cooperative_matrix::Instance::new(&entry, instance);
    #[cfg(feature = "ash_latest")]
    let functions = ash::khr::cooperative_matrix::Instance::load(&entry, instance);
    let get = functions
        .fp()
        .get_physical_device_cooperative_matrix_properties_khr;
    let mut count = 0;
    unsafe { get(pd, &mut count, std::ptr::null_mut()) }.result()?;
    let mut shapes = vec![vk::CooperativeMatrixPropertiesKHR::default(); count as usize];
    unsafe { get(pd, &mut count, shapes.as_mut_ptr()) }.result()?;
    shapes.truncate(count as usize);
    Ok(shapes)
}
