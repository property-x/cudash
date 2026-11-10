//! The Vulkan half: exportable semaphores on cudash's own device, and opaque fds for them
//! and for device memory. Nothing in here touches CUDA, so it can be tested on any Vulkan driver.

use std::{
    ffi::CStr,
    os::fd::{FromRawFd, OwnedFd},
    sync::Arc,
};

use ash::{khr, vk};

use crate::{
    driver::device::Device,
    interop::{Error, Result},
};

const MEMORY_HANDLE: vk::ExternalMemoryHandleTypeFlags =
    vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
const SEMAPHORE_HANDLE: vk::ExternalSemaphoreHandleTypeFlags =
    vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;

/// The device extensions interop needs. `cudash::driver` enables them when the GPU has them.
pub(crate) const EXPORT_EXTENSIONS: [&CStr; 2] = [
    khr::external_memory_fd::NAME,
    khr::external_semaphore_fd::NAME,
];

/// Creates exportable resources on a cudash device. They keep the device alive.
pub(crate) struct VulkanExporter {
    pub(crate) device: Arc<Device>,
    memory_fd: khr::external_memory_fd::Device,
    semaphore_fd: khr::external_semaphore_fd::Device,
}

impl VulkanExporter {
    pub(crate) fn new(device: &Arc<Device>) -> Result<Self> {
        if !EXPORT_EXTENSIONS
            .iter()
            .all(|name| device.extensions.contains(name))
        {
            return Err(Error::ExportUnsupported);
        }
        Ok(Self {
            device: device.clone(),
            memory_fd: khr::external_memory_fd::Device::new(&device.instance, &device.device),
            semaphore_fd: khr::external_semaphore_fd::Device::new(&device.instance, &device.device),
        })
    }

    pub(crate) fn create_semaphore(&self) -> Result<ExportedSemaphore> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
            .handle_type(SEMAPHORE_HANDLE)
            .push_next(&mut type_info);
        let mut external = vk::ExternalSemaphoreProperties::default();
        unsafe {
            self.device
                .instance
                .get_physical_device_external_semaphore_properties(
                    self.device.physical_device,
                    &info,
                    &mut external,
                )
        };
        if !external
            .external_semaphore_features
            .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
        {
            return Err(Error::ExportUnsupported);
        }

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut export_info =
            vk::ExportSemaphoreCreateInfo::default().handle_types(SEMAPHORE_HANDLE);
        let create_info = vk::SemaphoreCreateInfo::default()
            .push_next(&mut type_info)
            .push_next(&mut export_info);
        let semaphore = unsafe { self.device.device.create_semaphore(&create_info, None) }?;
        self.device
            .set_name(semaphore, format_args!("SharedSemaphore"));
        Ok(ExportedSemaphore {
            device: self.device.clone(),
            semaphore,
        })
    }

    /// Exports a new fd for `memory`, which must have been allocated as exportable.
    /// Each call returns a new fd.
    pub(crate) fn export_memory(&self, memory: vk::DeviceMemory) -> Result<OwnedFd> {
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(MEMORY_HANDLE);
        let fd = unsafe { self.memory_fd.get_memory_fd(&info) }?;
        // SAFETY: Vulkan transfers ownership of the returned fd to the caller.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Exports a new fd for `semaphore`. Each call returns a new fd.
    pub(crate) fn export_semaphore(&self, semaphore: &ExportedSemaphore) -> Result<OwnedFd> {
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore.semaphore)
            .handle_type(SEMAPHORE_HANDLE);
        let fd = unsafe { self.semaphore_fd.get_semaphore_fd(&info) }?;
        // SAFETY: Vulkan transfers ownership of the returned fd to the caller.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

pub(crate) struct ExportedSemaphore {
    pub(crate) device: Arc<Device>,
    pub(crate) semaphore: vk::Semaphore,
}

impl Drop for ExportedSemaphore {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_semaphore(self.semaphore, None) };
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::IntoRawFd;

    use super::*;
    use crate::driver::CudaContext;

    /// A context on the first GPU, if it can export.
    fn any_gpu() -> Option<(Arc<CudaContext>, VulkanExporter)> {
        let ctx = CudaContext::new(0)
            .inspect_err(|e| eprintln!("SKIPPED: {e}"))
            .ok()?;
        let exporter = VulkanExporter::new(&ctx.device)
            .inspect_err(|e| eprintln!("SKIPPED: {e}"))
            .ok()?;
        Some((ctx, exporter))
    }

    /// A timeline semaphore re-imported into Vulkan from its fd shares the original's value.
    #[test]
    fn exported_timeline_semaphore_round_trips() {
        let Some((ctx, exporter)) = any_gpu() else {
            return;
        };
        let device = &ctx.device.device;
        let original = exporter.create_semaphore().unwrap();
        let fd = exporter.export_semaphore(&original).unwrap();

        let mut type_info =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        unsafe {
            let imported = device.create_semaphore(&create_info, None).unwrap();
            let import = vk::ImportSemaphoreFdInfoKHR::default()
                .semaphore(imported)
                .handle_type(SEMAPHORE_HANDLE)
                .fd(fd.into_raw_fd());
            exporter.semaphore_fd.import_semaphore_fd(&import).unwrap();
            let signal = vk::SemaphoreSignalInfo::default()
                .semaphore(original.semaphore)
                .value(7);
            device.signal_semaphore(&signal).unwrap();
            assert_eq!(device.get_semaphore_counter_value(imported).unwrap(), 7);
            device.destroy_semaphore(imported, None);
        }
    }

    /// Resources must keep the device alive, or their Drop would use it after it's
    /// destroyed when the context is dropped first.
    #[test]
    fn resources_keep_the_device_alive() {
        let Some((ctx, exporter)) = any_gpu() else {
            return;
        };
        let semaphore = exporter.create_semaphore().unwrap();
        drop(exporter);
        let held = Arc::strong_count(&ctx.device);
        drop(semaphore);
        assert_eq!(Arc::strong_count(&ctx.device), held - 1);
    }
}
