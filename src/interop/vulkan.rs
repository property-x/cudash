//! The Vulkan half: exportable buffers and semaphores on cudash's own device, and their
//! opaque fds. Nothing in here touches CUDA, so it can be tested on any Vulkan driver.

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

pub(crate) fn acquire_barrier(
    buffer: vk::Buffer,
    queue_family: u32,
) -> vk::BufferMemoryBarrier<'static> {
    barrier(buffer, vk::QUEUE_FAMILY_EXTERNAL, queue_family)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
}

pub(crate) fn release_barrier(
    buffer: vk::Buffer,
    queue_family: u32,
) -> vk::BufferMemoryBarrier<'static> {
    barrier(buffer, queue_family, vk::QUEUE_FAMILY_EXTERNAL)
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::empty())
}

fn barrier(buffer: vk::Buffer, src: u32, dst: u32) -> vk::BufferMemoryBarrier<'static> {
    vk::BufferMemoryBarrier::default()
        .src_queue_family_index(src)
        .dst_queue_family_index(dst)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE)
}

/// Creates exportable resources on a cudash device. They keep the device alive.
pub(crate) struct VulkanExporter {
    device: Arc<Device>,
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

    pub(crate) fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<ExportedBuffer> {
        // Some drivers require dedicated allocations for certain usages, and
        // only this query reveals that.
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(MEMORY_HANDLE);
        let mut external = vk::ExternalBufferProperties::default();
        unsafe {
            self.device
                .instance
                .get_physical_device_external_buffer_properties(
                    self.device.physical_device,
                    &info,
                    &mut external,
                )
        };
        let features = external.external_memory_properties.external_memory_features;
        if !features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE) {
            return Err(Error::ExportUnsupported);
        }

        let mut external_info =
            vk::ExternalMemoryBufferCreateInfo::default().handle_types(MEMORY_HANDLE);
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_info);
        let buffer = unsafe { self.device.device.create_buffer(&create_info, None) }?;

        // From here on, Drop releases whatever has been created if we bail.
        let mut exported = ExportedBuffer {
            device: self.device.clone(),
            buffer,
            memory: vk::DeviceMemory::null(),
            size: 0,
            dedicated: false,
        };

        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let requirements_info = vk::BufferMemoryRequirementsInfo2::default().buffer(buffer);
        unsafe {
            self.device
                .device
                .get_buffer_memory_requirements2(&requirements_info, &mut requirements)
        };
        let requirements = requirements.memory_requirements;
        let use_dedicated = dedicated.prefers_dedicated_allocation == vk::TRUE
            || dedicated.requires_dedicated_allocation == vk::TRUE
            || features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);

        let memory_type_index = self
            .device
            .memory_type(
                requirements.memory_type_bits,
                memory_properties,
                vk::MemoryPropertyFlags::empty(),
            )
            .map_err(|_| Error::NoSuitableMemoryType)?;
        let mut export_info = vk::ExportMemoryAllocateInfo::default().handle_types(MEMORY_HANDLE);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export_info);
        if use_dedicated {
            allocate_info = allocate_info.push_next(&mut dedicated_info);
        }
        exported.memory = unsafe { self.device.device.allocate_memory(&allocate_info, None) }?;
        exported.size = requirements.size;
        exported.dedicated = use_dedicated;
        unsafe { (self.device.device).bind_buffer_memory(buffer, exported.memory, 0) }?;
        Ok(exported)
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
        Ok(ExportedSemaphore {
            device: self.device.clone(),
            semaphore,
        })
    }

    /// Exports a new fd for `buffer`'s memory. Each call returns a new fd.
    pub(crate) fn export_memory(&self, buffer: &ExportedBuffer) -> Result<OwnedFd> {
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(buffer.memory)
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

pub(crate) struct ExportedBuffer {
    device: Arc<Device>,
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    /// Size of the allocation, which can exceed the size asked for. CUDA must
    /// import exactly this size.
    pub(crate) size: u64,
    pub(crate) dedicated: bool,
}

impl Drop for ExportedBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
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

    /// Exported memory re-imported into Vulkan must alias the original bytes,
    /// which is the same contract CUDA's import relies on.
    #[test]
    fn exported_memory_round_trips() {
        let Some((ctx, exporter)) = any_gpu() else {
            return;
        };
        let device = &ctx.device.device;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let usage = vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = exporter.create_buffer(4096, usage, host).unwrap();
        assert!(buffer.size >= 4096);

        let fd = exporter.export_memory(&buffer).unwrap();
        let raw = fd.into_raw_fd();
        // Opaque fds must be imported with the memory type they were exported from.
        let type_bits =
            unsafe { device.get_buffer_memory_requirements(buffer.buffer) }.memory_type_bits;
        let none = vk::MemoryPropertyFlags::empty();
        let type_index = ctx.device.memory_type(type_bits, host, none).unwrap();

        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(MEMORY_HANDLE)
            .fd(raw);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer.buffer);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(buffer.size)
            .memory_type_index(type_index)
            .push_next(&mut import);
        if buffer.dedicated {
            info = info.push_next(&mut dedicated);
        }
        unsafe {
            let imported = device.allocate_memory(&info, None).unwrap();
            let map = vk::MemoryMapFlags::empty();
            let a = device.map_memory(buffer.memory, 0, 4, map).unwrap();
            let b = device.map_memory(imported, 0, 4, map).unwrap();
            a.cast::<u32>().write_volatile(0xC0DA_F00D);
            assert_eq!(b.cast::<u32>().read_volatile(), 0xC0DA_F00D);
            device.unmap_memory(imported);
            device.unmap_memory(buffer.memory);
            device.free_memory(imported, None);
        }
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

    #[test]
    fn missing_memory_type_is_an_error() {
        let Some((_ctx, exporter)) = any_gpu() else {
            return;
        };
        let usage = vk::BufferUsageFlags::TRANSFER_DST;
        let impossible = vk::MemoryPropertyFlags::from_raw(0x8000_0000);
        assert!(matches!(
            exporter.create_buffer(16, usage, impossible),
            Err(Error::NoSuitableMemoryType)
        ));
    }

    /// Resources must keep the device alive, or their Drop would use it after it's
    /// destroyed when the context is dropped first.
    #[test]
    fn resources_keep_the_device_alive() {
        let Some((ctx, exporter)) = any_gpu() else {
            return;
        };
        let usage = vk::BufferUsageFlags::TRANSFER_DST;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let buffer = exporter.create_buffer(64, usage, local).unwrap();
        let semaphore = exporter.create_semaphore().unwrap();
        drop(exporter);
        let held = Arc::strong_count(&ctx.device);
        drop(buffer);
        drop(semaphore);
        assert_eq!(Arc::strong_count(&ctx.device), held - 2);
    }
}
