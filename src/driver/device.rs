//! The Vulkan device behind a context, and buffers allocated from it.

use std::{
    ffi::CStr,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use ash::{ext, khr, vk};

use super::{DriverError, stream::Tracking};

// Mandates what the API relies on: buffer device addresses for kernel pointers, and timeline
// semaphores and synchronization2 for streams. Kernel args get as little as 128 bytes of push
// constants, and `launch` returns an error above the device's limit.
const API_VERSION: u32 = vk::API_VERSION_1_3;

pub(crate) struct Device {
    _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) queue: Mutex<vk::Queue>,
    pub(crate) queue_family: u32,
    pub(crate) properties: vk::PhysicalDeviceProperties,
    pub(crate) uuid: [u8; 16],
    memory: vk::PhysicalDeviceMemoryProperties,
    pub(crate) ordinal: usize,
    host_import: Option<HostImport>,
    /// The optional extensions that were enabled.
    pub(crate) extensions: Vec<&'static CStr>,
    /// The matching CUDA context, created by `interop` on first use.
    #[cfg(feature = "cudarc")]
    pub(crate) cuda: crate::interop::ContextSlot,
}

/// `VK_EXT_external_memory_host`, where the driver has it.
struct HostImport {
    functions: ext::external_memory_host::Device,
    alignment: u64,
}

/// Loads Vulkan and lists suitable GPUs, discrete ones first, so ordinal 0 is the
/// one a CUDA user would expect.
/// If portability extension is enabled, we list these devices as well, e.g. MoltenVK.
fn enumerate() -> Result<(ash::Entry, ash::Instance, Vec<vk::PhysicalDevice>), DriverError> {
    unsafe {
        let entry = ash::Entry::load().map_err(|e| DriverError::NoVulkan(e.to_string()))?;
        let app = vk::ApplicationInfo::default().api_version(API_VERSION);
        let extension = entry.enumerate_instance_extension_properties(None)?;
        let portability = [khr::portability_enumeration::NAME.as_ptr()];
        let mut info = vk::InstanceCreateInfo::default().application_info(&app);
        if extension
            .iter()
            .any(|ext| ext.extension_name_as_c_str() == Ok(khr::portability_enumeration::NAME))
        {
            info = info
                .enabled_extension_names(&portability)
                .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }
        let instance = entry.create_instance(&info, None)?;
        let mut devices = match instance.enumerate_physical_devices() {
            Ok(devices) => devices,
            Err(e) => {
                instance.destroy_instance(None);
                return Err(e.into());
            }
        };
        devices.retain(|&pd| {
            let properties = instance.get_physical_device_properties(pd);
            properties.api_version >= API_VERSION
                && properties.device_type != vk::PhysicalDeviceType::CPU
                && compute_queue_family(&instance, pd).is_some()
        });
        devices.sort_by_key(|&pd| {
            let kind = instance.get_physical_device_properties(pd).device_type;
            kind != vk::PhysicalDeviceType::DISCRETE_GPU
        });
        Ok((entry, instance, devices))
    }
}

/// Extensions cudash uses when the GPU has them: host import for registered memory, and
/// with the `cudarc` feature, exporting memory and semaphores to CUDA.
fn available_extensions(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Vec<&'static CStr> {
    let have = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
    let mut wanted = vec![
        ext::external_memory_host::NAME,
        khr::portability_subset::NAME,
    ];
    #[cfg(feature = "cudarc")]
    wanted.extend(crate::interop::EXPORT_EXTENSIONS);
    wanted.retain(|name| have.iter().any(|e| e.extension_name_as_c_str() == Ok(name)));
    wanted
}

fn compute_queue_family(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Option<u32> {
    unsafe { instance.get_physical_device_queue_family_properties(pd) }
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|family| family as u32)
}

/// The first feature cudash can't work without that the GPU doesn't report. Vulkan 1.3
/// mandates these, but a missing one would otherwise only surface as invalid calls later.
fn missing_required_feature(
    v12: &vk::PhysicalDeviceVulkan12Features,
    v13: &vk::PhysicalDeviceVulkan13Features,
) -> Option<&'static str> {
    [
        ("bufferDeviceAddress", v12.buffer_device_address),
        ("timelineSemaphore", v12.timeline_semaphore),
        ("synchronization2", v13.synchronization2),
    ]
        .into_iter()
        .find_map(|(name, enabled)| (enabled != vk::TRUE).then_some(name))
}

impl Device {
    pub(crate) fn count() -> Result<i32, DriverError> {
        let (_entry, instance, devices) = enumerate()?;
        unsafe { instance.destroy_instance(None) };
        Ok(devices.len() as i32)
    }

    pub(crate) fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        let (entry, instance, devices) = enumerate()?;
        let created = devices
            .get(ordinal)
            .ok_or(DriverError::NoSuchDevice(ordinal))
            .and_then(|&pd| {
                let extensions = available_extensions(&instance, pd);
                unsafe { Self::create(&instance, pd, &extensions) }.map(|c| (c, extensions))
            });
        let ((device, queue_family, physical_device), extensions) = match created {
            Ok(created) => created,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        unsafe {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut import = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            let has_host_import = extensions.contains(&ext::external_memory_host::NAME);
            if has_host_import {
                properties = properties.push_next(&mut import);
            }
            instance.get_physical_device_properties2(physical_device, &mut properties);
            let properties = properties.properties;
            let host_import = has_host_import.then(|| HostImport {
                functions: ext::external_memory_host::Device::new(&instance, &device),
                alignment: import.min_imported_host_pointer_alignment,
            });
            Ok(Arc::new(Self {
                queue: Mutex::new(device.get_device_queue(queue_family, 0)),
                memory: instance.get_physical_device_memory_properties(physical_device),
                uuid: id.device_uuid,
                properties,
                _entry: entry,
                instance,
                physical_device,
                device,
                queue_family,
                ordinal,
                host_import,
                extensions,
                #[cfg(feature = "cudarc")]
                cuda: Default::default(),
            }))
        }
    }

    /// Enables every feature the GPU has, since a kernel could use any of them.
    unsafe fn create(
        instance: &ash::Instance,
        pd: vk::PhysicalDevice,
        extensions: &[&CStr],
    ) -> Result<(ash::Device, u32, vk::PhysicalDevice), DriverError> {
        unsafe {
            let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut features = {
                let mut query = vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut v11)
                    .push_next(&mut v12)
                    .push_next(&mut v13);
                instance.get_physical_device_features2(pd, &mut query);
                query.features
            };
            if let Some(name) = missing_required_feature(&v12, &v13) {
                return Err(DriverError::MissingFeature(name));
            }
            // Slows everything down, and CUDA has no equivalent.
            features.robust_buffer_access = vk::FALSE;
            v11.p_next = std::ptr::null_mut();
            v12.p_next = std::ptr::null_mut();
            v13.p_next = std::ptr::null_mut();
            let mut features = vk::PhysicalDeviceFeatures2::default()
                .features(features)
                .push_next(&mut v11)
                .push_next(&mut v12)
                .push_next(&mut v13);

            let queue_family = compute_queue_family(instance, pd).expect("checked in enumerate");
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let extensions: Vec<_> = extensions.iter().map(|name| name.as_ptr()).collect();
            let info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&extensions)
                .push_next(&mut features);
            Ok((instance.create_device(pd, &info, None)?, queue_family, pd))
        }
    }

    #[cfg(test)]
    pub(crate) fn has_host_import(&self) -> bool {
        self.host_import.is_some()
    }

    pub(crate) fn name(&self) -> String {
        self.properties
            .device_name_as_c_str()
            .unwrap_or(c"")
            .to_string_lossy()
            .into_owned()
    }

    pub(crate) fn total_mem(&self) -> usize {
        let heaps = &self.memory.memory_heaps[..self.memory.memory_heap_count as usize];
        heaps
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size as usize)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn memory_type(
        &self,
        bits: u32,
        wanted: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Result<u32, DriverError> {
        let count = self.memory.memory_type_count;
        let find = |flags: vk::MemoryPropertyFlags| {
            (0..count).find(|&i| {
                bits & (1 << i) != 0
                    && self.memory.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
            })
        };
        find(wanted | preferred)
            .or_else(|| find(wanted))
            .ok_or(DriverError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A GPU buffer and its memory.
pub(crate) struct Buffer {
    pub(crate) device: Arc<Device>,
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) address: u64,
    mapped: Option<NonNull<u8>>,
    pub(crate) tracking: Mutex<Tracking>,
    /// Size of the allocation, which can exceed the buffer's. CUDA imports exactly this.
    #[cfg_attr(not(feature = "cudarc"), allow(dead_code))]
    pub(crate) allocation_size: u64,
    /// `Some` if the memory can be exported as an opaque fd, holding whether it's a
    /// dedicated allocation, which CUDA has to be told.
    #[cfg_attr(not(feature = "cudarc"), allow(dead_code))]
    pub(crate) export: Option<Dedicated>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dedicated(pub(crate) bool);

/// Whether memory for `usage` can be exported to CUDA, and whether the driver then only
/// allows dedicated allocations. Needs the export extensions, which the `cudarc` feature enables.
fn export_support(device: &Device, usage: vk::BufferUsageFlags) -> Option<Dedicated> {
    if !device.extensions.contains(&khr::external_memory_fd::NAME) {
        return None;
    }
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let mut properties = vk::ExternalBufferProperties::default();
    unsafe {
        device
            .instance
            .get_physical_device_external_buffer_properties(
                device.physical_device,
                &info,
                &mut properties,
            )
    };
    let features = properties
        .external_memory_properties
        .external_memory_features;
    use vk::ExternalMemoryFeatureFlags as F;
    features
        .contains(F::EXPORTABLE)
        .then_some(Dedicated(features.contains(F::DEDICATED_ONLY)))
}

// SAFETY: handles aren't tied to a thread, and `mapped` is only written through
// while the buffer is exclusively owned by the call that created it.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn empty(device: &Arc<Device>) -> Self {
        Self {
            device: device.clone(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            address: 0,
            mapped: None,
            tracking: Mutex::default(),
            allocation_size: 0,
            export: None,
        }
    }

    fn new(
        device: &Arc<Device>,
        size: u64,
        usage: vk::BufferUsageFlags,
        wanted: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
        exportable: bool,
    ) -> Result<Self, DriverError> {
        let mut result = Self::empty(device);
        // Empty slices are legal in cudarc, and Vulkan has no empty buffers.
        if size == 0 {
            return Ok(result);
        }
        let vk_device = &device.device;
        let handle = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        let export = exportable.then(|| export_support(device, usage)).flatten();
        unsafe {
            let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
            let mut info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            if export.is_some() {
                info = info.push_next(&mut external);
            }
            result.buffer = vk_device.create_buffer(&info, None)?;
            let mut dedicated = vk::MemoryDedicatedRequirements::default();
            let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
            let buffer_info = vk::BufferMemoryRequirementsInfo2::default().buffer(result.buffer);
            vk_device.get_buffer_memory_requirements2(&buffer_info, &mut requirements);
            let requirements = requirements.memory_requirements;
            result.export = export.map(|Dedicated(only)| {
                Dedicated(
                    only || dedicated.requires_dedicated_allocation == vk::TRUE
                        || dedicated.prefers_dedicated_allocation == vk::TRUE,
                )
            });
            let memory_type =
                device.memory_type(requirements.memory_type_bits, wanted, preferred)?;
            let mut flags = vk::MemoryAllocateFlagsInfo::default();
            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                flags = flags.flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            }
            // shortcut: one allocation per buffer, and Vulkan only guarantees 4096 of them.
            // Add a suballocator if real code hits that limit.
            let mut allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type)
                .push_next(&mut flags);
            let mut export_info = vk::ExportMemoryAllocateInfo::default().handle_types(handle);
            let mut dedicated_info =
                vk::MemoryDedicatedAllocateInfo::default().buffer(result.buffer);
            if let Some(Dedicated(dedicated)) = result.export {
                allocate = allocate.push_next(&mut export_info);
                if dedicated {
                    allocate = allocate.push_next(&mut dedicated_info);
                }
            }
            result.memory = vk_device.allocate_memory(&allocate, None)?;
            result.allocation_size = requirements.size;
            vk_device.bind_buffer_memory(result.buffer, result.memory, 0)?;
            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                let info = vk::BufferDeviceAddressInfo::default().buffer(result.buffer);
                result.address = vk_device.get_buffer_device_address(&info);
            }
        }
        Ok(result)
    }

    /// Device-local memory that kernels and copies can use, with undefined contents.
    pub(crate) fn device_local(device: &Arc<Device>, size: u64) -> Result<Arc<Self>, DriverError> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let none = vk::MemoryPropertyFlags::empty();
        // Exportable where the driver allows it, so `interop` can lend it to CUDA.
        Ok(Arc::new(Self::new(device, size, usage, local, none, true)?))
    }

    /// Host-visible memory for moving data to or from the GPU.
    pub(crate) fn staging(device: &Arc<Device>, size: u64) -> Result<Arc<Self>, DriverError> {
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // Cached memory makes reading back much faster, and doesn't hurt writes much.
        let mut buffer = Self::new(
            device,
            size,
            usage,
            host,
            vk::MemoryPropertyFlags::HOST_CACHED,
            false,
        )?;
        if size > 0 {
            let flags = vk::MemoryMapFlags::empty();
            let pointer = unsafe {
                device
                    .device
                    .map_memory(buffer.memory, 0, vk::WHOLE_SIZE, flags)
            }?;
            buffer.mapped = NonNull::new(pointer.cast());
        }
        Ok(Arc::new(buffer))
    }

    /// A buffer the GPU can copy from, over the whole pages around `bytes`, so the data itself
    /// needn't be aligned. Returns it with the offset of `bytes` inside it.
    ///
    /// # Safety
    /// `bytes` must stay allocated until the buffer is dropped. The GPU may only read the
    /// `bytes.len()` bytes at the returned offset, because the rest belongs to someone else.
    pub(crate) unsafe fn import_host(
        device: &Arc<Device>,
        bytes: &[u8],
    ) -> Result<(Arc<Self>, u64), DriverError> {
        let import = device
            .host_import
            .as_ref()
            .ok_or(DriverError::MissingFeature("VK_EXT_external_memory_host"))?;
        let alignment = import.alignment as usize;
        let address = bytes.as_ptr() as usize;
        let start = address & !(alignment - 1);
        let size = (address + bytes.len()).next_multiple_of(alignment) - start;
        let handle_type = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let vk_device = &device.device;
        let mut result = Self::empty(device);
        unsafe {
            // ash 0.38 has no wrapper for this one.
            let mut pointer = vk::MemoryHostPointerPropertiesEXT::default();
            (import.functions.fp().get_memory_host_pointer_properties_ext)(
                vk_device.handle(),
                handle_type,
                start as *const _,
                &mut pointer,
            )
                .result()?;
            let mut external =
                vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
            let info = vk::BufferCreateInfo::default()
                .size(size as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external);
            result.buffer = vk_device.create_buffer(&info, None)?;
            let requirements = vk_device.get_buffer_memory_requirements(result.buffer);
            // Coherent, so the GPU sees the caller's writes without a flush.
            let memory_type = device.memory_type(
                requirements.memory_type_bits & pointer.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::empty(),
            )?;
            let mut host = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(handle_type)
                .host_pointer(start as *mut _);
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(size as u64)
                .memory_type_index(memory_type)
                .push_next(&mut host);
            result.memory = vk_device.allocate_memory(&allocate, None)?;
            vk_device.bind_buffer_memory(result.buffer, result.memory, 0)?;
        }
        Ok((Arc::new(result), (address - start) as u64))
    }

    /// Where a staging buffer is mapped, if it isn't empty.
    pub(crate) fn mapped_ptr(&self) -> Option<NonNull<u8>> {
        self.mapped
    }

    /// # Safety
    /// This must be a staging buffer of at least `bytes.len()` bytes, not in use by the GPU.
    pub(crate) unsafe fn write_bytes(&self, bytes: &[u8]) {
        let mapped = self.mapped.expect("staging buffers are mapped");
        unsafe {
            mapped
                .as_ptr()
                .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len())
        };
    }

    /// # Safety
    /// This must be a staging buffer of at least `bytes.len()` bytes, not in use by the GPU.
    pub(crate) unsafe fn read_bytes(&self, bytes: &mut [u8]) {
        let mapped = self.mapped.expect("staging buffers are mapped");
        unsafe {
            bytes
                .as_mut_ptr()
                .copy_from_nonoverlapping(mapped.as_ptr(), bytes.len())
        };
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Submissions hold a reference until they finish, so the GPU is done here.
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the `cudarc` feature, device memory is exportable wherever the driver can export
    /// it, so `interop::lend` works. Staging memory never is.
    #[test]
    fn device_memory_is_exportable_with_the_export_extensions() {
        let device = Device::new(0).unwrap();
        let can_export = device.extensions.contains(&khr::external_memory_fd::NAME);
        assert!(cfg!(feature = "cudarc") || !can_export);
        let buffer = Buffer::device_local(&device, 1000).unwrap();
        assert_eq!(buffer.export.is_some(), can_export);
        assert!(buffer.allocation_size >= 1000);
        assert!(Buffer::staging(&device, 1000).unwrap().export.is_none());
        assert!(Buffer::device_local(&device, 0).unwrap().export.is_none());
    }

    /// Exported memory re-imported into Vulkan aliases the original bytes, which is the
    /// contract CUDA's import relies on. Host-visible here, so the test can compare bytes.
    #[test]
    fn exported_memory_round_trips() {
        let device = Device::new(0).unwrap();
        if !device.extensions.contains(&khr::external_memory_fd::NAME) {
            eprintln!("SKIPPED: no opaque fd export");
            return;
        }
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let none = vk::MemoryPropertyFlags::empty();
        let usage = vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = Buffer::new(&device, 4096, usage, host, none, true).unwrap();
        let Some(Dedicated(dedicated)) = buffer.export else {
            panic!("the driver has the extension, but can't export this buffer");
        };
        let vk_device = &device.device;
        let fd_functions = khr::external_memory_fd::Device::new(&device.instance, vk_device);
        let handle = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        unsafe {
            let info = vk::MemoryGetFdInfoKHR::default()
                .memory(buffer.memory)
                .handle_type(handle);
            let fd = fd_functions.get_memory_fd(&info).unwrap();
            // Opaque fds must be imported with the memory type they were exported from.
            let bits = vk_device
                .get_buffer_memory_requirements(buffer.buffer)
                .memory_type_bits;
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(handle)
                .fd(fd);
            let mut dedicated_info =
                vk::MemoryDedicatedAllocateInfo::default().buffer(buffer.buffer);
            let mut allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(buffer.allocation_size)
                .memory_type_index(device.memory_type(bits, host, none).unwrap())
                .push_next(&mut import);
            if dedicated {
                allocate = allocate.push_next(&mut dedicated_info);
            }
            let imported = vk_device.allocate_memory(&allocate, None).unwrap();
            let map = vk::MemoryMapFlags::empty();
            let a = vk_device.map_memory(buffer.memory, 0, 4, map).unwrap();
            let b = vk_device.map_memory(imported, 0, 4, map).unwrap();
            a.cast::<u32>().write_volatile(0xC0DA_F00D);
            assert_eq!(b.cast::<u32>().read_volatile(), 0xC0DA_F00D);
            vk_device.unmap_memory(imported);
            vk_device.unmap_memory(buffer.memory);
            vk_device.free_memory(imported, None);
        }
    }

    #[test]
    fn every_required_feature_is_checked() {
        let all_12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true);
        let all_13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        assert_eq!(missing_required_feature(&all_12, &all_13), None);

        let no_address = all_12.buffer_device_address(false);
        assert_eq!(
            missing_required_feature(&no_address, &all_13),
            Some("bufferDeviceAddress")
        );
        let no_timeline = all_12.timeline_semaphore(false);
        assert_eq!(
            missing_required_feature(&no_timeline, &all_13),
            Some("timelineSemaphore")
        );
        let no_sync2 = all_13.synchronization2(false);
        assert_eq!(
            missing_required_feature(&all_12, &no_sync2),
            Some("synchronization2")
        );
    }
}
