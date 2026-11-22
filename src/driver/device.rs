//! The Vulkan device behind a context, and buffers allocated from it.

#[cfg(feature = "cudarc")]
use std::collections::HashMap;

use std::{
    ffi::CStr,
    ops::Deref,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use ash::{ext, khr, vk};

use super::{DriverError, stream::Tracking};

// Mandates what the API relies on: buffer device addresses for kernel pointers, and timeline
// semaphores and synchronization2 for streams. Kernel args get as little as 128 bytes of push
// constants, and `launch` returns an error above the device's limit.
const API_VERSION: u32 = vk::API_VERSION_1_3;

/// A context's Vulkan device. Dropping it parks the device in [`IDLE`] for the next context on
/// its GPU instead of destroying it.
pub(crate) struct Device(Option<DeviceState>);

/// Devices whose contexts were dropped, which have no Vulkan objects left. On NVIDIA,
/// `vkDestroyDevice` deadlocks if another thread that used Vulkan exits meanwhile.
static IDLE: Mutex<Vec<DeviceState>> = Mutex::new(Vec::new());

pub(crate) struct DeviceState {
    /// A handle to the process-wide instance from [`instance`], which is never destroyed.
    pub(crate) instance: ash::Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) queue: Mutex<vk::Queue>,
    pub(crate) queue_family: u32,
    pub(crate) properties: vk::PhysicalDeviceProperties,
    pub(crate) uuid: [u8; 16],
    memory: vk::PhysicalDeviceMemoryProperties,
    /// `maxBufferSize` and `maxMemoryAllocationSize`, checked before Vulkan sees a bigger request.
    max_buffer: u64,
    max_allocation: u64,
    pub(crate) ordinal: usize,
    host_import: Option<HostImport>,
    /// The optional extensions that were enabled.
    pub(crate) extensions: Vec<&'static CStr>,
    /// SPIR-V capabilities a kernel may not declare, with the feature they need.
    pub(crate) unsupported_capabilities: Vec<(u32, &'static str)>,
    pub(crate) frame_boundary: bool,
    /// Labels operations for tools such as RenderDoc, where the instance has the extension.
    pub(crate) debug_utils: Option<ext::debug_utils::Device>,
    /// The matching CUDA context, created by `interop` on first use.
    #[cfg(feature = "cudarc")]
    pub(crate) cuda: crate::interop::ContextSlot,
}

/// `VK_EXT_external_memory_host`, where the driver has it.
struct HostImport {
    functions: ext::external_memory_host::Device,
    alignment: u64,
}

/// The loader and the one instance every context shares. It lives until the process exits,
/// because destroying the last instance unloads the driver, and NVIDIA's driver stops
/// loading after a few dozen unloads.
struct Vulkan {
    _entry: ash::Entry,
    instance: ash::Instance,
    /// Whether `VK_EXT_debug_utils` is on, for labels in tools such as RenderDoc.
    debug_utils: bool,
}

/// The shared instance, created on first use. A failure isn't cached, so later calls retry.
fn vulkan() -> Result<&'static Vulkan, DriverError> {
    static VULKAN: Mutex<Option<&'static Vulkan>> = Mutex::new(None);
    let mut slot = VULKAN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(vulkan) = *slot {
        return Ok(vulkan);
    }
    let vulkan: &'static Vulkan = Box::leak(Box::new(create_instance()?));
    *slot = Some(vulkan);
    Ok(vulkan)
}

fn create_instance() -> Result<Vulkan, DriverError> {
    unsafe {
        let entry = ash::Entry::load().map_err(|e| DriverError::NoVulkan(e.to_string()))?;
        let app = vk::ApplicationInfo::default().api_version(API_VERSION);
        let available = entry.enumerate_instance_extension_properties(None)?;
        let has = |name| {
            available
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let debug_utils = has(ext::debug_utils::NAME);
        let mut extensions = Vec::new();
        if debug_utils {
            extensions.push(ext::debug_utils::NAME.as_ptr());
        }
        let mut info = vk::InstanceCreateInfo::default().application_info(&app);
        if has(khr::portability_enumeration::NAME) {
            extensions.push(khr::portability_enumeration::NAME.as_ptr());
            info = info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }
        let instance = entry.create_instance(&info.enabled_extension_names(&extensions), None)?;
        Ok(Vulkan {
            _entry: entry,
            instance,
            debug_utils,
        })
    }
}

/// Lists every Vulkan 1.3 device with a compute queue, fastest kind first (discrete, integrated,
/// virtual, other, CPU), so ordinal 0 is the one a CUDA user would expect.
/// If portability extension is enabled, we list these devices as well, e.g. MoltenVK.
fn enumerate() -> Result<(&'static Vulkan, Vec<vk::PhysicalDevice>), DriverError> {
    let vulkan = vulkan()?;
    let instance = &vulkan.instance;
    unsafe {
        let mut devices = instance.enumerate_physical_devices()?;
        devices.retain(|&pd| {
            let properties = instance.get_physical_device_properties(pd);
            properties.api_version >= API_VERSION && compute_queue_family(instance, pd).is_some()
        });
        devices.sort_by_key(
            |&pd| match instance.get_physical_device_properties(pd).device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 4,
                _ => 3,
            },
        );
        Ok((vulkan, devices))
    }
}

/// Extensions cudash uses when the GPU has them: host import for registered memory, and
/// with the `cudarc` feature, exporting memory and semaphores to CUDA.
fn available_extensions(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Vec<&'static CStr> {
    let have = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
    let mut wanted = vec![
        ext::external_memory_host::NAME,
        khr::portability_subset::NAME,
        ext::frame_boundary::NAME,
    ];
    wanted.extend(OPTIONAL_EXTENSIONS);
    #[cfg(feature = "cudarc")]
    wanted.extend(crate::interop::EXPORT_EXTENSIONS);
    // For `CudaSlice::export_dma_buf`, which also needs `VK_KHR_external_memory_fd` from above.
    #[cfg(feature = "cudarc")]
    wanted.push(ext::external_memory_dma_buf::NAME);
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

/// SPIR-V `Capability` values for the core features, named and numbered as in Khronos's
/// `spirv.core.grammar.json`. `extensions.rs` gives those of the extensions with each entry.
#[allow(non_upper_case_globals)]
mod spirv_capability {
    pub const Float16: u32 = 9;
    pub const Float64: u32 = 10;
    pub const Int64: u32 = 11;
    pub const Int64Atomics: u32 = 12;
    pub const Int16: u32 = 22;
    pub const Int8: u32 = 39;
    pub const StorageBuffer16BitAccess: u32 = 4433;
    pub const StorageBuffer8BitAccess: u32 = 4448;
}

/// Defines what `extensions.rs` lists:
/// `OPTIONAL_EXTENSIONS`, and `OptionalFeatures` with one feature struct per extension.
macro_rules! optional_extensions {
    ($($name:literal => $features:ident {
        capabilities: [$($capability:ident($id:literal): [$($bit:ident),* $(,)?]),* $(,)?],
        off: [$($off:ident),* $(,)?] $(,)?
    }),* $(,)?) => {
        const OPTIONAL_EXTENSIONS: &[&CStr] = &[$($name),*];

        #[allow(non_snake_case)]
        #[derive(Default)]
        struct OptionalFeatures {
            $($features: vk::$features<'static>,)*
        }

        impl OptionalFeatures {
            /// Chains the struct of each extension in `extensions` onto `features`. Clears the
            /// `p_next` an earlier chain left, so the queried structs can be chained again.
            fn chain<'a>(
                &'a mut self,
                mut features: vk::PhysicalDeviceFeatures2<'a>,
                extensions: &[&CStr],
            ) -> vk::PhysicalDeviceFeatures2<'a> {
                $(if extensions.contains(&$name) {
                    self.$features.p_next = std::ptr::null_mut();
                    features = features.push_next(&mut self.$features);
                })*
                features
            }

            /// Clears the bits listed under `off`.
            fn turn_off(&mut self) {
                $($(self.$features.$off = vk::FALSE;)*)*
            }

            /// `(capability, name, whether any of its bits is on)` for every listed capability.
            fn capabilities(&self) -> Vec<(u32, &'static str, vk::Bool32)> {
                vec![$($((
                    $id,
                    stringify!($capability),
                    0 $(| self.$features.$bit)*,
                ),)*)*]
            }
        }
    };
}

include!("extensions.rs");

/// SPIR-V capabilities whose feature is off, as `(capability, name)`. `optional` holds the
/// extensions' capabilities from [`OptionalFeatures::capabilities`].
fn unsupported_capabilities(
    core: &vk::PhysicalDeviceFeatures,
    v11: &vk::PhysicalDeviceVulkan11Features,
    v12: &vk::PhysicalDeviceVulkan12Features,
    optional: Vec<(u32, &'static str, vk::Bool32)>,
) -> Vec<(u32, &'static str)> {
    use spirv_capability::*;
    [
        (Float16, "Float16", v12.shader_float16),
        (Float64, "Float64", core.shader_float64),
        (Int64, "Int64", core.shader_int64),
        (
            Int64Atomics,
            "Int64Atomics",
            v12.shader_buffer_int64_atomics | v12.shader_shared_int64_atomics,
        ),
        (Int16, "Int16", core.shader_int16),
        (Int8, "Int8", v12.shader_int8),
        (
            StorageBuffer16BitAccess,
            "StorageBuffer16BitAccess",
            v11.storage_buffer16_bit_access,
        ),
        (
            StorageBuffer8BitAccess,
            "StorageBuffer8BitAccess",
            v12.storage_buffer8_bit_access,
        ),
    ]
    .into_iter()
    .chain(optional)
    .filter(|&(_, _, on)| on != vk::TRUE)
    .map(|(capability, name, _)| (capability, name))
    .collect()
}

impl Device {
    /// Names `handle` for tools such as RenderDoc, prefixed with its Vulkan type, as in
    /// `[Pipeline] computeMain block (16, 16, 1)`. `name` is only formatted when
    /// `VK_EXT_debug_utils` is on, and failing to set it changes nothing else.
    pub(crate) fn set_name<H: vk::Handle>(&self, handle: H, name: std::fmt::Arguments<'_>) {
        if let Some(utils) = &self.debug_utils {
            let kind = match H::TYPE {
                vk::ObjectType::DEVICE => "Device".to_owned(),
                vk::ObjectType::QUEUE => "Queue".to_owned(),
                vk::ObjectType::SEMAPHORE => "Semaphore".to_owned(),
                vk::ObjectType::COMMAND_POOL => "Command Pool".to_owned(),
                vk::ObjectType::COMMAND_BUFFER => "Command Buffer".to_owned(),
                vk::ObjectType::BUFFER => "Buffer".to_owned(),
                vk::ObjectType::DEVICE_MEMORY => "Memory".to_owned(),
                vk::ObjectType::IMAGE => "Image".to_owned(),
                vk::ObjectType::IMAGE_VIEW => "Image View".to_owned(),
                vk::ObjectType::SHADER_MODULE => "Shader Module".to_owned(),
                vk::ObjectType::PIPELINE_LAYOUT => "Pipeline Layout".to_owned(),
                vk::ObjectType::PIPELINE => "Pipeline".to_owned(),
                other => format!("{other:?}"),
            };
            let name = std::ffi::CString::new(format!("[{kind}] {name}")).unwrap_or_default();
            let info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(handle)
                .object_name(&name);
            let _ = unsafe { utils.set_debug_utils_object_name(&info) };
        }
    }

    pub(crate) fn count() -> Result<i32, DriverError> {
        Ok(enumerate()?.1.len() as i32)
    }

    pub(crate) fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        let parked = {
            let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
            let index = idle.iter().position(|state| state.ordinal == ordinal);
            index.map(|i| idle.swap_remove(i))
        };
        let state = match parked {
            Some(state) => state,
            None => Self::create_state(ordinal)?,
        };
        let device = Arc::new(Self(Some(state)));
        device.set_name(
            device.device.handle(),
            format_args!("CudaContext {ordinal}"),
        );
        device.set_name(
            *device.queue.lock().unwrap(),
            format_args!("CudaContext {ordinal}"),
        );
        Ok(device)
    }

    fn create_state(ordinal: usize) -> Result<DeviceState, DriverError> {
        let (vulkan, devices) = enumerate()?;
        let instance = &vulkan.instance;
        let pd = *devices
            .get(ordinal)
            .ok_or(DriverError::NoSuchDevice(ordinal))?;
        let extensions = available_extensions(instance, pd);
        let (device, queue_family, physical_device, frame_boundary, unsupported_capabilities) =
            unsafe { Self::create(instance, pd, &extensions) }?;
        unsafe {
            let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut v13 = vk::PhysicalDeviceVulkan13Properties::default();
            let mut import = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut properties = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut v11)
                .push_next(&mut v13);
            let has_host_import = extensions.contains(&ext::external_memory_host::NAME);
            if has_host_import {
                properties = properties.push_next(&mut import);
            }
            instance.get_physical_device_properties2(physical_device, &mut properties);
            let properties = properties.properties;
            let debug_utils = vulkan
                .debug_utils
                .then(|| ext::debug_utils::Device::new(instance, &device));
            let host_import = has_host_import.then(|| HostImport {
                functions: ext::external_memory_host::Device::new(instance, &device),
                alignment: import.min_imported_host_pointer_alignment,
            });
            Ok(DeviceState {
                queue: Mutex::new(device.get_device_queue(queue_family, 0)),
                memory: instance.get_physical_device_memory_properties(physical_device),
                uuid: v11.device_uuid,
                max_buffer: v13.max_buffer_size,
                max_allocation: v11.max_memory_allocation_size,
                properties,
                instance: instance.clone(),
                physical_device,
                device,
                queue_family,
                ordinal,
                host_import,
                extensions,
                unsupported_capabilities,
                frame_boundary,
                debug_utils,
                #[cfg(feature = "cudarc")]
                cuda: Default::default(),
            })
        }
    }

    /// Enables every feature the GPU has, since a kernel could use any of them.
    #[allow(clippy::type_complexity)]
    unsafe fn create(
        instance: &ash::Instance,
        pd: vk::PhysicalDevice,
        extensions: &[&CStr],
    ) -> Result<
        (
            ash::Device,
            u32,
            vk::PhysicalDevice,
            bool,
            Vec<(u32, &'static str)>,
        ),
        DriverError,
    > {
        unsafe {
            let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut frame_boundary = vk::PhysicalDeviceFrameBoundaryFeaturesEXT::default();
            let mut optional = OptionalFeatures::default();
            // Builds the same chain twice, to query and then to enable, so it clears each
            // `p_next` the query left behind.
            macro_rules! chain {
                ($features:expr) => {{
                    for p_next in [
                        &mut v11.p_next,
                        &mut v12.p_next,
                        &mut v13.p_next,
                        &mut frame_boundary.p_next,
                    ] {
                        *p_next = std::ptr::null_mut();
                    }
                    let mut features = $features
                        .push_next(&mut v11)
                        .push_next(&mut v12)
                        .push_next(&mut v13);
                    if extensions.contains(&ext::frame_boundary::NAME) {
                        features = features.push_next(&mut frame_boundary);
                    }
                    optional.chain(features, extensions)
                }};
            }
            let mut features = {
                let mut query = chain!(vk::PhysicalDeviceFeatures2::default());
                instance.get_physical_device_features2(pd, &mut query);
                query.features
            };
            if let Some(name) = missing_required_feature(&v12, &v13) {
                return Err(DriverError::MissingFeature(name));
            }
            let unsupported =
                unsupported_capabilities(&features, &v11, &v12, optional.capabilities());
            // Slows everything down, and CUDA has no equivalent.
            features.robust_buffer_access = vk::FALSE;
            optional.turn_off();
            let frame_boundary_enabled = frame_boundary.frame_boundary == vk::TRUE;
            let mut features = chain!(vk::PhysicalDeviceFeatures2::default().features(features));

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
            Ok((
                instance.create_device(pd, &info, None)?,
                queue_family,
                pd,
                frame_boundary_enabled,
                unsupported,
            ))
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

impl Deref for Device {
    type Target = DeviceState;

    fn deref(&self) -> &DeviceState {
        self.0.as_ref().expect("only taken in drop")
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let state = self.0.take().expect("only taken in drop");
        let _ = unsafe { state.device.device_wait_idle() };
        IDLE.lock().unwrap_or_else(|e| e.into_inner()).push(state);
    }
}

/// A GPU buffer and its memory.
pub(crate) struct Buffer {
    pub(crate) device: Arc<Device>,
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    /// Index of `memory`'s type, for binding an image to the same memory.
    pub(crate) memory_type: u32,
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
    /// Whether the memory can also be exported as a dma-buf, for `CudaSlice::export_dma_buf`.
    #[cfg_attr(not(feature = "cudarc"), allow(dead_code))]
    pub(crate) dma_buf: bool,
    #[cfg(feature = "cudarc")]
    pub(crate) cuda_mappings: Mutex<HashMap<usize, Arc<crate::interop::buffer::Mapping>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dedicated(pub(crate) bool);

/// Whether memory for `usage` can be exported with all of `handles`, and whether the driver then
/// only allows dedicated allocations. Needs the export extensions, which the `cudarc` feature
/// enables.
fn export_support(
    device: &Device,
    usage: vk::BufferUsageFlags,
    handles: vk::ExternalMemoryHandleTypeFlags,
) -> Option<Dedicated> {
    let dma_buf = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    if handles.is_empty()
        || !device.extensions.contains(&khr::external_memory_fd::NAME)
        || handles.contains(dma_buf)
            && !device
                .extensions
                .contains(&ext::external_memory_dma_buf::NAME)
    {
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
    let properties = properties.external_memory_properties;
    let features = properties.external_memory_features;
    use vk::ExternalMemoryFeatureFlags as F;
    (features.contains(F::EXPORTABLE) && properties.compatible_handle_types.contains(handles))
        .then_some(Dedicated(features.contains(F::DEDICATED_ONLY)))
}

// SAFETY: handles aren't tied to a thread, and `mapped` is only written through
// while the buffer is exclusively owned by the call that created it.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Names the buffer and its memory, as [`Device::set_name`].
    pub(crate) fn set_name(&self, name: std::fmt::Arguments<'_>) {
        if self.device.debug_utils.is_some() && self.buffer != vk::Buffer::null() {
            let name = name.to_string();
            self.device.set_name(self.buffer, format_args!("{name}"));
            self.device.set_name(self.memory, format_args!("{name}"));
        }
    }

    fn empty(device: &Arc<Device>) -> Self {
        Self {
            device: device.clone(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            memory_type: 0,
            address: 0,
            mapped: None,
            tracking: Mutex::default(),
            allocation_size: 0,
            export: None,
            dma_buf: false,
            #[cfg(feature = "cudarc")]
            cuda_mappings: Mutex::default(),
        }
    }

    fn new(
        device: &Arc<Device>,
        size: u64,
        usage: vk::BufferUsageFlags,
        wanted: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
        handles: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<Self, DriverError> {
        let mut result = Self::empty(device);
        // Empty slices are legal in cudarc, and Vulkan has no empty buffers.
        if size == 0 {
            return Ok(result);
        }
        // fail instead of letting the driver spill device memory to system RAM (like Cuda).
        let too_big = Err(DriverError::Vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        if size > device.max_buffer {
            return too_big;
        }
        let vk_device = &device.device;
        let export = export_support(device, usage, handles);
        let handle = if export.is_some() {
            handles
        } else {
            vk::ExternalMemoryHandleTypeFlags::empty()
        };
        result.dma_buf = handle.contains(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
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
            let heap = device.memory.memory_types[memory_type as usize].heap_index;
            if requirements.size > device.max_allocation
                || requirements.size > device.memory.memory_heaps[heap as usize].size
            {
                return too_big;
            }
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
            result.memory_type = memory_type;
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
        // Exportable where the driver allows it, so `interop` can lend it to CUDA.
        let opaque = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        Self::device_local_exporting(device, size, opaque)
    }

    /// [`Self::device_local`], exportable with `handles` where the driver allows all of them.
    pub(crate) fn device_local_exporting(
        device: &Arc<Device>,
        size: u64,
        handles: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<Arc<Self>, DriverError> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let none = vk::MemoryPropertyFlags::empty();
        Ok(Arc::new(Self::new(
            device, size, usage, local, none, handles,
        )?))
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
            vk::ExternalMemoryHandleTypeFlags::empty(),
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
        #[cfg(feature = "cudarc")]
        self.cuda_mappings
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
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

    /// Every capability number cudash checks is the one Khronos's grammar gives its name.
    /// Fails if the number does not match the name according to khronos grammar.
    #[test]
    fn capability_numbers_match_spirv() {
        let all = unsupported_capabilities(
            &Default::default(),
            &Default::default(),
            &Default::default(),
            OptionalFeatures::default().capabilities(),
        );
        assert_eq!(all.len(), 19);
        for (number, name) in all {
            let capability: spirv::Capability = name.parse().unwrap();
            assert_eq!(capability as u32, number, "{name}");
        }
    }

    /// With the `cudarc` feature, device memory is exportable wherever the driver can export
    /// it, so `AsCuda` can lend it. Staging memory never is.
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
        let handle = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        let buffer = Buffer::new(&device, 4096, usage, host, none, handle).unwrap();
        let Some(Dedicated(dedicated)) = buffer.export else {
            panic!("the driver has the extension, but can't export this buffer");
        };
        let vk_device = &device.device;
        let fd_functions = khr::external_memory_fd::Device::new(&device.instance, vk_device);
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

    /// Where the driver can export a dma-buf, a slice allocated for it exports one that covers it.
    /// Other slices don't, so `as_image` can still bind their memory to any image.
    #[cfg(feature = "cudarc")]
    #[test]
    fn device_memory_exports_as_dma_buf() {
        use std::io::{Seek, SeekFrom};
        let ctx = crate::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        assert!(
            stream
                .alloc_zeros::<u8>(1000)
                .unwrap()
                .export_dma_buf()
                .is_err()
        );
        let slice = stream.alloc_zeros_dma_buf::<u8>(1000).unwrap();
        let fd = match slice.export_dma_buf() {
            Err(DriverError::MissingFeature(_)) => return eprintln!("SKIPPED: no dma-buf export"),
            fd => fd.unwrap(),
        };
        assert!(std::fs::File::from(fd).seek(SeekFrom::End(0)).unwrap() >= 1000);
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
