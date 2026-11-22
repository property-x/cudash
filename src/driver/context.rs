use std::sync::Arc;

use crate::ash::vk;

use super::{
    DriverError,
    device::Device,
    host::Registry,
    stream::{GpuStream, StreamCore},
};

/// A handle to one GPU, like cudarc's `GpuContext`.
pub struct GpuContext {
    pub(crate) device: Arc<Device>,
    default_core: Arc<StreamCore>,
    pub(crate) registered: Registry,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("ordinal", &self.device.ordinal)
            .field("name", &self.device.name())
            .finish_non_exhaustive()
    }
}

/// Contexts are equal when they share a Vulkan device. Unlike CUDA's primary context, two
/// `GpuContext::new(0)` calls create two devices, so they aren't equal.
impl PartialEq for GpuContext {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.device, &other.device)
    }
}
impl Eq for GpuContext {}

impl GpuContext {
    /// Creates a context on the GPU with this ordinal. Discrete GPUs come first.
    ///
    /// Requires Vulkan 1.3.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        let device = Device::new(ordinal)?;
        let default_core = StreamCore::new(&device, None, "default_stream")?;
        Ok(Arc::new(Self {
            device,
            default_core,
            registered: Registry::default(),
        }))
    }

    /// How many suitable GPUs there are.
    pub fn device_count() -> Result<i32, DriverError> {
        Device::count()
    }

    pub fn ordinal(&self) -> usize {
        self.device.ordinal
    }

    pub fn name(&self) -> Result<String, DriverError> {
        Ok(self.device.name())
    }

    pub fn uuid(&self) -> Result<[u8; 16], DriverError> {
        Ok(self.device.uuid)
    }

    /// Size of the largest device-local memory heap, in bytes.
    pub fn total_mem(&self) -> Result<usize, DriverError> {
        Ok(self.device.total_mem())
    }

    /// `(free, total)` bytes of the heap [`Self::total_mem`] reports, like `cuMemGetInfo`. Free
    /// comes from `VK_EXT_memory_budget`; without it, free is the whole heap.
    pub fn mem_get_info(&self) -> Result<(usize, usize), DriverError> {
        Ok(self.device.mem_get_info())
    }

    /// The context's shared default stream. All handles run on the same stream.
    pub fn default_stream(self: &Arc<Self>) -> Arc<GpuStream> {
        GpuStream::new(self, self.default_core.clone())
    }

    /// A new, independent stream.
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<GpuStream>, DriverError> {
        Ok(GpuStream::new(
            self,
            StreamCore::new(&self.device, None, "new_stream")?,
        ))
    }

    /// Blocks until all work on this GPU has finished.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        let queue = self.device.queue.lock().unwrap();
        unsafe { self.device.device.queue_wait_idle(*queue) }.map_err(|e| self.device.error(e))?;
        Ok(())
    }

    /// The device extensions cudash enabled: its own, and those of [`OPTIONAL_EXTENSIONS`](super::OPTIONAL_EXTENSIONS)
    /// the GPU has.
    pub fn extensions(&self) -> &[&'static std::ffi::CStr] {
        &self.device.extensions
    }

    /// The Vulkan instance, for use alongside cudash, such as rendering results.
    pub fn vk_instance(&self) -> &crate::ash::Instance {
        &self.device.instance
    }

    /// What RenderDoc calls this context's device for Vulkan: the instance's dispatch table
    /// pointer (`RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE`). Pass it to
    /// the `renderdoc` crate's `start_frame_capture` and `end_frame_capture`, as
    /// `examples/frame_capture.rs` does.
    pub fn renderdoc_device_pointer(&self) -> *mut std::ffi::c_void {
        let instance =
            vk::Handle::as_raw(self.vk_instance().handle()) as *const *mut std::ffi::c_void;
        // SAFETY: a dispatchable Vulkan handle points at its dispatch table pointer.
        unsafe { *instance }
    }

    /// The Vulkan physical device behind this context.
    pub fn vk_physical_device(&self) -> vk::PhysicalDevice {
        self.device.physical_device
    }

    /// The Vulkan device behind this context.
    pub fn vk_device(&self) -> &crate::ash::Device {
        &self.device.device
    }
}
