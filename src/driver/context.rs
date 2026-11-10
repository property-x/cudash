use std::sync::Arc;

use ash::vk;

use super::{
    DriverError,
    device::Device,
    host::Registry,
    stream::{CudaStream, StreamCore},
};

/// A handle to one GPU, like cudarc's `CudaContext`.
pub struct CudaContext {
    pub(crate) device: Arc<Device>,
    default_core: Arc<StreamCore>,
    pub(crate) registered: Registry,
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext")
            .field("ordinal", &self.device.ordinal)
            .field("name", &self.device.name())
            .finish_non_exhaustive()
    }
}

/// Contexts are equal when they share a Vulkan device. Unlike CUDA's primary context, two
/// `CudaContext::new(0)` calls create two devices, so they aren't equal.
impl PartialEq for CudaContext {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.device, &other.device)
    }
}
impl Eq for CudaContext {}

impl CudaContext {
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

    /// The context's shared default stream. All handles run on the same stream.
    pub fn default_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        CudaStream::new(self, self.default_core.clone())
    }

    /// A new, independent stream.
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
        Ok(CudaStream::new(
            self,
            StreamCore::new(&self.device, None, "new_stream")?,
        ))
    }

    /// Blocks until all work on this GPU has finished.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        let queue = self.device.queue.lock().unwrap();
        unsafe { self.device.device.queue_wait_idle(*queue) }?;
        Ok(())
    }

    /// The Vulkan instance, for use alongside cudash, such as rendering results.
    pub fn vk_instance(&self) -> &ash::Instance {
        &self.device.instance
    }

    /// What RenderDoc calls this context's device for Vulkan: the instance's dispatch table
    /// pointer (`RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE`). Pass it to
    /// [`RenderDoc::start`](crate::debug::RenderDoc::start) and `end`.
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
    pub fn vk_device(&self) -> &ash::Device {
        &self.device.device
    }
}
