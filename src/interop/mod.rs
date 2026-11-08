//! Zero-copy data exchange between CUDA ([`cudarc`]) and Vulkan ([`ash`]).
//!
//! Vulkan allocates and exports the memory, CUDA imports it, and both APIs then
//! see the same bytes. A [`SharedSemaphore`](crate::interop::SharedSemaphore) orders work between them.
//!
//! Memory and semaphores come from a [`cudash::driver::CudaContext`](crate::driver::CudaContext),
//! which enables the export extensions when the GPU has them. Pair it with a cudarc stream on
//! the same GPU through [`Interop`](crate::interop::Interop). Linux (opaque fd handles) only for now.

#[cfg(not(unix))]
compile_error!("cudash currently supports only Unix opaque fd handles");

mod buffer;
mod semaphore;
mod vulkan;

use std::{fmt, sync::Arc};

use ash::vk;
use cudarc::driver::{
    CudaContext, CudaStream, DeviceRepr, ValidAsZeroBits,
    result::{self, DriverError},
    sys,
};

pub use buffer::SharedBuffer;
pub use cudarc;
pub use semaphore::SharedSemaphore;
pub(crate) use vulkan::EXPORT_EXTENSIONS;

const NVIDIA_VENDOR_ID: u32 = 0x10de;

/// Everything that can go wrong while setting up or using shared resources.
#[derive(Debug)]
pub enum Error {
    Vulkan(vk::Result),
    Cuda(DriverError),
    /// From the cudash context the resources come from.
    Driver(crate::driver::DriverError),
    /// No CUDA device has the same UUID as the cudash context.
    NoMatchingCudaDevice,
    /// The CUDA stream and the cudash context are on different GPUs.
    DeviceMismatch,
    /// The driver can't export this kind of resource as an opaque fd.
    ExportUnsupported,
    /// No memory type satisfies both the buffer and the requested properties.
    NoSuitableMemoryType,
    /// The buffer would be empty, or its byte size overflows.
    InvalidLength,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(e) => write!(f, "Vulkan error: {e}"),
            Self::Cuda(e) => write!(f, "CUDA error: {e}"),
            Self::Driver(e) => write!(f, "cudash error: {e}"),
            Self::NoMatchingCudaDevice => f.write_str("no CUDA device matches the cudash context"),
            Self::DeviceMismatch => {
                f.write_str("CUDA stream and cudash context are on different GPUs")
            }
            Self::ExportUnsupported => f.write_str("driver cannot export this as an opaque fd"),
            Self::NoSuitableMemoryType => f.write_str("no suitable Vulkan memory type"),
            Self::InvalidLength => f.write_str("buffer length is zero or overflows"),
        }
    }
}

impl std::error::Error for Error {}

impl From<vk::Result> for Error {
    fn from(e: vk::Result) -> Self {
        Self::Vulkan(e)
    }
}

impl From<DriverError> for Error {
    fn from(e: DriverError) -> Self {
        Self::Cuda(e)
    }
}

impl From<crate::driver::DriverError> for Error {
    fn from(e: crate::driver::DriverError) -> Self {
        Self::Driver(e)
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Creates a CUDA context on the same GPU as a cudash context.
///
/// Returns [`Error::NoMatchingCudaDevice`] on non-NVIDIA devices without touching CUDA,
/// so it's safe to call on machines that have no CUDA installed.
pub fn cuda_context_for(ctx: &crate::driver::CudaContext) -> Result<Arc<CudaContext>> {
    if ctx.device.properties.vendor_id != NVIDIA_VENDOR_ID {
        return Err(Error::NoMatchingCudaDevice);
    }
    let uuid = ctx.uuid()?;
    result::init()?;
    for ordinal in 0..result::device::get_count()? {
        let device = result::device::get(ordinal)?;
        if uuid_bytes(result::device::get_uuid(device)?) == uuid {
            return Ok(CudaContext::new(ordinal as usize)?);
        }
    }
    Err(Error::NoMatchingCudaDevice)
}

fn uuid_bytes(uuid: sys::CUuuid) -> [u8; 16] {
    uuid.bytes.map(|b| b as u8)
}

/// A cudash context paired with a CUDA stream on the same GPU, that shared resources are
/// created from.
pub struct Interop {
    vulkan: vulkan::VulkanExporter,
    stream: Arc<CudaStream>,
}

impl Interop {
    /// Pairs `ctx` with a CUDA stream, checking they're on the same GPU.
    ///
    /// `stream` becomes the stream that [`SharedBuffer`]s are bound to.
    pub fn new(ctx: &Arc<crate::driver::CudaContext>, stream: Arc<CudaStream>) -> Result<Self> {
        if uuid_bytes(stream.context().uuid()?) != ctx.uuid()? {
            return Err(Error::DeviceMismatch);
        }
        let vulkan = vulkan::VulkanExporter::new(&ctx.device)?;
        Ok(Self { vulkan, stream })
    }

    /// The stream that shared buffers created from this value are bound to.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Creates a zero-initialized, device-local buffer of `len` elements.
    pub fn create_buffer<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<SharedBuffer<T>> {
        self.create_buffer_with(len, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    /// Like [`Self::create_buffer`], with explicit Vulkan memory properties.
    pub fn create_buffer_with<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<SharedBuffer<T>> {
        SharedBuffer::new(&self.vulkan, &self.stream, len, usage, memory_properties)
    }

    /// Creates a timeline semaphore usable from both APIs, starting at value 0.
    pub fn create_semaphore(&self) -> Result<SharedSemaphore> {
        SharedSemaphore::new(&self.vulkan, self.stream.context())
    }
}
