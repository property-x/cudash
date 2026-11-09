//! Zero-copy data exchange between CUDA ([`cudarc`]) and Vulkan ([`ash`]).
//!
//! Pair a [`cudash::driver::CudaContext`](crate::driver::CudaContext) with a cudarc stream on
//! the same GPU through [`Interop`](crate::interop::Interop), then lend ordinary cudash slices
//! to real cudarc calls with [`Interop::lend`](crate::interop::Interop::lend) and
//! [`Interop::lend_mut`](crate::interop::Interop::lend_mut). Vulkan exports the memory, CUDA
//! imports it, and both see the same bytes. A [`SharedSemaphore`](crate::interop::SharedSemaphore)
//! orders work between them by hand. Linux (opaque fd handles) only for now.

#[cfg(not(unix))]
compile_error!("cudash currently supports only Unix opaque fd handles");

mod buffer;
mod semaphore;
mod vulkan;

use std::{fmt, sync::Arc};

use ash::vk;
use cudarc::driver::{
    CudaContext, CudaStream,
    result::{self, DriverError},
    sys,
};

pub use buffer::{Lent, LentMut};
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

/// A cudash context paired with a CUDA stream on the same GPU, which lends cudash slices to
/// cudarc and creates shared semaphores.
pub struct Interop {
    vulkan: vulkan::VulkanExporter,
    stream: Arc<CudaStream>,
}

impl Interop {
    /// Pairs `ctx` with a CUDA stream, checking they're on the same GPU.
    ///
    /// Lent slices are bound to `stream`.
    pub fn new(ctx: &Arc<crate::driver::CudaContext>, stream: Arc<CudaStream>) -> Result<Self> {
        if uuid_bytes(stream.context().uuid()?) != ctx.uuid()? {
            return Err(Error::DeviceMismatch);
        }
        let vulkan = vulkan::VulkanExporter::new(&ctx.device)?;
        Ok(Self { vulkan, stream })
    }

    /// The stream that lent slices are bound to.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Lends `slice` to cudarc calls for reading, without copying. Blocks until cudash
    /// work that writes it has finished. Dropping the result blocks until the interop
    /// stream has finished, and the borrow keeps cudash from writing meanwhile.
    ///
    /// Returns [`Error::ExportUnsupported`] if the slice's memory isn't exportable,
    /// [`Error::InvalidLength`] if it's empty, and [`Error::DeviceMismatch`] if it's
    /// from another cudash context.
    pub fn lend<'l, T>(&self, slice: &'l crate::driver::CudaSlice<T>) -> Result<Lent<'l, T>> {
        Ok(Lent {
            cuda: self.import(slice)?,
            _slice: std::marker::PhantomData,
        })
    }

    /// [`Self::lend`] for writing as well. Blocks until all cudash work using the slice
    /// has finished, and the borrow keeps cudash from using it until the result is dropped.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use cudash::interop::{Interop, cuda_context_for};
    ///
    /// let ctx = cudash::driver::CudaContext::new(0)?;
    /// let stream = ctx.default_stream();
    /// let mut buf = stream.alloc_zeros::<f32>(4)?;
    ///
    /// let cuda = cuda_context_for(&ctx)?.default_stream();
    /// let interop = Interop::new(&ctx, cuda.clone())?;
    /// {
    ///     let mut lent = interop.lend_mut(&mut buf)?;
    ///     cuda.memcpy_htod(&[1.0f32, 2.0, 3.0, 4.0], &mut lent)?; // a real cudarc call
    /// }
    /// assert_eq!(stream.clone_dtoh(&buf)?, [1.0, 2.0, 3.0, 4.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn lend_mut<'l, T>(
        &self,
        slice: &'l mut crate::driver::CudaSlice<T>,
    ) -> Result<LentMut<'l, T>> {
        Ok(LentMut {
            cuda: self.import(slice)?,
            _slice: std::marker::PhantomData,
        })
    }

    // shortcut: exports and imports the memory on every lend. Cache the import per buffer if
    // measurements show the cost matters.
    // shortcut: no queue family ownership transfer to and from VK_QUEUE_FAMILY_EXTERNAL,
    // which the spec asks for; buffers have no layout, and NVIDIA's samples skip it too.
    fn import<T>(&self, slice: &crate::driver::CudaSlice<T>) -> Result<buffer::Imported<T>> {
        let buffer = crate::driver::DevicePtr::raw(slice).buffer;
        if !Arc::ptr_eq(&buffer.device, &self.vulkan.device) {
            return Err(Error::DeviceMismatch);
        }
        if slice.is_empty() {
            return Err(Error::InvalidLength);
        }
        let Some(crate::driver::device::Dedicated(dedicated)) = buffer.export else {
            return Err(Error::ExportUnsupported);
        };
        // Waits for reads too, which only `lend_mut` needs. Lending is rare enough.
        crate::driver::stream::wait_for_gpu(buffer)?;
        let fd = self.vulkan.export_memory(buffer.memory)?;
        buffer::Imported::new(
            &self.stream,
            fd,
            buffer.allocation_size,
            dedicated,
            slice.len(),
        )
    }

    /// Creates a timeline semaphore usable from both APIs, starting at value 0.
    pub fn create_semaphore(&self) -> Result<SharedSemaphore> {
        SharedSemaphore::new(&self.vulkan, self.stream.context())
    }
}
