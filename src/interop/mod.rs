//! Zero-copy data exchange between CUDA ([`cudarc`]) and Vulkan ([`ash`]).
//!
//! [`AsCuda`](crate::interop::AsCuda) lends ordinary cudash slices and views to real cudarc calls. Vulkan exports the memory,
//! CUDA imports it, and both see the same bytes. Each cudash stream is paired with its own
//! CUDA stream, created on first use. A [`SharedSemaphore`](crate::interop::SharedSemaphore)
//! orders work between them by hand. Linux (opaque fd handles) only for now.

#[cfg(not(unix))]
compile_error!("cudash currently supports only Unix opaque fd handles");

pub(crate) mod buffer;
pub(crate) mod graph;
mod semaphore;
mod vulkan;

use std::{
    fmt,
    sync::{Arc, Mutex},
};

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
    /// The CUDA stream and the cudash slice are on different GPUs.
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
                f.write_str("CUDA stream and cudash slice are on different GPUs")
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

/// Where a cudash device keeps its CUDA context.
pub(crate) type ContextSlot = Mutex<Option<Arc<CudaContext>>>;
/// Where a cudash stream keeps its paired CUDA stream.
pub(crate) type StreamSlot = Mutex<Option<Arc<CudaStream>>>;

/// The value in `slot`, created by `init` if it's empty. A failed `init` is retried next time.
fn cached<T>(
    slot: &Mutex<Option<Arc<T>>>,
    init: impl FnOnce() -> Result<Arc<T>>,
) -> Result<Arc<T>> {
    let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(value) = &*slot {
        return Ok(value.clone());
    }
    let value = init()?;
    *slot = Some(value.clone());
    Ok(value)
}

impl crate::driver::CudaContext {
    /// The CUDA context on the same GPU. It's created on the first call and then shared by
    /// everything in this module.
    ///
    /// Returns [`Error::NoMatchingCudaDevice`] on non-NVIDIA devices without touching CUDA,
    /// so it's safe to call on machines that have no CUDA installed.
    pub fn as_cuda(&self) -> Result<Arc<CudaContext>> {
        cuda_context(&self.device)
    }
}

impl crate::driver::CudaStream {
    /// The CUDA stream paired with this one: a fresh one, never the legacy null stream, which
    /// would serialize unrelated CUDA work. It's created on the first call.
    ///
    /// This only hands out the stream. It doesn't order its work against cudash work; lending
    /// a slice with [`AsCuda`] or a [`SharedSemaphore`] does that.
    pub fn as_cuda(&self) -> Result<Arc<CudaStream>> {
        let stream = cached(&self.core.cuda, || {
            Ok(self.context().as_cuda()?.new_stream()?)
        })?;
        // While this stream captures, so does its CUDA stream.
        graph::join(&self.core, &stream)?;
        Ok(stream)
    }
}

fn cuda_context(device: &crate::driver::device::Device) -> Result<Arc<CudaContext>> {
    cached(&device.cuda, || {
        if device.properties.vendor_id != NVIDIA_VENDOR_ID {
            return Err(Error::NoMatchingCudaDevice);
        }
        result::init()?;
        for ordinal in 0..result::device::get_count()? {
            let cu_device = result::device::get(ordinal)?;
            if uuid_bytes(result::device::get_uuid(cu_device)?) == device.uuid {
                return Ok(CudaContext::new(ordinal as usize)?);
            }
        }
        Err(Error::NoMatchingCudaDevice)
    })
}

fn uuid_bytes(uuid: sys::CUuuid) -> [u8; 16] {
    uuid.bytes.map(|b| b as u8)
}

/// Lends a cudash slice or view to cudarc calls, without copying.
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use cudash::interop::AsCuda;
///
/// let ctx = cudash::driver::CudaContext::new(0)?;
/// let stream = ctx.default_stream();
/// let mut buf = stream.alloc_zeros::<f32>(4)?;
/// {
///     let mut lent = buf.as_cuda_mut()?;
///     let cuda = lent.stream().clone(); // the CUDA stream paired with `stream`
///     cuda.memcpy_htod(&[1.0f32, 2.0, 3.0, 4.0], &mut lent)?; // a real cudarc call
/// }
/// assert_eq!(stream.clone_dtoh(&buf)?, [1.0, 2.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub trait AsCuda<T>: crate::driver::DevicePtr<T> {
    /// Lends the slice for reading, bound to the CUDA stream paired with the slice's cudash
    /// stream. Blocks until cudash work that writes it has finished. Dropping the result
    /// blocks until that CUDA stream has finished, and the borrow keeps cudash from writing
    /// meanwhile.
    ///
    /// Returns [`Error::NoMatchingCudaDevice`] without CUDA, [`Error::ExportUnsupported`] if
    /// the slice's memory isn't exportable, and [`Error::InvalidLength`] if it's empty.
    fn as_cuda(&self) -> Result<Lent<'_, T>> {
        self.as_cuda_with(&self.stream().as_cuda()?)
    }

    /// [`Self::as_cuda`] for writing as well. Blocks until all cudash work using the slice
    /// has finished, and the borrow keeps cudash from using it until the result is dropped.
    fn as_cuda_mut(&mut self) -> Result<LentMut<'_, T>>
    where
        Self: crate::driver::DevicePtrMut<T>,
    {
        let stream = self.stream().as_cuda()?;
        self.as_cuda_mut_with(&stream)
    }

    /// [`Self::as_cuda`] bound to `stream` instead. Returns [`Error::DeviceMismatch`] if
    /// `stream` is on another GPU.
    fn as_cuda_with(&self, stream: &Arc<CudaStream>) -> Result<Lent<'_, T>> {
        Ok(Lent {
            cuda: import(self, stream, false)?,
            _slice: std::marker::PhantomData,
        })
    }

    /// [`Self::as_cuda_mut`] bound to `stream` instead.
    fn as_cuda_mut_with(&mut self, stream: &Arc<CudaStream>) -> Result<LentMut<'_, T>>
    where
        Self: crate::driver::DevicePtrMut<T>,
    {
        Ok(LentMut {
            cuda: import(self, stream, true)?,
            _slice: std::marker::PhantomData,
        })
    }
}

/// Slices and views. Views lend only their own range of the allocation.
impl<T, S: crate::driver::DevicePtr<T> + ?Sized> AsCuda<T> for S {}

// The first lend of a buffer to a CUDA context exports and imports the whole allocation; the
// buffer keeps that mapping, so later lends only wait and wrap it.
// shortcut: no queue family ownership transfer to and from VK_QUEUE_FAMILY_EXTERNAL,
// which the spec asks for; buffers have no layout, and NVIDIA's samples skip it too.
fn import<T>(
    slice: &(impl crate::driver::DevicePtr<T> + ?Sized),
    stream: &Arc<CudaStream>,
    write: bool,
) -> Result<buffer::Imported<T>> {
    let raw = slice.raw();
    let buffer = raw.buffer;
    if uuid_bytes(stream.context().uuid()?) != buffer.device.uuid {
        return Err(Error::DeviceMismatch);
    }
    if slice.is_empty() {
        return Err(Error::InvalidLength);
    }
    let Some(crate::driver::device::Dedicated(dedicated)) = buffer.export else {
        return Err(Error::ExportUnsupported);
    };
    // Waits for reads too, which only `as_cuda_mut` needs. Lending is rare enough.
    crate::driver::stream::wait_for_gpu(buffer)?;
    // A lend inside a capture is replayed with the graph, which has to keep and order it.
    let core = &slice.stream().core;
    if graph::lend_is_captured(core, stream) {
        if let Some(capture) = &mut core.state.lock().unwrap().capture {
            capture.uses(buffer, write);
        }
    } else if !matches!(
        stream.capture_status()?,
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    ) {
        // Nothing would keep the buffer alive for the CUDA graph's replays.
        let unsupported = sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
        return Err(Error::Cuda(DriverError(unsupported)));
    }
    let context_id = Arc::as_ptr(stream.context()) as usize;
    let mut mappings = buffer
        .cuda_mappings
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mapping = if let Some(mapping) = mappings.get(&context_id) {
        mapping.clone()
    } else {
        let exporter = vulkan::VulkanExporter::new(&buffer.device)?;
        let fd = exporter.export_memory(buffer.memory)?;
        // Importing queues no stream work, but CUDA refuses it during a GLOBAL or THREAD_LOCAL
        // capture unless this thread opts out.
        let mapping = Arc::new(graph::relaxed(|| {
            buffer::Mapping::new(stream, fd, buffer.allocation_size, dedicated)
        })?);
        mappings.insert(context_id, mapping.clone());
        mapping
    };
    Ok(buffer::Imported::new(
        stream,
        mapping,
        raw.offset,
        slice.len(),
    ))
}
