use std::{
    mem::MaybeUninit,
    os::fd::{AsRawFd, OwnedFd},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use ash::vk;
use cudarc::driver::{CudaContext, CudaStream, result::DriverError, sys};

use crate::interop::{
    Result,
    vulkan::{ExportedSemaphore, VulkanExporter},
};

/// A timeline semaphore that both CUDA and Vulkan can signal and wait on.
///
/// Its value only ever goes up, and signal values are handed out by this type
/// so they stay increasing: each signal returns the value to wait for. Signals
/// must reach the GPU in the order their values were reserved, so don't race
/// them across threads.
///
/// Before dropping this value, make sure neither API still has work using it.
pub struct SharedSemaphore {
    last_reserved: AtomicU64,
    imported: ImportedSemaphore,
    vulkan: ExportedSemaphore,
}

impl SharedSemaphore {
    /// Creates a timeline semaphore on `ctx`'s GPU, usable from both APIs, starting at 0.
    pub fn new(ctx: &crate::driver::CudaContext) -> Result<Self> {
        let exporter = VulkanExporter::new(&ctx.device)?;
        let vulkan = exporter.create_semaphore()?;
        let fd = exporter.export_semaphore(&vulkan)?;
        let imported = ImportedSemaphore::new(&ctx.as_cuda()?, fd)?;
        Ok(Self {
            last_reserved: AtomicU64::new(0),
            imported,
            vulkan,
        })
    }

    /// The handle to use in Vulkan submissions, via
    /// `vk::TimelineSemaphoreSubmitInfo` or `vk::SemaphoreSubmitInfo`.
    pub fn vk_semaphore(&self) -> vk::Semaphore {
        self.vulkan.semaphore
    }

    /// Reserves the next value, for signaling from a Vulkan queue submission.
    pub fn next_value(&self) -> u64 {
        self.last_reserved.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Enqueues a signal on `stream`, after all work already on it. Returns the
    /// value that will be reached.
    pub fn cuda_signal(&self, stream: &CudaStream) -> Result<u64> {
        let value = self.next_value();
        // SAFETY: both params types are plain C data for which all zeros is valid.
        let mut params: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS =
            unsafe { MaybeUninit::zeroed().assume_init() };
        params.params.fence.value = value;
        stream.context().bind_to_thread()?;
        // SAFETY: the semaphore and stream are valid, and one element is passed.
        unsafe {
            sys::cuSignalExternalSemaphoresAsync(&self.imported.raw, &params, 1, stream.cu_stream())
        }
        .result()?;
        Ok(value)
    }

    /// Makes work enqueued on `stream` after this wait until `value` is reached.
    ///
    /// CUDA forbids issuing a wait before its signal has been issued, so submit the Vulkan
    /// work that signals `value` before calling this. See the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html),
    /// section 4.22.2.
    pub fn cuda_wait(&self, stream: &CudaStream, value: u64) -> Result<()> {
        // SAFETY: both params types are plain C data for which all zeros is valid.
        let mut params: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS =
            unsafe { MaybeUninit::zeroed().assume_init() };
        params.params.fence.value = value;
        stream.context().bind_to_thread()?;
        // SAFETY: the semaphore and stream are valid, and one element is passed.
        unsafe {
            sys::cuWaitExternalSemaphoresAsync(&self.imported.raw, &params, 1, stream.cu_stream())
        }
        .result()?;
        Ok(())
    }

    /// Signals from the host through Vulkan. Returns the value reached.
    pub fn vk_signal(&self) -> Result<u64> {
        let value = self.next_value();
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.vulkan.semaphore)
            .value(value);
        unsafe { self.vulkan.device.device.signal_semaphore(&info) }?;
        Ok(value)
    }

    /// Blocks the host until `value` is reached. Returns false on timeout.
    pub fn vk_wait(&self, value: u64, timeout_ns: u64) -> Result<bool> {
        let semaphores = [self.vulkan.semaphore];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        match unsafe { self.vulkan.device.device.wait_semaphores(&info, timeout_ns) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// The value the semaphore has reached so far.
    pub fn completed_value(&self) -> Result<u64> {
        Ok(unsafe {
            (self.vulkan.device.device).get_semaphore_counter_value(self.vulkan.semaphore)
        }?)
    }
}

struct ImportedSemaphore {
    raw: sys::CUexternalSemaphore,
    ctx: Arc<CudaContext>,
}

// SAFETY: CUDA handles aren't tied to a thread, and every use binds the context.
unsafe impl Send for ImportedSemaphore {}
unsafe impl Sync for ImportedSemaphore {}

impl ImportedSemaphore {
    fn new(ctx: &Arc<CudaContext>, fd: OwnedFd) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let desc = sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
            type_: sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD,
            handle: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1 { fd: fd.as_raw_fd() },
            flags: 0,
            reserved: [0; 16],
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: `desc` is fully initialized and describes a timeline semaphore.
        unsafe { sys::cuImportExternalSemaphore(&mut raw, &desc) }.result()?;
        // CUDA owns the fd now, but only on success. On failure `fd` closes above.
        std::mem::forget(fd);
        Ok(Self {
            raw,
            ctx: ctx.clone(),
        })
    }
}

impl Drop for ImportedSemaphore {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        // SAFETY: `raw` is valid and this is its only destroy.
        self.ctx
            .record_err(unsafe { sys::cuDestroyExternalSemaphore(self.raw) }.result());
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SharedSemaphore>();
};
