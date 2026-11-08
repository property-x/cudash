use std::{
    mem::ManuallyDrop,
    ops::{Deref, RangeBounds},
    os::fd::{AsRawFd, OwnedFd},
    sync::Arc,
};

use ash::vk;
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaViewMut, DevicePtr, DevicePtrMut, DeviceRepr,
    DeviceSlice, LaunchArgs, PushKernelArg, SyncOnDrop, ValidAsZeroBits,
    result::{self, DriverError},
    sys,
};

use crate::interop::{
    Error, Result,
    vulkan::{ExportedBuffer, VulkanExporter, acquire_barrier, release_barrier},
};

/// One block of GPU memory visible as both a `vk::Buffer` and a [`CudaSlice`].
///
/// Neither API knows about the other's work, so order access with a
/// [`SharedSemaphore`](crate::interop::SharedSemaphore). Before dropping this value,
/// make sure Vulkan work you submitted yourself has finished. CUDA work on its stream
/// is waited on automatically.
pub struct SharedBuffer<T> {
    // Unmapped in Drop, because CudaSlice's own Drop may use cuMemFreeAsync,
    // while mapped buffers must be released with cuMemFree. Fields then drop in
    // order: CUDA's import first, the Vulkan memory it refers to last.
    slice: ManuallyDrop<CudaSlice<T>>,
    _imported: ImportedMemory,
    vulkan: ExportedBuffer,
}

impl<T: DeviceRepr + ValidAsZeroBits> SharedBuffer<T> {
    pub(crate) fn new(
        exporter: &VulkanExporter,
        stream: &Arc<CudaStream>,
        len: usize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        // SAFETY: zeroed below, before anyone can read it.
        let mut buffer =
            unsafe { Self::new_uninit(exporter, stream, len, usage, memory_properties) }?;
        stream.memset_zeros(&mut buffer)?;
        stream.synchronize()?;
        Ok(buffer)
    }
}

impl<T> SharedBuffer<T> {
    /// # Safety
    /// Vulkan allocations start out undefined, so the caller must initialize
    /// every element before it's read.
    pub(crate) unsafe fn new_uninit(
        exporter: &VulkanExporter,
        stream: &Arc<CudaStream>,
        len: usize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        let num_bytes = len
            .checked_mul(size_of::<T>())
            .filter(|&n| n > 0)
            .ok_or(Error::InvalidLength)? as u64;
        let vulkan = exporter.create_buffer(num_bytes, usage, memory_properties)?;
        let fd = exporter.export_memory(&vulkan)?;
        let imported = ImportedMemory::new(stream.context(), fd, vulkan.size, vulkan.dedicated)?;

        // SAFETY: `imported` is valid, and the range fits in the allocation.
        let device_ptr =
            unsafe { result::external_memory::get_mapped_buffer(imported.raw, 0, num_bytes) }?;
        // SAFETY: the mapping holds `len` elements.
        let slice = unsafe { stream.upgrade_device_ptr::<T>(device_ptr, len) };
        Ok(Self {
            slice: ManuallyDrop::new(slice),
            _imported: imported,
            vulkan,
        })
    }
}

impl<T> SharedBuffer<T> {
    /// The buffer handle for use in Vulkan commands.
    pub fn vk_buffer(&self) -> vk::Buffer {
        self.vulkan.buffer
    }

    /// The memory backing [`Self::vk_buffer`].
    pub fn vk_memory(&self) -> vk::DeviceMemory {
        self.vulkan.memory
    }

    /// A mutable view, like `CudaSlice::as_view_mut`. Immutable views come from
    /// deref, as do `len`, `slice`, `split_at` and the rest of `CudaSlice`.
    pub fn as_view_mut(&mut self) -> CudaViewMut<'_, T> {
        self.slice.as_view_mut()
    }

    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        self.slice.slice_mut(bounds)
    }

    pub fn try_slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<CudaViewMut<'_, T>> {
        self.slice.try_slice_mut(bounds)
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (CudaViewMut<'_, T>, CudaViewMut<'_, T>) {
        self.slice.split_at_mut(mid)
    }

    pub fn try_split_at_mut(
        &mut self,
        mid: usize,
    ) -> Option<(CudaViewMut<'_, T>, CudaViewMut<'_, T>)> {
        self.slice.try_split_at_mut(mid)
    }

    /// A barrier that hands the buffer from CUDA to `queue_family`. Record it
    /// with `cmd_pipeline_barrier` before the first Vulkan use after CUDA work.
    ///
    /// Stages aren't part of the barrier. Pass `TOP_OF_PIPE` as the source and
    /// the stages that will use the buffer as the destination.
    pub fn vk_acquire_barrier(&self, queue_family: u32) -> vk::BufferMemoryBarrier<'static> {
        acquire_barrier(self.vulkan.buffer, queue_family)
    }

    /// A barrier that hands the buffer from `queue_family` to CUDA. Record it
    /// with `cmd_pipeline_barrier` after the last Vulkan use before CUDA work.
    ///
    /// Stages aren't part of the barrier. Pass the stages that used the buffer
    /// as the source and `BOTTOM_OF_PIPE` as the destination.
    pub fn vk_release_barrier(&self, queue_family: u32) -> vk::BufferMemoryBarrier<'static> {
        release_barrier(self.vulkan.buffer, queue_family)
    }
}

impl<T> Deref for SharedBuffer<T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &CudaSlice<T> {
        &self.slice
    }
}

// These let a SharedBuffer go anywhere cudarc accepts a CudaSlice. There's no
// DerefMut to CudaSlice, since swapping one in would break the Drop below.
impl<T> DeviceSlice<T> for SharedBuffer<T> {
    fn len(&self) -> usize {
        DeviceSlice::len(&*self.slice)
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.slice.stream()
    }
}

impl<T> DevicePtr<T> for SharedBuffer<T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        self.slice.device_ptr(stream)
    }
}

impl<T> DevicePtrMut<T> for SharedBuffer<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        self.slice.device_ptr_mut(stream)
    }
}

// SAFETY: forwards to cudarc's impls for the inner slice, which outlives 'b.
unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b SharedBuffer<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b SharedBuffer<T>) -> &mut Self {
        self.arg(&*arg.slice)
    }
}

// SAFETY: forwards to cudarc's impls for the inner slice, which outlives 'b.
unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b mut SharedBuffer<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut SharedBuffer<T>) -> &mut Self {
        self.arg(&mut *arg.slice)
    }
}

impl<T> Drop for SharedBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: `slice` isn't used again after this.
        let slice = unsafe { ManuallyDrop::take(&mut self.slice) };
        let stream = slice.stream().clone();
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let device_ptr = slice.leak();
        ctx.record_err(stream.synchronize());
        // SAFETY: the stream is idle, and `leak` made us the pointer's only owner.
        ctx.record_err(unsafe { result::memory_free(device_ptr) });
    }
}

struct ImportedMemory {
    raw: sys::CUexternalMemory,
    ctx: Arc<CudaContext>,
}

// SAFETY: CUDA handles aren't tied to a thread, and Drop binds the context.
unsafe impl Send for ImportedMemory {}
unsafe impl Sync for ImportedMemory {}

impl ImportedMemory {
    fn new(
        ctx: &Arc<CudaContext>,
        fd: OwnedFd,
        size: u64,
        dedicated: bool,
    ) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let desc = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 { fd: fd.as_raw_fd() },
            size,
            flags: if dedicated {
                sys::CUDA_EXTERNAL_MEMORY_DEDICATED
            } else {
                0
            },
            reserved: [0; 16],
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: `desc` is fully initialized and `size` is the allocation size.
        unsafe { sys::cuImportExternalMemory(&mut raw, &desc) }.result()?;
        // CUDA owns the fd now, but only on success. On failure `fd` closes above.
        std::mem::forget(fd);
        Ok(Self {
            raw,
            ctx: ctx.clone(),
        })
    }
}

impl Drop for ImportedMemory {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        // SAFETY: the mapping is freed in SharedBuffer::drop, before this runs.
        self.ctx
            .record_err(unsafe { result::external_memory::destroy_external_memory(self.raw) });
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SharedBuffer<f32>>();
};
