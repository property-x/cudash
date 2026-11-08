//! Host memory the GPU copies to and from directly: pinned slices, like cudarc's
//! `PinnedHostSlice`, and registered ranges, like CUDA's `cuMemHostRegister`.

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use super::{
    DriverError,
    context::CudaContext,
    device::Buffer,
    slice::{DeviceRepr, HostSlice, RawRange, ValidAsZeroBits},
    stream::wait_for_gpu,
};

/// Host memory the GPU can copy to and from without staging, like cudarc's
/// `PinnedHostSlice`. Allocate with [`CudaContext::alloc_pinned`].
///
/// Copies into and out of it don't wait for the GPU. The accessors do, so they never see
/// a copy half done.
pub struct PinnedHostSlice<T> {
    buffer: Arc<Buffer>,
    len: usize,
    ctx: Arc<CudaContext>,
    marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for PinnedHostSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedHostSlice")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

// SAFETY: the memory is only reached through `&self` and `&mut self`, as for a `Vec<T>`.
unsafe impl<T: Send> Send for PinnedHostSlice<T> {}
unsafe impl<T: Sync> Sync for PinnedHostSlice<T> {}

impl CudaContext {
    /// Allocates host memory that copies use directly, without staging.
    ///
    /// # Safety
    /// The memory is uninitialized.
    pub unsafe fn alloc_pinned<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<PinnedHostSlice<T>, DriverError> {
        let bytes = len.checked_mul(size_of::<T>()).ok_or(DriverError::Vulkan(
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ))?;
        Ok(PinnedHostSlice {
            buffer: Buffer::staging(&self.device, bytes as u64)?,
            len,
            ctx: self.clone(),
            marker: PhantomData,
        })
    }
}

impl<T> PinnedHostSlice<T> {
    /// The context this was created in.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn ptr(&self) -> *mut T {
        // Empty buffers aren't mapped, and slices still need an aligned pointer.
        self.buffer
            .mapped_ptr()
            .map_or(std::ptr::dangling_mut(), |p| p.as_ptr().cast())
    }
}

impl<T: ValidAsZeroBits> PinnedHostSlice<T> {
    /// Waits for copies using this memory, then returns a pointer to it.
    pub fn as_ptr(&self) -> Result<*const T, DriverError> {
        wait_for_gpu(&self.buffer)?;
        Ok(self.ptr())
    }

    /// Waits for copies using this memory, then returns a pointer to it.
    pub fn as_mut_ptr(&mut self) -> Result<*mut T, DriverError> {
        wait_for_gpu(&self.buffer)?;
        Ok(self.ptr())
    }

    /// Waits for copies using this memory, then returns it.
    pub fn as_slice(&self) -> Result<&[T], DriverError> {
        wait_for_gpu(&self.buffer)?;
        // SAFETY: mapped for `len` elements, the GPU is done, and `T` is valid as any bits.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr(), self.len) })
    }

    /// Waits for copies using this memory, then returns it.
    pub fn as_mut_slice(&mut self) -> Result<&mut [T], DriverError> {
        wait_for_gpu(&self.buffer)?;
        // SAFETY: as above, and `&mut self` makes it exclusive.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr(), self.len) })
    }
}

impl<T> HostSlice<T> for PinnedHostSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn as_host_slice(&self) -> &[T] {
        wait_for_gpu(&self.buffer).expect("waiting for the GPU");
        // SAFETY: as in `as_slice`. Only reachable through `memcpy_*`, which write `T`s.
        unsafe { std::slice::from_raw_parts(self.ptr(), self.len) }
    }
    fn as_host_slice_mut(&mut self) -> &mut [T] {
        wait_for_gpu(&self.buffer).expect("waiting for the GPU");
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts_mut(self.ptr(), self.len) }
    }
    fn pinned(&self) -> Option<RawRange<'_>> {
        Some(RawRange {
            buffer: &self.buffer,
            offset: 0,
        })
    }
}

/// Host memory imported for the GPU by [`CudaContext::host_register`].
pub(crate) struct Registration {
    /// The registered bytes, `start..end`.
    pub(crate) start: usize,
    pub(crate) end: usize,
    /// The imported pages around them, and where `start` is inside.
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) offset: u64,
}

impl Drop for Registration {
    fn drop(&mut self) {
        // Copies don't keep the import alive, so it must outlast them here. An error
        // means the device is lost, and nothing runs anyway.
        let _ = wait_for_gpu(&self.buffer);
    }
}

pub(crate) type Registry = Mutex<Vec<Registration>>;

impl CudaContext {
    /// Lets copies read `data` directly, without staging, like CUDA's
    /// `cuMemHostRegister` (cudarc only has it in `sys`). After this, any `memcpy_htod` or
    /// `clone_htod` from inside `data` returns without waiting for the GPU or copying on
    /// the CPU. Empty slices aren't registered.
    ///
    /// Needs `VK_EXT_external_memory_host`, and doesn't work for every kind of memory: on
    /// RADV, `static` data fails with a Vulkan error.
    ///
    /// # Safety
    /// Until [`Self::host_unregister`] returns, `data` must stay allocated and not move,
    /// and mustn't be written while a copy from it may be running: synchronize the stream
    /// first. As with CUDA, a copy reads the memory when the GPU gets to it, not when it's
    /// queued.
    pub unsafe fn host_register<T: DeviceRepr>(&self, data: &[T]) -> Result<(), DriverError> {
        let start = data.as_ptr() as usize;
        let end = start + size_of_val(data);
        if start == end {
            return Ok(());
        }
        let mut registry = self.registered.lock().unwrap();
        if registry.iter().any(|r| r.start < end && start < r.end) {
            return Err(DriverError::HostMemoryAlreadyRegistered);
        }
        // SAFETY: the caller keeps `data` alive until it's unregistered, which drops this,
        // and copies only read registered bytes.
        let (buffer, offset) = unsafe {
            let bytes = std::slice::from_raw_parts(start as *const u8, end - start);
            Buffer::import_host(&self.device, bytes)
        }?;
        registry.push(Registration {
            start,
            end,
            buffer,
            offset,
        });
        Ok(())
    }

    /// Undoes [`Self::host_register`] for the range starting at `ptr`, like CUDA's
    /// `cuMemHostUnregister`. Waits for copies still reading it, so the memory is free to
    /// change or be freed once this returns.
    pub fn host_unregister<T>(&self, ptr: *const T) -> Result<(), DriverError> {
        let mut registry = self.registered.lock().unwrap();
        let index = registry
            .iter()
            .position(|r| r.start == ptr as usize)
            .ok_or(DriverError::HostMemoryNotRegistered)?;
        let registration = registry.swap_remove(index);
        drop(registry);
        wait_for_gpu(&registration.buffer)
    }
}
