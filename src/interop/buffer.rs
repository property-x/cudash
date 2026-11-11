use std::{
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::Deref,
    os::fd::{AsRawFd, OwnedFd},
    sync::Arc,
};

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaViewMut, DevicePtr, DevicePtrMut, DeviceSlice,
    LaunchArgs, PushKernelArg, SyncOnDrop,
    result::{self, DriverError},
    sys,
};

use crate::interop::Result;

/// A cudash slice lent to cudarc for reading, from
/// [`AsCuda::as_cuda`](crate::interop::AsCuda::as_cuda). It goes wherever cudarc takes a
/// `&CudaSlice`. Dropping it waits for the CUDA stream it's bound to.
pub struct Lent<'l, T> {
    pub(crate) cuda: Imported<T>,
    pub(crate) _slice: PhantomData<&'l crate::driver::CudaSlice<T>>,
}

/// A cudash slice lent to cudarc for writing, from
/// [`AsCuda::as_cuda_mut`](crate::interop::AsCuda::as_cuda_mut). It goes wherever cudarc
/// takes a `&mut CudaSlice` too. Dropping it waits for the CUDA stream it's bound to.
pub struct LentMut<'l, T> {
    pub(crate) cuda: Imported<T>,
    pub(crate) _slice: PhantomData<&'l mut crate::driver::CudaSlice<T>>,
}

impl<T> LentMut<'_, T> {
    /// A mutable view, for cudarc calls on part of the slice.
    pub fn as_view_mut(&mut self) -> CudaViewMut<'_, T> {
        self.cuda.slice.as_view_mut()
    }
}

/// A CUDA mapping retained by the Vulkan buffer for one CUDA context.
pub(crate) struct Mapping {
    device_ptr: sys::CUdeviceptr,
    memory: ImportedMemory,
}

impl Mapping {
    /// Imports and maps the full Vulkan allocation so later lends can use any slice length.
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        fd: OwnedFd,
        size: u64,
        dedicated: bool,
    ) -> Result<Self> {
        let memory = ImportedMemory::new(stream.context(), fd, size, dedicated)?;
        // SAFETY: `memory` is valid, and the range fits in the allocation.
        let device_ptr =
            unsafe { result::external_memory::get_mapped_buffer(memory.raw, 0, size) }?;
        Ok(Self { device_ptr, memory })
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        let ctx = &self.memory.ctx;
        ctx.record_err(ctx.bind_to_thread());
        // Every lend synchronizes before releasing its wrapper, so no CUDA work can use this
        // mapping once the owning Vulkan buffer drops it.
        ctx.record_err(unsafe { result::memory_free(self.device_ptr) });
    }
}

/// A per-lend cudarc slice wrapper around a buffer-owned CUDA mapping.
pub(crate) struct Imported<T> {
    // The mapping is retained by the Vulkan buffer; this Arc keeps it alive for the lend too.
    slice: ManuallyDrop<CudaSlice<T>>,
    mapping: Arc<Mapping>,
}

impl<T> Imported<T> {
    pub(crate) fn new(stream: &Arc<CudaStream>, mapping: Arc<Mapping>, len: usize) -> Self {
        // SAFETY: the cached mapping covers the full Vulkan allocation and the slice borrow
        // keeps that allocation alive for this wrapper.
        let slice = unsafe { stream.upgrade_device_ptr::<T>(mapping.device_ptr, len) };
        Self {
            slice: ManuallyDrop::new(slice),
            mapping,
        }
    }
}

impl<T> Drop for Imported<T> {
    fn drop(&mut self) {
        // SAFETY: the temporary wrapper is no longer used and must not free the cached pointer.
        let slice = unsafe { ManuallyDrop::take(&mut self.slice) };
        let stream = slice.stream().clone();
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let _ = slice.leak();
        ctx.record_err(stream.synchronize());
        let _ = &self.mapping;
    }
}

// These let the types go anywhere cudarc accepts a CudaSlice. There's no DerefMut, since
// swapping another CudaSlice in would break `Imported`'s Drop.
macro_rules! forward_to_cuda_slice {
    ($name:ident $(<$l:lifetime>)?) => {
        impl<$($l,)? T> Deref for $name<$($l,)? T> {
            type Target = CudaSlice<T>;
            fn deref(&self) -> &CudaSlice<T> {
                &self.cuda.slice
            }
        }

        impl<$($l,)? T> DeviceSlice<T> for $name<$($l,)? T> {
            fn len(&self) -> usize {
                DeviceSlice::len(&*self.cuda.slice)
            }
            fn stream(&self) -> &Arc<CudaStream> {
                self.cuda.slice.stream()
            }
        }

        impl<$($l,)? T> DevicePtr<T> for $name<$($l,)? T> {
            fn device_ptr<'a>(
                &'a self,
                stream: &'a CudaStream,
            ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
                self.cuda.slice.device_ptr(stream)
            }
        }

        // SAFETY: forwards to cudarc's impl for the inner slice, which outlives 'b.
        unsafe impl<'a, 'b: 'a, $($l,)? T> PushKernelArg<&'b $name<$($l,)? T>> for LaunchArgs<'a> {
            #[inline(always)]
            fn arg(&mut self, arg: &'b $name<$($l,)? T>) -> &mut Self {
                self.arg(&*arg.cuda.slice)
            }
        }
    };
}

macro_rules! forward_to_cuda_slice_mut {
    ($name:ident $(<$l:lifetime>)?) => {
        impl<$($l,)? T> DevicePtrMut<T> for $name<$($l,)? T> {
            fn device_ptr_mut<'a>(
                &'a mut self,
                stream: &'a CudaStream,
            ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
                self.cuda.slice.device_ptr_mut(stream)
            }
        }

        // SAFETY: forwards to cudarc's impl for the inner slice, which outlives 'b.
        unsafe impl<'a, 'b: 'a, $($l,)? T> PushKernelArg<&'b mut $name<$($l,)? T>>
            for LaunchArgs<'a>
        {
            #[inline(always)]
            fn arg(&mut self, arg: &'b mut $name<$($l,)? T>) -> &mut Self {
                self.arg(&mut *arg.cuda.slice)
            }
        }
    };
}

forward_to_cuda_slice!(Lent<'l>);
forward_to_cuda_slice!(LentMut<'l>);
forward_to_cuda_slice_mut!(LentMut<'l>);

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
        // SAFETY: the mapping is freed in Imported::drop, before this runs.
        self.ctx
            .record_err(unsafe { result::external_memory::destroy_external_memory(self.raw) });
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LentMut<'static, f32>>();
};
