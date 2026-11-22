//! Device memory: slices, views, and the traits that let calls take any of them.

use std::{
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    sync::Arc,
};

use crate::ash::vk;
#[cfg(feature = "cuda")]
use crate::compat::load_ext;

use super::{
    DriverError,
    context::GpuContext,
    device::Buffer,
    stream::{GpuStream, SyncOnDrop},
};

/// Types that are plain data with the same layout on the GPU and the host.
///
/// # Safety
/// The type must be `Copy`-like plain data with no pointers to host memory, and with
/// a layout matching what kernels expect. Use `#[repr(C)]` for structs.
pub unsafe trait DeviceRepr {}

/// Types for which all-zero bytes are a valid value.
///
/// # Safety
/// All-zero bytes must be a valid value of the type.
pub unsafe trait ValidAsZeroBits {}

macro_rules! impl_plain_data {
    ($($t:ty),*) => {$(
        unsafe impl DeviceRepr for $t {}
        unsafe impl ValidAsZeroBits for $t {}
    )*};
}
impl_plain_data!(
    bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64
);

unsafe impl<T: DeviceRepr, const N: usize> DeviceRepr for [T; N] {}
unsafe impl<T: ValidAsZeroBits, const N: usize> ValidAsZeroBits for [T; N] {}

/// Host memory that copies can read from or write to.
pub trait HostSlice<T> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[doc(hidden)]
    fn as_host_slice(&self) -> &[T];
    #[doc(hidden)]
    fn as_host_slice_mut(&mut self) -> &mut [T];
    /// The GPU buffer behind pinned memory, which copies use directly.
    #[doc(hidden)]
    fn pinned(&self) -> Option<RawRange<'_>> {
        None
    }
}

impl<T> HostSlice<T> for [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    fn as_host_slice(&self) -> &[T] {
        self
    }
    fn as_host_slice_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T, const N: usize> HostSlice<T> for [T; N] {
    fn len(&self) -> usize {
        N
    }
    fn as_host_slice(&self) -> &[T] {
        self
    }
    fn as_host_slice_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T> HostSlice<T> for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn as_host_slice(&self) -> &[T] {
        self
    }
    fn as_host_slice_mut(&mut self) -> &mut [T] {
        self
    }
}

/// Where a slice or view's bytes live.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct RawRange<'a> {
    pub(crate) buffer: &'a Arc<Buffer>,
    pub(crate) offset: u64,
}

/// Device memory of some length: a slice or a view.
pub trait DeviceSlice<T> {
    fn len(&self) -> usize;
    fn num_bytes(&self) -> usize {
        self.len() * size_of::<T>()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn stream(&self) -> &Arc<GpuStream>;
}

/// Device memory that can be read: [`GpuSlice`], [`GpuView`] or [`GpuViewMut`].
pub trait DevicePtr<T>: DeviceSlice<T> {
    #[doc(hidden)]
    fn raw(&self) -> RawRange<'_>;

    /// The buffer device address of the first element, for reading on `stream`, as in
    /// cudarc. `stream` waits for earlier writes from other streams. Drop the guard after
    /// queuing the work that uses the address: the stream then records the read and keeps
    /// the memory alive until that work has finished.
    ///
    /// The address is only valid on this context's Vulkan device, not in CUDA.
    fn device_ptr<'a>(&'a self, stream: &'a GpuStream) -> (u64, SyncOnDrop<'a>) {
        let raw = self.raw();
        let address = raw.buffer.address + raw.offset;
        (address, SyncOnDrop::new(stream, raw.buffer, false))
    }

    /// A `width * height` image over this memory, in rows, without copying. See [`GpuImage`](super::GpuImage).
    ///
    /// # Panics
    /// If this doesn't hold exactly `width * height` elements.
    fn as_image(&self, width: u32, height: u32) -> Result<super::GpuImage<T>, DriverError>
    where
        T: super::ImageFormat,
        Self: Sized,
    {
        super::image::image_over(self, width, height)
    }

    /// A copy of this memory as a new `width * height` image, on this slice's stream. For a
    /// view that follows the slice instead, see [`Self::as_image`].
    ///
    /// # Panics
    /// If this doesn't hold exactly `width * height` elements.
    fn to_image(&self, width: u32, height: u32) -> Result<super::GpuImage<T>, DriverError>
    where
        T: super::ImageFormat,
        Self: Sized,
    {
        self.stream().clone_to_image(self, width, height)
    }
}

/// Device memory that can be written: [`GpuSlice`] or [`GpuViewMut`].
pub trait DevicePtrMut<T>: DeviceSlice<T> {
    #[doc(hidden)]
    fn raw_mut(&mut self) -> RawRange<'_>;

    /// [`DevicePtr::device_ptr`] for writing: `stream` also waits for earlier reads.
    fn device_ptr_mut<'a>(&'a mut self, stream: &'a GpuStream) -> (u64, SyncOnDrop<'a>) {
        let raw = self.raw_mut();
        let address = raw.buffer.address + raw.offset;
        (address, SyncOnDrop::new(stream, raw.buffer, true))
    }
}

/// Whether `len` elements of `S` fit in `bytes`. Unlike cudarc, an overflowing `len` doesn't fit.
fn fits<S>(len: usize, bytes: usize) -> bool {
    len.checked_mul(size_of::<S>())
        .is_some_and(|needed| needed <= bytes)
}

/// Resolves range bounds against a length, returning `None` if they're out of range.
fn resolve(bounds: impl RangeBounds<usize>, len: usize) -> Option<(usize, usize)> {
    let start = match bounds.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n.checked_add(1)?,
        Bound::Unbounded => 0,
    };
    let end = match bounds.end_bound() {
        Bound::Included(&n) => n.checked_add(1)?,
        Bound::Excluded(&n) => n,
        Bound::Unbounded => len,
    };
    (start <= end && end <= len).then_some((start, end - start))
}

/// `[T]` on the GPU, like cudarc's `GpuSlice`. Freed once work using it has finished,
/// without blocking.
pub struct GpuSlice<T> {
    buffer: Arc<Buffer>,
    len: usize,
    stream: Arc<GpuStream>,
    marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for GpuSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuSlice")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<T> std::fmt::Debug for GpuView<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuView")
            .field("byte_offset", &self.offset)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<T> std::fmt::Debug for GpuViewMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuViewMut")
            .field("byte_offset", &self.offset)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

// SAFETY: this only holds GPU memory, never a host `T`.
unsafe impl<T> Send for GpuSlice<T> {}
unsafe impl<T> Sync for GpuSlice<T> {}

impl<T> GpuSlice<T> {
    pub(crate) fn from_buffer(buffer: Arc<Buffer>, len: usize, stream: Arc<GpuStream>) -> Self {
        Self {
            buffer,
            len,
            stream,
            marker: PhantomData,
        }
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

    pub fn ordinal(&self) -> usize {
        self.stream.context().ordinal()
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        self.stream.context()
    }

    /// The stream this slice was allocated on.
    pub fn stream(&self) -> &Arc<GpuStream> {
        &self.stream
    }

    /// The Vulkan buffer behind this slice, for use alongside cudash. Order that work
    /// against this slice's stream yourself.
    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer.buffer
    }

    /// Exports the memory of a slice from [`GpuStream::alloc_zeros_dma_buf`] as a new dma-buf fd, so another device on the same GPU, such
    /// as a compositor or UI renderer, can read it without a copy. The slice starts at offset 0
    /// of the dma-buf. Order the reader against this slice's stream yourself, and keep the slice
    /// alive while the dma-buf is in use.
    #[cfg(feature = "cuda")]
    pub fn export_dma_buf(&self) -> Result<std::os::fd::OwnedFd, DriverError> {
        use std::os::fd::FromRawFd;
        if !self.buffer.dma_buf {
            return Err(DriverError::MissingFeature(
                "VK_EXT_external_memory_dma_buf",
            ));
        }
        let device = &self.buffer.device;
        let functions = load_ext!(
            crate::ash::khr::external_memory_fd::Device,
            &device.instance,
            &device.device
        );
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(self.buffer.memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let fd = unsafe { functions.get_memory_fd(&info) }?;
        // SAFETY: Vulkan transfers ownership of the returned fd to the caller.
        Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) })
    }

    pub fn as_view(&self) -> GpuView<'_, T> {
        GpuView {
            buffer: &self.buffer,
            offset: 0,
            len: self.len,
            stream: &self.stream,
            marker: PhantomData,
        }
    }

    pub fn as_view_mut(&mut self) -> GpuViewMut<'_, T> {
        GpuViewMut {
            buffer: &self.buffer,
            offset: 0,
            len: self.len,
            stream: &self.stream,
            marker: PhantomData,
        }
    }

    /// # Panics
    /// If `bounds` is out of range.
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> GpuView<'_, T> {
        self.as_view().slice(bounds)
    }

    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<GpuView<'_, T>> {
        self.as_view().try_slice(bounds)
    }

    /// The same memory as `len` elements of `S`, or `None` if that needs more bytes than
    /// `self` has.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`, and the view's start must be aligned for `S`
    /// wherever a kernel reads it through a pointer.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<GpuView<'_, S>> {
        unsafe { self.as_view().transmute(len) }
    }

    /// [`Self::transmute`] for writing.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`, and the view's start must be aligned for `S`
    /// wherever a kernel reads it through a pointer.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<GpuViewMut<'_, S>> {
        unsafe { self.as_view_mut().into_transmute_mut(len) }
    }

    /// # Panics
    /// If `bounds` is out of range.
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> GpuViewMut<'_, T> {
        self.try_slice_mut(bounds).expect("range out of bounds")
    }

    pub fn try_slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<GpuViewMut<'_, T>> {
        self.as_view_mut().try_into_slice_mut(bounds)
    }

    /// # Panics
    /// If `mid` is out of range.
    pub fn split_at(&self, mid: usize) -> (GpuView<'_, T>, GpuView<'_, T>) {
        self.as_view().split_at(mid)
    }

    pub fn try_split_at(&self, mid: usize) -> Option<(GpuView<'_, T>, GpuView<'_, T>)> {
        self.as_view().try_split_at(mid)
    }

    /// # Panics
    /// If `mid` is out of range.
    pub fn split_at_mut(&mut self, mid: usize) -> (GpuViewMut<'_, T>, GpuViewMut<'_, T>) {
        self.try_split_at_mut(mid).expect("index out of bounds")
    }

    pub fn try_split_at_mut(
        &mut self,
        mid: usize,
    ) -> Option<(GpuViewMut<'_, T>, GpuViewMut<'_, T>)> {
        self.as_view_mut().try_into_split_at_mut(mid)
    }
}

impl<T: DeviceRepr> GpuSlice<T> {
    /// Allocates a copy of this slice and schedules a device-to-device copy.
    pub fn try_clone(&self) -> Result<Self, DriverError> {
        self.stream.clone_dtod(self)
    }
}

impl<T: DeviceRepr> Clone for GpuSlice<T> {
    fn clone(&self) -> Self {
        self.try_clone().unwrap()
    }
}

impl<T: DeviceRepr> TryFrom<GpuSlice<T>> for Vec<T> {
    type Error = DriverError;
    fn try_from(value: GpuSlice<T>) -> Result<Self, DriverError> {
        value.stream.clone_dtoh(&value)
    }
}

impl<T> DeviceSlice<T> for GpuSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<GpuStream> {
        &self.stream
    }
}

impl<T> DevicePtr<T> for GpuSlice<T> {
    fn raw(&self) -> RawRange<'_> {
        RawRange {
            buffer: &self.buffer,
            offset: 0,
        }
    }
}

impl<T> DevicePtrMut<T> for GpuSlice<T> {
    fn raw_mut(&mut self) -> RawRange<'_> {
        RawRange {
            buffer: &self.buffer,
            offset: 0,
        }
    }
}

/// `&[T]` on the GPU, like cudarc's `GpuView`.
pub struct GpuView<'a, T> {
    buffer: &'a Arc<Buffer>,
    offset: u64,
    len: usize,
    stream: &'a Arc<GpuStream>,
    marker: PhantomData<&'a [T]>,
}

impl<T> Clone for GpuView<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for GpuView<'_, T> {}

// SAFETY: this only refers to GPU memory, never a host `T`.
unsafe impl<T> Send for GpuView<'_, T> {}
unsafe impl<T> Sync for GpuView<'_, T> {}

impl<'a, T> GpuView<'a, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn stream(&self) -> &'a Arc<GpuStream> {
        self.stream
    }

    /// # Panics
    /// If `bounds` is out of range.
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> Self {
        self.try_slice(bounds).expect("range out of bounds")
    }

    /// The same memory as `len` elements of `S`, or `None` if that needs more bytes than
    /// `self` has.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`, and the view's start must be aligned for `S`
    /// wherever a kernel reads it through a pointer.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<GpuView<'a, S>> {
        fits::<S>(len, self.num_bytes()).then_some(GpuView {
            buffer: self.buffer,
            offset: self.offset,
            len,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<Self> {
        let (start, len) = resolve(bounds, self.len)?;
        let offset = self.offset + (start * size_of::<T>()) as u64;
        Some(Self {
            offset,
            len,
            ..*self
        })
    }

    /// # Panics
    /// If `mid` is out of range.
    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        self.try_split_at(mid).expect("index out of bounds")
    }

    pub fn try_split_at(&self, mid: usize) -> Option<(Self, Self)> {
        Some((self.try_slice(..mid)?, self.try_slice(mid..)?))
    }
}

impl<T> DeviceSlice<T> for GpuView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<GpuStream> {
        self.stream
    }
}

impl<T> DevicePtr<T> for GpuView<'_, T> {
    fn raw(&self) -> RawRange<'_> {
        RawRange {
            buffer: self.buffer,
            offset: self.offset,
        }
    }
}

/// `&mut [T]` on the GPU, like cudarc's `GpuViewMut`.
pub struct GpuViewMut<'a, T> {
    buffer: &'a Arc<Buffer>,
    offset: u64,
    len: usize,
    stream: &'a Arc<GpuStream>,
    marker: PhantomData<&'a mut [T]>,
}

// SAFETY: this only refers to GPU memory, never a host `T`.
unsafe impl<T> Send for GpuViewMut<'_, T> {}
unsafe impl<T> Sync for GpuViewMut<'_, T> {}

impl<'a, T> GpuViewMut<'a, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn stream(&self) -> &Arc<GpuStream> {
        self.stream
    }

    pub fn as_view(&self) -> GpuView<'_, T> {
        GpuView {
            buffer: self.buffer,
            offset: self.offset,
            len: self.len,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    pub fn as_view_mut(&mut self) -> GpuViewMut<'_, T> {
        GpuViewMut {
            buffer: self.buffer,
            offset: self.offset,
            len: self.len,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// [`GpuView::transmute`] of this view.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`, and the view's start must be aligned for `S`
    /// wherever a kernel reads it through a pointer.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<GpuView<'_, S>> {
        unsafe { self.as_view().transmute(len) }
    }

    /// [`GpuView::transmute`] for writing.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`, and the view's start must be aligned for `S`
    /// wherever a kernel reads it through a pointer.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<GpuViewMut<'_, S>> {
        unsafe { self.as_view_mut().into_transmute_mut(len) }
    }

    /// # Safety
    /// As [`Self::transmute_mut`].
    unsafe fn into_transmute_mut<S>(self, len: usize) -> Option<GpuViewMut<'a, S>> {
        fits::<S>(len, self.num_bytes()).then_some(GpuViewMut {
            buffer: self.buffer,
            offset: self.offset,
            len,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    fn try_into_slice_mut(self, bounds: impl RangeBounds<usize>) -> Option<Self> {
        let (start, len) = resolve(bounds, self.len)?;
        let offset = self.offset + (start * size_of::<T>()) as u64;
        Some(Self {
            offset,
            len,
            ..self
        })
    }

    fn try_into_split_at_mut(self, mid: usize) -> Option<(Self, Self)> {
        if mid > self.len {
            return None;
        }
        let tail = Self {
            buffer: self.buffer,
            offset: self.offset + (mid * size_of::<T>()) as u64,
            len: self.len - mid,
            stream: self.stream,
            marker: PhantomData,
        };
        Some((Self { len: mid, ..self }, tail))
    }

    /// # Panics
    /// If `bounds` is out of range.
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> GpuView<'_, T> {
        self.as_view().slice(bounds)
    }

    /// # Panics
    /// If `bounds` is out of range.
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> GpuViewMut<'_, T> {
        self.try_slice_mut(bounds).expect("range out of bounds")
    }

    pub fn try_slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<GpuViewMut<'_, T>> {
        self.as_view_mut().try_into_slice_mut(bounds)
    }

    /// # Panics
    /// If `mid` is out of range.
    pub fn split_at_mut(&mut self, mid: usize) -> (GpuViewMut<'_, T>, GpuViewMut<'_, T>) {
        self.try_split_at_mut(mid).expect("index out of bounds")
    }

    pub fn try_split_at_mut(
        &mut self,
        mid: usize,
    ) -> Option<(GpuViewMut<'_, T>, GpuViewMut<'_, T>)> {
        self.as_view_mut().try_into_split_at_mut(mid)
    }
}

impl<T> DeviceSlice<T> for GpuViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<GpuStream> {
        self.stream
    }
}

impl<T> DevicePtr<T> for GpuViewMut<'_, T> {
    fn raw(&self) -> RawRange<'_> {
        RawRange {
            buffer: self.buffer,
            offset: self.offset,
        }
    }
}

impl<T> DevicePtrMut<T> for GpuViewMut<'_, T> {
    fn raw_mut(&mut self) -> RawRange<'_> {
        RawRange {
            buffer: self.buffer,
            offset: self.offset,
        }
    }
}
