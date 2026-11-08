//! Device memory: slices, views, and the traits that let calls take any of them.

use std::{
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    sync::Arc,
};

use ash::vk;

use super::{
    DriverError,
    context::CudaContext,
    device::Buffer,
    stream::{CudaStream, SyncOnDrop},
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
    fn stream(&self) -> &Arc<CudaStream>;
}

/// Device memory that can be read: [`CudaSlice`], [`CudaView`] or [`CudaViewMut`].
pub trait DevicePtr<T>: DeviceSlice<T> {
    #[doc(hidden)]
    fn raw(&self) -> RawRange<'_>;

    /// The buffer device address of the first element, for reading on `stream`, as in
    /// cudarc. `stream` waits for earlier writes from other streams. Drop the guard after
    /// queuing the work that uses the address: the stream then records the read and keeps
    /// the memory alive until that work has finished.
    ///
    /// The address is only valid on this context's Vulkan device, not in CUDA.
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (u64, SyncOnDrop<'a>) {
        let raw = self.raw();
        let address = raw.buffer.address + raw.offset;
        (address, SyncOnDrop::new(stream, raw.buffer, false))
    }
}

/// Device memory that can be written: [`CudaSlice`] or [`CudaViewMut`].
pub trait DevicePtrMut<T>: DeviceSlice<T> {
    #[doc(hidden)]
    fn raw_mut(&mut self) -> RawRange<'_>;

    /// [`DevicePtr::device_ptr`] for writing: `stream` also waits for earlier reads.
    fn device_ptr_mut<'a>(&'a mut self, stream: &'a CudaStream) -> (u64, SyncOnDrop<'a>) {
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

/// `[T]` on the GPU, like cudarc's `CudaSlice`. Freed once work using it has finished,
/// without blocking.
pub struct CudaSlice<T> {
    buffer: Arc<Buffer>,
    len: usize,
    stream: Arc<CudaStream>,
    marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for CudaSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaSlice")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<T> std::fmt::Debug for CudaView<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaView")
            .field("byte_offset", &self.offset)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<T> std::fmt::Debug for CudaViewMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaViewMut")
            .field("byte_offset", &self.offset)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

// SAFETY: this only holds GPU memory, never a host `T`.
unsafe impl<T> Send for CudaSlice<T> {}
unsafe impl<T> Sync for CudaSlice<T> {}

impl<T> CudaSlice<T> {
    pub(crate) fn from_buffer(buffer: Arc<Buffer>, len: usize, stream: Arc<CudaStream>) -> Self {
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

    pub fn context(&self) -> &Arc<CudaContext> {
        self.stream.context()
    }

    /// The stream this slice was allocated on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The Vulkan buffer behind this slice, for use alongside cudash. Order that work
    /// against this slice's stream yourself.
    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer.buffer
    }

    pub fn as_view(&self) -> CudaView<'_, T> {
        CudaView {
            buffer: &self.buffer,
            offset: 0,
            len: self.len,
            stream: &self.stream,
            marker: PhantomData,
        }
    }

    pub fn as_view_mut(&mut self) -> CudaViewMut<'_, T> {
        CudaViewMut {
            buffer: &self.buffer,
            offset: 0,
            len: self.len,
            stream: &self.stream,
            marker: PhantomData,
        }
    }

    /// # Panics
    /// If `bounds` is out of range.
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> CudaView<'_, T> {
        self.as_view().slice(bounds)
    }

    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<CudaView<'_, T>> {
        self.as_view().try_slice(bounds)
    }

    /// The same memory as `len` elements of `S`, or `None` if that needs more bytes than
    /// `self` has.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`, and the view's start must be aligned for `S`
    /// wherever a kernel reads it through a pointer.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'_, S>> {
        unsafe { self.as_view().transmute(len) }
    }

    /// [`Self::transmute`] for writing.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`, and the view's start must be aligned for `S`
    /// wherever a kernel reads it through a pointer.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'_, S>> {
        unsafe { self.as_view_mut().into_transmute_mut(len) }
    }

    /// # Panics
    /// If `bounds` is out of range.
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        self.try_slice_mut(bounds).expect("range out of bounds")
    }

    pub fn try_slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<CudaViewMut<'_, T>> {
        self.as_view_mut().try_into_slice_mut(bounds)
    }

    /// # Panics
    /// If `mid` is out of range.
    pub fn split_at(&self, mid: usize) -> (CudaView<'_, T>, CudaView<'_, T>) {
        self.as_view().split_at(mid)
    }

    pub fn try_split_at(&self, mid: usize) -> Option<(CudaView<'_, T>, CudaView<'_, T>)> {
        self.as_view().try_split_at(mid)
    }

    /// # Panics
    /// If `mid` is out of range.
    pub fn split_at_mut(&mut self, mid: usize) -> (CudaViewMut<'_, T>, CudaViewMut<'_, T>) {
        self.try_split_at_mut(mid).expect("index out of bounds")
    }

    pub fn try_split_at_mut(
        &mut self,
        mid: usize,
    ) -> Option<(CudaViewMut<'_, T>, CudaViewMut<'_, T>)> {
        self.as_view_mut().try_into_split_at_mut(mid)
    }
}

impl<T: DeviceRepr> CudaSlice<T> {
    /// Allocates a copy of this slice and schedules a device-to-device copy.
    pub fn try_clone(&self) -> Result<Self, DriverError> {
        self.stream.clone_dtod(self)
    }
}

impl<T: DeviceRepr> Clone for CudaSlice<T> {
    fn clone(&self) -> Self {
        self.try_clone().unwrap()
    }
}

impl<T: DeviceRepr> TryFrom<CudaSlice<T>> for Vec<T> {
    type Error = DriverError;
    fn try_from(value: CudaSlice<T>) -> Result<Self, DriverError> {
        value.stream.clone_dtoh(&value)
    }
}

impl<T> DeviceSlice<T> for CudaSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T> DevicePtr<T> for CudaSlice<T> {
    fn raw(&self) -> RawRange<'_> {
        RawRange {
            buffer: &self.buffer,
            offset: 0,
        }
    }
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn raw_mut(&mut self) -> RawRange<'_> {
        RawRange {
            buffer: &self.buffer,
            offset: 0,
        }
    }
}

/// `&[T]` on the GPU, like cudarc's `CudaView`.
pub struct CudaView<'a, T> {
    buffer: &'a Arc<Buffer>,
    offset: u64,
    len: usize,
    stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a [T]>,
}

impl<T> Clone for CudaView<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for CudaView<'_, T> {}

// SAFETY: this only refers to GPU memory, never a host `T`.
unsafe impl<T> Send for CudaView<'_, T> {}
unsafe impl<T> Sync for CudaView<'_, T> {}

impl<'a, T> CudaView<'a, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn stream(&self) -> &'a Arc<CudaStream> {
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
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'a, S>> {
        fits::<S>(len, self.num_bytes()).then_some(CudaView {
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

impl<T> DeviceSlice<T> for CudaView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

impl<T> DevicePtr<T> for CudaView<'_, T> {
    fn raw(&self) -> RawRange<'_> {
        RawRange {
            buffer: self.buffer,
            offset: self.offset,
        }
    }
}

/// `&mut [T]` on the GPU, like cudarc's `CudaViewMut`.
pub struct CudaViewMut<'a, T> {
    buffer: &'a Arc<Buffer>,
    offset: u64,
    len: usize,
    stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a mut [T]>,
}

// SAFETY: this only refers to GPU memory, never a host `T`.
unsafe impl<T> Send for CudaViewMut<'_, T> {}
unsafe impl<T> Sync for CudaViewMut<'_, T> {}

impl<'a, T> CudaViewMut<'a, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }

    pub fn as_view(&self) -> CudaView<'_, T> {
        CudaView {
            buffer: self.buffer,
            offset: self.offset,
            len: self.len,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    pub fn as_view_mut(&mut self) -> CudaViewMut<'_, T> {
        CudaViewMut {
            buffer: self.buffer,
            offset: self.offset,
            len: self.len,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// [`CudaView::transmute`] of this view.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`, and the view's start must be aligned for `S`
    /// wherever a kernel reads it through a pointer.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'_, S>> {
        unsafe { self.as_view().transmute(len) }
    }

    /// [`CudaView::transmute`] for writing.
    ///
    /// # Safety
    /// The bytes must be valid values of `S`, and the view's start must be aligned for `S`
    /// wherever a kernel reads it through a pointer.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<CudaViewMut<'_, S>> {
        unsafe { self.as_view_mut().into_transmute_mut(len) }
    }

    /// # Safety
    /// As [`Self::transmute_mut`].
    unsafe fn into_transmute_mut<S>(self, len: usize) -> Option<CudaViewMut<'a, S>> {
        fits::<S>(len, self.num_bytes()).then_some(CudaViewMut {
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
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> CudaView<'_, T> {
        self.as_view().slice(bounds)
    }

    /// # Panics
    /// If `bounds` is out of range.
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        self.try_slice_mut(bounds).expect("range out of bounds")
    }

    pub fn try_slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<CudaViewMut<'_, T>> {
        self.as_view_mut().try_into_slice_mut(bounds)
    }

    /// # Panics
    /// If `mid` is out of range.
    pub fn split_at_mut(&mut self, mid: usize) -> (CudaViewMut<'_, T>, CudaViewMut<'_, T>) {
        self.try_split_at_mut(mid).expect("index out of bounds")
    }

    pub fn try_split_at_mut(
        &mut self,
        mid: usize,
    ) -> Option<(CudaViewMut<'_, T>, CudaViewMut<'_, T>)> {
        self.as_view_mut().try_into_split_at_mut(mid)
    }
}

impl<T> DeviceSlice<T> for CudaViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

impl<T> DevicePtr<T> for CudaViewMut<'_, T> {
    fn raw(&self) -> RawRange<'_> {
        RawRange {
            buffer: self.buffer,
            offset: self.offset,
        }
    }
}

impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {
    fn raw_mut(&mut self) -> RawRange<'_> {
        RawRange {
            buffer: self.buffer,
            offset: self.offset,
        }
    }
}
