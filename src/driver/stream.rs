//! Streams: ordered GPU work, each operation chained on a timeline semaphore.

use std::{
    any::Any,
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use ash::vk;

use super::{
    DriverError,
    context::CudaContext,
    device::{Buffer, Device},
    slice::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, HostSlice, RawRange, ValidAsZeroBits},
};

struct Timeline {
    device: Arc<Device>,
    semaphore: vk::Semaphore,
}

impl Drop for Timeline {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_semaphore(self.semaphore, None) };
    }
}

/// A moment on some stream's timeline.
#[derive(Clone)]
pub(crate) struct Point {
    timeline: Arc<Timeline>,
    value: u64,
}

impl Point {
    fn wait(&self) -> Result<(), DriverError> {
        let (semaphores, values) = ([self.timeline.semaphore], [self.value]);
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.timeline.device.device.wait_semaphores(&info, u64::MAX) }?;
        Ok(())
    }
}

/// Blocks until the GPU has finished every submission that reads or writes `buffer`.
pub(crate) fn wait_for_gpu(buffer: &Buffer) -> Result<(), DriverError> {
    let points: Vec<Point> = {
        let tracking = buffer.tracking.lock().unwrap();
        tracking
            .write
            .iter()
            .chain(&tracking.reads)
            .cloned()
            .collect()
    };
    points.iter().try_for_each(Point::wait)
}

/// What a buffer's next user has to wait for. This plays the part of cudarc's read
/// and write events, and only matters when a buffer is used from several streams.
#[derive(Default)]
pub(crate) struct Tracking {
    write: Option<Point>,
    reads: Vec<Point>,
}

/// Anything that has to stay alive until the GPU has finished a submission.
pub(crate) type Keep = Arc<dyn Any + Send + Sync>;

struct Pending {
    value: u64,
    command_buffer: vk::CommandBuffer,
    _keep: Vec<Keep>,
}

struct State {
    last_value: u64,
    /// What the next submission waits for. Starts on the source stream if forked.
    last: Option<Point>,
    pending: VecDeque<Pending>,
}

/// The part of a stream that's shared by every handle to it.
pub(crate) struct StreamCore {
    device: Arc<Device>,
    timeline: Arc<Timeline>,
    pool: vk::CommandPool,
    state: Mutex<State>,
    /// The staging buffer `memcpy_htod_reused` used last, and its size.
    reusable: Mutex<Option<(Arc<Buffer>, u64)>>,
    /// Staging buffers `memcpy_htod` reuses, and their sizes. One is free once this holds
    /// the only reference, because finished submissions drop theirs.
    staging: Mutex<Vec<(Arc<Buffer>, u64)>>,
}

/// shortcut: at most this many staging buffers kept per stream, whatever their size.
/// Bound it by bytes too if streams copy many sizes.
const STAGING_KEPT: usize = 4;

impl StreamCore {
    pub(crate) fn new(
        device: &Arc<Device>,
        after: Option<Point>,
    ) -> Result<Arc<Self>, DriverError> {
        let vk_device = &device.device;
        unsafe {
            let mut kind =
                vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
            let timeline = Arc::new(Timeline {
                device: device.clone(),
                semaphore: vk_device.create_semaphore(&info, None)?,
            });
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(device.queue_family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let pool = vk_device.create_command_pool(&info, None)?;
            Ok(Arc::new(Self {
                device: device.clone(),
                timeline,
                pool,
                state: Mutex::new(State {
                    last_value: 0,
                    last: after,
                    pending: VecDeque::new(),
                }),
                reusable: Mutex::new(None),
                staging: Mutex::new(Vec::new()),
            }))
        }
    }

    fn reclaim(&self, state: &mut State) -> Result<(), DriverError> {
        let device = &self.device.device;
        let done = unsafe { device.get_semaphore_counter_value(self.timeline.semaphore) }?;
        while state.pending.front().is_some_and(|p| p.value <= done) {
            let finished = state.pending.pop_front().expect("checked by the loop");
            unsafe { device.free_command_buffers(self.pool, &[finished.command_buffer]) };
        }
        Ok(())
    }

    fn last_point(&self) -> Option<Point> {
        self.state.lock().unwrap().last.clone()
    }

    fn synchronize(&self) -> Result<(), DriverError> {
        if let Some(point) = self.last_point() {
            point.wait()?;
        }
        self.reclaim(&mut self.state.lock().unwrap())
    }

    /// A free staging buffer of at least `bytes`, the smallest there is, or a new one.
    /// The caller has the only reference until it calls [`Self::return_staging`].
    fn take_staging(&self, bytes: u64) -> Result<(Arc<Buffer>, u64), DriverError> {
        self.reclaim(&mut self.state.lock().unwrap())?;
        let mut free = self.staging.lock().unwrap();
        let best = (free.iter().enumerate())
            .filter(|(_, (buffer, size))| *size >= bytes && Arc::strong_count(buffer) == 1)
            .min_by_key(|(_, (_, size))| *size)
            .map(|(index, _)| index);
        match best {
            Some(index) => Ok(free.remove(index)),
            None => Ok((Buffer::staging(&self.device, bytes)?, bytes)),
        }
    }

    fn return_staging(&self, buffer: Arc<Buffer>, size: u64) {
        let mut free = self.staging.lock().unwrap();
        free.push((buffer, size));
        if free.len() > STAGING_KEPT {
            free.remove(0);
        }
    }

    /// An empty submission that orders this stream after `buffer`'s earlier uses, records
    /// a use of it, and keeps it alive until everything queued so far has finished. If
    /// submitting fails, which means the device is lost or out of memory, it blocks instead.
    fn mark_use(&self, buffer: &Arc<Buffer>, write: bool) {
        let (reads, writes) = if write {
            (vec![], vec![buffer])
        } else {
            (vec![buffer], vec![])
        };
        if self.submit(&reads, &writes, Vec::new(), |_, _| {}).is_err() {
            let _ = wait_for_gpu(buffer).and_then(|()| self.synchronize());
        }
    }

    pub(crate) fn submit(
        &self,
        reads: &[&Arc<Buffer>],
        writes: &[&Arc<Buffer>],
        keep: Vec<Keep>,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), DriverError> {
        self.submit_at(reads, writes, keep, record).map(drop)
    }

    /// [`Self::submit`], returning the point the submission signals.
    fn submit_at(
        &self,
        reads: &[&Arc<Buffer>],
        writes: &[&Arc<Buffer>],
        mut keep: Vec<Keep>,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<Point, DriverError> {
        let device = &self.device.device;
        let mut state = self.state.lock().unwrap();
        self.reclaim(&mut state)?;

        let waits = required_waits(state.last.as_ref(), reads, writes);
        let wait_infos: Vec<_> = waits
            .iter()
            .map(|point| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(point.timeline.semaphore)
                    .value(point.value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();

        let value = state.last_value + 1;
        let signal_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline.semaphore)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];

        // shortcut: one queue submission per operation, batch them if this shows up in
        // profiles. Timeline waits give every operation a full memory dependency on what
        // came before, so no pipeline barriers are needed.
        let command_buffer = unsafe {
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .command_buffer_count(1);
            device.allocate_command_buffers(&allocate)?[0]
        };
        let submitted = unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(command_buffer, &begin)
                .and_then(|()| {
                    record(device, command_buffer);
                    device.end_command_buffer(command_buffer)?;
                    let command_buffers =
                        [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
                    let submit = vk::SubmitInfo2::default()
                        .wait_semaphore_infos(&wait_infos)
                        .signal_semaphore_infos(&signal_infos)
                        .command_buffer_infos(&command_buffers);
                    let queue = self.device.queue.lock().unwrap();
                    device.queue_submit2(*queue, &[submit], vk::Fence::null())
                })
        };
        if let Err(e) = submitted {
            unsafe { device.free_command_buffers(self.pool, &[command_buffer]) };
            return Err(e.into());
        }

        let point = Point {
            timeline: self.timeline.clone(),
            value,
        };
        for buffer in reads {
            record_read(buffer, &point);
        }
        for buffer in writes {
            let mut tracking = buffer.tracking.lock().unwrap();
            tracking.write = Some(point.clone());
            tracking.reads.clear();
        }
        keep.extend(
            reads
                .iter()
                .chain(writes)
                .map(|&buffer| buffer.clone() as Keep),
        );
        state.last_value = value;
        state.last = Some(point.clone());
        state.pending.push_back(Pending {
            value,
            command_buffer,
            _keep: keep,
        });
        Ok(point)
    }
}

/// Returned with an address by `device_ptr`, like cudarc's `SyncOnDrop`. On drop, the stream
/// records that its work so far used the buffer, and keeps the buffer alive until that work
/// has finished.
#[must_use]
pub struct SyncOnDrop<'a> {
    stream: &'a CudaStream,
    buffer: &'a Arc<Buffer>,
    write: bool,
}

impl<'a> SyncOnDrop<'a> {
    pub(crate) fn new(stream: &'a CudaStream, buffer: &'a Arc<Buffer>, write: bool) -> Self {
        stream.core.mark_use(buffer, write);
        Self {
            stream,
            buffer,
            write,
        }
    }
}

impl Drop for SyncOnDrop<'_> {
    fn drop(&mut self) {
        self.stream.core.mark_use(self.buffer, self.write);
    }
}

impl std::fmt::Debug for SyncOnDrop<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncOnDrop")
            .field("write", &self.write)
            .finish_non_exhaustive()
    }
}

fn record_read(buffer: &Buffer, point: &Point) {
    let mut tracking = buffer.tracking.lock().unwrap();
    tracking
        .reads
        .retain(|p| !Arc::ptr_eq(&p.timeline, &point.timeline));
    tracking.reads.push(point.clone());
}

/// What an operation must wait for: the previous work on its stream, the last write to
/// everything it touches, and earlier reads of everything it writes.
fn required_waits(
    last: Option<&Point>,
    reads: &[&Arc<Buffer>],
    writes: &[&Arc<Buffer>],
) -> Vec<Point> {
    let mut waits: Vec<Point> = last.into_iter().cloned().collect();
    for buffer in reads {
        waits.extend(buffer.tracking.lock().unwrap().write.clone());
    }
    for buffer in writes {
        let tracking = buffer.tracking.lock().unwrap();
        waits.extend(tracking.write.clone());
        waits.extend(tracking.reads.iter().cloned());
    }
    waits
}

impl Drop for StreamCore {
    fn drop(&mut self) {
        // Everything queued on this stream must finish before its semaphore and
        // command buffers go away. Errors here mean the device is lost anyway.
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(point) = &state.last {
            let _ = point.wait();
        }
        state.pending.clear();
        unsafe { self.device.device.destroy_command_pool(self.pool, None) };
    }
}

/// Ordered GPU work, like cudarc's `CudaStream`: everything runs in the order it was
/// called, and most calls return before the GPU has finished.
///
/// Create with [`CudaContext::default_stream`], [`CudaContext::new_stream`] or
/// [`Self::fork`].
pub struct CudaStream {
    pub(crate) core: Arc<StreamCore>,
    ctx: Arc<CudaContext>,
}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("ctx", &self.ctx)
            .finish_non_exhaustive()
    }
}

/// Streams are equal when they are the same queue of work, as in cudarc.
impl PartialEq for CudaStream {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.core, &other.core)
    }
}
impl Eq for CudaStream {}

impl CudaStream {
    pub(crate) fn new(ctx: &Arc<CudaContext>, core: Arc<StreamCore>) -> Arc<Self> {
        Arc::new(Self {
            core,
            ctx: ctx.clone(),
        })
    }

    pub(crate) fn device(&self) -> &Arc<Device> {
        &self.core.device
    }

    /// The context this stream belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Blocks until everything queued on this stream has finished.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.core.synchronize()
    }

    /// A new stream whose work starts after work already queued on this one.
    pub fn fork(&self) -> Result<Arc<Self>, DriverError> {
        let core = StreamCore::new(&self.core.device, self.core.last_point())?;
        Ok(Self::new(&self.ctx, core))
    }

    /// Allocates `len` elements with undefined contents.
    ///
    /// # Safety
    /// Every element must be written before it's read.
    pub unsafe fn alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(DriverError::Vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))?;
        let buffer = Buffer::device_local(self.device(), bytes as u64)?;
        Ok(CudaSlice::from_buffer(buffer, len, self.clone()))
    }

    /// Allocates `len` zeroed elements.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        // SAFETY: zeroed right below, and the stream keeps later reads in order.
        let mut slice = unsafe { self.alloc(len) }?;
        self.memset_zeros(&mut slice)?;
        Ok(slice)
    }

    /// An empty slice.
    pub fn null<T>(self: &Arc<Self>) -> Result<CudaSlice<T>, DriverError> {
        let buffer = Buffer::device_local(self.device(), 0)?;
        Ok(CudaSlice::from_buffer(buffer, 0, self.clone()))
    }

    /// Zeroes a slice or view, in stream order.
    pub fn memset_zeros<T: DeviceRepr + ValidAsZeroBits, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        let bytes = dst.num_bytes() as u64;
        let range = dst.raw_mut();
        if bytes == 0 {
            return Ok(());
        }
        // vkCmdFillBuffer only takes 4-byte aligned ranges.
        if range.offset.is_multiple_of(4) && bytes.is_multiple_of(4) {
            let (buffer, offset) = (range.buffer.buffer, range.offset);
            return self.core.submit(
                &[],
                &[range.buffer],
                Vec::new(),
                |device, commands| unsafe {
                    device.cmd_fill_buffer(commands, buffer, offset, bytes, 0);
                },
            );
        }
        self.copy_staged(&vec![0; bytes as usize], range)
    }

    /// Queues a copy of `bytes` bytes between two buffers. `reads` and `writes` are
    /// tracked and kept alive as for [`StreamCore::submit`], and `keep` is kept alive too.
    fn copy(
        &self,
        from: RawRange<'_>,
        to: RawRange<'_>,
        bytes: u64,
        reads: &[&Arc<Buffer>],
        writes: &[&Arc<Buffer>],
        keep: Vec<Keep>,
    ) -> Result<Point, DriverError> {
        let region = vk::BufferCopy::default()
            .src_offset(from.offset)
            .dst_offset(to.offset)
            .size(bytes);
        let (from, to) = (from.buffer.buffer, to.buffer.buffer);
        self.core
            .submit_at(reads, writes, keep, |device, commands| unsafe {
                device.cmd_copy_buffer(commands, from, to, &[region]);
            })
    }

    /// Copies `source[source_offset..]` into `to`, keeping `source` alive until it's done.
    fn copy_to_device(
        &self,
        source: Arc<Buffer>,
        source_offset: u64,
        to: RawRange<'_>,
        bytes: u64,
    ) -> Result<(), DriverError> {
        let from = RawRange {
            buffer: &source,
            offset: source_offset,
        };
        self.copy(from, to, bytes, &[], &[to.buffer], vec![source.clone()])
            .map(drop)
    }

    /// Copies `data` into `to` through one of the stream's reused staging buffers.
    fn copy_staged(&self, data: &[u8], to: RawRange<'_>) -> Result<(), DriverError> {
        let (staging, size) = self.core.take_staging(data.len() as u64)?;
        // SAFETY: nothing else holds `staging`, so the GPU isn't using it.
        unsafe { staging.write_bytes(data) };
        self.core.return_staging(staging.clone(), size);
        self.copy_to_device(staging, 0, to, data.len() as u64)
    }

    /// Copies host data into a slice or view, in stream order, without waiting for the GPU.
    ///
    /// A [`PinnedHostSlice`](super::PinnedHostSlice), or memory inside a
    /// [`host_register`](CudaContext::host_register)ed range, is read by the GPU directly.
    /// Anything else is copied into a reused staging buffer first, so it's free to change
    /// as soon as this returns.
    ///
    /// # Panics
    /// If `dst` is shorter than `src`.
    pub fn memcpy_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        let bytes = (src.len() * size_of::<T>()) as u64;
        if bytes == 0 {
            return Ok(());
        }
        let range = dst.raw_mut();
        if let Some(pinned) = src.pinned() {
            let (reads, writes) = (&[pinned.buffer], &[range.buffer]);
            return self
                .copy(pinned, range, bytes, reads, writes, Vec::new())
                .map(drop);
        }
        let src = src.as_host_slice();
        let start = src.as_ptr() as usize;
        {
            // Held while submitting, so `host_unregister` waits for this copy.
            let registry = self.ctx.registered.lock().unwrap();
            let covering = registry
                .iter()
                .find(|r| r.start <= start && start + bytes as usize <= r.end);
            if let Some(registration) = covering {
                let from = RawRange {
                    buffer: &registration.buffer,
                    offset: registration.offset + (start - registration.start) as u64,
                };
                // Not passed as a read, which would keep the import alive past
                // `host_unregister`. The read is recorded below instead, for it to wait on.
                let point = self.copy(from, range, bytes, &[], &[range.buffer], Vec::new())?;
                record_read(&registration.buffer, &point);
                return Ok(());
            }
        }
        // SAFETY: `T: DeviceRepr` is plain data.
        let raw = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), bytes as usize) };
        self.copy_staged(raw, range)
    }

    /// Experimental, and may be removed. Like
    /// [`Self::memcpy_htod`], but has the GPU read `src` directly instead of copying it on the
    /// CPU first. Blocks until all earlier work on this stream and the copy have finished.
    /// Returns an error if the driver can't import this memory, as with `static` data.
    ///
    /// # Panics
    /// If `dst` is shorter than `src`.
    #[doc(hidden)]
    pub fn memcpy_htod_unstaged<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        let src = src.as_host_slice();
        let bytes = size_of_val(src);
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: `T: DeviceRepr` is plain data. `synchronize` below drops the import before
        // `src` can be freed, and the copy only reads `bytes` bytes at `source_offset`.
        // shortcut: if the wait fails, the device is lost and the import outlives this call.
        // Handle that properly if this becomes more than an experiment.
        let (source, source_offset) = unsafe {
            let raw = std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), bytes);
            Buffer::import_host(self.device(), raw)
        }?;
        self.copy_to_device(source, source_offset, dst.raw_mut(), bytes as u64)?;
        self.synchronize()
    }

    /// Experimental, and may be removed. Like
    /// [`Self::memcpy_htod`], but reuses the stream's last staging buffer if it's big enough and
    /// the GPU has finished with it. The contract is the same, including not blocking.
    ///
    /// # Panics
    /// If `dst` is shorter than `src`.
    #[doc(hidden)]
    pub fn memcpy_htod_reused<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        let src = src.as_host_slice();
        let bytes = size_of_val(src) as u64;
        if bytes == 0 {
            return Ok(());
        }
        let cached = self.core.reusable.lock().unwrap().take();
        self.core.reclaim(&mut self.core.state.lock().unwrap())?;
        // Finished submissions have been reclaimed, so holding the only reference means the
        // GPU is done with it. Taking it from the cache keeps other threads away.
        let (staging, size) = match cached {
            Some((buffer, size)) if size >= bytes && Arc::strong_count(&buffer) == 1 => {
                (buffer, size)
            }
            _ => (Buffer::staging(self.device(), bytes)?, bytes),
        };
        // SAFETY: `T: DeviceRepr` is plain data, and the GPU isn't using `staging`, as above.
        unsafe {
            let raw = std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), bytes as usize);
            staging.write_bytes(raw);
        }
        *self.core.reusable.lock().unwrap() = Some((staging.clone(), size));
        self.copy_to_device(staging, 0, dst.raw_mut(), bytes)
    }

    /// Allocates a slice holding a copy of host data.
    pub fn clone_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        // SAFETY: the copy below initializes every element, in stream order.
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    /// Copies a slice or view to host memory, blocking until it's done.
    ///
    /// # Panics
    /// If `dst` is shorter than `src`.
    pub fn memcpy_dtoh<T: DeviceRepr, Src: DevicePtr<T>, Dst: HostSlice<T> + ?Sized>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        let bytes = src.num_bytes() as u64;
        if bytes == 0 {
            return Ok(());
        }
        let staging = Buffer::staging(self.device(), bytes)?;
        let (from, to) = (
            src.raw(),
            RawRange {
                buffer: &staging,
                offset: 0,
            },
        );
        self.copy(from, to, bytes, &[from.buffer], &[], vec![staging.clone()])?;
        self.synchronize()?;
        let dst = dst.as_host_slice_mut();
        // SAFETY: the GPU has finished, `dst` holds at least `bytes` and `T` is plain data.
        unsafe {
            let raw = std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), bytes as usize);
            staging.read_bytes(raw);
        }
        Ok(())
    }

    /// Copies a slice or view into a new `Vec`, blocking until it's done.
    pub fn clone_dtoh<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<Vec<T>, DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        // SAFETY: `T: DeviceRepr` is plain data, and `memcpy_dtoh` overwrites all of it.
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len())
        };
        self.memcpy_dtoh(src, &mut dst)?;
        Ok(dst)
    }

    /// Copies between slices or views, in stream order.
    ///
    /// # Panics
    /// If `dst` is shorter than `src`.
    pub fn memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        assert!(dst.len() >= src.len());
        let bytes = src.num_bytes() as u64;
        if bytes == 0 {
            return Ok(());
        }
        let (from, to) = (src.raw(), dst.raw_mut());
        self.copy(from, to, bytes, &[from.buffer], &[to.buffer], Vec::new())
            .map(drop)
    }

    /// Allocates a slice holding a copy of a slice or view.
    pub fn clone_dtod<T: DeviceRepr, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        // SAFETY: the copy below initializes every element, in stream order.
        let mut dst = unsafe { self.alloc(src.len()) }?;
        self.memcpy_dtod(src, &mut dst)?;
        Ok(dst)
    }
}

/// These prove ordering, which other tests can't: they pass with no ordering at all on a GPU
/// that happens to finish each operation before the next one starts. Some hold a stream
/// behind a semaphore only the test can signal, then check what has and hasn't run.
#[cfg(test)]
mod tests {
    use std::{thread::sleep, time::Duration};

    use super::*;
    use crate::driver::DevicePtr;

    struct Gate {
        core: Arc<StreamCore>,
    }

    impl Gate {
        fn new(ctx: &Arc<CudaContext>) -> Self {
            Self {
                core: StreamCore::new(&ctx.device, None).unwrap(),
            }
        }

        /// A stream whose work can't start until [`Self::open`].
        fn stream(&self, ctx: &Arc<CudaContext>) -> Arc<CudaStream> {
            let after = Point {
                timeline: self.core.timeline.clone(),
                value: 1,
            };
            CudaStream::new(ctx, StreamCore::new(&ctx.device, Some(after)).unwrap())
        }

        fn open(&self) {
            let info = vk::SemaphoreSignalInfo::default()
                .semaphore(self.core.timeline.semaphore)
                .value(1);
            unsafe { self.core.device.device.signal_semaphore(&info) }.unwrap();
        }
    }

    /// Copies `src` to the GPU without staging, and checks what arrived.
    /// Returns false if this driver has no host import.
    fn imports(stream: &Arc<CudaStream>, src: &[u8]) -> bool {
        let mut dst = stream.alloc_zeros::<u8>(src.len()).unwrap();
        match stream.memcpy_htod_unstaged(src, &mut dst) {
            Err(DriverError::MissingFeature(name)) => {
                println!("skipped: no {name}");
                false
            }
            result => {
                result.unwrap();
                assert!(stream.clone_dtoh(&dst).unwrap() == src);
                true
            }
        }
    }

    #[test]
    fn device_ptr_keeps_the_buffer_until_the_stream_is_done() {
        let ctx = CudaContext::new(0).unwrap();
        let (stream, other) = (ctx.default_stream(), ctx.new_stream().unwrap());
        let slice = stream.clone_htod(&[1u32, 2, 3]).unwrap();
        stream.synchronize().unwrap();
        let buffer = slice.raw().buffer.clone();
        let view = slice.slice(1..);
        let (address, guard) = view.device_ptr(&other);
        assert_eq!(address, buffer.address + 4);
        drop(guard);
        drop(slice);
        assert!(
            Arc::strong_count(&buffer) > 1,
            "the guard's submission holds it"
        );
        other.synchronize().unwrap();
        assert_eq!(Arc::strong_count(&buffer), 1);
    }

    #[test]
    fn transmute_reinterprets_the_same_bytes() {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let mut slice = stream.clone_htod(&[1u32, 0, 2, 0, 3]).unwrap();
        unsafe {
            assert!(slice.transmute::<u64>(3).is_none(), "needs 24 of 20 bytes");
            assert!(slice.transmute::<u64>(usize::MAX).is_none(), "overflows");
            let view = slice.slice(2..).transmute::<u64>(1).unwrap();
            assert_eq!(stream.clone_dtoh(&view).unwrap(), [2]);
            let mut wide = slice.transmute_mut::<u64>(2).unwrap();
            stream.memcpy_htod(&[7u64, 8], &mut wide).unwrap();
        }
        assert_eq!(stream.clone_dtoh(&slice).unwrap(), [7, 0, 8, 0, 3]);
    }

    #[test]
    fn equality_is_identity() {
        let ctx = CudaContext::new(0).unwrap();
        assert_eq!(ctx.default_stream(), ctx.default_stream());
        assert_ne!(ctx.default_stream(), ctx.new_stream().unwrap());
        assert_ne!(ctx, CudaContext::new(0).unwrap());
        assert!(format!("{:?}", ctx.default_stream()).contains("ordinal: 0"));
    }

    #[test]
    fn unstaged_copies_any_ordinary_host_memory() {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let page = 4096;
        let data: Vec<u8> = (0..3 * page).map(|i| (i % 251) as u8).collect();
        for src in [
            &data[..],                 // the allocator's own alignment
            &data[17..3 * page - 5],   // misaligned at both ends
            &data[page - 3..page + 3], // crossing one page boundary
            &data[100..101],           // a single byte
            &data[page..2 * page],     // exactly one page
        ] {
            if !imports(&stream, src) {
                return;
            }
        }
        let heap = vec![5u8; 64]; // a small allocation of its own
        assert!(imports(&stream, &heap));
        // calloc'd memory, whose pages may not exist until something touches them.
        assert!(imports(&stream, &vec![0u8; 1 << 22]));
    }

    #[test]
    fn unstaged_writes_only_the_part_of_dst_it_should() {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let mut dst = stream.clone_htod(&[9u8; 16]).unwrap();
        let src = vec![1u8; 4];
        let mut view = dst.slice_mut(4..8);
        match stream.memcpy_htod_unstaged(&src, &mut view) {
            Err(DriverError::MissingFeature(_)) => return,
            result => result.unwrap(),
        }
        let mut expected = [9u8; 16];
        expected[4..8].fill(1);
        assert_eq!(stream.clone_dtoh(&dst).unwrap(), expected);
    }

    #[test]
    fn unstaged_neighbours_in_one_page_can_copy_at_once() {
        let ctx = CudaContext::new(0).unwrap();
        let data: Vec<u8> = (0..1000).map(|i| (i % 251) as u8).collect();
        std::thread::scope(|scope| {
            for chunk in data.chunks(100) {
                let ctx = &ctx;
                scope.spawn(move || {
                    let stream = ctx.new_stream().unwrap();
                    for _ in 0..50 {
                        imports(&stream, chunk);
                    }
                });
            }
        });
    }

    #[test]
    fn unstaged_is_an_error_or_correct_for_static_data() {
        static DATA: [u8; 1 << 16] = [7; 1 << 16];
        let stream = CudaContext::new(0).unwrap().default_stream();
        let mut dst = stream.alloc_zeros::<u8>(DATA.len()).unwrap();
        if stream.memcpy_htod_unstaged(&DATA, &mut dst).is_ok() {
            assert_eq!(stream.clone_dtoh(&dst).unwrap(), DATA);
        }
    }

    #[test]
    fn unstaged_waits_for_earlier_work_and_its_copy() {
        let ctx = CudaContext::new(0).unwrap();
        let gate = Gate::new(&ctx);
        let stream = gate.stream(&ctx);
        let mut dst = stream.clone_htod(&[1u8; 64]).unwrap();
        let src = vec![2u8; 64];
        std::thread::scope(|scope| {
            let copy = scope.spawn(|| stream.memcpy_htod_unstaged(&src, &mut dst));
            sleep(Duration::from_millis(100));
            assert!(!copy.is_finished(), "returned before earlier work ran");
            gate.open();
            match copy.join().unwrap() {
                Err(DriverError::MissingFeature(_)) | Ok(()) => {}
                Err(e) => panic!("{e}"),
            }
        });
        let expected = if ctx.device.has_host_import() { 2 } else { 1 };
        assert_eq!(stream.clone_dtoh(&dst).unwrap(), [expected; 64]);
    }

    #[test]
    fn reused_staging_never_overwrites_a_copy_in_flight() {
        let ctx = CudaContext::new(0).unwrap();
        let gate = Gate::new(&ctx);
        let stream = gate.stream(&ctx);
        let mut one = stream.alloc_zeros::<u32>(64).unwrap();
        let mut two = stream.alloc_zeros::<u32>(64).unwrap();
        stream.memcpy_htod_reused(&[1u32; 64], &mut one).unwrap();
        stream.memcpy_htod_reused(&[2u32; 64], &mut two).unwrap();
        gate.open();
        assert_eq!(stream.clone_dtoh(&one).unwrap(), [1; 64]);
        assert_eq!(stream.clone_dtoh(&two).unwrap(), [2; 64]);
    }

    #[test]
    fn reused_staging_is_reused_when_finished_and_big_enough() {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let cached = || stream.core.reusable.lock().unwrap().clone().unwrap();
        let mut dst = stream.alloc_zeros::<u32>(64).unwrap();

        stream.memcpy_htod_reused(&[1u32; 32], &mut dst).unwrap();
        let first = Arc::as_ptr(&cached().0);
        stream.synchronize().unwrap();
        stream.memcpy_htod_reused(&[2u32; 16], &mut dst).unwrap();
        assert_eq!(first, Arc::as_ptr(&cached().0), "smaller copy reuses it");
        assert_eq!(cached().1, 32 * 4, "keeps its full size");

        stream.synchronize().unwrap();
        let before = Arc::as_ptr(&cached().0);
        stream.memcpy_htod_reused(&[3u32; 64], &mut dst).unwrap();
        assert_ne!(before, Arc::as_ptr(&cached().0), "larger copy replaces it");
        assert_eq!(cached().1, 64 * 4);

        let mut expected = [3u32; 64];
        assert_eq!(stream.clone_dtoh(&dst).unwrap(), expected);
        stream.memcpy_htod_reused(&[4u32; 8], &mut dst).unwrap();
        expected[..8].fill(4);
        assert_eq!(stream.clone_dtoh(&dst).unwrap(), expected);
    }

    /// Pointers to the staging buffers a stream keeps, oldest first.
    fn kept(stream: &CudaStream) -> Vec<*const Buffer> {
        let staging = stream.core.staging.lock().unwrap();
        staging.iter().map(|(b, _)| Arc::as_ptr(b)).collect()
    }

    #[test]
    fn staging_is_reused_once_the_gpu_is_done_and_never_before() {
        let ctx = CudaContext::new(0).unwrap();
        let gate = Gate::new(&ctx);
        let stream = gate.stream(&ctx);
        let mut one = stream.alloc_zeros::<u32>(64).unwrap();
        let mut two = stream.alloc_zeros::<u32>(64).unwrap();
        stream.memcpy_htod(&[1u32; 64], &mut one).unwrap();
        stream.memcpy_htod(&[2u32; 64], &mut two).unwrap();
        assert_eq!(
            kept(&stream).len(),
            2,
            "the first was busy, so the second is new"
        );
        gate.open();
        assert_eq!(stream.clone_dtoh(&one).unwrap(), [1; 64]);
        assert_eq!(stream.clone_dtoh(&two).unwrap(), [2; 64]);

        let before = kept(&stream);
        stream.memcpy_htod(&[3u32; 32], &mut one).unwrap();
        assert_eq!(kept(&stream).len(), 2, "a finished one was reused");
        assert!(before.contains(kept(&stream).last().unwrap()));
        let copied = stream.clone_dtoh(&one).unwrap();
        assert_eq!((copied[..32].to_vec(), copied[32]), (vec![3; 32], 1));

        // A gate of its own, since `gate` is open: every copy must still be running.
        let held = Gate::new(&ctx);
        let gated = held.stream(&ctx);
        let mut dst = gated.alloc_zeros::<u8>(16).unwrap();
        for i in 0..10u8 {
            gated.memcpy_htod(&[i; 16], &mut dst).unwrap();
        }
        assert_eq!(kept(&gated).len(), STAGING_KEPT, "keeps at most this many");
        held.open();
        assert_eq!(gated.clone_dtoh(&dst).unwrap(), [9; 16]);
    }

    #[test]
    fn pinned_memory_is_copied_directly_and_waits_before_access() {
        let ctx = CudaContext::new(0).unwrap();
        let gate = Gate::new(&ctx);
        let stream = gate.stream(&ctx);
        let mut pinned = unsafe { ctx.alloc_pinned::<u32>(64) }.unwrap();
        pinned.as_mut_slice().unwrap().fill(1);
        let mut dst = stream.alloc_zeros::<u32>(64).unwrap();
        stream.memcpy_htod(&pinned, &mut dst).unwrap();
        assert!(
            kept(&stream).is_empty(),
            "staged instead of copying directly"
        );

        std::thread::scope(|scope| {
            let write = scope.spawn(|| pinned.as_mut_slice().unwrap().fill(2));
            sleep(Duration::from_millis(100));
            assert!(!write.is_finished(), "wrote while a copy was reading it");
            gate.open();
            write.join().unwrap();
        });
        assert_eq!(stream.clone_dtoh(&dst).unwrap(), [1; 64]);
        assert_eq!(pinned.as_slice().unwrap(), [2; 64]);

        let mut back = unsafe { ctx.alloc_pinned::<u32>(64) }.unwrap();
        stream.memcpy_dtoh(&dst, &mut back).unwrap();
        assert_eq!(back.as_slice().unwrap(), [1; 64]);
        let empty = unsafe { ctx.alloc_pinned::<u128>(0) }.unwrap();
        assert!(empty.as_slice().unwrap().is_empty());
        stream
            .memcpy_htod(&empty, &mut stream.alloc_zeros::<u128>(0).unwrap())
            .unwrap();
    }

    /// Registers `data`, or returns false if this driver can't.
    fn register(ctx: &CudaContext, data: &[u8]) -> bool {
        match unsafe { ctx.host_register(data) } {
            Err(DriverError::MissingFeature(_)) => false,
            result => {
                result.unwrap();
                true
            }
        }
    }

    #[test]
    fn registered_memory_is_copied_directly() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let data: Vec<u8> = (0..10_000).map(|i| (i % 251) as u8).collect();
        if !register(&ctx, &data) {
            return;
        }
        let mut dst = stream.alloc_zeros::<u8>(data.len()).unwrap();
        for range in [0..data.len(), 17..5000, 4095..4097] {
            let mut view = dst.slice_mut(range.clone());
            stream.memcpy_htod(&data[range.clone()], &mut view).unwrap();
            assert_eq!(stream.clone_dtoh(&view).unwrap(), data[range]);
        }
        assert!(
            kept(&stream).is_empty(),
            "staged instead of copying directly"
        );

        // Overlapping the registered range, so this one is staged.
        let other = vec![1u8; 100];
        stream
            .memcpy_htod(&other, &mut dst.slice_mut(..100))
            .unwrap();
        assert_eq!(kept(&stream).len(), 1);

        ctx.host_unregister(data.as_ptr()).unwrap();
        stream.memcpy_htod(&data[..10], &mut dst).unwrap();
        assert_eq!(stream.clone_dtoh(&dst).unwrap()[..10], data[..10]);
    }

    #[test]
    fn registering_follows_cudas_rules() {
        let ctx = CudaContext::new(0).unwrap();
        let data = vec![0u8; 8192];
        if !register(&ctx, &data[100..200]) {
            return;
        }
        let overlapping = unsafe { ctx.host_register(&data[150..300]) };
        assert_eq!(overlapping, Err(DriverError::HostMemoryAlreadyRegistered));
        assert!(
            register(&ctx, &data[200..300]),
            "touching but not overlapping"
        );
        assert_eq!(
            ctx.host_unregister(data[150..].as_ptr()),
            Err(DriverError::HostMemoryNotRegistered),
            "only the start address unregisters"
        );
        ctx.host_unregister(data[100..].as_ptr()).unwrap();
        ctx.host_unregister(data[200..].as_ptr()).unwrap();
        assert_eq!(
            ctx.host_unregister(data[100..].as_ptr()),
            Err(DriverError::HostMemoryNotRegistered)
        );
        unsafe { ctx.host_register::<u8>(&[]) }.unwrap();
    }

    #[test]
    fn registered_copies_return_at_once_and_unregister_waits_for_them() {
        let ctx = CudaContext::new(0).unwrap();
        let gate = Gate::new(&ctx);
        let stream = gate.stream(&ctx);
        let data = vec![5u8; 4096 * 3];
        if !register(&ctx, &data) {
            return;
        }
        let mut dst = stream.alloc_zeros::<u8>(data.len()).unwrap();
        stream.memcpy_htod(&data, &mut dst).unwrap();
        std::thread::scope(|scope| {
            let unregister = scope.spawn(|| ctx.host_unregister(data.as_ptr()));
            sleep(Duration::from_millis(100));
            assert!(
                !unregister.is_finished(),
                "unregistered while a copy was reading"
            );
            gate.open();
            unregister.join().unwrap().unwrap();
        });
        drop(data);
        assert_eq!(stream.clone_dtoh(&dst).unwrap(), [5; 4096 * 3]);
    }

    /// How many operations have finished on a stream, after giving them time to run.
    fn finished(stream: &CudaStream) -> u64 {
        sleep(Duration::from_millis(100));
        let (device, semaphore) = (&stream.core.device.device, stream.core.timeline.semaphore);
        unsafe { device.get_semaphore_counter_value(semaphore) }.unwrap()
    }

    #[test]
    fn work_waits_for_earlier_work_on_its_stream() {
        let ctx = CudaContext::new(0).unwrap();
        let gate = Gate::new(&ctx);
        let stream = gate.stream(&ctx);
        let a = stream.clone_htod(&[7u32; 16]).unwrap();
        let b = stream.clone_dtod(&a).unwrap();
        assert_eq!(finished(&stream), 0, "ran before the gate opened");

        gate.open();
        assert_eq!(stream.clone_dtoh(&b).unwrap(), [7; 16]);
    }

    #[test]
    fn forks_wait_for_their_source() {
        let ctx = CudaContext::new(0).unwrap();
        let gate = Gate::new(&ctx);
        let source = gate.stream(&ctx);
        let a = source.clone_htod(&[7u32; 16]).unwrap();
        let fork = source.fork().unwrap();
        let b = fork.clone_dtod(&a).unwrap();
        assert_eq!(finished(&fork), 0, "fork ran before its source");

        gate.open();
        assert_eq!(fork.clone_dtoh(&b).unwrap(), [7; 16]);
    }

    /// The (stream, value) pairs an operation would wait for. Streams share a queue, so a
    /// held-back stream stalls all of them, and only checking the list itself can show
    /// that cross-stream tracking works.
    fn waits(reads: &[&Arc<Buffer>], writes: &[&Arc<Buffer>]) -> Vec<(*const Timeline, u64)> {
        let mut waits: Vec<_> = required_waits(None, reads, writes)
            .iter()
            .map(|point| (Arc::as_ptr(&point.timeline), point.value))
            .collect();
        waits.sort();
        waits
    }

    fn at(stream: &CudaStream, value: u64) -> (*const Timeline, u64) {
        (Arc::as_ptr(&stream.core.timeline), value)
    }

    #[test]
    fn cross_stream_waits_follow_each_buffers_reads_and_writes() {
        let ctx = CudaContext::new(0).unwrap();
        let (one, two) = (ctx.new_stream().unwrap(), ctx.new_stream().unwrap());
        let mut a = one.clone_htod(&[1u32; 16]).unwrap(); // `one` value 1 writes it
        let buffer = a.raw().buffer.clone();
        let written_by_one = vec![at(&one, 1)];

        assert_eq!(
            waits(&[&buffer], &[]),
            written_by_one,
            "reads wait for the last write"
        );
        assert_eq!(
            waits(&[], &[&buffer]),
            written_by_one,
            "writes wait for the last write"
        );

        let copy = two.clone_dtod(&a).unwrap(); // `two` value 1 reads it
        let mut expected = vec![at(&one, 1), at(&two, 1)];
        expected.sort();
        assert_eq!(
            waits(&[&buffer], &[]),
            written_by_one,
            "reads don't wait for other reads"
        );
        assert_eq!(
            waits(&[], &[&buffer]),
            expected,
            "writes wait for reads too"
        );

        two.memcpy_dtod(&copy, &mut a).unwrap(); // `two` value 2 writes it
        let written_by_two = vec![at(&two, 2)];
        assert_eq!(
            waits(&[], &[&buffer]),
            written_by_two,
            "a write replaces what came before"
        );
    }

    #[test]
    fn repeated_reads_on_a_stream_keep_only_the_latest() {
        let ctx = CudaContext::new(0).unwrap();
        let (one, two) = (ctx.new_stream().unwrap(), ctx.new_stream().unwrap());
        let a = one.clone_htod(&[1u32; 16]).unwrap();
        let buffer = a.raw().buffer.clone();
        let _first = two.clone_dtod(&a).unwrap(); // value 1
        let _second = two.clone_dtod(&a).unwrap(); // value 2
        let mut expected = vec![at(&one, 1), at(&two, 2)];
        expected.sort();
        assert_eq!(waits(&[], &[&buffer]), expected);
    }

    #[test]
    fn slices_can_drop_while_their_work_is_held_back() {
        let ctx = CudaContext::new(0).unwrap();
        let gate = Gate::new(&ctx);
        let stream = gate.stream(&ctx);
        let a = stream.clone_htod(&[7u32; 16]).unwrap();
        let b = stream.clone_dtod(&a).unwrap();
        drop(a);
        drop(b);
        gate.open();
        stream.synchronize().unwrap();
    }
}
