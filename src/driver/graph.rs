//! Graphs through stream capture, like cudarc's `graph.rs`: work queued while a stream captures
//! is recorded into reusable command buffers, and [`CudaGraph::launch`] submits them again.

use std::{fmt, sync::Arc};

#[cfg(feature = "cudarc")]
use super::stream::semaphore_info;

use ash::vk;

use super::{
    DriverError,
    device::{Buffer, Device},
    stream::{CudaStream, Keep, StreamCore, required_waits},
    sys,
};

/// One part of a graph, replayed in order.
pub(crate) enum Part {
    /// Recorded Vulkan work. `None` only orders the parts around it.
    Vulkan(Option<vk::CommandBuffer>),
    /// Work captured on the paired CUDA stream.
    #[cfg(feature = "cudarc")]
    Cuda(crate::interop::graph::CudaPart),
}

/// What a stream has recorded since `begin_capture`.
pub(crate) struct Capture {
    /// Becomes the graph's, and owns the command pool, so dropping a capture frees it.
    rec: Recorded,
    /// Passed on to the paired CUDA stream's capture.
    #[cfg_attr(not(feature = "cudarc"), allow(dead_code))]
    mode: sys::CUstreamCaptureMode,
    /// Why the capture was invalidated, if it was.
    invalid: Option<String>,
    /// The command buffer being recorded into, once something has been.
    current: Option<vk::CommandBuffer>,
    /// The paired CUDA stream, once it's capturing too.
    #[cfg(feature = "cudarc")]
    pub(crate) cuda: Option<Arc<cudarc::driver::CudaStream>>,
}

impl Capture {
    /// Marks the capture invalid, and returns the error for the call that did it.
    pub(crate) fn invalidate(&mut self, op: &str) -> DriverError {
        self.invalid.get_or_insert_with(|| op.to_owned());
        DriverError::StreamCapture(format!(
            "{op} can't be captured, so the capture is invalidated"
        ))
    }

    #[cfg(feature = "cudarc")]
    pub(crate) fn mode(&self) -> sys::CUstreamCaptureMode {
        self.mode
    }

    /// Records one operation after everything recorded so far.
    pub(crate) fn record(
        &mut self,
        core: &StreamCore,
        label: fmt::Arguments<'_>,
        reads: &[&Arc<Buffer>],
        writes: &[&Arc<Buffer>],
        keep: Vec<Keep>,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), DriverError> {
        if let Some(op) = &self.invalid {
            return Err(DriverError::StreamCapture(format!(
                "the capture was invalidated by {op}"
            )));
        }
        // CUDA work queued since the last operation goes between it and this one.
        #[cfg(feature = "cudarc")]
        match crate::interop::graph::end_part(self) {
            Ok(Some(part)) => {
                self.close_vulkan()?;
                self.rec.parts.push(Part::Cuda(part));
            }
            Ok(None) => {}
            Err(e) => return Err(self.invalidate(&format!("CUDA work ({e})"))),
        }
        let device = &self.rec.device.device;
        let command_buffer = match self.current {
            Some(command_buffer) => {
                // Separate submissions get this from their timeline waits.
                let barrier = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(
                        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                    )];
                let dependency = vk::DependencyInfo::default().memory_barriers(&barrier);
                unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency) };
                command_buffer
            }
            None => unsafe {
                let allocate = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.rec.pool)
                    .command_buffer_count(1);
                let command_buffer = device.allocate_command_buffers(&allocate)?[0];
                // Replays can be queued before the last one has finished.
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                device.begin_command_buffer(command_buffer, &begin)?;
                self.current = Some(command_buffer);
                command_buffer
            },
        };
        core.record_labeled(command_buffer, label, record);
        self.rec.reads.extend(reads.iter().map(|&b| b.clone()));
        self.rec.writes.extend(writes.iter().map(|&b| b.clone()));
        self.rec.keep.extend(keep);
        Ok(())
    }

    /// Ends the command buffer being recorded, if any, as a part of its own.
    fn close_vulkan(&mut self) -> Result<(), DriverError> {
        if let Some(command_buffer) = self.current.take() {
            unsafe { self.rec.device.device.end_command_buffer(command_buffer) }?;
            self.rec.parts.push(Part::Vulkan(Some(command_buffer)));
        }
        Ok(())
    }

    /// Registers a buffer that work outside these command buffers uses, such as a lend.
    #[cfg(feature = "cudarc")]
    pub(crate) fn uses(&mut self, buffer: &Arc<Buffer>, write: bool) {
        let list = if write {
            &mut self.rec.writes
        } else {
            &mut self.rec.reads
        };
        list.push(buffer.clone());
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        #[cfg(feature = "cudarc")]
        crate::interop::graph::abandon(self);
    }
}

/// Everything a graph's replays need, kept alive until the last replay has finished.
struct Recorded {
    device: Arc<Device>,
    pool: vk::CommandPool,
    /// Once the capture has ended, starts and ends with a Vulkan part, and CUDA parts sit
    /// between Vulkan ones.
    parts: Vec<Part>,
    reads: Vec<Arc<Buffer>>,
    writes: Vec<Arc<Buffer>>,
    keep: Vec<Keep>,
    /// Joins Vulkan and CUDA parts, once the graph has any.
    #[cfg(feature = "cudarc")]
    shared: Option<crate::interop::SharedSemaphore>,
}

impl Recorded {
    fn new(device: Arc<Device>, pool: vk::CommandPool) -> Self {
        Self {
            device,
            pool,
            parts: Vec::new(),
            reads: Vec::new(),
            writes: Vec::new(),
            keep: Vec::new(),
            #[cfg(feature = "cudarc")]
            shared: None,
        }
    }
}

impl Drop for Recorded {
    fn drop(&mut self) {
        // CUDA parts go first, while their stream is still alive. Recorded command buffers that
        // were never submitted go with their pool.
        self.parts.clear();
        unsafe { self.device.device.destroy_command_pool(self.pool, None) };
    }
}

/// A replayable graph, like cudarc's `CudaGraph`. Create with [`CudaStream::begin_capture`] and
/// [`CudaStream::end_capture`], and replay with [`Self::launch`].
pub struct CudaGraph {
    recorded: Arc<Recorded>,
    stream: Arc<CudaStream>,
}

impl fmt::Debug for CudaGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaGraph")
            .field("parts", &self.recorded.parts.len())
            .finish_non_exhaustive()
    }
}

impl CudaStream {
    /// Starts recording this stream's work instead of running it. See "Graphs" in the crate's
    /// `plan/api.md` for which calls can be captured. Under `cudarc`, the paired CUDA stream
    /// captures too, with the same `mode`.
    pub fn begin_capture(&self, mode: sys::CUstreamCaptureMode) -> Result<(), DriverError> {
        let core = &self.core;
        let mut state = core.state.lock().unwrap();
        if state.capture.is_some() {
            return Err(DriverError::StreamCapture(
                "the stream is already capturing".into(),
            ));
        }
        let info =
            vk::CommandPoolCreateInfo::default().queue_family_index(core.device.queue_family);
        let pool = unsafe { core.device.device.create_command_pool(&info, None) }?;
        core.device
            .set_name(pool, format_args!("{} capture", core.name));
        let capture = state.capture.insert(Capture {
            rec: Recorded::new(core.device.clone(), pool),
            mode,
            invalid: None,
            current: None,
            #[cfg(feature = "cudarc")]
            cuda: None,
        });
        #[cfg(feature = "cudarc")]
        if let Err(e) = crate::interop::graph::begin(core, capture) {
            state.capture = None;
            return Err(DriverError::Cuda(e.to_string()));
        }
        #[cfg(not(feature = "cudarc"))]
        let _ = capture;
        Ok(())
    }

    /// Stops recording, and returns the graph. Returns `Ok(None)` if a call that can't be
    /// captured invalidated the capture, as cudarc does. `flags` only reach CUDA parts.
    pub fn end_capture(
        self: &Arc<Self>,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<Option<CudaGraph>, DriverError> {
        let (capture, invalid) = self.take_capture()?;
        if invalid.is_some() {
            return Ok(None);
        }
        self.build(capture, flags).map(Some)
    }

    /// cudash-only: [`Self::end_capture`], but if the capture was invalidated, returns a graph of
    /// what was recorded before that, with the error that invalidated it. The call that failed and
    /// everything after it are missing from the graph, CUDA work included. When CUDA invalidated
    /// the capture, the CUDA work since the last cudash operation is missing too, since CUDA
    /// discards it.
    pub fn end_capture_partial(
        self: &Arc<Self>,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<(CudaGraph, Option<DriverError>), DriverError> {
        let (capture, invalid) = self.take_capture()?;
        Ok((self.build(capture, flags)?, invalid))
    }

    /// Ends the capture, and returns it with the error that invalidated it, if any.
    fn take_capture(&self) -> Result<(Capture, Option<DriverError>), DriverError> {
        let mut capture = {
            let mut state = self.core.state.lock().unwrap();
            state
                .capture
                .take()
                .ok_or_else(|| DriverError::StreamCapture("the stream isn't capturing".into()))?
        };
        #[cfg(feature = "cudarc")]
        {
            // Always ends the CUDA capture. Its work after a cudash call was refused doesn't
            // belong in a partial graph.
            let refused = capture.invalid.is_some();
            match crate::interop::graph::finish(&mut capture) {
                Ok(Some(part)) if !refused => {
                    capture.close_vulkan()?;
                    capture.rec.parts.push(Part::Cuda(part));
                }
                Ok(_) => {}
                Err(e) => {
                    capture.invalidate(&format!("CUDA work ({e})"));
                }
            }
        }
        let invalid = (capture.invalid.as_ref())
            .map(|op| DriverError::StreamCapture(format!("the capture was invalidated by {op}")));
        Ok((capture, invalid))
    }

    /// Turns an ended capture into a graph.
    fn build(
        self: &Arc<Self>,
        mut capture: Capture,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<CudaGraph, DriverError> {
        #[cfg(not(feature = "cudarc"))]
        let _ = flags;
        capture.close_vulkan()?;
        let device = capture.rec.device.clone();
        // The pool moves to the graph, so the capture mustn't destroy it.
        let null = Recorded::new(device, vk::CommandPool::null());
        let mut rec = std::mem::replace(&mut capture.rec, null);
        if !matches!(rec.parts.first(), Some(Part::Vulkan(_))) {
            rec.parts.insert(0, Part::Vulkan(None));
        }
        if !matches!(rec.parts.last(), Some(Part::Vulkan(_))) {
            rec.parts.push(Part::Vulkan(None));
        }
        #[cfg(feature = "cudarc")]
        {
            for part in &mut rec.parts {
                if let Part::Cuda(part) = part {
                    part.instantiate(flags)
                        .map_err(|e| DriverError::Cuda(e.to_string()))?;
                }
            }
            if rec.parts.iter().any(|p| matches!(p, Part::Cuda(_))) {
                let shared = crate::interop::SharedSemaphore::new(self.context())
                    .map_err(|e| DriverError::Cuda(e.to_string()))?;
                rec.shared = Some(shared);
            }
        }
        Ok(CudaGraph {
            recorded: Arc::new(rec),
            stream: self.clone(),
        })
    }

    /// Whether this stream is capturing, and whether the capture has been invalidated.
    pub fn capture_status(&self) -> Result<sys::CUstreamCaptureStatus, DriverError> {
        use sys::CUstreamCaptureStatus::*;
        Ok(match &self.core.state.lock().unwrap().capture {
            None => CU_STREAM_CAPTURE_STATUS_NONE,
            Some(capture) if capture.invalid.is_some() => CU_STREAM_CAPTURE_STATUS_INVALIDATED,
            Some(_) => CU_STREAM_CAPTURE_STATUS_ACTIVE,
        })
    }
}

impl CudaGraph {
    /// Queues a replay on the stream it was captured on, after the stream's earlier work and
    /// other streams' uses of the graph's buffers. Replays of one graph never overlap.
    pub fn launch(&self) -> Result<(), DriverError> {
        let core = &self.stream.core;
        let recorded = &self.recorded;
        let mut state = core.state.lock().unwrap();
        core.reclaim(&mut state)?;
        if let Some(capture) = &mut state.capture {
            return Err(capture.invalidate("CudaGraph::launch"));
        }
        let reads: Vec<_> = recorded.reads.iter().collect();
        let writes: Vec<_> = recorded.writes.iter().collect();
        let last = recorded.parts.len() - 1;
        // The value the next Vulkan part waits for, which CUDA signals.
        #[cfg(feature = "cudarc")]
        let mut cuda_done: Option<u64> = None;
        #[cfg(feature = "cudarc")]
        let mut vulkan_done = 0;
        for (i, part) in recorded.parts.iter().enumerate() {
            match part {
                Part::Vulkan(command_buffer) => {
                    let (first, final_part) = (i == 0, i == last);
                    let waits = if first {
                        required_waits(state.last.as_ref(), &reads, &writes)
                    } else {
                        state.last.iter().cloned().collect()
                    };
                    // Waits for the CUDA part before, and signals the one after.
                    let (mut extra_waits, mut extra_signals) = (Vec::new(), Vec::new());
                    #[cfg(feature = "cudarc")]
                    if let Some(shared) = &recorded.shared {
                        let semaphore = shared.vk_semaphore();
                        extra_waits.extend(cuda_done.take().map(|v| semaphore_info(semaphore, v)));
                        if !final_part {
                            vulkan_done = shared.next_value();
                            extra_signals.push(semaphore_info(semaphore, vulkan_done));
                        }
                    }
                    #[cfg(not(feature = "cudarc"))]
                    let _ = (&mut extra_waits, &mut extra_signals);
                    let tracked: (&[_], &[_]) = if final_part {
                        (&reads, &writes)
                    } else {
                        (&[], &[])
                    };
                    core.queue(
                        &mut state,
                        *command_buffer,
                        false,
                        &waits,
                        &extra_waits,
                        &extra_signals,
                        tracked,
                        vec![recorded.clone() as Keep],
                        false,
                    )?;
                }
                #[cfg(feature = "cudarc")]
                Part::Cuda(part) => {
                    let shared = recorded.shared.as_ref().expect("made with every CUDA part");
                    let done = part
                        .launch(shared, vulkan_done)
                        .map_err(|e| DriverError::Cuda(e.to_string()))?;
                    cuda_done = Some(done);
                }
            }
        }
        Ok(())
    }

    /// Does nothing for Vulkan parts, which need no upload. Under `cudarc`, uploads CUDA parts.
    pub fn upload(&self) -> Result<(), DriverError> {
        // Parts aren't thread-safe, so this takes the lock `launch` does.
        #[cfg(feature = "cudarc")]
        {
            let _state = self.stream.core.state.lock().unwrap();
            for part in &self.recorded.parts {
                if let Part::Cuda(part) = part {
                    part.upload()
                        .map_err(|e| DriverError::Cuda(e.to_string()))?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{CudaContext, LaunchConfig, PushKernelArg, Spirv};
    use sys::{CUgraphInstantiate_flags as Flags, CUstreamCaptureMode, CUstreamCaptureStatus};

    const DOUBLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/double_all.spv"));
    const RELAXED: CUstreamCaptureMode = CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;

    /// cudarc's `test_end_capture_flags`: replays repeat the captured work, and run after
    /// whatever was queued before them.
    #[test]
    fn replays_repeat_the_captured_work() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut data = stream.clone_htod(&[1u32; 4]).unwrap();
        stream.begin_capture(RELAXED).unwrap();
        stream.memset_zeros(&mut data).unwrap();
        let graph = stream.end_capture(Flags(0)).unwrap().unwrap();
        assert_eq!(
            stream.clone_dtoh(&data).unwrap(),
            [1; 4],
            "capture ran the work"
        );

        graph.launch().unwrap();
        assert_eq!(stream.clone_dtoh(&data).unwrap(), [0; 4]);
        stream.memcpy_htod(&[2u32; 4], &mut data).unwrap();
        graph.launch().unwrap();
        assert_eq!(stream.clone_dtoh(&data).unwrap(), [0; 4]);
    }

    /// Operations in one command buffer are ordered by barriers, and queued replays don't
    /// overlap: each one doubles twice, then copies.
    #[test]
    fn recorded_operations_and_replays_stay_in_order() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let module = ctx.load_module(Spirv::from_binary(DOUBLE)).unwrap();
        let double = module.load_function("double_all").unwrap();
        let n = 1 << 16;
        let n_arg = n as i32;
        let mut x = stream.clone_htod(&vec![1.0f32; n]).unwrap();
        let mut copy = stream.alloc_zeros::<f32>(n).unwrap();

        stream.begin_capture(RELAXED).unwrap();
        for _ in 0..2 {
            let mut builder = stream.launch_builder(&double);
            builder.arg(&mut x).arg(&n_arg);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.unwrap();
        }
        stream.memcpy_dtod(&x, &mut copy).unwrap();
        let graph = stream.end_capture(Flags(0)).unwrap().unwrap();

        for _ in 0..3 {
            graph.launch().unwrap();
        }
        assert!(stream.clone_dtoh(&copy).unwrap().iter().all(|&v| v == 64.0));
    }

    #[test]
    fn other_streams_wait_for_replays() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let other = ctx.new_stream().unwrap();
        let mut data = stream.clone_htod(&[7u8; 1 << 20]).unwrap();
        stream.synchronize().unwrap();
        stream.begin_capture(RELAXED).unwrap();
        stream.memset_zeros(&mut data).unwrap();
        let graph = stream.end_capture(Flags(0)).unwrap().unwrap();
        graph.launch().unwrap();
        assert!(other.clone_dtoh(&data).unwrap().iter().all(|&v| v == 0));
    }

    #[test]
    fn calls_that_cant_be_captured_invalidate_it() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut data = stream.alloc_zeros::<u32>(4).unwrap();
        stream.begin_capture(RELAXED).unwrap();
        stream.memset_zeros(&mut data).unwrap();
        assert!(matches!(
            stream.clone_dtoh(&data),
            Err(DriverError::StreamCapture(_))
        ));
        let status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED;
        assert_eq!(stream.capture_status().unwrap(), status);
        assert!(stream.memset_zeros(&mut data).is_err());
        assert!(stream.end_capture(Flags(0)).unwrap().is_none());

        // The stream works normally again.
        let status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        assert_eq!(stream.capture_status().unwrap(), status);
        stream.memcpy_htod(&[3u32; 4], &mut data).unwrap();
        assert_eq!(stream.clone_dtoh(&data).unwrap(), [3; 4]);
    }

    /// A partial capture replays what was recorded before the call that invalidated it.
    #[test]
    fn partial_captures_keep_what_was_recorded() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut a = stream.clone_htod(&[1u32; 4]).unwrap();
        let mut b = stream.clone_htod(&[1u32; 4]).unwrap();
        stream.begin_capture(RELAXED).unwrap();
        stream.memset_zeros(&mut a).unwrap();
        assert!(stream.clone_dtoh(&a).is_err());
        assert!(stream.memset_zeros(&mut b).is_err());
        let (graph, invalid) = stream.end_capture_partial(Flags(0)).unwrap();
        assert!(
            matches!(invalid, Some(DriverError::StreamCapture(e)) if e.contains("memcpy_dtoh"))
        );

        graph.launch().unwrap();
        assert_eq!(stream.clone_dtoh(&a).unwrap(), [0; 4]);
        assert_eq!(stream.clone_dtoh(&b).unwrap(), [1; 4]);

        stream.begin_capture(RELAXED).unwrap();
        stream.memset_zeros(&mut b).unwrap();
        let (graph, invalid) = stream.end_capture_partial(Flags(0)).unwrap();
        assert!(invalid.is_none());
        graph.launch().unwrap();
        assert_eq!(stream.clone_dtoh(&b).unwrap(), [0; 4]);
    }

    #[test]
    fn capture_calls_out_of_turn_are_errors() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        assert!(stream.end_capture(Flags(0)).is_err());
        stream.begin_capture(RELAXED).unwrap();
        assert!(stream.begin_capture(RELAXED).is_err());
        let status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE;
        assert_eq!(stream.capture_status().unwrap(), status);
        let empty = stream.end_capture(Flags(0)).unwrap().unwrap();
        empty.launch().unwrap();
        stream.synchronize().unwrap();
    }

    /// The graph keeps what it uses, including staging made at capture time.
    #[test]
    fn graphs_keep_their_buffers_alive() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut data = stream.clone_htod(&[9u8; 7]).unwrap();
        let mut pinned = unsafe { ctx.alloc_pinned::<u8>(7) }.unwrap();
        pinned.as_mut_slice().unwrap().fill(5);
        stream.begin_capture(RELAXED).unwrap();
        stream.memcpy_htod(&pinned, &mut data).unwrap();
        let mut copy = stream.clone_dtod(&data).unwrap();
        // Seven bytes can't be filled by `vkCmdFillBuffer`, so this goes through staging.
        stream.memset_zeros(&mut data).unwrap();
        stream
            .memcpy_dtod(&data.slice(0..3), &mut copy.slice_mut(0..3))
            .unwrap();
        let graph = stream.end_capture(Flags(0)).unwrap().unwrap();
        drop(data);

        graph.launch().unwrap();
        assert_eq!(stream.clone_dtoh(&copy).unwrap(), [0, 0, 0, 5, 5, 5, 5]);
        pinned.as_mut_slice().unwrap().fill(6);
        graph.launch().unwrap();
        assert_eq!(stream.clone_dtoh(&copy).unwrap(), [0, 0, 0, 6, 6, 6, 6]);
        drop(graph);
        stream.synchronize().unwrap();
    }
}
