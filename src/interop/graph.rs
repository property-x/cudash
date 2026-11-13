//! The CUDA parts of a mixed graph: while a cudash stream captures, its paired CUDA stream
//! captures too, and each cudash operation ends the CUDA part so far. See "Graphs" in
//! `plan/api.md`.

use std::sync::Arc;

use cudarc::driver::{CudaStream, result, sys};

use crate::{
    driver::{graph::Capture, stream::StreamCore, sys as cudash_sys},
    interop::{Result, SharedSemaphore},
};

/// Work captured on a CUDA stream, between two Vulkan parts.
pub(crate) struct CudaPart {
    graph: sys::CUgraph,
    /// Null until [`Self::instantiate`].
    exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

// SAFETY: CUDA handles aren't tied to a thread, and every use binds the context. CUDA graphs
// aren't internally synchronized, but a part is only used by `CudaGraph::launch` and `upload`
// on its own cudash stream, and launch holds that stream's lock.
unsafe impl Send for CudaPart {}
unsafe impl Sync for CudaPart {}

impl CudaPart {
    pub(crate) fn instantiate(
        &mut self,
        flags: cudash_sys::CUgraphInstantiate_flags,
    ) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        let flags = sys::CUgraphInstantiate_flags(flags.0);
        // SAFETY: `graph` is a valid captured graph.
        self.exec = unsafe { result::graph::instantiate(self.graph, flags) }?;
        Ok(())
    }

    /// Queues this part after `after` on the shared semaphore, and returns the value it signals
    /// when done. `after` must already be submitted, as CUDA requires.
    pub(crate) fn launch(&self, shared: &SharedSemaphore, after: u64) -> Result<u64> {
        shared.cuda_wait(&self.stream, after)?;
        self.stream.context().bind_to_thread()?;
        // SAFETY: `exec` was instantiated in `end_capture`, and the stream is valid.
        unsafe { result::graph::launch(self.exec, self.stream.cu_stream()) }?;
        shared.cuda_signal(&self.stream)
    }

    pub(crate) fn upload(&self) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: as in `launch`.
        unsafe { result::graph::upload(self.exec, self.stream.cu_stream()) }?;
        Ok(())
    }
}

impl Drop for CudaPart {
    fn drop(&mut self) {
        let ctx = self.stream.context();
        ctx.record_err(ctx.bind_to_thread());
        if !self.exec.is_null() {
            // SAFETY: the graph keeps parts until its last replay has finished.
            ctx.record_err(unsafe { result::graph::exec_destroy(self.exec) });
        }
        ctx.record_err(unsafe { result::graph::destroy(self.graph) });
    }
}

fn mode(mode: cudash_sys::CUstreamCaptureMode) -> sys::CUstreamCaptureMode {
    use sys::CUstreamCaptureMode::*;
    match mode {
        cudash_sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL => {
            CU_STREAM_CAPTURE_MODE_GLOBAL
        }
        cudash_sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL => {
            CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        }
        cudash_sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED => {
            CU_STREAM_CAPTURE_MODE_RELAXED
        }
    }
}

/// Runs `f` with this thread's capture mode set to RELAXED, for CUDA calls that queue no
/// stream work but that GLOBAL and THREAD_LOCAL captures refuse.
pub(crate) fn relaxed<R>(f: impl FnOnce() -> R) -> R {
    let mut mode = sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;
    // SAFETY: swaps the calling thread's mode, and the second call restores it.
    unsafe { sys::cuThreadExchangeStreamCaptureMode(&mut mode) };
    let result = f();
    unsafe { sys::cuThreadExchangeStreamCaptureMode(&mut mode) };
    result
}

/// Starts capturing on the paired CUDA stream, if the cudash stream has one yet.
pub(crate) fn begin(core: &StreamCore, capture: &mut Capture) -> Result<()> {
    let stream = core.cuda.lock().unwrap_or_else(|e| e.into_inner()).clone();
    match stream {
        // Drains CUDA work queued before the capture, which replays can't wait for.
        Some(stream) => {
            stream.synchronize()?;
            start(&stream, capture)
        }
        None => Ok(()),
    }
}

/// Starts capturing on `stream`, the paired CUDA stream, if the cudash stream is capturing and
/// it isn't yet. Called whenever the paired stream is handed out.
pub(crate) fn join(core: &StreamCore, stream: &Arc<CudaStream>) -> Result<()> {
    let mut state = core.state.lock().unwrap();
    match &mut state.capture {
        Some(capture) if capture.cuda.is_none() => start(stream, capture).inspect_err(|e| {
            capture.invalidate(&format!("starting the CUDA capture ({e})"));
        }),
        _ => Ok(()),
    }
}

/// Whether `stream` is the CUDA stream capturing for `core`. Lends bound to it are part of the
/// capture.
pub(crate) fn lend_is_captured(core: &StreamCore, stream: &Arc<CudaStream>) -> bool {
    let state = core.state.lock().unwrap();
    state
        .capture
        .as_ref()
        .and_then(|c| c.cuda.as_ref())
        .is_some_and(|s| Arc::ptr_eq(s, stream))
}

fn start(stream: &Arc<CudaStream>, capture: &mut Capture) -> Result<()> {
    stream.begin_capture(mode(capture.mode()))?;
    capture.cuda = Some(stream.clone());
    Ok(())
}

/// Ends the CUDA capture. Returns the work captured, or `None` if there was none.
fn end(stream: &Arc<CudaStream>) -> Result<Option<CudaPart>> {
    stream.context().bind_to_thread()?;
    // SAFETY: the stream is valid and capturing.
    let graph = unsafe { result::stream::end_capture(stream.cu_stream()) }?;
    if graph.is_null() {
        let invalidated = sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
        return Err(result::DriverError(invalidated).into());
    }
    let mut nodes = 0;
    // SAFETY: a null node array asks for the count only.
    let counted = unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut nodes) }.result();
    let part = CudaPart {
        graph,
        exec: std::ptr::null_mut(),
        stream: stream.clone(),
    };
    counted?;
    Ok((nodes > 0).then_some(part))
}

/// Ends the CUDA part captured since the last cudash operation, and starts the next one.
pub(crate) fn end_part(capture: &mut Capture) -> Result<Option<CudaPart>> {
    let Some(stream) = capture.cuda.take() else {
        return Ok(None);
    };
    let part = end(&stream)?;
    start(&stream, capture)?;
    Ok(part)
}

/// Ends the CUDA capture for good, at `end_capture`.
pub(crate) fn finish(capture: &mut Capture) -> Result<Option<CudaPart>> {
    match capture.cuda.take() {
        Some(stream) => end(&stream),
        None => Ok(None),
    }
}

/// Ends a CUDA capture whose cudash capture was dropped or invalidated, discarding the work.
pub(crate) fn abandon(capture: &mut Capture) {
    if let Some(stream) = capture.cuda.take() {
        let _ = end(&stream);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::{LaunchConfig, PushKernelArg};

    use crate::{
        driver::{
            self, CudaContext, PushKernelArg as _, Spirv, sys::CUgraphInstantiate_flags as Flags,
        },
        interop::{AsCuda, Error},
    };
    use cudash_sys::CUstreamCaptureMode::*;

    use super::*;

    const DOUBLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/double_all.spv"));
    const ADD_ONE: &str = r#"extern "C" __global__ void add_one(float* x, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) x[i] += 1.0f;
    }"#;
    const N: usize = 1 << 12;

    /// A GLOBAL capture makes CUDA refuse other threads' unsafe calls, and those invalidate
    /// captures, so these tests take turns.
    static CUDA_CAPTURES: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The CUDA and Vulkan kernels, or `None` without an NVIDIA GPU, unless
    /// `CUDASH_REQUIRE_NVIDIA` is set.
    fn kernels(
        ctx: &Arc<CudaContext>,
    ) -> Option<(driver::CudaFunction, cudarc::driver::CudaFunction)> {
        let cuda = match ctx.as_cuda() {
            Err(Error::NoMatchingCudaDevice)
                if std::env::var_os("CUDASH_REQUIRE_NVIDIA").is_none() =>
            {
                println!("SKIPPED: no CUDA device matches the cudash context");
                return None;
            }
            Err(e) => panic!(
                "CUDASH_REQUIRE_NVIDIA is set, but GPU 0 is {} ({e}). Is the NVIDIA Vulkan driver \
                 visible, for example through VK_DRIVER_FILES?",
                ctx.name().unwrap_or_default()
            ),
            Ok(cuda) => cuda,
        };
        let ptx = cudarc::nvrtc::compile_ptx(ADD_ONE).unwrap();
        let add_one = cuda
            .load_module(ptx)
            .unwrap()
            .load_function("add_one")
            .unwrap();
        let module = ctx.load_module(Spirv::from_binary(DOUBLE)).unwrap();
        Some((module.load_function("double_all").unwrap(), add_one))
    }

    fn double(
        stream: &driver::CudaStream,
        f: &driver::CudaFunction,
        x: &mut driver::CudaSlice<f32>,
    ) {
        let n = N as i32;
        let mut builder = stream.launch_builder(f);
        builder.arg(x).arg(&n);
        unsafe { builder.launch(driver::LaunchConfig::for_num_elems(N as u32)) }.unwrap();
    }

    fn add_one(f: &cudarc::driver::CudaFunction, x: &mut driver::CudaSlice<f32>) -> Result<()> {
        let mut lent = x.as_cuda_mut()?;
        let cuda = lent.stream().clone();
        let n = N as i32;
        let mut builder = cuda.launch_builder(f);
        builder.arg(&mut lent).arg(&n);
        unsafe { builder.launch(LaunchConfig::for_num_elems(N as u32)) }?;
        Ok(())
    }

    /// Vulkan, CUDA, Vulkan in one graph, in every capture mode, with the paired CUDA stream
    /// made before the capture (`early`) or during it.
    #[test]
    fn mixed_graphs_replay_in_order() {
        let _turn = CUDA_CAPTURES.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = CudaContext::new(0).unwrap();
        let Some((vk_double, cuda_add_one)) = kernels(&ctx) else {
            return;
        };
        let mut failures = Vec::new();
        for mode in [
            CU_STREAM_CAPTURE_MODE_RELAXED,
            CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            CU_STREAM_CAPTURE_MODE_GLOBAL,
        ] {
            for early in [false, true] {
                let stream = ctx.new_stream().unwrap();
                let mut x = stream.clone_htod(&vec![1.0f32; N]).unwrap();
                if early {
                    stream.as_cuda().unwrap();
                }
                let mut run = || -> std::result::Result<Vec<f32>, String> {
                    stream.begin_capture(mode).map_err(|e| e.to_string())?;
                    double(&stream, &vk_double, &mut x);
                    let lent = add_one(&cuda_add_one, &mut x).map_err(|e| format!("lend: {e}"));
                    double(&stream, &vk_double, &mut x);
                    let graph = stream.end_capture(Flags(0)).map_err(|e| e.to_string())?;
                    lent?;
                    let graph = graph.ok_or("end_capture returned None")?;
                    let mut firsts = Vec::new();
                    for _ in 0..3 {
                        graph.launch().map_err(|e| format!("launch: {e}"))?;
                        let host = stream.clone_dtoh(&x).map_err(|e| e.to_string())?;
                        if host.iter().any(|&v| v != host[0]) {
                            return Err(format!("uneven result {:?}", &host[..4]));
                        }
                        firsts.push(host[0]);
                    }
                    Ok(firsts)
                };
                let result = run();
                println!("{mode:?} early={early}: {result:?}");
                if result != Ok(vec![6.0, 26.0, 106.0]) {
                    failures.push((mode, early, result));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    /// A lend taken before the capture and dropped during it, and one bound to another
    /// stream's capturing CUDA stream.
    #[test]
    fn lends_across_the_capture_boundary() {
        let _turn = CUDA_CAPTURES.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = CudaContext::new(0).unwrap();
        let Some((vk_double, cuda_add_one)) = kernels(&ctx) else {
            return;
        };
        let stream = ctx.new_stream().unwrap();
        let mut x = stream.clone_htod(&vec![1.0f32; N]).unwrap();
        {
            let mut lent = x.as_cuda_mut().unwrap();
            let cuda = lent.stream().clone();
            let n = N as i32;
            let mut builder = cuda.launch_builder(&cuda_add_one);
            builder.arg(&mut lent).arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(N as u32)) }.unwrap();
            stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED)
                .unwrap();
        }
        double(&stream, &vk_double, &mut x);
        let graph = stream.end_capture(Flags(0)).unwrap().unwrap();
        graph.launch().unwrap();
        assert_eq!(stream.clone_dtoh(&x).unwrap()[..2], [4.0, 4.0]);

        let other = ctx.new_stream().unwrap();
        let mut y = other.clone_htod(&vec![1.0f32; N]).unwrap();
        stream
            .begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED)
            .unwrap();
        let capturing = stream.as_cuda().unwrap();
        let refused = y.as_cuda_mut_with(&capturing).map(drop);
        println!("lend bound to another stream's capture: {refused:?}");
        assert!(refused.is_err());
        stream.end_capture(Flags(0)).unwrap();
    }
}
