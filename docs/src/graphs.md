# Graphs

cudash has cudarc's stream capture API: record a stream's work once, then replay it.

```rust
use cudash::driver::*;
use cudash::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

# fn main() -> Result<(), DriverError> {
let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;
let mut data = stream.clone_htod(&[1u32; 4])?;

stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
stream.memset_zeros(&mut data)?;                 // recorded, not run
let graph = stream.end_capture(CUgraphInstantiate_flags(0))?.unwrap();
assert_eq!(stream.clone_dtoh(&data)?, [1; 4]);   // nothing ran yet

graph.launch()?;
assert_eq!(stream.clone_dtoh(&data)?, [0; 4]);
stream.memcpy_htod(&[2u32; 4], &mut data)?;
graph.launch()?;                                  // runs the memset again
assert_eq!(stream.clone_dtoh(&data)?, [0; 4]);
# Ok(())
# }
```

| Call | What it does |
|---|---|
| `stream.begin_capture(mode)` | Starts recording the stream's work instead of running it |
| `stream.end_capture(flags)` | Stops recording. Returns the graph, or `Ok(None)` if the capture was invalidated |
| `stream.end_capture_partial(flags)` | cudash-only. Like `end_capture`, but an invalidated capture still returns a graph of what was recorded, with the error |
| `stream.capture_status()` | `CU_STREAM_CAPTURE_STATUS_NONE`, `_ACTIVE` or `_INVALIDATED` |
| `graph.launch()` | Replays the graph on the stream it was captured on |
| `graph.upload()` | Does nothing for Vulkan work. Uploads the CUDA parts of a mixed graph |

The mode, flags and status types are in `cudash::driver::sys`, with the same names and values as in
`cudarc::driver::sys`, so capture code ports unchanged. Vulkan ignores `mode` and `flags`; they only reach the CUDA
parts of a [mixed graph](#mixed-graphs).

## What a capture records

| Call while capturing | Result |
|---|---|
| Kernel launches, `memcpy_dtod`, `clone_dtod`, `memset_zeros`, `alloc_zeros`, `device_ptr`, image copies | Recorded |
| `memcpy_htod` from a `PinnedHostSlice` | Recorded. Each replay reads the pinned memory as it is then |
| `memcpy_htod` from other host memory, `memcpy_dtoh`, `clone_dtoh`, `synchronize`, `frame`, `stop_frame`, launching another graph | Returns `DriverError::StreamCapture`, and invalidates the capture |
| A second `begin_capture` on the same stream | Returns `DriverError::StreamCapture` |

As in CUDA, a call that can't be captured invalidates the capture: later calls on the stream return errors until
`end_capture`, which returns `Ok(None)`. The stream then works normally again.

Things that are fixed at capture time, as in CUDA:

- **Kernel arguments.** Values and buffer addresses are recorded. To launch with other values, capture again.
- **Allocations.** `alloc_zeros` or `clone_dtod` inside a capture allocates once, at capture time. Every replay
  reuses that memory, and `alloc_zeros` zeroes it again.

## Keeping a partial capture

`end_capture_partial` returns `(CudaGraph, Option<DriverError>)`. If the capture was invalidated, the graph holds
what was recorded before that, and the error says what invalidated it. The call that failed and everything after it
are missing, so check the error before replaying: kernels that were meant to follow a refused copy would run on
stale input.

In a [mixed graph](#mixed-graphs), CUDA can invalidate the capture too, for example when something synchronizes the
paired CUDA stream. CUDA then discards its work since the last cudash operation, and earlier CUDA parts are kept.

```rust
# use cudash::driver::*;
# use cudash::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
# fn main() -> Result<(), DriverError> {
# let stream = CudaContext::new(0)?.new_stream()?;
# let mut data = stream.alloc_zeros::<u32>(4)?;
stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
stream.memset_zeros(&mut data)?;
let _ = stream.clone_dtoh(&data);   // refused: invalidates the capture
let (graph, invalid) = stream.end_capture_partial(CUgraphInstantiate_flags(0))?;
assert!(invalid.is_some());
graph.launch()?;                     // replays the memset
# Ok(())
# }
```

## Replays

`launch` queues the recorded work on the capture stream. Like any other call, it waits for the stream's earlier work
and for other streams' uses of the buffers the graph touches. Replays of one graph never overlap, as in CUDA.

The graph keeps every buffer it uses alive, so slices used in a capture can be dropped before the graph.

On Vulkan, a graph is a set of reusable command buffers recorded with `SIMULTANEOUS_USE`, with a full memory barrier
between operations. A replay is one queue submission instead of one per operation.

## Mixed graphs

With the `cudarc` feature, CUDA work can be part of a cudash graph. While a cudash stream captures, its paired CUDA
stream captures too, in the same mode. cudarc calls on a lent slice, inside the capture, are recorded by CUDA:

```rust,no_run
use cudash::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudash::interop::{AsCuda, cudarc};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let ctx = cudash::driver::CudaContext::new(0)?;
# let stream = ctx.new_stream()?;
# let mut x = stream.alloc_zeros::<f32>(1024)?;
# let cuda_kernel: cudarc::driver::CudaFunction = todo!();
stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
stream.memset_zeros(&mut x)?;                    // Vulkan part
{
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let mut lent = x.as_cuda_mut()?;             // CUDA part
    let cuda = lent.stream().clone();
    let mut builder = cuda.launch_builder(&cuda_kernel);
    builder.arg(&mut lent);
    unsafe { builder.launch(LaunchConfig::for_num_elems(1024)) }?;
}
stream.memset_zeros(&mut x)?;                    // Vulkan part
let graph = stream.end_capture(CUgraphInstantiate_flags(0))?.unwrap();
graph.launch()?;
# Ok(())
# }
```

Each cudash call ends the CUDA part captured so far, so the graph becomes alternating Vulkan and CUDA parts. `launch`
replays them in order, joined through a semaphore both APIs share, with new values for every replay.

- **All three capture modes work.** In `GLOBAL` mode, CUDA refuses unsafe CUDA calls from other threads during the
  capture, and those calls can invalidate it. That's CUDA's own rule.
- **A lent buffer is kept alive by the graph.** Dropping a lend while capturing doesn't synchronize the CUDA stream,
  since CUDA forbids that.
- **Only lends bound to the paired stream are captured.** A lend bound to a CUDA stream that captures for another
  cudash stream returns `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, since nothing would keep its memory alive.
- **CUDA work queued before `begin_capture`** has to finish first: `begin_capture` synchronizes the paired stream.

## Not supported

cudarc only has these in `sys`, and most have no Vulkan equivalent:

- Building graphs node by node, updating a graph's parameters, enabling and disabling nodes.
- Conditional nodes (`IF`, `WHILE`, `SWITCH`) and launching graphs from the device.
- Capturing several streams into one graph through events.
