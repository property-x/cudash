# Differences from cudarc

Names and signatures follow `cudarc::driver` 0.19.10 unless this page says otherwise.

## Different by necessity

| Area | cudarc | cudash |
|---|---|---|
| Kernels | PTX, `Ptx::from_src` | SPIR-V, `Spirv::from_file` and `Spirv::from_binary`. No text form |
| Block size | Chosen at launch | Patched into the SPIR-V per `block_dim`, with a pipeline cached for each |
| Shared memory | `shared_mem_bytes` | Ignored: shaders declare it at compile time |
| Kernel arguments | Kernel parameters | Push constants, at least 128 bytes |
| `device_ptr` | A `CUdeviceptr` | A Vulkan buffer device address. Kernels can use it, CUDA can't |
| Element traits | cudarc's `DeviceRepr`, `ValidAsZeroBits` | cudash's own, with the same names |
| Errors | `DriverError(CUresult)` | `DriverError` is an enum of Vulkan and cudash errors |
| `uuid` | A `CUuuid` | `[u8; 16]` |
| Graph `mode` and `flags` | Used by CUDA | Ignored by Vulkan. They only reach a mixed graph's CUDA parts |
| `graph.upload()` | Uploads | Does nothing for Vulkan work |

## Not provided

- **Events.** `launch` returns `()`, and there's no `record_event` or timing.
- **`per_thread_stream`,** context limits and cache settings, per-function attributes.
- **Half-precision types,** and tuples as `DeviceRepr`.
- **Unified memory** and the profiling APIs.
- **cudarc's other libraries:** cuBLAS, cuFFT, cuRAND, NCCL, cuDNN. Through [interop](interop.md), the real ones
  can run on cudash memory on NVIDIA GPUs.
- **`cudarc::driver::sys` and `result`.** `cudash::driver::sys` only holds the three types the graph API takes.
- **Graph building, updating and conditional nodes,** which cudarc only has in `sys`.

## Added by cudash

- **Partial captures:** `end_capture_partial` keeps what was recorded before a capture was invalidated.
- **Host registration:** `host_register` and `host_unregister`, which cudarc only has in `sys`.
- **Images:** `CudaImage`, `to_image`, `as_image`, `clone_to_image`, `imgcpy_dtod`.
- **Vulkan access:** `vk_instance`, `vk_device`, `vk_buffer` and the rest.
- **Debugging:** `frame`, `stop_frame`, `renderdoc_device_pointer`, and labels on every operation.
- **Interop:** `AsCuda` and `SharedSemaphore`.

## Behaving differently

- **Streams share one Vulkan queue,** so a stream that's waiting can hold up the others.
- **`ctx.synchronize()`** waits for the whole queue to go idle.
- **Allocations** are limited to 4096 per device, and one allocation's size to the driver's maximum.
- **Speed differs.** Each operation is one queue submission, which costs more than a CUDA launch. Graphs batch them.
