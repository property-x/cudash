# Vulkan and debugging

## Vulkan handles

cudash exposes the Vulkan objects behind its types, through the same `ash` version it uses (`cudash::ash`):

| Call | Gives |
|---|---|
| `ctx.vk_instance()`, `ctx.vk_physical_device()`, `ctx.vk_device()` | The instance, physical device and device |
| `slice.vk_buffer()` | The buffer behind a slice. Views start at an offset into it |
| `image.vk_image()`, `image.vk_image_view()` | An image and its view |

Use them to render results, or to record your own commands. cudash doesn't know about that work, so order it against
cudash streams yourself, for example by calling `stream.synchronize()` first.

## Labels

When the instance has `VK_EXT_debug_utils`, which validation layers and capture tools add, every operation carries a
label such as `memcpy_htod 22.1 MB` or `launch double_all grid (1, 1, 1) block (1024, 1, 1)`. Vulkan objects are
named after the code that made them: `CudaSlice<f32> 3 len 1024`, `default_stream 0`, `CudaImage<f32> 0 64x32`.
Debuggers and profilers show these names.

## RenderDoc

A compute-only program never presents a frame, so RenderDoc's capture key does nothing. Bracket the work with
RenderDoc's in-application API instead, as `examples/frame_capture.rs` does with the `renderdoc` crate.
`ctx.renderdoc_device_pointer()` gives the device pointer its `start_frame_capture` and `end_frame_capture` take.

With the `kernel_debug` feature, which is on by default, the repository's kernels are compiled with `slangc -g`, so
RenderDoc shows the Slang source and can step through a thread.

## Frame boundaries

Tools such as Nsight Graphics capture by frame. `stream.frame()` starts an off-screen frame on a stream, and
`stream.stop_frame()` ends it, through `VK_EXT_frame_boundary`. `stream.supports_frame_boundaries()` says whether the
device has the extension.

## Validation

`just validate` runs the test suite under the Khronos validation layers, and fails on any validation error. Passing
validation doesn't prove ordering is right: some drivers, RADV among them, run correctly even with ordering removed,
so ordering bugs only show on other drivers.
