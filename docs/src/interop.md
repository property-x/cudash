# Working with cudarc

The `cudarc` feature adds `cudash::interop`, which lends cudash memory to real cudarc calls on NVIDIA GPUs, without
copying. Vulkan exports the memory, CUDA imports it, and both see the same bytes. This lets you run cuBLAS, cuFFT or
PTX kernels on cudash data.

```rust,no_run
use cudash::interop::AsCuda;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = cudash::driver::CudaContext::new(0)?;
let stream = ctx.default_stream();
let mut buf = stream.alloc_zeros::<f32>(4)?;
{
    let mut lent = buf.as_cuda_mut()?;           // waits for cudash work on `buf`
    let cuda = lent.stream().clone();            // the CUDA stream paired with `stream`
    cuda.memcpy_htod(&[1.0f32, 2.0, 3.0, 4.0], &mut lent)?;   // a real cudarc call
}                                                // waits for the CUDA work
assert_eq!(stream.clone_dtoh(&buf)?, [1.0, 2.0, 3.0, 4.0]);
# Ok(())
# }
```

## Lending

| Call | What it does |
|---|---|
| `slice.as_cuda()` | Lends for reading. Waits for cudash work that writes the slice |
| `slice.as_cuda_mut()` | Lends for writing. Waits for all cudash work using the slice |
| `as_cuda_with(&stream)`, `as_cuda_mut_with(&stream)` | The same, bound to a CUDA stream of your choice |

The lent objects, `Lent` and `LentMut`, go anywhere cudarc accepts a `&CudaSlice` (or `&mut`): they implement
cudarc's `DevicePtr`, `DevicePtrMut` and `PushKernelArg`. Dropping one waits for its CUDA stream.

- **The borrow checker orders the two APIs.** While a slice is lent, cudash can't use it.
- **Views lend their own range.** A `CudaView` can only be lent for reading.
- **One CUDA stream per cudash stream.** Each cudash stream gets a fresh CUDA stream on first use, never the legacy
  null stream, which would serialize unrelated work. `stream.as_cuda()` returns it.
- **The first lend of a buffer** exports and imports the whole allocation. The buffer keeps that mapping, so later
  lends are cheap.

## Contexts and semaphores

- `ctx.as_cuda()` returns the cudarc `CudaContext` on the same GPU, matched by UUID. Nothing touches CUDA before the
  first call, and on a non-NVIDIA GPU it returns `Error::NoMatchingCudaDevice` without loading CUDA.
- `SharedSemaphore::new(&ctx)` creates a timeline semaphore both APIs can signal and wait on, for ordering your own
  Vulkan and CUDA work by hand: `cuda_signal`, `cuda_wait`, `vk_signal`, `vk_wait` and `vk_semaphore`.

## Graphs

cudarc work on a lent slice while the cudash stream captures becomes part of the cudash graph. See
[mixed graphs](graphs.md#mixed-graphs).

## Errors

Interop calls return `cudash::interop::Error`:

| Error | When |
|---|---|
| `NoMatchingCudaDevice` | No CUDA device has this GPU's UUID, for example on AMD |
| `DeviceMismatch` | The CUDA stream is on another GPU |
| `ExportUnsupported` | The slice's memory can't be exported. Staging and pinned memory never can |
| `InvalidLength` | The slice is empty |
| `Cuda`, `Vulkan`, `Driver` | An error from CUDA, Vulkan or `cudash::driver` |

## Limits

- **Linux only,** through opaque file descriptors.
- **One way:** cudash memory can be lent to CUDA, but cudarc memory can't be used by cudash.
- **No copying fallback.** If zero-copy isn't possible, the lend returns an error.
- **`device_ptr` addresses aren't CUDA addresses.** Use a lend to get memory CUDA can use.
- **Only NVIDIA GPUs.** CUDA can't see another vendor's memory.

## NVIDIA in a distrobox

On a Fedora-based host such as Bazzite, `distrobox --nvidia` mounts the host's NVIDIA Vulkan manifest into an Ubuntu
box, and it points at `/usr/lib64`, which doesn't exist there. Vulkan then can't see the GPU. `just setup` installs a
manifest with Ubuntu's path, under `/etc/vulkan/icd.d/`.
