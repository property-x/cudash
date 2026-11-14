# Memory

## Contexts and slices

A `CudaContext` is one GPU: a Vulkan instance and device. `CudaSlice<T>` is device memory holding `len` elements of
`T`, like cudarc's. Allocation, copies and frees all go through a stream.

```rust
use cudash::driver::*;

# fn main() -> Result<(), DriverError> {
let ctx = CudaContext::new(0)?;
println!("{} with {} bytes", ctx.name()?, ctx.total_mem()?);
let stream = ctx.default_stream();

let zeros = stream.alloc_zeros::<f32>(1024)?;      // zeroed on the GPU
let data = stream.clone_htod(&[1u32, 2, 3])?;      // copied from the host
let copy = stream.clone_dtod(&data)?;              // copied on the GPU
let uninit = unsafe { stream.alloc::<u8>(64) }?;   // contents undefined until written
let host: Vec<u32> = stream.clone_dtoh(&copy)?;    // back to the host, blocking
assert_eq!(host, [1, 2, 3]);
# let _ = (zeros, uninit);
# Ok(())
# }
```

Element types implement `DeviceRepr`, and `alloc_zeros` and `memset_zeros` also need `ValidAsZeroBits`. These are
cudash's own traits, since the base crate doesn't depend on cudarc. A struct used with both crates needs both
crates' impls.

A slice is freed when it's dropped. The stream keeps the memory alive until the GPU has finished with it, so dropping
a slice with work still queued is safe.

## Copies

| Call | Direction | Returns |
|---|---|---|
| `clone_htod(&src)`, `memcpy_htod(&src, &mut dst)` | Host to device | At once. The data is copied on the CPU first, so `src` is free to change, except for [registered memory](#host-memory-without-staging) |
| `clone_dtoh(&src)`, `memcpy_dtoh(&src, &mut dst)` | Device to host | When the copy has finished |
| `clone_dtod(&src)`, `memcpy_dtod(&src, &mut dst)` | Device to device | At once |
| `memset_zeros(&mut dst)` | Fill with zeros | At once |

The `memcpy_*` calls panic if `dst` is shorter than `src`, as in cudarc. Every call runs in stream order: a copy
queued after a kernel sees the kernel's results.

## Views

Views are ranges of a slice, as in cudarc. Read-only views are `CudaView`, writable ones `CudaViewMut`, and every
copy and launch accepts both.

```rust
# use cudash::driver::*;
# fn main() -> Result<(), DriverError> {
# let stream = CudaContext::new(0)?.default_stream();
let mut buf = stream.clone_htod(&[0u32; 8])?;
{
    let (mut left, mut right) = buf.split_at_mut(4);
    stream.memcpy_htod(&[1u32; 4], &mut left)?;
    stream.memcpy_htod(&[2u32; 4], &mut right)?;
}
let middle = buf.slice(2..6);
assert_eq!(stream.clone_dtoh(&middle)?, [1, 1, 2, 2]);
assert!(buf.try_slice(4..20).is_none());
# Ok(())
# }
```

`slice`, `slice_mut`, `split_at` and `split_at_mut` panic on a bad range, and their `try_` versions return `None`.
`transmute` and `transmute_mut` reinterpret the bytes as another type. They're `unsafe`, as in cudarc, and return
`None` if `len` elements of the new type don't fit.

## Host memory without staging

`memcpy_htod` normally copies through a staging buffer, which costs a CPU copy. Two kinds of host memory skip it:

- **Pinned memory,** `ctx.alloc_pinned::<T>(len)`, returns a `PinnedHostSlice<T>`, as in cudarc. Copies read and
  write it directly. Its accessors (`as_slice`, `as_mut_slice`, `as_ptr`, `as_mut_ptr`) wait for copies that use it,
  so they never see one half done.
- **Registered memory,** `unsafe { ctx.host_register(&data) }`, imports existing host memory, like CUDA's
  `cuMemHostRegister` (cudarc only has that in `sys`). Copies from inside it return without copying on the CPU, and
  read the memory when the GPU gets to them, so don't write it while a copy may be running: synchronize the stream
  first. Until `ctx.host_unregister(ptr)`, which waits for those copies, the memory must stay allocated and not move. Registering needs `VK_EXT_external_memory_host`, and not
  every kind of memory can be registered: on RADV, `static` data can't.

```rust
# use cudash::driver::*;
# fn main() -> Result<(), DriverError> {
# let ctx = CudaContext::new(0)?;
# let stream = ctx.default_stream();
let mut pinned = unsafe { ctx.alloc_pinned::<f32>(256) }?;
pinned.as_mut_slice()?.fill(3.0);
let on_gpu = stream.clone_htod(&pinned)?;   // no CPU copy
stream.memcpy_dtoh(&on_gpu, &mut pinned)?;
assert!(pinned.as_slice()?.iter().all(|&v| v == 3.0));
# Ok(())
# }
```

## Device addresses

`device_ptr(&stream)` and `device_ptr_mut(&stream)` return the buffer device address as a `u64`, with a `SyncOnDrop`
guard, as in cudarc. The address is a Vulkan buffer device address: kernels can use it, but CUDA can't. Getting it
and dropping the guard each order the stream against the buffer's other uses, and keep the buffer alive until the
stream's work so far is done.

## Limits

- **4096 allocations.** Vulkan only guarantees this many per device, and each slice is one allocation.
- **One allocation's size** is capped by the driver, often just under 4 GiB. Larger `alloc` calls return an error.
