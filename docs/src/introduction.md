# Introduction

cudash is [cudarc](https://github.com/chelsea0x3b/cudarc)'s basic driver API, implemented on Vulkan with
[ash](https://github.com/ash-project/ash). Code written against `cudarc::driver` runs on any GPU with a Vulkan 1.3
driver, with no CUDA installed. For code that only uses memory and copies, changing the import is enough:

```rust
use cudash::driver::*; // was: use cudarc::driver::*;

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let a: CudaSlice<f32> = stream.clone_htod(&[1.0f32; 100])?;
    let mut b = stream.alloc_zeros::<f32>(100)?;
    stream.memcpy_dtod(&a, &mut b)?;
    assert_eq!(stream.clone_dtoh(&b)?, vec![1.0; 100]);
    Ok(())
}
```

Kernels are the one real difference: Vulkan runs SPIR-V, not PTX. Writing kernels in
[Slang](https://shader-slang.org) gives one source that compiles to both, with the same launch arguments.

The crate has two modules:

| Module | What it is | Needs |
|---|---|---|
| `cudash::driver` | cudarc's driver API on Vulkan | Any Vulkan 1.3 GPU |
| `cudash::interop` | Lends cudash memory to real cudarc calls, without copying | The `cudarc` feature (on by default), and an NVIDIA GPU at run time |

cudash is a prototype. Its tests pass on an AMD GPU with RADV and on an NVIDIA GTX 1650 Ti, and haven't run on other
drivers.

## How to read this book

- [Getting started](getting-started.md) covers requirements, features and building.
- [Memory](memory.md), [Streams](streams.md), [Kernels](kernels.md) and [Graphs](graphs.md) follow cudarc's API and
  say where Vulkan behaves differently.
- [Images](images.md) and [Vulkan and debugging](vulkan.md) are cudash-only.
- [Working with cudarc](interop.md) covers sharing memory with CUDA on NVIDIA GPUs.
- [Differences from cudarc](differences.md) lists every difference in one place.

The examples in this book are compiled by `cargo test`, and most of them run on a GPU too.
