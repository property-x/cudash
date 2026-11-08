# cudash: minimal and safe api over the vulkan toolkit

[cudarc](https://github.com/chelsea0x3b/cudarc)'s basic driver API, implemented on Vulkan, so the same code runs on GPUs
without CUDA via [ash](https://github.com/ash-project/ash), hence the name cudash.

```rust
// switch over to cudash
use cudash::driver::*; // use cudarc::driver::*;

// do it like they do on the cudarc channel
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

Hint: This is a prototype to help me learn and understand Vulkan compute from a Cuda developers perspective.

**Contributions welcome!**

## What's covered

- **Memory:** `alloc`, `alloc_zeros`, `clone_htod`, `clone_dtoh`, `clone_dtod`, `memcpy_htod`,
  `memcpy_dtoh`, `memcpy_dtod`, `memset_zeros`, `null`, and `Clone` on slices.
- **Views:** `slice`, `slice_mut`, `split_at`, `split_at_mut`, their `try_` versions, `as_view` and
  `as_view_mut`. `device_ptr` and `device_ptr_mut` return the buffer device address.
- **Streams:** `default_stream`, `new_stream`, `fork` and `synchronize`. Work runs in call order, and
  buffers used from several streams are ordered automatically.
- **Kernels:** `load_module`, `load_function`, `launch_builder`, `arg`, `launch` and `LaunchConfig`.

## Current Limitations

It should work on any system given these limitations:

* Unix (`fd` handles).
* Vulkan 1.3+ capable GPUs -- _almost_ 1.2 capable (only `queue_submit2`).
* interop needs Cuda capable hardware.

## Kernels

Any Spir-V capable language is fine. Using [Slang](https://docs.shader-slang.org/en/latest/) allows you to check if
cudash and cudarc behave the same.

```slang
[shader("compute")]
[numthreads(1, 1, 1)]
void double_all(uniform float* x, uniform int n, uint3 id : SV_DispatchThreadID)
{
    if (id.x < n) x[id.x] *= 2.0;
}
```

```sh
slangc double_all.slang -target spirv -fvk-use-entrypoint-name -o double_all.spv
slangc double_all.slang -target ptx -o double_all.ptx
```

Without `-fvk-use-entrypoint-name`, Slang renames the entry point to `main`. Launching the same as
in cudarc, with `Spirv` in place of `Ptx`:

```rust
let module = ctx.load_module(Spirv::from_file("double_all.spv")) ?;
let f = module.load_function("double_all") ?;
let mut builder = stream.launch_builder( & f);
builder.arg( & mut x).arg( & n);
unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) } ?;
```

- **Arguments** are passed in order, with slices and views as GPU pointers. They must be `uniform`
  parameters. `RWStructuredBuffer` won't work, because it isn't a launch argument on CUDA and needs
  descriptor sets on Vulkan. Use `Ptr<float, Access.Read>` for a read-only pointer.
- **Block size:** a kernel can declare any size, and the launch's `block_dim` replaces it. So use
  `SV_DispatchThreadID` for indices.
- **cudarc:** the same source compiles to PTX with the same signature, but that PTX hasn't run yet,
  since it needs an NVIDIA GPU. `examples/slang_spirv.rs` and `examples/slang_cudarc.rs` are the same
  program on each.
- **PTX files** end in a NUL byte if you compile by hand, which makes `Ptx::from_src` panic. `build.rs`
  strips it.

## Examples

Each of cudarc's examples has a version in `examples/` under the same name, except where noted.
Run one with `just example 01-allocate`.
The goal is to have parity wherever it makes sense.

| Example                                           | Difference from cudarc                                                                         |
|---------------------------------------------------|------------------------------------------------------------------------------------------------|
| `01-allocate`, `02-copy`                          | Only the import                                                                                |
| `03-launch-kernel`, `04-streams`, `matmul-kernel` | Slang kernels, loaded as SPIR-V                                                                |
| `05-device-repr`, `06-threading`                  | Shaders have no `assert` or `printf`, so kernels write results to a buffer                     |
| `07-build-workflow`                               | One file using cudash's `build.rs`. No `nvcc` or `bindgen`, so the struct is written twice     |
| `09-constant-memory`                              | No `get_global`, so the coefficients are a slice argument                                      |
| `10-function-attributes`, `12-context-config`     | Print Vulkan's device limits, read-only. Per-kernel attributes and settable limits don't exist |
| `13-copy-multi-gpu`                               | Copies through the host. Needs two GPUs, and software renderers don't count                    |
| `cufile-copy`                                     | Reads the file through host memory, not GPUDirect Storage                                      |
| `nvrtc-compile`                                   | Runs `slangc` at runtime                                                                       |

`08-cupti` has no version, because profiling APIs are out of scope.

## Using it with Vulkan

`ctx.vk_instance()`, `ctx.vk_physical_device()`, `ctx.vk_device()` and `slice.vk_buffer()` give you the
Vulkan objects, for example to render results. Ordering that work against streams is up to you.

## Using it with cudarc

The `cudarc` feature adds `cudash::interop`, for sharing memory with real CUDA on NVIDIA hardware.
`Interop::new(&ctx, cuda_stream)` pairs a cudash context with a cudarc stream on the same GPU, and
creates buffers and semaphores both can use. `cuda_context_for(&ctx)` finds that GPU in CUDA. It has
never been run on NVIDIA, and `lend` (`plan/api.md`) will replace the separate buffer type.

## Building and testing

Everything is available using [just](https://just.systems).

Run `just` to list every command. The main ones:

| Command            | What it does                                                   |
|--------------------|----------------------------------------------------------------|
| `just doctor`      | Shows the tools and GPUs this machine has                      |
| `just test`        | Runs the tests. `just test ptx` runs only those matching `ptx` |
| `just validate`    | Runs the tests under Vulkan validation, and fails on any error |
| `just lint`        | Checks formatting and clippy                                   |
| `just ci`          | Runs everything that works without an NVIDIA GPU               |
| `just test-nvidia` | Runs the real CUDA tests. Needs an NVIDIA GPU                  |

`build.rs` compiles the kernels in `tests/kernels` and `examples/slang` into `OUT_DIR`, as SPIR-V and
PTX. Without NVRTC, you get a warning and no PTX. `SLANGC` and `NVRTC_PATH` override where the tools
are.
