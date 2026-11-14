# Kernels

Vulkan runs SPIR-V, not PTX, so CUDA C++ kernels have to be rewritten. [Slang](https://shader-slang.org) compiles
one source to both SPIR-V for cudash and PTX for cudarc, with the same launch arguments.

## Writing a kernel

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

- **Pass `-fvk-use-entrypoint-name`,** or Slang renames the entry point to `main`.
- **Arguments are `uniform` parameters.** `RWStructuredBuffer` won't work: it isn't a launch argument on CUDA, and it
  needs descriptor sets on Vulkan.
- **Pointers** are `float*`. A read-only pointer is `Ptr<float, Access.Read>`, since Slang rejects `const float*`.
- **Use `SV_DispatchThreadID` for indices.** The `numthreads` a kernel declares is replaced by the launch's
  `block_dim`, so don't read the block size from a constant.
- **Struct layout** follows std430, which matches `#[repr(C)]` for scalars. Vectors such as `float3` align to 16
  bytes, so check structs that use them.

In this repository, `build.rs` compiles the `.slang` files in `tests/kernels` and `examples/slang` to SPIR-V and
PTX in `OUT_DIR`.

## Launching

The same as cudarc, with `Spirv` in place of `Ptx`:

```rust,no_run
use cudash::driver::*;

# fn main() -> Result<(), DriverError> {
let ctx = CudaContext::new(0)?;
let stream = ctx.default_stream();
let module = ctx.load_module(Spirv::from_file("double_all.spv"))?;
let f = module.load_function("double_all")?;

let n = 1000i32;
let mut x = stream.clone_htod(&vec![1.0f32; n as usize])?;
let mut builder = stream.launch_builder(&f);
builder.arg(&mut x).arg(&n);
unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }?;
assert!(stream.clone_dtoh(&x)?.iter().all(|&v| v == 2.0));
# Ok(())
# }
```

`Spirv::from_binary` takes a `Vec<u8>` or `include_bytes!` output without copying it. SPIR-V has no text form that
drivers accept, so there's no equivalent of `Ptx::from_src`.

## How arguments are passed

Arguments go into push constants, packed in `arg` order like C function arguments. A slice or view becomes its 64-bit
buffer device address, plus the view's offset. Other values are copied by value. No descriptor sets are involved.

Vulkan 1.3 guarantees 128 bytes of push constants, and many GPUs have more. `launch` returns
`DriverError::LaunchLimit` if the arguments don't fit, or if the grid or block exceeds the device's limits.

## Block size

CUDA picks the block size at launch, and Vulkan at compile time. cudash rewrites the workgroup size in the SPIR-V
for each `block_dim` it sees, and caches a pipeline for each.

`LaunchConfig::for_num_elems(n)` uses blocks of 1024 threads, as in cudarc. Some GPUs allow fewer, and then `launch`
returns an error, so pick a smaller `block_dim` yourself.

## What doesn't carry over

- **`shared_mem_bytes` is ignored.** A shader declares its shared memory at compile time.
- **`f64`, 8-bit and 16-bit types** depend on optional GPU features. cudash enables all that the GPU has.
- **No `printf` or `assert`** in shaders. Write results to a buffer instead.
- **No `get_global`,** so constant memory becomes a slice argument.
- **Kernels can't take images,** which would need descriptor sets.
- **Pointer arithmetic beyond a buffer, recursion, and CUDA-only intrinsics** don't exist in SPIR-V.
