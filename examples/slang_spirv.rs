//! Runs `examples/slang/kernel.slang` on Vulkan, on any Vulkan 1.3 GPU. No CUDA or NVIDIA
//! driver is needed. `slang_cudarc.rs` is the same program on real CUDA. The two differ in
//! the imports, the module type (`Spirv` vs `Ptx`) and the kernel path, and nothing else.
//!
//! Run: `cargo run --example slang_spirv`

use cudash::driver::*;

const KERNEL: &str = concat!(env!("OUT_DIR"), "/kernel.spv");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Spirv::from_file(KERNEL))?;
    let f = module.load_function("computeMain")?;

    let n = 256i32;
    let mut data = stream.alloc_zeros::<f32>(n as usize)?;
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut data).arg(&n);
    unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }?;

    for (i, value) in stream.clone_dtoh(&data)?.iter().enumerate() {
        assert_eq!(*value, i as f32 * 2.0 + 1.0);
    }
    println!("ok: {n} elements computed on {}", ctx.name()?);
    Ok(())
}
