//! Runs `examples/slang/kernel.slang` on real CUDA, which needs an NVIDIA GPU. The PTX is
//! built when NVRTC is installed. `slang_spirv.rs` is the same program on Vulkan. The two
//! differ in the imports, the module type (`Ptx` vs `Spirv`) and the kernel path, and nothing else.
//!
//! Run: `cargo run --example slang_cudarc`

use cudarc::{driver::*, nvrtc::Ptx};

const KERNEL: &str = concat!(env!("OUT_DIR"), "/kernel.ptx");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_file(KERNEL))?;
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
