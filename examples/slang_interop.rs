//! cudash owns the data, and real cudarc runs the PTX build of `examples/slang/kernel.slang` on
//! it, through `cudash::interop` without a copy. Needs an NVIDIA GPU with CUDA, and the PTX,
//! which is built when NVRTC is installed.
//!
//! The code is in three kinds of part: cudash as in `slang_spirv.rs`, cudarc as in
//! `slang_cudarc.rs`, and the interop that joins them.
//!
//! Run: `cargo run --example slang_interop`

use cudarc::{driver::*, nvrtc::Ptx};
use cudash::interop::{Interop, cuda_context_for};

const KERNEL: &str = concat!(env!("OUT_DIR"), "/kernel.ptx");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Data Allocation: Vulkan
    let cudash_ctx = cudash::driver::CudaContext::new(0)?;
    let cudash_stream = cudash_ctx.default_stream();
    let len = 512;
    let mut data = cudash_stream.clone_htod(&vec![-1.0f32; len])?;

    // Function Definition: Cuda
    let cudarc_stream = cuda_context_for(&cudash_ctx)?.default_stream();
    let interop = Interop::new(&cudash_ctx, cudarc_stream.clone())?;
    let module = cudarc_stream.context().load_module(Ptx::from_file(KERNEL))?;
    let f = module.load_function("computeMain")?;
    let n = (len / 2) as i32; // the kernel writes 2i + 1 into the first half only

    // Interop: Calling Cuda function, borrowing data from Vulkan
    let cuda_result: Vec<f32> = {
        let mut lent = interop.lend_mut(&mut data)?;
        let mut builder = cudarc_stream.launch_builder(&f);
        builder.arg(&mut lent).arg(&n);
        unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        cudarc_stream.clone_dtoh(&lent)?  // return cuda copy of result
    };

    //  Vulkan: Copy result back to host
    let vulkan_result = cudash_stream.clone_dtoh(&data)?;
    for (i, (value_vulkan, value_cuda)) in vulkan_result.iter().zip(cuda_result.iter()).enumerate() {
        let expected = if i < n as usize { i as f32 * 2.0 + 1.0 } else { -1.0 };
        assert_eq!(*value_vulkan, expected, "element {i} does not match: vulkan={value_vulkan}, expected={expected}");
        assert_eq!(*value_cuda, expected, "element {i} does not match: vulkan={value_cuda}, expected={expected}");
    }
    println!("ok: CUDA wrote {n} of {len} cudash elements on {}", cudash_ctx.name()?);
    Ok(())
}
