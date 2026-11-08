//! cudarc's `examples/nvrtc-compile.rs` (MIT OR Apache-2.0) on Vulkan. cudash has no runtime
//! compiler, so this runs `slangc`. Its `-fp-mode fast` stands in for NVRTC's relaxed
//! `ftz`, `prec_div`, `prec_sqrt` and `fmad` options.

use std::process::Command;

use cudash::driver::{CudaContext, Spirv};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let src = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/kernels/sin_kernel.slang"
    );
    let spv = std::env::temp_dir().join("cudash_nvrtc_compile.spv");

    let output = Command::new("slangc")
        .arg(src)
        .args(["-target", "spirv", "-fvk-use-entrypoint-name"])
        .args(["-fp-mode", "fast", "-o"])
        .arg(&spv)
        .output()?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into());
    }

    // Unlike NVRTC, slangc can't tell us the result is loadable, so check that too.
    let ctx = CudaContext::new(0)?;
    let module = ctx.load_module(Spirv::from_file(&spv))?;
    module.load_function("sin_kernel")?;
    std::fs::remove_file(&spv)?;

    println!("Compilation succeeded!");
    Ok(())
}
