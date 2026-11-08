//! The Slang kernels from `tests/driver.rs`, compiled to PTX and launched by real cudarc
//! with the same argument list. Needs an NVIDIA GPU, and NVRTC when building, so that
//! `build.rs` writes the PTX. Elsewhere this prints SKIPPED (or panics, with
//! CUDASH_REQUIRE_NVIDIA set).

mod common;

use std::path::Path;

use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

const PTX: &str = concat!(env!("OUT_DIR"), "/double_all.ptx");

fn launches_like_cudash() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_file(PTX))?;
    let f = module.load_function("double_all")?;

    let mut x = stream.clone_htod(&vec![1.0f32; 5000])?;
    let n = 5000i32;
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut x).arg(&n);
    unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }?;
    assert_eq!(stream.clone_dtoh(&x)?, vec![2.0; 5000]);
    Ok(())
}

#[test]
fn same_slang_source_runs_on_cudarc() {
    if !common::have_nvidia() {
        return;
    }
    if !Path::new(PTX).exists() {
        common::skip(format!("no {PTX}, which needs NVRTC when building"));
        return;
    }
    launches_like_cudash().unwrap();
}
