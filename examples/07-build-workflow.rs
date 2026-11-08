//! cudarc's `examples/07-build-workflow` (MIT OR Apache-2.0) on Vulkan. There, a separate
//! crate's `build.rs` runs `nvcc` and `bindgen`. Here, cudash's own `build.rs` compiles
//! `examples/slang/my_struct_kernel.slang` with `slangc`, and this file embeds the result.
//! Slang has no bindgen, so `MyStruct` is written out by hand on both sides.

use std::time::Instant;

use cudash::driver::*;

#[repr(C)]
#[derive(Clone)]
struct MyStruct {
    data: [f32; 4],
}

unsafe impl DeviceRepr for MyStruct {}

// include the compiled SPIR-V in the binary
const KERNEL_MY_STRUCT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/my_struct_kernel.spv"));

fn main() -> Result<(), DriverError> {
    // setup GPU device
    let now = Instant::now();

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    println!("Time taken to initialise Vulkan: {:.2?}", now.elapsed());

    // load SPIR-V
    let now = Instant::now();

    let my_module = ctx.load_module(Spirv::from_binary(KERNEL_MY_STRUCT))?;
    let my_function = my_module.load_function("my_struct_kernel")?;

    println!("Time taken to load SPIR-V: {:.2?}", now.elapsed());

    // create data
    let now = Instant::now();

    let n = 10_usize;
    let my_structs = vec![MyStruct { data: [1.0; 4] }; n];

    // copy to GPU
    let mut gpu_my_structs = stream.clone_htod(&my_structs)?;

    println!("Time taken to initialise data: {:.2?}", now.elapsed());

    let now = Instant::now();
    let mut launch_args = stream.launch_builder(&my_function);
    launch_args.arg(&mut gpu_my_structs);
    launch_args.arg(&n);
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe { launch_args.launch(cfg) }?;

    println!("Time taken to call kernel: {:.2?}", now.elapsed());

    let my_structs = stream.clone_dtoh(&gpu_my_structs)?;

    assert!(my_structs.iter().all(|i| i.data == [2.0; 4]));

    Ok(())
}
