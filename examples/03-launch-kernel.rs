//! cudarc's `examples/03-launch-kernel.rs` (MIT OR Apache-2.0) on Vulkan. The kernel is
//! `tests/kernels/sin_kernel.slang`, loaded as SPIR-V instead of PTX.

use cudash::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg, Spirv};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // You can load a function from pre-compiled SPIR-V like so:
    let module = ctx.load_module(Spirv::from_file(concat!(
        env!("OUT_DIR"),
        "/sin_kernel.spv"
    )))?;

    // and then load a function from it:
    let f = module.load_function("sin_kernel")?;

    let a_host = [1.0, 2.0, 3.0];

    let a_dev = stream.clone_htod(&a_host)?;
    let mut b_dev = a_dev.clone();

    // we use a buidler pattern to launch kernels.
    let n = 3i32;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch_args = stream.launch_builder(&f);
    launch_args.arg(&mut b_dev);
    launch_args.arg(&a_dev);
    launch_args.arg(&n);
    unsafe { launch_args.launch(cfg) }?;

    let a_host_2 = stream.clone_dtoh(&a_dev)?;
    let b_host = stream.clone_dtoh(&b_dev)?;

    println!("Found {b_host:?}");
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(&a_host, a_host_2.as_slice());

    Ok(())
}
