//! cudarc's `examples/05-device-repr.rs` (MIT OR Apache-2.0) on Vulkan. Shaders have no
//! `assert`, so `examples/slang/device_repr.slang` writes its checks to a buffer instead.

use cudash::driver::*;

/// Here's the struct in rust, note that we have #[repr(C)]
/// here which allows us to pass it to the kernel.
#[repr(C)]
struct MyCoolRustStruct {
    a: f32,
    b: f64,
    c: u32,
    d: usize,
}

/// We have to implement this to send it to the kernel!
unsafe impl DeviceRepr for MyCoolRustStruct {}

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(Spirv::from_file(concat!(
    env!("OUT_DIR"),
    "/device_repr.spv"
    )))?;
    let f = module.load_function("my_custom_kernel")?;

    // try changing some of these values to see the assert below fail
    let thing = MyCoolRustStruct {
        a: 1.0,
        b: 2.34,
        c: 57,
        d: 420,
    };

    let mut ok = stream.alloc_zeros::<u32>(1)?;
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut ok);
    // since MyCoolRustStruct implements DeviceRepr, we can pass it to launch.
    builder.arg(&thing);
    unsafe { builder.launch(LaunchConfig::for_num_elems(1)) }?;

    assert_eq!(stream.clone_dtoh(&ok)?, [1]);

    Ok(())
}
