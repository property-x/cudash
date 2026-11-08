//! cudarc's `examples/06-threading.rs` (MIT OR Apache-2.0) on Vulkan. Shaders have no
//! `printf`, so `examples/slang/hello_world.slang` hands `i` back for the host to print.

use std::{sync::Arc, thread};

use cudash::driver::*;

const KERNEL: &str = concat!(env!("OUT_DIR"), "/hello_world.spv");

fn hello(stream: &Arc<CudaStream>, f: &CudaFunction, i: i32) -> Result<(), DriverError> {
    let mut out = stream.alloc_zeros::<i32>(1)?;
    let mut builder = stream.launch_builder(f);
    builder.arg(&mut out).arg(&i);
    unsafe { builder.launch(LaunchConfig::for_num_elems(1)) }?;
    println!(
        "Hello from the kernel in thread {}",
        stream.clone_dtoh(&out)?[0]
    );
    Ok(())
}

fn main() -> Result<(), DriverError> {
    {
        // Option 1: sharing ctx & module between threads
        let ctx = CudaContext::new(0)?;
        let module = ctx.load_module(Spirv::from_file(KERNEL))?;
        thread::scope(|s| {
            for i in 0..10i32 {
                let thread_ctx = ctx.clone();
                let thread_module = module.clone();
                s.spawn(move || {
                    let stream = thread_ctx.default_stream();
                    let f = thread_module.load_function("hello_world").unwrap();
                    hello(&stream, &f, i).unwrap();
                });
            }
        });
    }

    {
        // Option 2: initializing different context in each
        // Unlike on CUDA, each context here is its own Vulkan device with its own default stream.
        thread::scope(|s| {
            for i in 0..10i32 {
                s.spawn(move || {
                    let ctx = CudaContext::new(0).unwrap();
                    let module = ctx.load_module(Spirv::from_file(KERNEL)).unwrap();
                    let stream = ctx.default_stream();
                    let f = module.load_function("hello_world").unwrap();
                    hello(&stream, &f, i).unwrap();
                });
            }
        });
    }

    Ok(())
}
