//! `cudash::driver` used like `cudarc::driver`: cudarc's own examples with only the import and kernel changed,
//! plus what no feature scenario covers (the view API, PTX signatures). Behaviour lives in `features/`.
//! Runs on any Vulkan 1.3 GPU.

use std::sync::Arc;

use cudash::driver::*;

const SIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sin_kernel.spv"));
const DOUBLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/double_all.spv"));
const WRITE_STRUCT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/write_struct.spv"));

fn load(ctx: &Arc<CudaContext>, spirv: &[u8], name: &str) -> CudaFunction {
    let module = ctx.load_module(Spirv::from_binary(spirv.to_vec())).unwrap();
    module.load_function(name).unwrap()
}

fn double(stream: &CudaStream, f: &CudaFunction, x: &mut CudaSlice<f32>) {
    let n = x.len() as i32;
    let mut builder = stream.launch_builder(f);
    builder.arg(x).arg(&n);
    unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.unwrap();
}

/// cudarc's examples/01-allocate.rs
#[test]
fn example_allocate() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // unsafe initialization of unset memory
    let _: CudaSlice<f32> = unsafe { stream.alloc::<f32>(10) }?;

    // this will have memory initialized as 0
    let _: CudaSlice<f64> = stream.alloc_zeros::<f64>(10)?;

    // initialize with slices!
    let _: CudaSlice<usize> = stream.clone_htod(&[0; 10])?;
    let _: CudaSlice<u32> = stream.clone_htod(&[1, 2, 3])?;

    Ok(())
}

/// cudarc's examples/02-copy.rs
#[test]
fn example_copy() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let a: CudaSlice<f64> = stream.alloc_zeros::<f64>(10)?;
    let mut b = stream.alloc_zeros::<f64>(10)?;

    // you can do device to device copies of course
    stream.memcpy_dtod(&a, &mut b)?;

    // but also host to device copys with already allocated buffers
    stream.memcpy_htod(&vec![2.0; b.len()], &mut b)?;
    // you can use any type of slice
    stream.memcpy_htod(&[3.0; 10], &mut b)?;

    // you can transfer back using clone_dtoh
    let mut a_host: Vec<f64> = stream.clone_dtoh(&a)?;
    assert_eq!(a_host, [0.0; 10]);

    let b_host = stream.clone_dtoh(&b)?;
    assert_eq!(b_host, [3.0; 10]);

    // or transfer into a pre allocated slice
    stream.memcpy_dtoh(&b, &mut a_host)?;
    assert_eq!(a_host, b_host);

    Ok(())
}

/// cudarc's examples/03-launch-kernel.rs
#[test]
fn example_launch_kernel() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(Spirv::from_binary(SIN.to_vec()))?;
    let f = module.load_function("sin_kernel")?;

    let a_host = [1.0, 2.0, 3.0];
    let a_dev = stream.clone_htod(&a_host)?;
    let mut b_dev = a_dev.clone();

    let n = 3i32;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch_args = stream.launch_builder(&f);
    launch_args.arg(&mut b_dev);
    launch_args.arg(&a_dev);
    launch_args.arg(&n);
    unsafe { launch_args.launch(cfg) }?;

    let a_host_2 = stream.clone_dtoh(&a_dev)?;
    let b_host = stream.clone_dtoh(&b_dev)?;

    let expected = a_host.map(f32::sin);
    for (found, want) in b_host.iter().zip(expected) {
        assert!((found - want).abs() < 1e-4, "{found} vs {want}");
    }
    assert_eq!(&a_host, a_host_2.as_slice());
    Ok(())
}

/// cudarc's examples/04-streams.rs
#[test]
fn example_streams() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(Spirv::from_binary(SIN.to_vec()))?;
    let f = module.load_function("sin_kernel")?;

    let n = 3i32;
    let a_host = [1.0, 2.0, 3.0];
    let a_dev = stream.clone_htod(&a_host)?;
    let mut b_dev = stream.alloc_zeros::<f32>(n as usize)?;

    // the copy and alloc_zeros above complete before work on this stream starts
    let stream2 = stream.fork()?;

    let mut builder = stream2.launch_builder(&f);
    builder.arg(&mut b_dev);
    builder.arg(&a_dev);
    builder.arg(&n);
    unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }?;

    // this must wait for stream2, because b_dev was mutated there
    let a_host_2 = stream.clone_dtoh(&a_dev)?;
    let b_host = stream.clone_dtoh(&b_dev)?;

    let expected = a_host.map(f32::sin);
    for (found, want) in b_host.iter().zip(expected) {
        assert!((found - want).abs() < 1e-4, "{found} vs {want}");
    }
    assert_eq!(&a_host, a_host_2.as_slice());
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct MyCoolRustStruct {
    a: f32,
    b: f64,
    c: u32,
    d: usize,
}

// SAFETY: #[repr(C)] and plain data, matching write_struct.slang.
unsafe impl DeviceRepr for MyCoolRustStruct {}

/// cudarc's examples/05-device-repr.rs
#[test]
fn example_device_repr() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let f = load(&ctx, WRITE_STRUCT, "write_struct");

    let thing = MyCoolRustStruct {
        a: 1.0,
        b: 2.0,
        c: 3,
        d: 4,
    };
    let mut out = stream.alloc_zeros::<f64>(4)?;
    let mut builder = stream.launch_builder(&f);
    builder.arg(&mut out);
    builder.arg(&thing);
    unsafe { builder.launch(LaunchConfig::for_num_elems(1)) }?;
    assert_eq!(stream.clone_dtoh(&out)?, [1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn kernels_cover_every_element_across_many_blocks() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let f = load(&ctx, DOUBLE, "double_all");
    // More than one block of 1024, in a shader that declares a block of 1.
    let mut a = stream.clone_htod(&vec![1.0f32; 5000]).unwrap();
    double(&stream, &f, &mut a);
    assert_eq!(stream.clone_dtoh(&a).unwrap(), vec![2.0; 5000]);
}

#[test]
fn views_work_like_slices() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut a = stream.clone_htod(&[1u32, 2, 3, 4]).unwrap();

    stream.memcpy_htod(&[9u32], &mut a.slice_mut(1..2)).unwrap();
    assert_eq!(stream.clone_dtoh(&a.slice(1..3)).unwrap(), [9, 3]);
    assert_eq!(stream.clone_dtoh(&a.slice(..=1)).unwrap(), [1, 9]);

    let (mut head, mut tail) = a.split_at_mut(2);
    stream.memcpy_htod(&[7u32, 7], &mut head).unwrap();
    stream.memcpy_htod(&[8u32, 8], &mut tail).unwrap();
    let (head, tail) = a.split_at(1);
    assert_eq!((head.len(), tail.len()), (1, 3));
    assert_eq!(stream.clone_dtoh(&tail.slice(1..)).unwrap(), [8, 8]);

    assert!(a.try_slice(3..9).is_none());
    assert!(a.try_split_at_mut(5).is_none());
}

#[test]
fn clones_are_independent() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let a = stream.clone_htod(&[1u32, 2]).unwrap();
    let mut b = a.clone();
    stream.memcpy_htod(&[5u32, 6], &mut b).unwrap();
    let c = stream.clone_dtod(&a).unwrap();
    assert_eq!(stream.clone_dtoh(&a).unwrap(), [1, 2]);
    assert_eq!(stream.clone_dtoh(&c).unwrap(), [1, 2]);
    assert_eq!(Vec::try_from(b).unwrap(), [5, 6]);
}

/// `build.rs` writes these if NVRTC is installed. Both backends must keep the kernel's
/// name, so `load_function` takes the same string, and take the same plain arguments.
#[test]
fn ptx_matches_the_spirv_kernels() {
    for (name, params) in [("double_all", 2), ("sin_kernel", 3), ("write_struct", 2)] {
        let path = format!("{}/{name}.ptx", env!("OUT_DIR"));
        let Ok(ptx) = std::fs::read_to_string(&path) else {
            eprintln!("SKIPPED: no {path}, which needs NVRTC");
            return;
        };
        assert!(!ptx.contains('\0'), "Ptx::from_src would panic on this");
        let (_, signature) = ptx
            .split_once(&format!(".entry {name}("))
            .unwrap_or_else(|| panic!("no entry point {name} in:\n{ptx}"));
        let signature = signature.split_once(')').expect("signature ends").0;
        assert_eq!(signature.matches(".param").count(), params, "{signature}");
    }
}
