//! cudarc's `examples/09-constant-memory.rs` (MIT OR Apache-2.0) on Vulkan. cudash has no
//! `get_global`, so the coefficients are a read-only slice passed as a kernel argument.

use cudash::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg, Spirv};

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(Spirv::from_file(concat!(
        env!("OUT_DIR"),
        "/constant_memory.spv"
    )))?;

    // Set up polynomial coefficients: 1.0 + 2.0*x + 3.0*x^2 + 4.0*x^3
    let coefficients = [1.0f32, 2.0, 3.0, 4.0];
    let coefficients_dev = stream.clone_htod(&coefficients)?;

    // Load the kernel function
    let polynomial_kernel = module.load_function("polynomial_kernel")?;

    // Prepare input data
    let input = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let n = input.len();

    // Copy input to device
    let input_dev = stream.clone_htod(&input)?;
    let mut output_dev = stream.alloc_zeros::<f32>(n)?;

    // Launch kernel
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        stream
            .launch_builder(&polynomial_kernel)
            .arg(&mut output_dev)
            .arg(&input_dev)
            .arg(&(n as i32))
            .arg(&coefficients_dev)
            .launch(cfg)
    }?;

    // Copy results back
    let output = stream.clone_dtoh(&output_dev)?;

    // Verify results
    println!("\nPolynomial evaluation (1.0 + 2.0*x + 3.0*x^2 + 4.0*x^3):");
    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        let expected = coefficients[0]
            + coefficients[1] * x
            + coefficients[2] * x * x
            + coefficients[3] * x * x * x;
        println!("  f({:.1}) = {:.1} (expected {:.1})", x, y, expected);
        assert!(
            (y - expected).abs() < 1e-4,
            "Mismatch at index {}: got {}, expected {}",
            i,
            y,
            expected
        );
    }

    println!("\nAll results match expected values!");

    Ok(())
}
