//! cudarc's `examples/cufile-copy.rs` (MIT OR Apache-2.0) on Vulkan. cuFile reads straight
//! into GPU memory, which cudash doesn't do, so the file goes through host memory.

use std::fs;

use cudash::driver::CudaContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 100000;
    let data: Vec<u8> = (0..N).flat_map(|x| (x as f32).to_le_bytes()).collect();
    let data_sz = data.len();
    let src_file = std::env::temp_dir().join("cudash_file_test.bin");
    fs::write(&src_file, &data)?;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let mut buf = stream.alloc_zeros::<u8>(data_sz)?;

    stream.memcpy_htod(&fs::read(&src_file)?, &mut buf)?;
    fs::remove_file(&src_file)?;

    let verify_dst = stream.clone_dtoh(&buf)?;
    assert_eq!(verify_dst, data);

    Ok(())
}
