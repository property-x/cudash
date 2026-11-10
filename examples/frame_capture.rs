//! Processes one 2560x2160 `f32` frame (the bench size) with a 3x3 box blur, copies the input and
//! the result into images so RenderDoc's texture viewer shows both, and marks it all as a frame
//! for RenderDoc. A compute-only program never presents, so RenderDoc can't find the frame
//! by itself: this calls its in-application API through the `renderdoc` crate.
//!
//! Run: `just renderdoc-example frame_capture`, then open the `.rdc` it prints in `qrenderdoc`. Without RenderDoc it runs the same and captures nothing.

use cudash::driver::*;
use renderdoc::{RenderDoc, V110};

const KERNEL: &str = concat!(env!("OUT_DIR"), "/frame.spv");
const WIDTH: usize = 2560;
const HEIGHT: usize = 2160;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Spirv::from_file(KERNEL))?;
    let f = module.load_function("computeMain")?;

    // A checkerboard of 8-pixel squares in 0 and 1, so the blur's softened edges show up.
    let host: Vec<f32> = (0..WIDTH * HEIGHT)
        .map(|i| ((i % WIDTH / 8 + i / WIDTH / 8) % 2) as f32)
        .collect();
    let mut renderdoc = RenderDoc::<V110>::new().ok();
    let device = ctx.renderdoc_device_pointer();
    let captures = renderdoc.as_ref().map_or(0, |rd| rd.get_num_captures());
    if let Some(rd) = &mut renderdoc {
        rd.start_frame_capture(device, std::ptr::null());
    }

    let src = stream.clone_htod(&host)?;
    let mut dst = stream.alloc_zeros::<f32>(WIDTH * HEIGHT)?;
    let (width, height) = (WIDTH as i32, HEIGHT as i32);
    let cfg = LaunchConfig {
        grid_dim: ((WIDTH / 16) as u32, (HEIGHT / 16) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&f);
    builder.arg(&src).arg(&mut dst).arg(&width).arg(&height);
    unsafe { builder.launch(cfg) }?;
    // Kept until the end of the capture, so RenderDoc lists them.
    let _images = (
        src.as_image(WIDTH as u32, HEIGHT as u32)?,
        dst.as_image(WIDTH as u32, HEIGHT as u32)?,
    );
    let out = stream.clone_dtoh(&dst)?;

    if let Some(rd) = &mut renderdoc {
        rd.end_frame_capture(device, std::ptr::null());
        assert_eq!(
            rd.get_num_captures(),
            captures + 1,
            "RenderDoc didn't save the capture"
        );
    }
    // Away from a square's edge, the 3x3 average is the square's own value.
    let (x, y) = (101, 203);
    assert_eq!(out[y * WIDTH + x], host[y * WIDTH + x]);
    // On an edge it's a mix: 3 of the 9 neighbours of (104, 203) are in the square to the left.
    assert_eq!(
        out[203 * WIDTH + 104],
        (6.0 * host[203 * WIDTH + 104] + 3.0 * host[203 * WIDTH + 103]) / 9.0
    );
    let captured = if renderdoc.is_some() {
        "captured"
    } else {
        "not under RenderDoc, nothing captured"
    };
    println!(
        "ok: blurred a {WIDTH}x{HEIGHT} frame on {}, {captured}",
        ctx.name()?
    );
    Ok(())
}
