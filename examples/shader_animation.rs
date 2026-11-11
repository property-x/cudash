//! Renders a few frames of iced_shader's "shader-animation" as a Slang compute shader and writes
//! each as a PPM to `target/shader_animation/`. Frames are a half second apart in the widget's
//! clock, the pixel format is RGBA8, and the last frame is also shown as an image in RenderDoc.
//!
//! Run: `cargo run --example shader_animation`, or `just renderdoc-example shader_animation` to
//! capture all frames as one RenderDoc capture.

use cudash::driver::*;
use renderdoc::{RenderDoc, V110};

const KERNEL: &str = concat!(env!("OUT_DIR"), "/shader_animation.spv");
const WIDTH: usize = 960;
const HEIGHT: usize = 540;
const FRAMES: usize = 60;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;

    // attach RenderDoc before any Vulkan objects are created, so it can hook them all
    let mut renderdoc = RenderDoc::<V110>::new().ok();
    let device = ctx.renderdoc_device_pointer();
    let captures = renderdoc.as_ref().map_or(0, |rd| rd.get_num_captures());
    if let Some(rd) = &mut renderdoc {
        rd.start_frame_capture(device, std::ptr::null());
    }

    let stream = ctx.default_stream();
    let module = ctx.load_module(Spirv::from_file(KERNEL))?;
    let f = module.load_function("computeMain")?;

    let out_dir = std::path::Path::new("target/shader_animation");
    std::fs::create_dir_all(out_dir)?;

    let mut dst = stream.alloc_zeros::<[u8; 4]>(WIDTH * HEIGHT)?;
    let cfg = LaunchConfig {
        grid_dim: (WIDTH.div_ceil(16) as u32, HEIGHT.div_ceil(16) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    // iced_shader's defaults.
    let (width, height) = (WIDTH as i32, HEIGHT as i32);
    let (line_width, lines, ring_scale, mosaic, color_spread) =
        (0.002f32, 5i32, 5.0f32, 0.2f32, 0.01f32);

    for frame in 0..FRAMES {
        let _image_view = dst.as_image(WIDTH as u32, HEIGHT as u32);
        stream.frame()?;
        // The widget's `time` is 1.0 + seconds * 3 at speed 1.
        let time = 1.0 + frame as f32 * 0.5 * 3.0;
        let mut builder = stream.launch_builder(&f);
        builder
            .arg(&mut dst)
            .arg(&width)
            .arg(&height)
            .arg(&time)
            .arg(&line_width)
            .arg(&lines)
            .arg(&ring_scale)
            .arg(&mosaic)
            .arg(&color_spread);
        unsafe { builder.launch(cfg) }?;
        /*{
            let pixels = stream.clone_dtoh(&dst)?;
            assert!(pixels.iter().all(|p| p[3] == 255), "output is opaque");
            assert!(
                pixels.iter().any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0),
                "frame {frame} is black"
            );

            let mut ppm = format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
            ppm.extend(pixels.iter().flat_map(|p| [p[0], p[1], p[2]]));
            let path = out_dir.join(format!("frame_{frame}.ppm"));
            std::fs::write(&path, ppm)?;
            println!("t = {time:>4}: {}", path.display());
        }*/
        stream.stop_frame()?;
        stream.synchronize()?;
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    // Kept until the end of the capture, so RenderDoc lists it.
    let _image = dst.as_image(WIDTH as u32, HEIGHT as u32)?;
    if let Some(rd) = &mut renderdoc {
        rd.end_frame_capture(device, std::ptr::null());
        assert_eq!(
            rd.get_num_captures(),
            captures + 1,
            "RenderDoc didn't save the capture"
        );
    }
    println!("ok: {FRAMES} frames of {WIDTH}x{HEIGHT} on {}", ctx.name()?);
    Ok(())
}
