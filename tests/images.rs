//! `CudaImage` from outside the crate: every format, size limits, drops and cross-stream sources.
//! Not a feature file, because reading an image back needs hand-written Vulkan (`read_back`).
//!
//! `read_back` submits to cudash's own queue (the first compute family, queue 0) without cudash's
//! lock, so it only runs when the context is idle and nothing else is running.

use std::sync::Arc;

use cudash::{
    ash::vk,
    driver::{
        CudaContext, CudaImage, CudaStream, DriverError, ImageFormat, LaunchConfig, PushKernelArg,
        Spirv, ValidAsZeroBits,
    },
};

const DOUBLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/double_all.spv"));

/// Copies the image into a slice with `vkCmdCopyImageToBuffer` and reads that.
fn read_back<T: ImageFormat + ValidAsZeroBits + Copy>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    image: &CudaImage<T>,
) -> Vec<T> {
    ctx.synchronize().unwrap();
    let out = stream
        .alloc_zeros::<T>((image.width() * image.height()) as usize)
        .unwrap();
    stream.synchronize().unwrap();
    let device = ctx.vk_device();
    let family = unsafe {
        ctx.vk_instance()
            .get_physical_device_queue_family_properties(ctx.vk_physical_device())
    }
    .iter()
    .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
    .unwrap() as u32;
    unsafe {
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )
            .unwrap();
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .command_buffer_count(1);
        let cmd = device.allocate_command_buffers(&info).unwrap()[0];
        device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: image.width(),
                height: image.height(),
                depth: 1,
            });
        device.cmd_copy_image_to_buffer(
            cmd,
            image.vk_image(),
            vk::ImageLayout::GENERAL,
            out.vk_buffer(),
            &[region],
        );
        device.end_command_buffer(cmd).unwrap();
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        let queue = device.get_device_queue(family, 0);
        let cmds = [cmd];
        device
            .queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                fence,
            )
            .unwrap();
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
    }
    stream.clone_dtoh(&out).unwrap()
}

fn setup() -> (Arc<CudaContext>, Arc<CudaStream>) {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    (ctx, stream)
}

macro_rules! format_roundtrip {
    ($($name:ident: $t:ty = |$i:ident| $value:expr, $format:ident;)*) => {$(
        #[test]
        fn $name() {
            let (ctx, stream) = setup();
            let data: Vec<$t> = (0..7 * 5u32).map(|$i| $value).collect();
            let slice = stream.clone_htod(&data).unwrap();
            let image = match stream.clone_to_image(&slice, 7, 5) {
                Ok(image) => image,
                Err(DriverError::InvalidImage(why)) => return eprintln!("SKIPPED: {why}"),
                Err(e) => panic!("{e}"),
            };
            assert_eq!(image.vk_format(), vk::Format::$format);
            assert_eq!(read_back(&ctx, &stream, &image), data, "rows are row-major, unpadded");
        }
    )*};
}

format_roundtrip! {
    f32_image: f32 = |i| i as f32 * 0.5, R32_SFLOAT;
    f32x2_image: [f32; 2] = |i| [i as f32, -(i as f32)], R32G32_SFLOAT;
    f32x4_image: [f32; 4] = |i| [i as f32, 1.0, 2.0, 3.5], R32G32B32A32_SFLOAT;
    u8_image: u8 = |i| i as u8 * 7, R8_UNORM;
    u8x2_image: [u8; 2] = |i| [i as u8, 255 - i as u8], R8G8_UNORM;
    u8x4_image: [u8; 4] = |i| [i as u8, 1, 2, 255], R8G8B8A8_UNORM;
    u16_image: u16 = |i| i as u16 * 1000, R16_UNORM;
    u32_image: u32 = |i| i * 123_457, R32_UINT;
    i32_image: i32 = |i| i as i32 - 20, R32_SINT;
}

#[test]
fn sides_must_be_between_one_and_the_device_limit() {
    let (ctx, stream) = setup();
    let max = 1 << 20;
    let limit = unsafe {
        ctx.vk_instance()
            .get_physical_device_properties(ctx.vk_physical_device())
    }
    .limits
    .max_image_dimension2_d;
    assert!(limit < max);
    for (w, h) in [(limit, 1), (1, limit)] {
        let data = stream.clone_htod(&vec![1.0f32; (w * h) as usize]).unwrap();
        let image = stream.clone_to_image(&data, w, h).unwrap();
        assert!(
            read_back(&ctx, &stream, &image).iter().all(|&x| x == 1.0),
            "{w}x{h}"
        );
    }
    let data = stream.alloc_zeros::<f32>(limit as usize + 1).unwrap();
    for (w, h) in [(limit + 1, 1), (1, limit + 1), (0, 1), (1, 0)] {
        let r = stream.clone_to_image(&data, w, h);
        assert!(matches!(r, Err(DriverError::InvalidImage(_))), "{w}x{h}");
    }
}

#[test]
fn dropping_an_image_while_its_copy_is_in_flight_is_fine() {
    let (ctx, stream) = setup();
    let slice = stream.clone_htod(&vec![3.0f32; 2048 * 2048]).unwrap();
    for _ in 0..20 {
        drop(stream.clone_to_image(&slice, 2048, 2048).unwrap());
    }
    drop(slice);
    ctx.synchronize().unwrap();
}

#[test]
fn a_kernel_on_another_stream_finishes_before_the_copy() {
    let (ctx, stream) = setup();
    let other = ctx.new_stream().unwrap();
    let module = ctx.load_module(Spirv::from_binary(DOUBLE)).unwrap();
    let double = module.load_function("double_all").unwrap();
    let n = 1 << 20;
    let mut slice = other.clone_htod(&vec![1.0f32; n]).unwrap();
    for _ in 0..4 {
        let n_arg = n as i32;
        let mut builder = other.launch_builder(&double);
        builder.arg(&mut slice).arg(&n_arg);
        unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.unwrap();
    }
    let image = stream.clone_to_image(&slice, 1024, 1024).unwrap();
    assert!(read_back(&ctx, &stream, &image).iter().all(|&x| x == 16.0));
}

#[test]
fn a_write_after_the_copy_does_not_change_the_image() {
    let (ctx, stream) = setup();
    let other = ctx.new_stream().unwrap();
    let mut slice = stream.clone_htod(&vec![5.0f32; 1 << 20]).unwrap();
    let image = stream.clone_to_image(&slice, 1024, 1024).unwrap();
    other.memset_zeros(&mut slice).unwrap();
    assert!(read_back(&ctx, &stream, &image).iter().all(|&x| x == 5.0));
}
