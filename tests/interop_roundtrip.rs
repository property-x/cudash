//! `SharedSemaphore` and `AsCuda`, with hand-written Vulkan submissions. These tests need an NVIDIA GPU
//! with CUDA, and print SKIPPED (or panic, with CUDASH_REQUIRE_NVIDIA set) on any other machine.

mod common;

use ash::vk;
use common::{Nvidia, nvidia};
use cudash::interop::{AsCuda, SharedSemaphore};

const TIMEOUT_NS: u64 = 5_000_000_000;

/// The queue the test submits on: the compute queue family cudash picks (the first with
/// COMPUTE), queue 0. Not exposed by `driver::CudaContext`.
struct Vk {
    gpu: Nvidia,
    queue: vk::Queue,
    queue_family: u32,
}

fn vulkan() -> Option<Vk> {
    let gpu = nvidia()?;
    let queue_family = unsafe {
        gpu.ctx
            .vk_instance()
            .get_physical_device_queue_family_properties(gpu.ctx.vk_physical_device())
    }
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .expect("cudash needs a compute queue") as u32;
    let queue = unsafe { gpu.ctx.vk_device().get_device_queue(queue_family, 0) };
    Some(Vk {
        gpu,
        queue,
        queue_family,
    })
}

/// Records and submits one command buffer, waiting on and signaling `semaphore`.
fn submit(
    vk: &Vk,
    semaphore: &SharedSemaphore,
    wait: u64,
    record: impl FnOnce(vk::CommandBuffer),
) -> u64 {
    let signal = semaphore.next_value();
    let device = vk.gpu.ctx.vk_device();
    unsafe {
        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(vk.queue_family);
        let pool = device.create_command_pool(&pool_info, None).unwrap();
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&allocate).unwrap()[0];
        device
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        record(command_buffer);
        device.end_command_buffer(command_buffer).unwrap();

        let semaphores = [semaphore.vk_semaphore()];
        let wait_values = [wait];
        let signal_values = [signal];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let command_buffers = [command_buffer];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let info = vk::SubmitInfo::default()
            .wait_semaphores(&semaphores)
            .wait_dst_stage_mask(&stages)
            .signal_semaphores(&semaphores)
            .command_buffers(&command_buffers)
            .push_next(&mut timeline);
        device
            .queue_submit(vk.queue, &[info], vk::Fence::null())
            .unwrap();
        assert!(
            semaphore.vk_wait(signal, TIMEOUT_NS).unwrap(),
            "Vulkan work timed out"
        );
        device.destroy_command_pool(pool, None);
    }
    signal
}

/// Copies `size` bytes between two cudash slices' Vulkan buffers.
fn copy_buffer(vk: &Vk, cb: vk::CommandBuffer, src: vk::Buffer, dst: vk::Buffer, size: u64) {
    let region = vk::BufferCopy::default().size(size);
    unsafe {
        vk.gpu
            .ctx
            .vk_device()
            .cmd_copy_buffer(cb, src, dst, &[region])
    };
}

#[test]
fn cuda_writes_vulkan_reads() {
    let Some(vk) = vulkan() else {
        return;
    };
    let stream = &vk.gpu.stream;
    let mut shared = stream.alloc_zeros::<u32>(1024).unwrap();
    let out = stream.alloc_zeros::<u32>(1024).unwrap();
    let semaphore = SharedSemaphore::new(&vk.gpu.ctx).unwrap();

    let expected: Vec<u32> = (0..1024).collect();
    let ready = {
        let mut lent = shared.as_cuda_mut().unwrap();
        let cuda = lent.stream().clone();
        cuda.memcpy_htod(&expected, &mut lent).unwrap();
        semaphore.cuda_signal(&cuda).unwrap()
    };

    submit(&vk, &semaphore, ready, |cb| {
        copy_buffer(&vk, cb, shared.vk_buffer(), out.vk_buffer(), 4 * 1024)
    });

    assert_eq!(stream.clone_dtoh(&out).unwrap(), expected);
}

#[test]
fn vulkan_writes_cuda_reads() {
    let Some(vk) = vulkan() else {
        return;
    };
    let device = vk.gpu.ctx.vk_device();
    let stream = &vk.gpu.stream;
    let shared = stream.alloc_zeros::<u32>(1024).unwrap();
    let semaphore = SharedSemaphore::new(&vk.gpu.ctx).unwrap();

    let ready = submit(&vk, &semaphore, 0, |cb| unsafe {
        device.cmd_fill_buffer(cb, shared.vk_buffer(), 0, 4 * 1024, 0xC0DA_F00D);
    });

    let lent = shared.as_cuda().unwrap();
    semaphore.cuda_wait(lent.stream(), ready).unwrap();
    assert_eq!(
        lent.stream().clone_dtoh(&lent).unwrap(),
        vec![0xC0DA_F00D_u32; 1024]
    );
}
