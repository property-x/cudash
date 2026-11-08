//! The low-level `Interop` API, with hand-written Vulkan submissions. These tests need an NVIDIA GPU
//! with CUDA, and print SKIPPED (or panic, with CUDASH_REQUIRE_NVIDIA set) on any other machine.

mod common;

use ash::vk;
use common::{Nvidia, nvidia};
use cudash::interop::{Error, SharedSemaphore};

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

#[test]
fn cuda_writes_vulkan_reads() {
    let Some(vk) = vulkan() else {
        return;
    };
    let device = vk.gpu.ctx.vk_device();
    let interop = &vk.gpu.interop;
    let usage = vk::BufferUsageFlags::TRANSFER_SRC;
    let mut shared = interop.create_buffer::<u32>(1024, usage).unwrap();
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let staging = interop
        .create_buffer_with::<u32>(1024, vk::BufferUsageFlags::TRANSFER_DST, host)
        .unwrap();
    let semaphore = interop.create_semaphore().unwrap();

    let expected: Vec<u32> = (0..1024).collect();
    interop
        .stream()
        .memcpy_htod(&expected, &mut shared)
        .unwrap();
    let ready = semaphore.cuda_signal(interop.stream()).unwrap();

    submit(&vk, &semaphore, ready, |cb| unsafe {
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[shared.vk_acquire_barrier(vk.queue_family)],
            &[],
        );
        let region = vk::BufferCopy::default().size(4 * 1024);
        device.cmd_copy_buffer(cb, shared.vk_buffer(), staging.vk_buffer(), &[region]);
        let to_host = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[to_host],
            &[],
            &[],
        );
    });

    unsafe {
        let flags = vk::MemoryMapFlags::empty();
        let mapped = device
            .map_memory(staging.vk_memory(), 0, vk::WHOLE_SIZE, flags)
            .unwrap();
        let actual = std::slice::from_raw_parts(mapped.cast::<u32>(), 1024);
        assert_eq!(actual, expected);
        device.unmap_memory(staging.vk_memory());
    }
}

#[test]
fn vulkan_writes_cuda_reads() {
    let Some(vk) = vulkan() else {
        return;
    };
    let device = vk.gpu.ctx.vk_device();
    let interop = &vk.gpu.interop;
    let shared = interop
        .create_buffer::<u32>(1024, vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    let semaphore = interop.create_semaphore().unwrap();

    let ready = submit(&vk, &semaphore, 0, |cb| unsafe {
        device.cmd_fill_buffer(cb, shared.vk_buffer(), 0, vk::WHOLE_SIZE, 0xC0DA_F00D);
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[shared.vk_release_barrier(vk.queue_family)],
            &[],
        );
    });

    semaphore.cuda_wait(interop.stream(), ready).unwrap();
    assert_eq!(
        interop.stream().clone_dtoh(&shared).unwrap(),
        vec![0xC0DA_F00D_u32; 1024]
    );
}

#[test]
fn new_buffers_are_zeroed() {
    let Some(gpu) = nvidia() else {
        return;
    };
    let interop = &gpu.interop;
    let shared = interop
        .create_buffer::<u64>(64, vk::BufferUsageFlags::TRANSFER_SRC)
        .unwrap();
    assert_eq!(interop.stream().clone_dtoh(&shared).unwrap(), vec![0; 64]);
}

#[test]
fn invalid_lengths_are_rejected() {
    let Some(gpu) = nvidia() else {
        return;
    };
    let interop = &gpu.interop;
    let usage = vk::BufferUsageFlags::TRANSFER_SRC;
    assert!(matches!(
        interop.create_buffer::<u32>(0, usage),
        Err(Error::InvalidLength)
    ));
    assert!(matches!(
        interop.create_buffer::<u32>(usize::MAX, usage),
        Err(Error::InvalidLength)
    ));
}
