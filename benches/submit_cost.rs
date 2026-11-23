//! Where the time of a launch goes, with raw `ash` and no cudash: one compute dispatch of an
//! empty kernel, N times in a row, on one queue with a timeline semaphore.
//!
//! Variants (all: pre-allocated command buffers, recording and submitting inside the timed
//! region, timed from the first record until a host wait on the last timeline value returns):
//! - `timeline wait`: one command buffer and submit per launch, waits on the previous value, signals the next.
//! - `barrier`: the same, but no semaphore wait; a global memory barrier starts each command buffer.
//! - `batched`: N dispatches in one command buffer with a global barrier between them, one submit.
//! - `nosync`: one submit per launch, no wait and no barrier. The submit-cost floor.
//!
//! ```text
//! just bench submit_cost
//! ```

use std::{ffi::CStr, time::Instant};

use ash::vk;

const KERNEL: &str = concat!(env!("OUT_DIR"), "/double_all.spv");
const NS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
const PASSES: usize = 5;
const VARIANTS: [&str; 4] = ["timeline wait", "barrier", "batched", "nosync"];

struct Ctx {
    device: ash::Device,
    queue: vk::Queue,
    timeline: vk::Semaphore,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    pool: vk::CommandPool,
    cbs: Vec<vk::CommandBuffer>,
    value: u64,
}

/// Microseconds spent recording, submitting and in the final wait.
#[derive(Clone, Copy, Default)]
struct Split {
    record: f64,
    submit: f64,
    wait: f64,
}

impl Ctx {
    fn record(&self, cb: vk::CommandBuffer, dispatches: usize, barrier_first: bool, between: bool) {
        let d = &self.device;
        let mem = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
        let dep = vk::DependencyInfo::default().memory_barriers(&mem);
        let push = [0u8; 16]; // null pointer and n = 0: the body never runs
        unsafe {
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .unwrap();
            d.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            if barrier_first {
                d.cmd_pipeline_barrier2(cb, &dep);
            }
            for i in 0..dispatches {
                if between && i > 0 {
                    d.cmd_pipeline_barrier2(cb, &dep);
                }
                d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
                d.cmd_dispatch(cb, 1, 1, 1);
            }
            d.end_command_buffer(cb).unwrap();
        }
    }

    fn submit(&self, cb: vk::CommandBuffer, wait: Option<u64>, signal: u64) {
        let info = |v| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline)
                .value(v)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        };
        let waits: Vec<_> = wait.map(info).into_iter().collect();
        let signals = [info(signal)];
        let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .signal_semaphore_infos(&signals)
            .command_buffer_infos(&cbs);
        unsafe {
            self.device
                .queue_submit2(self.queue, &[submit], vk::Fence::null())
                .unwrap()
        };
    }

    fn run(&mut self, variant: &str, n: usize) -> Split {
        let mut s = Split::default();
        let first = self.value;
        let batched = variant == "batched";
        let launches = if batched { 1 } else { n };
        for i in 0..launches {
            let cb = self.cbs[i];
            let t0 = Instant::now();
            match variant {
                "batched" => self.record(cb, n, false, true),
                "barrier" => self.record(cb, 1, true, false),
                _ => self.record(cb, 1, false, false),
            }
            let t1 = Instant::now();
            let wait = (variant == "timeline wait").then(|| first + i as u64);
            self.submit(
                cb,
                wait,
                if batched {
                    first + n as u64
                } else {
                    first + 1 + i as u64
                },
            );
            s.record += (t1 - t0).as_secs_f64();
            s.submit += t1.elapsed().as_secs_f64();
        }
        let last = first + n as u64;
        let t = Instant::now();
        let sems = [self.timeline];
        let values = [last];
        unsafe {
            self.device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&sems)
                        .values(&values),
                    u64::MAX,
                )
                .unwrap()
        };
        s.wait = t.elapsed().as_secs_f64();
        self.value = last;
        Split {
            record: s.record * 1e6,
            submit: s.submit * 1e6,
            wait: s.wait * 1e6,
        }
    }
}

fn main() {
    let entry = unsafe { ash::Entry::load().unwrap() };
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )
    }
    .unwrap();
    let pd = unsafe { instance.enumerate_physical_devices().unwrap() }[0];
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let family = unsafe { instance.get_physical_device_queue_family_properties(pd) }
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .unwrap() as u32;
    let prio = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&prio)];
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(true)
        .buffer_device_address(true)
        .vulkan_memory_model(true)
        .vulkan_memory_model_device_scope(true);
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
    let feats = vk::PhysicalDeviceFeatures::default().shader_int64(true);
    let device = unsafe {
        instance.create_device(
            pd,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_features(&feats)
                .push_next(&mut f12)
                .push_next(&mut f13),
            None,
        )
    }
    .unwrap();
    let queue = unsafe { device.get_device_queue(family, 0) };
    let mut tl = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let timeline = unsafe {
        device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tl), None)
    }
    .unwrap();

    let bytes = std::fs::read(KERNEL).unwrap();
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
    }
    .unwrap();
    let ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(16)];
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
            None,
        )
    }
    .unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"double_all");
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)],
            None,
        )
    }
    .unwrap()[0];
    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
    }
    .unwrap();
    let cbs = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(128),
        )
    }
    .unwrap();
    let mut ctx = Ctx {
        device,
        queue,
        timeline,
        pipeline,
        layout,
        pool,
        cbs,
        value: 0,
    };
    let _ = ctx.pool;

    println!(
        "device: {}",
        unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy()
    );
    println!(
        "driver version: {:#x}, passes: {PASSES}",
        props.driver_version
    );

    // totals[pass][n][variant] and splits, in µs, medians over reps.
    let mut totals = vec![vec![[0f64; 4]; NS.len()]; PASSES];
    let mut splits = vec![vec![[Split::default(); 4]; NS.len()]; PASSES];
    for pass in 0..PASSES {
        for (ni, &n) in NS.iter().enumerate() {
            let reps = (8000 / n).clamp(40, 2000);
            // Interleave variants rep by rep so a slow phase hits all of them.
            let mut t = vec![vec![]; 4];
            let mut sp = vec![vec![]; 4];
            for rep in 0..reps + 10 {
                for (vi, v) in VARIANTS.iter().enumerate() {
                    let start = Instant::now();
                    let s = ctx.run(v, n);
                    let total = start.elapsed().as_secs_f64() * 1e6;
                    if rep >= 10 {
                        t[vi].push(total);
                        sp[vi].push(s);
                    }
                }
            }
            for (vi, _) in VARIANTS.iter().enumerate() {
                totals[pass][ni][vi] = median(&mut t[vi]);
                let part =
                    |f: fn(&Split) -> f64| median(&mut sp[vi].iter().map(f).collect::<Vec<f64>>());
                splits[pass][ni][vi] = Split {
                    record: part(|s| s.record),
                    submit: part(|s| s.submit),
                    wait: part(|s| s.wait),
                };
            }
        }
    }

    println!("\nper-launch µs, median of passes [min-max over passes] (total µs)\n");
    print!("| N |");
    for v in VARIANTS {
        print!(" {v} |");
    }
    println!("\n|---|---|---|---|---|");
    for (ni, &n) in NS.iter().enumerate() {
        print!("| {n} |");
        for (vi, _) in VARIANTS.iter().enumerate() {
            let mut p: Vec<f64> = (0..PASSES).map(|p| totals[p][ni][vi] / n as f64).collect();
            p.sort_by(f64::total_cmp);
            print!(
                " {:.1} [{:.1}-{:.1}] ({:.0}) |",
                p[PASSES / 2],
                p[0],
                p[PASSES - 1],
                p[PASSES / 2] * n as f64
            );
        }
        println!();
    }
    println!(
        "\nsplit per launch µs (median over passes): record / submit / final wait (wait is per N, not per launch)\n"
    );
    print!("| N |");
    for v in VARIANTS {
        print!(" {v} |");
    }
    println!("\n|---|---|---|---|---|");
    for (ni, &n) in NS.iter().enumerate() {
        print!("| {n} |");
        for (vi, _) in VARIANTS.iter().enumerate() {
            let mut r: Vec<f64> = (0..PASSES).map(|p| splits[p][ni][vi].record).collect();
            let mut s: Vec<f64> = (0..PASSES).map(|p| splits[p][ni][vi].submit).collect();
            let mut w: Vec<f64> = (0..PASSES).map(|p| splits[p][ni][vi].wait).collect();
            print!(
                " {:.1} / {:.1} / {:.0} |",
                median(&mut r) / n as f64,
                median(&mut s) / n as f64,
                median(&mut w)
            );
        }
        println!();
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
