//! Does this machine have AMD user-mode queues (doorbell submission without a submit ioctl)?
//! Needs the kernel (`amdgpu.user_queue`, `userq_ip_mask`) and Mesa (`AMD_USERQ=1`). Reports what it finds,
//! and fails only if the kernel offers queues and RADV can't run a kernel with them on.
//! Needs `vulkaninfo`. Skips on a non-AMD GPU.
//!
//!     cargo test --test amd_user_queues -- --nocapture

use cudash::driver::*;
use std::process::Command;

const AXPY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/double_all.spv"));

/// `userq_ip_mask` from RADV's `RADV_DEBUG=info`: bit per IP (gfx, compute, sdma) the kernel can give user queues.
fn userq_ip_mask() -> Option<u32> {
    let out = Command::new("vulkaninfo")
        .arg("--summary")
        .env("RADV_DEBUG", "info")
        .output()
        .ok()?;
    let text =
        String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout);
    let rest = text.split("userq_ip_mask =").nth(1)?;
    u32::from_str_radix(rest.split_whitespace().next()?, 16).ok()
}

#[test]
fn user_mode_queues_are_available_and_work() {
    let Some(mask) = userq_ip_mask() else {
        eprintln!("skipped: no RADV `userq_ip_mask` (not AMD, or no vulkaninfo)");
        return;
    };
    let param = std::fs::read_to_string("/sys/module/amdgpu/parameters/user_queue")
        .map(|s| s.trim().to_string());
    eprintln!("kernel amdgpu.user_queue = {param:?} (-1 auto, 0 off, 1 on)");
    eprintln!(
        "kernel userq_ip_mask     = {mask:#x} -> user queues: {}",
        if mask != 0 { "YES" } else { "NO" }
    );
    if mask == 0 {
        eprintln!(
            "RADV will use kernel queues (CS ioctl). Try booting with amdgpu.user_queue=1 on kernel >= 6.16."
        );
        return;
    }
    // Kernel has them: a launch in a child with AMD_USERQ=1 must still work.
    if std::env::var("AMD_USERQ").as_deref() != Ok("1") {
        let me = std::env::current_exe().unwrap();
        let status = Command::new(me)
            .args([
                "--nocapture",
                "--exact",
                "user_mode_queues_are_available_and_work",
            ])
            .env("AMD_USERQ", "1")
            .status()
            .unwrap();
        assert!(status.success(), "launch with AMD_USERQ=1 failed");
        return;
    }
    let ctx = GpuContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut data = stream.clone_htod(&[1.0f32, 2.0, 3.0]).unwrap();
    let f = ctx
        .load_module(Spirv::from_binary(AXPY.to_vec()))
        .unwrap()
        .load_function("double_all")
        .unwrap();
    let mut b = stream.launch_builder(&f);
    b.arg(&mut data).arg(&3u32);
    unsafe { b.launch(LaunchConfig::for_num_elems(3)) }.unwrap();
    assert_eq!(stream.clone_dtoh(&data).unwrap(), [2.0, 4.0, 6.0]);
}
