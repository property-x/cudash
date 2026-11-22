//! Step definitions for the feature files, and the filter that decides which scenarios run.

pub mod driver;
pub mod extensions;
pub mod graph;
pub mod interop;
pub mod slang;
pub mod world;

use std::sync::OnceLock;

use cucumber::gherkin;

pub use world::World;

fn nvidia_present() -> bool {
    static PRESENT: OnceLock<bool> = OnceLock::new();
    *PRESENT
        .get_or_init(|| cudash::driver::CudaContext::new(0).is_ok_and(|ctx| ctx.as_cuda().is_ok()))
}

fn non_nvidia_present() -> bool {
    static PRESENT: OnceLock<bool> = OnceLock::new();
    *PRESENT.get_or_init(|| {
        let count = cudash::driver::CudaContext::device_count().unwrap_or(0);
        (0..count as usize)
            .filter_map(|ordinal| cudash::driver::CudaContext::new(ordinal).ok())
            .any(|ctx| {
                let props = unsafe {
                    ctx.vk_instance()
                        .get_physical_device_properties(ctx.vk_physical_device())
                };
                props.vendor_id != 0x10de
            })
    })
}

/// Whether the first GPU reports an optional shader feature, by the short names the
/// `@gpu-has-<name>` and `@gpu-lacks-<name>` tags use.
fn gpu_has(name: &str) -> bool {
    use cudash::ash::vk;
    let ctx = cudash::driver::CudaContext::new(0).expect("a Vulkan 1.3 GPU");
    let (mut atomic, mut atomic2, mut clock, mut rotate, mut quad, mut layout, mut matrix) = (
        vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default(),
        vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT::default(),
        vk::PhysicalDeviceShaderClockFeaturesKHR::default(),
        vk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR::default(),
        vk::PhysicalDeviceShaderQuadControlFeaturesKHR::default(),
        vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default(),
        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default(),
    );
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut atomic)
        .push_next(&mut atomic2)
        .push_next(&mut clock)
        .push_next(&mut rotate)
        .push_next(&mut quad)
        .push_next(&mut layout)
        .push_next(&mut matrix);
    // Structs of extensions the driver lacks stay zeroed, which reads as "lacks".
    unsafe {
        ctx.vk_instance()
            .get_physical_device_features2(ctx.vk_physical_device(), &mut features)
    };
    let on = |b: vk::Bool32| b == vk::TRUE;
    match name {
        "float32-atomic-add" => on(atomic.shader_buffer_float32_atomic_add),
        "float64-atomic-add" => on(atomic.shader_buffer_float64_atomic_add),
        "float16-atomic-add" => on(atomic2.shader_buffer_float16_atomic_add),
        "float32-atomic-min-max" => on(atomic2.shader_buffer_float32_atomic_min_max),
        "shader-clock" => on(clock.shader_device_clock),
        "subgroup-rotate" => on(rotate.shader_subgroup_rotate),
        "quad-control" => on(quad.shader_quad_control),
        "explicit-layout" => on(layout.workgroup_memory_explicit_layout),
        "cooperative-matrix" => on(matrix.cooperative_matrix),
        other => panic!("unknown GPU feature {other:?} in a tag"),
    }
}

/// Decides whether a scenario runs here. A skipped scenario prints why.
pub fn selected(
    feature: &gherkin::Feature,
    rule: Option<&gherkin::Rule>,
    scenario: &gherkin::Scenario,
) -> bool {
    let mut tags: Vec<&str> = feature.tags.iter().map(String::as_str).collect();
    if let Some(rule) = rule {
        tags.extend(rule.tags.iter().map(String::as_str));
    }
    tags.extend(scenario.tags.iter().map(String::as_str));
    let has = |tag: &str| tags.iter().any(|t| t.trim_start_matches('@') == tag);
    let gpu_tag = |prefix: &str| {
        tags.iter()
            .filter_map(|t| t.trim_start_matches('@').strip_prefix(prefix))
            .next()
    };

    // Without CUDASH_REQUIRE_NVIDIA, a missing NVIDIA GPU skips. With it, the scenario runs
    // and fails, so a machine that should have a GPU can't pass silently.
    let require_nvidia = std::env::var_os("CUDASH_REQUIRE_NVIDIA").is_some();

    let skip = if has("planned") {
        Some("designed but not implemented")
    } else if has("open-question") {
        Some("open question")
    } else if has("known-gap") {
        Some("known gap: see the comment and todo/ item for this scenario")
    } else if has("two-gpus") {
        Some("needs two NVIDIA GPUs")
    } else if has("no-vulkan") {
        Some("needs a machine without a Vulkan loader")
    } else if gpu_tag("gpu-has-").is_some_and(|f| !gpu_has(f)) {
        Some("needs a GPU feature this one lacks")
    } else if gpu_tag("gpu-lacks-").is_some_and(gpu_has) {
        Some("needs a GPU without a feature this one has")
    } else if has("non-nvidia") && !non_nvidia_present() {
        Some("needs a non-NVIDIA Vulkan 1.3 GPU")
    } else if has("nvidia") && !require_nvidia && !nvidia_present() {
        Some("needs an NVIDIA GPU with CUDA")
    } else {
        None
    };

    match skip {
        None => true,
        Some(reason) => {
            eprintln!("skipped {:?}: {reason}", scenario.name);
            false
        }
    }
}
