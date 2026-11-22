//! Steps for `driver_extensions.feature`: modules whose SPIR-V is written by hand, so a scenario
//! can declare any capability without a compiler that emits it.

use cucumber::{then, when};
use cudash::driver::Spirv;

use super::world::World;

const OP_CAPABILITY: u32 = 17;
const OP_MEMORY_MODEL: u32 = 14;

fn instruction(op: u32, operands: &[u32]) -> Vec<u32> {
    let mut words = vec![(operands.len() as u32 + 1) << 16 | op];
    words.extend_from_slice(operands);
    words
}

/// A module with a SPIR-V 1.6 header and these instructions, which has no function bodies.
fn module(instructions: &[Vec<u32>]) -> Spirv {
    let mut words = vec![0x0723_0203, 0x0001_0600, 0, 1, 0];
    words.extend(instructions.iter().flatten());
    Spirv::from_binary(
        words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
}

fn capabilities(list: &str) -> Vec<Vec<u32>> {
    list.replace(" and ", ", ")
        .split(", ")
        .map(|n| {
            instruction(
                OP_CAPABILITY,
                &[n.trim().parse().expect("a capability number")],
            )
        })
        .collect()
}

/// Loads `spirv`, recording the error. A panic is recorded as `Panicked`, so a scenario can
/// expect an error and fail on a crash.
fn load(w: &mut World, spirv: Spirv) {
    let ctx = w.ctx().clone();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.load_module(spirv)
            .map(|_| ())
            .map_err(|e| format!("{e:?}"))
    }))
    .unwrap_or_else(|_| Err("Panicked".to_owned()));
    let _ = w.record(outcome);
}

#[when(regex = r"^I load a module declaring the capabilities (\d+(?:(?:, | and )\d+)*)$")]
fn when_load_capabilities(w: &mut World, list: String) {
    load(w, module(&capabilities(&list)));
}

#[when(
    regex = r"^I load a module declaring Shader, then its memory model, then the capability (\d+)$"
)]
fn when_load_late_capability(w: &mut World, number: u32) {
    let mut body = capabilities("1");
    body.push(instruction(OP_MEMORY_MODEL, &[0, 1]));
    body.extend(capabilities(&number.to_string()));
    load(w, module(&body));
}

#[when(regex = r"^I load a module whose OpCapability has no operand$")]
fn when_load_empty_capability(w: &mut World) {
    load(w, module(&[instruction(OP_CAPABILITY, &[])]));
}

#[when(regex = r#"^I load the Slang SPIR-V build of "(\w+)"$"#)]
fn when_load_slang(w: &mut World, file: String) {
    let path = std::path::Path::new(env!("OUT_DIR")).join(format!("{file}.spv"));
    load(w, Spirv::from_file(path));
}

// ---- What the context says it enabled ----------------------------------------------------

/// `(extension, extension it needs)`, as `requires` in `src/driver/extensions.rs` plus cudash's
/// own. Copied, because the library keeps its table private.
const REQUIRES: &[(&str, &str)] = &[
    ("VK_EXT_shader_atomic_float2", "VK_EXT_shader_atomic_float"),
    (
        "VK_KHR_shader_quad_control",
        "VK_KHR_shader_maximal_reconvergence",
    ),
    ("VK_NV_cooperative_matrix2", "VK_KHR_cooperative_matrix"),
    (
        "VK_EXT_external_memory_dma_buf",
        "VK_KHR_external_memory_fd",
    ),
];

fn reported(w: &World) -> Vec<String> {
    let ctx = w.ctx();
    unsafe {
        ctx.vk_instance()
            .enumerate_device_extension_properties(ctx.vk_physical_device())
    }
    .expect("listing the GPU's extensions")
    .iter()
    .map(|e| {
        e.extension_name_as_c_str()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    })
    .collect()
}

fn enabled(w: &World) -> Vec<String> {
    w.ctx()
        .extensions()
        .iter()
        .map(|e| e.to_string_lossy().into_owned())
        .collect()
}

#[then("every enabled extension is one the GPU reports, once")]
fn then_enabled_subset(w: &mut World) {
    let (have, on) = (reported(w), enabled(w));
    for e in &on {
        assert!(
            have.contains(e),
            "{e} is enabled but the GPU doesn't report it"
        );
        assert_eq!(
            on.iter().filter(|o| *o == e).count(),
            1,
            "{e} is listed twice"
        );
    }
}

#[then("every optional extension the GPU reports with its requirements is enabled")]
fn then_optional_enabled(w: &mut World) {
    let (have, on) = (reported(w), enabled(w));
    // A requirement can itself have one, so a chain is met only if each link is reported.
    let met = |mut e: &str| loop {
        if !have.iter().any(|h| h == e) {
            return false;
        }
        match REQUIRES.iter().find(|(x, _)| *x == e) {
            Some((_, r)) => e = r,
            None => return true,
        }
    };
    for e in cudash::driver::OPTIONAL_EXTENSIONS {
        let e = e.to_str().unwrap();
        assert_eq!(
            met(e),
            on.iter().any(|o| o == e),
            "{e}: met {} but enabled {}",
            met(e),
            on.iter().any(|o| o == e)
        );
    }
}

#[then("every enabled extension has its requirement enabled")]
fn then_requirements_enabled(w: &mut World) {
    let on = enabled(w);
    for (e, r) in REQUIRES {
        if on.iter().any(|o| o == e) {
            assert!(on.iter().any(|o| o == r), "{e} is enabled without {r}");
        }
    }
}

#[then("the feature tags agree with the enabled extensions")]
fn then_tags_agree(w: &mut World) {
    let on = enabled(w);
    for (feature, ext) in [
        ("float32-atomic-add", "VK_EXT_shader_atomic_float"),
        ("float16-atomic-add", "VK_EXT_shader_atomic_float2"),
        ("float32-atomic-min-max", "VK_EXT_shader_atomic_float2"),
        ("shader-clock", "VK_KHR_shader_clock"),
        ("subgroup-rotate", "VK_KHR_shader_subgroup_rotate"),
        ("quad-control", "VK_KHR_shader_quad_control"),
        ("explicit-layout", "VK_KHR_workgroup_memory_explicit_layout"),
        (
            "float16-vector-atomics",
            "VK_NV_shader_atomic_float16_vector",
        ),
        ("cooperative-matrix", "VK_KHR_cooperative_matrix"),
    ] {
        // Mesa fills structs of extensions it doesn't expose, so only "has" must imply enabled when
        // the extension is reported; a reported-and-wanted extension must be on either way.
        if super::gpu_has(feature) && reported(w).iter().any(|r| r == ext) {
            assert!(
                on.iter().any(|o| o == ext),
                "{feature} is on but {ext} isn't enabled"
            );
        }
    }
}

#[then("the caps example exits successfully")]
fn then_caps_example(_w: &mut World) {
    let out = std::process::Command::new(std::env::var("CARGO").unwrap_or("cargo".into()))
        .args(["run", "-q", "--example", "caps"])
        .output()
        .expect("running cargo");
    assert!(
        out.status.success(),
        "caps failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.stdout.is_empty(), "caps printed nothing");
}
