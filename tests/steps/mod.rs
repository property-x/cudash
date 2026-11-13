//! Step definitions for the feature files, and the filter that decides which scenarios run.

pub mod driver;
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
