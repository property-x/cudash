//! Steps for `driver_extensions.feature`: modules whose SPIR-V is written by hand, so a scenario
//! can declare any capability without a compiler that emits it.

use cucumber::when;
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
