//! Steps for the Slang feature: the SPIR-V builds `build.rs` writes to `OUT_DIR`. The PTX steps need
//! NVIDIA and aren't written yet.

use std::sync::Arc;

use cucumber::{given, when};
use cudash::driver::{self, LaunchConfig, PushKernelArg, Spirv};

use super::world::World;

/// The struct of `write_struct.slang`, field for field.
#[repr(C)]
#[derive(Clone, Copy)]
struct MyCoolStruct {
    a: f32,
    b: f64,
    c: u32,
    d: usize,
}

// SAFETY: #[repr(C)] and plain data, matching write_struct.slang.
unsafe impl driver::DeviceRepr for MyCoolStruct {}

#[given(regex = r#"^the Slang SPIR-V build of "(\w+)" as "(\w+)" with entry point "(\w+)"$"#)]
fn given_slang_spirv(w: &mut World, file: String, name: String, entry: String) {
    let path = std::path::Path::new(env!("OUT_DIR")).join(format!("{file}.spv"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let function = w
        .ctx()
        .load_module(Spirv::from_binary(bytes))
        .and_then(|module| module.load_function(&entry));
    if let Some(function) = w.record(function) {
        w.kernels.insert(name, Arc::new(function));
    }
}

#[when(
    regex = r#"^I launch "(\w+)" on "(\w+)" with a struct of a float (\S+), a double (\S+), a uint (\d+) and a size_t (\d+)$"#
)]
fn when_launch_struct(
    w: &mut World,
    kernel: String,
    buffer: String,
    a: f32,
    b: f64,
    c: u32,
    d: usize,
) {
    let stream = w.default_stream();
    let function = w.kernels[&kernel].clone();
    let thing = MyCoolStruct { a, b, c, d };
    let out = match w.dbufs.get_mut(&buffer) {
        Some(super::world::DBuf::F64(s)) => s,
        _ => panic!("{buffer:?} must be an f64 driver buffer"),
    };
    let mut builder = stream.launch_builder(&function);
    builder.arg(out).arg(&thing);
    let outcome = unsafe { builder.launch(LaunchConfig::for_num_elems(1)) };
    let _ = w.record(outcome);
}
