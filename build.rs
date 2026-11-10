//! Compiles Slang kernels in `tests/kernels` and `examples/slang` into `OUT_DIR`:
//! `name.slang` becomes `name.spv` for cudash, and `name.ptx` for cudarc.
//!
//! Kernels need to be declared with `[shader("compute")]`, so `slangc` finds their entry points.
//! SPIR-V failures stop the build but PTX failure only warns.
//!
//! Env: `SLANGC` overrides the compiler. `NVRTC_PATH` overrides the NVRTC library.

use std::{path::Path, process::Command};

const KERNEL_DIRS: [&str; 2] = ["tests/kernels", "examples/slang"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SLANGC");
    println!("cargo:rerun-if-env-changed=NVRTC_PATH");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_KERNEL_DEBUG");
    let out_dir = std::env::var("OUT_DIR").expect("cargo sets OUT_DIR");
    let slangc = std::env::var("SLANGC").unwrap_or_else(|_| "slangc".into());

    // shortcut: these only matter to this repo's tests and examples, but cargo has no
    // dev-only build scripts. Exclude both directories when publishing, so this does nothing.
    for dir in KERNEL_DIRS {
        println!("cargo:rerun-if-changed={dir}");
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries {
            let src = entry.expect("read kernel directory").path();
            if src.extension().is_some_and(|e| e == "slang") {
                compile(&slangc, &src, Path::new(&out_dir));
            }
        }
    }
}

fn compile(slangc: &str, src: &Path, out_dir: &Path) {
    let stem = src.file_stem().expect("kernel has a file name");
    let spv = out_dir.join(stem).with_extension("spv");
    let ptx = out_dir.join(stem).with_extension("ptx");

    // Without -fvk-use-entrypoint-name, Slang renames every SPIR-V entry point to "main".
    // With the `kernel_debug` feature, -g (debug level 2) gives RenderDoc the Slang source and per-line info.
    // It embeds the kernel source in the .spv, so anyone who has the binary or the SPIR-V can read the Slang source.
    // Keep that in mind before shipping kernels built with the feature on.
    // PTX stays without -g: it adds .loc/.file lines, and we can't load PTX on NVIDIA here to check.
    let mut spirv_command = Command::new(slangc);
    spirv_command
        .arg(src)
        .args(["-target", "spirv", "-fvk-use-entrypoint-name"]);
    if std::env::var_os("CARGO_FEATURE_KERNEL_DEBUG").is_some() {
        spirv_command.arg("-g");
    }
    let spirv = spirv_command.arg("-o").arg(&spv).output();
    let spirv = match spirv {
        Ok(output) => output,
        Err(e) => {
            println!(
                "cargo:warning=can't run {slangc}: {e}. Skipping {}",
                src.display()
            );
            return;
        }
    };
    if !spirv.status.success() {
        panic!(
            "slangc failed on {}:\n{}",
            src.display(),
            String::from_utf8_lossy(&spirv.stderr)
        );
    }

    let mut command = Command::new(slangc);
    command.arg(src).args(["-target", "ptx", "-o"]).arg(&ptx);
    if let Ok(nvrtc) = std::env::var("NVRTC_PATH") {
        command.arg("-nvrtc-path").arg(nvrtc);
    }
    let output = command.output().expect("slangc ran a moment ago");
    if output.status.success() {
        strip_trailing_nul(&ptx);
    } else {
        let _ = std::fs::remove_file(&ptx);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let reason = stderr.lines().next().unwrap_or("");
        println!("cargo:warning=no PTX for {}: {reason}", src.display());
    }
}

/// Fix: Slang ends PTX with a \0 byte, causing panic in `Ptx::from_src`.
fn strip_trailing_nul(ptx: &Path) {
    let mut bytes = std::fs::read(ptx).expect("slangc wrote this file");
    if bytes.last() == Some(&0) {
        bytes.pop();
        std::fs::write(ptx, bytes).expect("rewrite PTX");
    }
}
