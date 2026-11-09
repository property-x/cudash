//! Helpers shared by the GPU test files. Not a test target itself.
#![allow(dead_code)]

use std::{fmt::Display, sync::Arc};

use cudash::{
    driver,
    interop::{
        cuda_context_for,
        cudarc::{
            driver::{CudaContext, CudaFunction},
            nvrtc::compile_ptx,
        },
    },
};

/// Compiles `src` with NVRTC and returns the kernel `name` from it.
pub fn load_kernel(
    ctx: &Arc<CudaContext>,
    src: &str,
    name: &str,
) -> cudash::interop::Result<CudaFunction> {
    let ptx = compile_ptx(src).expect("NVRTC failed to compile the test kernel");
    let module = ctx.load_module(ptx)?;
    Ok(module.load_function(name)?)
}

/// Reports that a GPU test didn't run.
///
/// Prints `SKIPPED` and lets the test pass, unless `CUDASH_REQUIRE_NVIDIA` is set, in
/// which case it panics. Set it on any machine that's meant to have an NVIDIA GPU,
/// so a missing GPU fails the run instead of passing silently.
pub fn skip(reason: impl Display) {
    if std::env::var_os("CUDASH_REQUIRE_NVIDIA").is_some() {
        panic!("CUDASH_REQUIRE_NVIDIA is set but the GPU test can't run: {reason}");
    }
    eprintln!("SKIPPED: {reason}");
}

/// A cudash context and its default stream, and the CUDA context on the same GPU. cudash
/// slices come from `stream`, and `as_cuda` lends them to cudarc.
pub struct Nvidia {
    pub ctx: Arc<driver::CudaContext>,
    pub stream: Arc<driver::CudaStream>,
    pub cuda: Arc<CudaContext>,
}

/// The NVIDIA GPU's contexts, or `None` after [`skip`]. Doesn't touch CUDA on other GPUs.
pub fn nvidia() -> Option<Nvidia> {
    let ctx = driver::CudaContext::new(0).inspect_err(|e| skip(e)).ok()?;
    let cuda = cuda_context_for(&ctx).inspect_err(|e| skip(e)).ok()?;
    let stream = ctx.default_stream();
    Some(Nvidia { ctx, stream, cuda })
}

/// `false` (after [`skip`]) when there's no NVIDIA GPU with CUDA.
pub fn have_nvidia() -> bool {
    nvidia().is_some()
}

/// cudarc's `examples/sin.cu`.
pub const SIN_CU: &str = r#"
extern "C" __global__ void sin_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}
"#;

/// `y[i] = a * x[i] + y[i]`, with a float scalar argument.
pub const AXPY_CU: &str = r#"
extern "C" __global__ void axpy(float *y, const float *x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
"#;

/// `out[i] = i`, so every element shows which thread wrote it.
pub const FILL_INDEX_CU: &str = r#"
extern "C" __global__ void fill_index(unsigned long long *out, int n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}
"#;

/// `data[i] *= 2`, for launching on a view of a larger buffer.
pub const DOUBLE_CU: &str = r#"
extern "C" __global__ void double_in_place(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= 2.0f;
    }
}
"#;

/// `data[i] += k`, with a 64-bit signed scalar argument.
pub const ADD_I64_CU: &str = r#"
extern "C" __global__ void add_i64(long long *data, long long k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] += k;
    }
}
"#;
