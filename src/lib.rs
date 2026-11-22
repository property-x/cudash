//! cudash is heavily inspired by cudarc's basic driver API,
//! implemented on Vulkan, so it runs without CUDA.
//!
//! [`driver`] mirrors `cudarc::driver`, with `Gpu*` type names. [`cudarc_compat`] adds cudarc's
//! `Cuda*` names. The `cuda` feature (on by default) enables `interop`, which works with real
//! cudarc on NVIDIA hardware, and adds `cudarc_compat` to the [`prelude`]. Build with
//! `--no-default-features` to drop it.

mod compat;
pub mod driver;

/// The driver API under cudarc's names, so `use cudarc::driver::*` ports by changing it to
/// `use cudash::cudarc_compat::*`.
pub mod cudarc_compat {
    pub use crate::driver::{
        GpuContext as CudaContext, GpuFunction as CudaFunction, GpuGraph as CudaGraph,
        GpuModule as CudaModule, GpuSlice as CudaSlice, GpuStream as CudaStream,
        GpuView as CudaView, GpuViewMut as CudaViewMut, *,
    };
}

/// The driver API, plus cudarc's `Cuda*` names with the `cuda` feature.
pub mod prelude {
    #[cfg(feature = "cuda")]
    pub use crate::cudarc_compat::*;
    #[cfg(not(feature = "cuda"))]
    pub use crate::driver::*;
}

/// Shares cudash memory with real cudarc on NVIDIA GPUs.
#[cfg(feature = "cuda")]
pub mod interop;

/// The ash version cudash's Vulkan handles (`vk_device`, `vk_buffer`, ...) come from: the release,
/// or the git version with the `ash_latest` feature.
#[cfg(not(feature = "ash_latest"))]
pub use ash;
#[cfg(feature = "ash_latest")]
pub use ash_latest as ash;

/// The book's chapters in `docs/src`, so `cargo test` compiles and runs their examples.
#[cfg(doctest)]
mod book {
    macro_rules! chapters {
        ($($name:ident = $file:literal),* $(,)?) => {$(
            #[doc = include_str!(concat!("../docs/src/", $file))]
            struct $name;
        )*};
    }
    chapters!(
        Introduction = "introduction.md",
        GettingStarted = "getting-started.md",
        Memory = "memory.md",
        Streams = "streams.md",
        Kernels = "kernels.md",
        Graphs = "graphs.md",
        Images = "images.md",
        Vulkan = "vulkan.md",
        Interop = "interop.md",
        Differences = "differences.md",
    );
}
