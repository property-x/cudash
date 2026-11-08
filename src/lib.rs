//! cudash is heavily inspired by cudarc's basic driver API,
//! implemented on Vulkan, so it runs without CUDA.
//!
//! [`driver`] mirrors `cudarc::driver`. The `cudarc` feature (on by default) enables `interop`,
//! which works with real cudarc on NVIDIA hardware. Build with `--no-default-features` to drop it.

pub mod driver;

/// Shares cudash memory with real cudarc on NVIDIA GPUs.
#[cfg(feature = "cudarc")]
pub mod interop;

/// The ash version cudash's Vulkan handles (`vk_device`, `vk_buffer`, ...) come from.
pub use ash;
