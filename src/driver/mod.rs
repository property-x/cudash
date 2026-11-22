//! cudarc's basic driver API, on Vulkan, with `Gpu*` type names (`GpuSlice` for `CudaSlice`). Changing
//! `use cudarc::driver::*` to `use cudash::cudarc_compat::*` is enough for code that only uses memory
//! and copies.
//!
//! ```no_run
//! use cudash::driver::*;
//!
//! # fn main() -> Result<(), DriverError> {
//! let ctx = GpuContext::new(0)?;
//! let stream = ctx.default_stream();
//! let a: GpuSlice<f32> = stream.clone_htod(&[1.0f32; 100])?;
//! let mut b = stream.alloc_zeros::<f32>(100)?;
//! stream.memcpy_dtod(&a, &mut b)?;
//! assert_eq!(stream.clone_dtoh(&b)?, vec![1.0; 100]);
//! # Ok(())
//! # }
//! ```
//!
//! Kernels are the one difference: they're SPIR-V, loaded with [`Spirv`] where cudarc
//! uses `Ptx`. Arguments arrive in push constants, in the order they're passed, with
//! slices as 64-bit GPU addresses. Everything else about launching matches cudarc.

mod context;
pub(crate) mod device;
mod error;
pub(crate) mod graph;
mod host;
mod image;
mod kernel;
mod slice;
pub(crate) mod stream;
pub mod sys;

pub use context::GpuContext;
pub use device::OPTIONAL_EXTENSIONS;
pub use error::DriverError;
pub use graph::GpuGraph;
pub use host::PinnedHostSlice;
pub use image::{GpuImage, ImageFormat};
pub use kernel::{
    GpuFunction, GpuModule, KernelStatistic, LaunchArgs, LaunchConfig, PushKernelArg, Spirv,
    StatisticValue,
};
pub use slice::{
    DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, GpuSlice, GpuView, GpuViewMut, HostSlice,
    RawRange, ValidAsZeroBits,
};
pub use stream::{GpuStream, SyncOnDrop};
