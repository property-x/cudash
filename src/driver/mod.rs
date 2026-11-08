//! cudarc's basic driver API, on Vulkan. Changing `use cudarc::driver::*` to
//! `use cudash::driver::*` is enough for code that only uses memory and copies.
//!
//! ```no_run
//! use cudash::driver::*;
//!
//! # fn main() -> Result<(), DriverError> {
//! let ctx = CudaContext::new(0)?;
//! let stream = ctx.default_stream();
//! let a: CudaSlice<f32> = stream.clone_htod(&[1.0f32; 100])?;
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
mod host;
mod kernel;
mod slice;
mod stream;

pub use context::CudaContext;
pub use error::DriverError;
pub use host::PinnedHostSlice;
pub use kernel::{CudaFunction, CudaModule, LaunchArgs, LaunchConfig, PushKernelArg, Spirv};
pub use slice::{
    CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, HostSlice,
    RawRange, ValidAsZeroBits,
};
pub use stream::CudaStream;
