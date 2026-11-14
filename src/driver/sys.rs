//! The three types cudarc's safe graph API takes from `cudarc::driver::sys`, with the same names
//! and values, so capture code ports unchanged. Nothing else from `sys` is provided.
//! We copy them to stay consistent with the `cudarc` API, and to avoid having to depend on `cudarc` for the safe graph API.
#![allow(non_camel_case_types)]

/// How a capture treats calls that aren't safe while capturing. Vulkan parts ignore it; under
/// `cudarc` it's passed on to the paired CUDA stream's capture.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamCaptureMode_enum {
    CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    CU_STREAM_CAPTURE_MODE_RELAXED = 2,
}
pub use self::CUstreamCaptureMode_enum as CUstreamCaptureMode;

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamCaptureStatus_enum {
    CU_STREAM_CAPTURE_STATUS_NONE = 0,
    CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
}
pub use self::CUstreamCaptureStatus_enum as CUstreamCaptureStatus;

/// Flags for instantiating a captured graph. Only CUDA parts of a graph use them.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUgraphInstantiate_flags_enum(pub ::core::ffi::c_uint);
pub use self::CUgraphInstantiate_flags_enum as CUgraphInstantiate_flags;

impl CUgraphInstantiate_flags_enum {
    pub const CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH: Self = Self(1);
    pub const CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD: Self = Self(2);
    pub const CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH: Self = Self(4);
    pub const CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY: Self = Self(8);
}

impl std::ops::BitOr for CUgraphInstantiate_flags_enum {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}
