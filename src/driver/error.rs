use std::fmt;

use crate::ash::vk;

/// Everything that can go wrong in [`crate::driver`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverError {
    Vulkan(vk::Result),
    /// The Vulkan library couldn't be loaded.
    NoVulkan(String),
    /// There's no suitable GPU with this ordinal.
    NoSuchDevice(usize),
    /// No suitable GPU has this PCI bus ID, or it isn't one.
    NoSuchBusId(String),
    /// The GPU lacks a Vulkan feature cudash relies on.
    MissingFeature(&'static str),
    /// No frame boundary is active on this stream.
    NoActiveFrame,
    /// A module couldn't be read or isn't valid SPIR-V.
    InvalidModule(String),
    /// The module has no compute entry point with this name.
    FunctionNotFound(String),
    /// A launch exceeds what the device allows.
    LaunchLimit(String),
    /// Part of this host memory is already registered.
    HostMemoryAlreadyRegistered,
    /// No registered host memory starts at this address.
    HostMemoryNotRegistered,
    /// An image can't have this size or format on this GPU.
    InvalidImage(String),
    /// A call that can't be captured, or a capture call out of turn. Like CUDA, a call that
    /// can't be captured also invalidates the capture.
    StreamCapture(String),
    /// From the CUDA part of a graph, under the `cuda` feature.
    Cuda(String),
    /// The GPU stopped this context, as after CUDA's "illegal memory access". Every later call
    /// on the context fails. Holds what the driver reported, where it has `VK_EXT_device_fault`
    /// and reported anything.
    DeviceLost(Option<DeviceFault>),
}

/// What the driver reported about a lost device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFault {
    pub description: String,
    /// The accesses that faulted, or were running when the device was lost.
    pub addresses: Vec<FaultAddress>,
    /// Driver-specific entries, as `(description, code, data)`.
    pub vendor: Vec<(String, u64, u64)>,
}

/// One address in a [`DeviceFault`]. The access was somewhere in `address..address + precision`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaultAddress {
    /// Read, write, execute, or an instruction pointer.
    pub kind: vk::DeviceFaultAddressTypeEXT,
    pub address: u64,
    pub precision: u64,
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(e) => write!(f, "Vulkan error: {e}"),
            Self::NoVulkan(e) => write!(f, "couldn't load Vulkan: {e}"),
            Self::NoSuchDevice(n) => write!(f, "no suitable GPU with ordinal {n}"),
            Self::NoSuchBusId(id) => write!(f, "no suitable GPU at PCI bus ID {id:?}"),
            Self::MissingFeature(name) => write!(f, "GPU lacks the Vulkan feature {name}"),
            Self::NoActiveFrame => f.write_str("no frame boundary is active"),
            Self::InvalidModule(e) => write!(f, "invalid module: {e}"),
            Self::FunctionNotFound(n) => write!(f, "no compute entry point named {n}"),
            Self::LaunchLimit(e) => write!(f, "launch exceeds a device limit: {e}"),
            Self::HostMemoryAlreadyRegistered => f.write_str("host memory is already registered"),
            Self::HostMemoryNotRegistered => f.write_str("no registered host memory starts here"),
            Self::InvalidImage(e) => write!(f, "invalid image: {e}"),
            Self::StreamCapture(e) => write!(f, "stream capture: {e}"),
            Self::Cuda(e) => write!(f, "CUDA error: {e}"),
            Self::DeviceLost(None) => f.write_str("device lost"),
            Self::DeviceLost(Some(fault)) => {
                write!(f, "device lost: {}", fault.description)?;
                for a in &fault.addresses {
                    write!(
                        f,
                        "; {:?} at {:#x} (±{:#x})",
                        a.kind, a.address, a.precision
                    )?;
                }
                for (description, code, data) in &fault.vendor {
                    write!(f, "; {description} (code {code:#x}, data {data:#x})")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for DriverError {}

impl From<vk::Result> for DriverError {
    fn from(e: vk::Result) -> Self {
        Self::Vulkan(e)
    }
}
