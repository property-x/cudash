use std::fmt;

use ash::vk;

/// Everything that can go wrong in [`crate::driver`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverError {
    Vulkan(vk::Result),
    /// The Vulkan library couldn't be loaded.
    NoVulkan(String),
    /// There's no suitable GPU with this ordinal.
    NoSuchDevice(usize),
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
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(e) => write!(f, "Vulkan error: {e}"),
            Self::NoVulkan(e) => write!(f, "couldn't load Vulkan: {e}"),
            Self::NoSuchDevice(n) => write!(f, "no suitable GPU with ordinal {n}"),
            Self::MissingFeature(name) => write!(f, "GPU lacks the Vulkan feature {name}"),
            Self::NoActiveFrame => f.write_str("no frame boundary is active"),
            Self::InvalidModule(e) => write!(f, "invalid module: {e}"),
            Self::FunctionNotFound(n) => write!(f, "no compute entry point named {n}"),
            Self::LaunchLimit(e) => write!(f, "launch exceeds a device limit: {e}"),
            Self::HostMemoryAlreadyRegistered => f.write_str("host memory is already registered"),
            Self::HostMemoryNotRegistered => f.write_str("no registered host memory starts here"),
            Self::InvalidImage(e) => write!(f, "invalid image: {e}"),
        }
    }
}

impl std::error::Error for DriverError {}

impl From<vk::Result> for DriverError {
    fn from(e: vk::Result) -> Self {
        Self::Vulkan(e)
    }
}
