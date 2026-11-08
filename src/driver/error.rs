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
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(e) => write!(f, "Vulkan error: {e}"),
            Self::NoVulkan(e) => write!(f, "couldn't load Vulkan: {e}"),
            Self::NoSuchDevice(n) => write!(f, "no suitable GPU with ordinal {n}"),
            Self::MissingFeature(name) => write!(f, "GPU lacks the Vulkan feature {name}"),
            Self::InvalidModule(e) => write!(f, "invalid module: {e}"),
            Self::FunctionNotFound(n) => write!(f, "no compute entry point named {n}"),
            Self::LaunchLimit(e) => write!(f, "launch exceeds a device limit: {e}"),
            Self::HostMemoryAlreadyRegistered => f.write_str("host memory is already registered"),
            Self::HostMemoryNotRegistered => f.write_str("no registered host memory starts here"),
        }
    }
}

impl std::error::Error for DriverError {}

impl From<vk::Result> for DriverError {
    fn from(e: vk::Result) -> Self {
        Self::Vulkan(e)
    }
}
