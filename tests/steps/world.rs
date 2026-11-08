//! Shared state for the feature steps, and the value helpers they use.
//!
//! Feature files write numbers as text. `Scalar` keeps integers and decimals apart until a
//! buffer's element type is known, and `Ty::normalize` turns an expected value into the
//! form the buffer stores, so comparisons are exact for every type.

// The interop half (`IBuf`, `sem`, `write_interop`, ...) waits for the steps in `interop.rs`, which
// aren't written yet. Remove this once they use it.
#![allow(dead_code, unused_imports)]

use std::{collections::HashMap, fmt, sync::Arc};

use cudash::{
    driver, interop,
    interop::cudarc::driver::{
        DeviceRepr as CuDeviceRepr, DeviceSlice as _, ValidAsZeroBits as CuZeroBits,
    },
};

/// A number from a feature file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Int(i128),
    Float(f64),
}

impl Scalar {
    /// Parses `12`, `-3`, `0xC0DAF00D`, `2.5` or `3.4028235e38`.
    pub fn parse(text: &str) -> Scalar {
        let t = text.trim();
        if let Some(hex) = t.strip_prefix("0x") {
            return Scalar::Int(i128::from_str_radix(hex, 16).expect("hex literal"));
        }
        if t.contains(['.', 'e', 'E']) {
            Scalar::Float(t.parse().expect("decimal literal"))
        } else {
            Scalar::Int(t.parse().expect("integer literal"))
        }
    }

    /// The value as a float, for formulas in feature files.
    pub fn as_f64(self) -> f64 {
        match self {
            Scalar::Int(i) => i as f64,
            Scalar::Float(f) => f,
        }
    }
}

/// Parses `1, 2 and 3` or `-5, 0 and 7` into scalars.
pub fn parse_list(text: &str) -> Vec<Scalar> {
    text.replace(" and ", ", ")
        .split(',')
        .map(Scalar::parse)
        .collect()
}

/// Element types a feature file can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    U8,
    U16,
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
}

impl Ty {
    pub fn parse(name: &str) -> Ty {
        match name {
            "u8" => Ty::U8,
            "u16" => Ty::U16,
            "u32" => Ty::U32,
            "u64" => Ty::U64,
            "i32" => Ty::I32,
            "i64" => Ty::I64,
            "f32" => Ty::F32,
            "f64" => Ty::F64,
            other => panic!("unknown element type {other:?}"),
        }
    }

    /// The form a value of this type is stored in, so it can be compared with `==`.
    pub fn normalize(self, value: Scalar) -> Scalar {
        match self {
            Ty::F32 => Scalar::Float(f64::from(value.as_f64() as f32)),
            Ty::F64 => Scalar::Float(value.as_f64()),
            _ => match value {
                Scalar::Int(i) => Scalar::Int(i),
                Scalar::Float(f) => Scalar::Int(f as i128),
            },
        }
    }
}

/// Element types that can be stored in a device buffer.
pub trait Element:
Copy
+ PartialEq
+ fmt::Debug
+ 'static
+ driver::DeviceRepr
+ driver::ValidAsZeroBits
+ CuDeviceRepr
+ CuZeroBits
{
    const TY: Ty;
    fn from_scalar(value: Scalar) -> Self;
    fn to_scalar(self) -> Scalar;
    fn wrap_driver(slice: driver::CudaSlice<Self>) -> DBuf;
    fn wrap_interop(slice: interop::SharedBuffer<Self>) -> IBuf;
}

macro_rules! int_element {
    ($t:ty, $ty:ident, $dv:ident, $iv:ident) => {
        impl Element for $t {
            const TY: Ty = Ty::$ty;
            fn from_scalar(value: Scalar) -> Self {
                match value {
                    Scalar::Int(i) => {
                        <$t>::try_from(i).expect("value out of range for the element type")
                    }
                    Scalar::Float(f) => panic!("{f} is not an integer, for an integer element"),
                }
            }
            fn to_scalar(self) -> Scalar {
                Scalar::Int(i128::from(self))
            }
            fn wrap_driver(slice: driver::CudaSlice<Self>) -> DBuf {
                DBuf::$dv(slice)
            }
            fn wrap_interop(slice: interop::SharedBuffer<Self>) -> IBuf {
                IBuf::$iv(slice)
            }
        }
    };
}

macro_rules! float_element {
    ($t:ty, $ty:ident, $dv:ident, $iv:ident) => {
        impl Element for $t {
            const TY: Ty = Ty::$ty;
            fn from_scalar(value: Scalar) -> Self {
                value.as_f64() as $t
            }
            fn to_scalar(self) -> Scalar {
                Scalar::Float(f64::from(self))
            }
            fn wrap_driver(slice: driver::CudaSlice<Self>) -> DBuf {
                DBuf::$dv(slice)
            }
            fn wrap_interop(slice: interop::SharedBuffer<Self>) -> IBuf {
                IBuf::$iv(slice)
            }
        }
    };
}

int_element!(u8, U8, U8, U8);
int_element!(u16, U16, U16, U16);
int_element!(u32, U32, U32, U32);
int_element!(u64, U64, U64, U64);
int_element!(i32, I32, I32, I32);
int_element!(i64, I64, I64, I64);
float_element!(f32, F32, F32, F32);
float_element!(f64, F64, F64, F64);

/// Runs `$body` with `$t` bound to the Rust type for a [`Ty`].
macro_rules! with_ty {
    ($ty:expr, $t:ident => $body:expr) => {
        match $ty {
            $crate::steps::world::Ty::U8 => {
                type $t = u8;
                $body
            }
            $crate::steps::world::Ty::U16 => {
                type $t = u16;
                $body
            }
            $crate::steps::world::Ty::U32 => {
                type $t = u32;
                $body
            }
            $crate::steps::world::Ty::U64 => {
                type $t = u64;
                $body
            }
            $crate::steps::world::Ty::I32 => {
                type $t = i32;
                $body
            }
            $crate::steps::world::Ty::I64 => {
                type $t = i64;
                $body
            }
            $crate::steps::world::Ty::F32 => {
                type $t = f32;
                $body
            }
            $crate::steps::world::Ty::F64 => {
                type $t = f64;
                $body
            }
        }
    };
}
pub(crate) use with_ty;

/// A buffer allocated by `cudash::driver`, of any element type.
pub enum DBuf {
    U8(driver::CudaSlice<u8>),
    U16(driver::CudaSlice<u16>),
    U32(driver::CudaSlice<u32>),
    U64(driver::CudaSlice<u64>),
    I32(driver::CudaSlice<i32>),
    I64(driver::CudaSlice<i64>),
    F32(driver::CudaSlice<f32>),
    F64(driver::CudaSlice<f64>),
}

/// A buffer allocated by `cudash::interop`, of any element type.
pub enum IBuf {
    U8(interop::SharedBuffer<u8>),
    U16(interop::SharedBuffer<u16>),
    U32(interop::SharedBuffer<u32>),
    U64(interop::SharedBuffer<u64>),
    I32(interop::SharedBuffer<i32>),
    I64(interop::SharedBuffer<i64>),
    F32(interop::SharedBuffer<f32>),
    F64(interop::SharedBuffer<f64>),
}

/// Matches a [`DBuf`] or [`IBuf`] and runs `$body` with the inner buffer bound to `$s`.
macro_rules! on_buf {
    ($buf:expr, $s:ident => $body:expr) => {
        match $buf {
            DBuf::U8($s) => $body,
            DBuf::U16($s) => $body,
            DBuf::U32($s) => $body,
            DBuf::U64($s) => $body,
            DBuf::I32($s) => $body,
            DBuf::I64($s) => $body,
            DBuf::F32($s) => $body,
            DBuf::F64($s) => $body,
        }
    };
}
pub(crate) use on_buf;

macro_rules! on_ibuf {
    ($buf:expr, $s:ident => $body:expr) => {
        match $buf {
            IBuf::U8($s) => $body,
            IBuf::U16($s) => $body,
            IBuf::U32($s) => $body,
            IBuf::U64($s) => $body,
            IBuf::I32($s) => $body,
            IBuf::I64($s) => $body,
            IBuf::F32($s) => $body,
            IBuf::F64($s) => $body,
        }
    };
}
pub(crate) use on_ibuf;

impl DBuf {
    pub fn ty(&self) -> Ty {
        match self {
            DBuf::U8(_) => Ty::U8,
            DBuf::U16(_) => Ty::U16,
            DBuf::U32(_) => Ty::U32,
            DBuf::U64(_) => Ty::U64,
            DBuf::I32(_) => Ty::I32,
            DBuf::I64(_) => Ty::I64,
            DBuf::F32(_) => Ty::F32,
            DBuf::F64(_) => Ty::F64,
        }
    }

    pub fn len(&self) -> usize {
        on_buf!(self, s => s.len())
    }

    pub fn num_bytes(&self) -> usize {
        on_buf!(self, s => s.num_bytes())
    }

    pub fn vk_buffer(&self) -> cudash::ash::vk::Buffer {
        on_buf!(self, s => s.vk_buffer())
    }

    /// Reads every element back to the host, using the buffer's own stream.
    pub fn read(&self) -> Result<Vec<Scalar>, String> {
        on_buf!(self, s => {
            let host = s.stream().clone_dtoh(s).map_err(|e| format!("{e:?}"))?;
            Ok(host.into_iter().map(Element::to_scalar).collect())
        })
    }

    /// Reads every element back to the host, on `stream`.
    pub fn read_on(&self, stream: &Arc<driver::CudaStream>) -> Result<Vec<Scalar>, String> {
        on_buf!(self, s => {
            let host = stream.clone_dtoh(s).map_err(|e| format!("{e:?}"))?;
            Ok(host.into_iter().map(Element::to_scalar).collect())
        })
    }

    /// Copies `values` into the start of the buffer, on `stream`.
    pub fn write_on(
        &mut self,
        stream: &Arc<driver::CudaStream>,
        values: &[Scalar],
    ) -> Result<(), String> {
        on_buf!(self, s => write_driver(stream, s, 0, values))
    }

    /// Copies `values` starting at element `start`, on `stream`.
    pub fn write_at_on(
        &mut self,
        stream: &Arc<driver::CudaStream>,
        start: usize,
        values: &[Scalar],
    ) -> Result<(), String> {
        on_buf!(self, s => write_driver(stream, s, start, values))
    }
}

impl IBuf {
    pub fn ty(&self) -> Ty {
        match self {
            IBuf::U8(_) => Ty::U8,
            IBuf::U16(_) => Ty::U16,
            IBuf::U32(_) => Ty::U32,
            IBuf::U64(_) => Ty::U64,
            IBuf::I32(_) => Ty::I32,
            IBuf::I64(_) => Ty::I64,
            IBuf::F32(_) => Ty::F32,
            IBuf::F64(_) => Ty::F64,
        }
    }

    pub fn len(&self) -> usize {
        on_ibuf!(self, s => s.len())
    }

    pub fn num_bytes(&self) -> usize {
        on_ibuf!(self, s => s.num_bytes())
    }

    pub fn vk_buffer(&self) -> cudash::ash::vk::Buffer {
        on_ibuf!(self, s => s.vk_buffer())
    }

    pub fn read_on(
        &self,
        stream: &Arc<interop::cudarc::driver::CudaStream>,
    ) -> Result<Vec<Scalar>, String> {
        on_ibuf!(self, s => {
            let host = stream.clone_dtoh(s).map_err(|e| format!("{e:?}"))?;
            Ok(host.into_iter().map(Element::to_scalar).collect())
        })
    }

    pub fn write_on(
        &mut self,
        stream: &Arc<interop::cudarc::driver::CudaStream>,
        values: &[Scalar],
    ) -> Result<(), String> {
        on_ibuf!(self, s => write_interop(stream, s, values))
    }
}

/// Wraps a typed driver slice in the [`DBuf`] for its element type.
pub fn wrap<T: Element>(slice: driver::CudaSlice<T>) -> DBuf {
    T::wrap_driver(slice)
}

/// Copies host values into a driver buffer starting at `start`.
fn write_driver<T: Element>(
    stream: &Arc<driver::CudaStream>,
    slice: &mut driver::CudaSlice<T>,
    start: usize,
    values: &[Scalar],
) -> Result<(), String> {
    let host: Vec<T> = values.iter().map(|v| T::from_scalar(*v)).collect();
    let end = start + host.len();
    assert!(
        end <= slice.len(),
        "{end} elements don't fit in a buffer of {}",
        slice.len()
    );
    let mut view = slice.slice_mut(start..end);
    stream
        .memcpy_htod(&host, &mut view)
        .map_err(|e| format!("{e:?}"))
}

/// Copies host values into an interop buffer, from its own cudarc stream.
fn write_interop<T: Element>(
    stream: &Arc<interop::cudarc::driver::CudaStream>,
    slice: &mut interop::SharedBuffer<T>,
    values: &[Scalar],
) -> Result<(), String> {
    let host: Vec<T> = values.iter().map(|v| T::from_scalar(*v)).collect();
    stream
        .memcpy_htod(&host, slice)
        .map_err(|e| format!("{e:?}"))
}

/// Allocates a driver buffer of `len` zeroed elements of type `T`.
pub fn alloc_driver<T: Element>(
    stream: &Arc<driver::CudaStream>,
    len: usize,
) -> Result<driver::CudaSlice<T>, String> {
    stream.alloc_zeros::<T>(len).map_err(|e| format!("{e:?}"))
}

/// Allocates a driver buffer holding `values`, uploaded on `stream`.
pub fn upload_driver<T: Element>(
    stream: &Arc<driver::CudaStream>,
    values: &[Scalar],
) -> Result<driver::CudaSlice<T>, String> {
    let host: Vec<T> = values.iter().map(|v| T::from_scalar(*v)).collect();
    stream.clone_htod(&host).map_err(|e| format!("{e:?}"))
}

/// The world each scenario starts with, and everything its steps touch.
#[derive(Default, cucumber::World)]
pub struct World {
    // cudash::driver, for the Vulkan-only driver features
    pub drv: Option<Arc<driver::CudaContext>>,
    pub streams: HashMap<String, Arc<driver::CudaStream>>,
    pub dbufs: HashMap<String, DBuf>,
    pub kernels: HashMap<String, Arc<driver::CudaFunction>>,
    pub spirv: HashMap<String, Vec<u8>>,

    // Host data and the last value read back, shared by both layers
    pub host: Vec<Scalar>,
    pub host_ty: Option<Ty>,
    pub result: Vec<Scalar>,
    pub result_ty: Option<Ty>,
    pub error: Option<String>,
    pub panicked: bool,

    // cudash::interop, for the CUDA–Vulkan features
    pub icx: Option<Arc<interop::cudarc::driver::CudaContext>>,
    pub ibufs: HashMap<String, IBuf>,
    pub sem: Option<interop::SharedSemaphore>,
    pub signal: u64,
    pub wait: Option<bool>,
    pub cuda_kernels: HashMap<String, Arc<cudash::interop::cudarc::driver::CudaFunction>>,
    pub all_ok: bool,
}

impl fmt::Debug for World {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("World")
            .field("dbufs", &self.dbufs.len())
            .field("ibufs", &self.ibufs.len())
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl World {
    /// The driver stream named `name`.
    pub fn stream(&self, name: &str) -> Arc<driver::CudaStream> {
        self.streams
            .get(name)
            .unwrap_or_else(|| panic!("no stream named {name:?}"))
            .clone()
    }

    pub fn default_stream(&self) -> Arc<driver::CudaStream> {
        self.stream("default")
    }

    pub fn ctx(&self) -> &Arc<driver::CudaContext> {
        self.drv.as_ref().expect("a context was created")
    }

    pub fn dbuf(&self, name: &str) -> &DBuf {
        self.dbufs
            .get(name)
            .unwrap_or_else(|| panic!("no driver buffer named {name:?}"))
    }

    pub fn dbuf_mut(&mut self, name: &str) -> &mut DBuf {
        self.dbufs
            .get_mut(name)
            .unwrap_or_else(|| panic!("no driver buffer named {name:?}"))
    }

    pub fn ibuf(&self, name: &str) -> &IBuf {
        self.ibufs
            .get(name)
            .unwrap_or_else(|| panic!("no interop buffer named {name:?}"))
    }

    pub fn icx(&self) -> &Arc<interop::cudarc::driver::CudaContext> {
        self.icx.as_ref().expect("an NVIDIA GPU was set up")
    }

    pub fn ibuf_mut(&mut self, name: &str) -> &mut IBuf {
        self.ibufs
            .get_mut(name)
            .unwrap_or_else(|| panic!("no interop buffer named {name:?}"))
    }

    /// The values and element type of any named buffer, from either layer.
    pub fn read_named(&self, name: &str) -> (Vec<Scalar>, Ty) {
        if let Some(buf) = self.dbufs.get(name) {
            let values = buf
                .read()
                .unwrap_or_else(|e| panic!("reading {name:?}: {e}"));
            return (values, buf.ty());
        }
        let buf = self.ibuf(name);
        let stream = self.icx().default_stream();
        let values = buf
            .read_on(&stream)
            .unwrap_or_else(|e| panic!("reading {name:?}: {e}"));
        (values, buf.ty())
    }

    /// Stores an error message, or clears the last error.
    pub fn record<T, E: fmt::Debug>(&mut self, outcome: Result<T, E>) -> Option<T> {
        match outcome {
            Ok(value) => {
                self.error = None;
                Some(value)
            }
            Err(e) => {
                self.error = Some(format!("{e:?}"));
                None
            }
        }
    }
}

/// Checks `actual` against `expected`, after normalizing `expected` to `ty`.
pub fn expect_values(actual: &[Scalar], ty: Ty, expected: &[Scalar]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "expected {} values, found {}",
            expected.len(),
            actual.len()
        ));
    }
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        let want = ty.normalize(*want);
        if *got != want {
            return Err(format!("element {i}: expected {want:?}, found {got:?}"));
        }
    }
    Ok(())
}

/// Checks that every element equals `value`.
pub fn expect_every(actual: &[Scalar], ty: Ty, value: Scalar) -> Result<(), String> {
    let want = ty.normalize(value);
    match actual.iter().position(|got| *got != want) {
        None => Ok(()),
        Some(i) => Err(format!(
            "element {i}: expected {want:?}, found {:?}",
            actual[i]
        )),
    }
}

/// Checks that element `i` equals `f(i)`, normalized to `ty`.
pub fn expect_by_index(
    actual: &[Scalar],
    ty: Ty,
    f: impl Fn(usize) -> Scalar,
) -> Result<(), String> {
    for (i, got) in actual.iter().enumerate() {
        let want = ty.normalize(f(i));
        if *got != want {
            return Err(format!("element {i}: expected {want:?}, found {got:?}"));
        }
    }
    Ok(())
}
