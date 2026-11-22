//! Kernels: SPIR-V compute shaders, launched like cudarc launches PTX.

use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::CString,
    io::Cursor,
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[cfg(feature = "ash_latest")]
use crate::compat::PushNext as _;
use crate::{
    ash::{khr, vk},
    compat::load_ext,
};

use super::{
    DriverError,
    context::GpuContext,
    device::{Buffer, Device},
    slice::{DevicePtr, DevicePtrMut, DeviceRepr, GpuSlice, GpuView, GpuViewMut, RawRange},
    stream::GpuStream,
};

const OP_CAPABILITY: u32 = 17;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_FUNCTION: u32 = 54;
const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;
const SPIRV_HEADER_WORDS: usize = 5;

/// A compiled kernel module, the counterpart of cudarc's `Ptx`.
#[derive(Debug, Clone)]
pub struct Spirv(SpirvKind);

#[derive(Debug, Clone)]
enum SpirvKind {
    File(PathBuf),
    Binary(Cow<'static, [u8]>),
}

impl Spirv {
    /// Loads from a file when passed to `load_module`.
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Self {
        Self(SpirvKind::File(path.into()))
    }

    /// Takes `include_bytes!` output without copying it.
    pub fn from_binary(data: impl Into<Cow<'static, [u8]>>) -> Self {
        Self(SpirvKind::Binary(data.into()))
    }
}

pub struct GpuModule {
    ctx: Arc<GpuContext>,
    words: Vec<u32>,
}

impl GpuContext {
    /// Loads a kernel module.
    pub fn load_module(self: &Arc<Self>, module: Spirv) -> Result<Arc<GpuModule>, DriverError> {
        let bytes = match module.0 {
            SpirvKind::File(path) => std::fs::read(&path)
                .map_err(|e| DriverError::InvalidModule(format!("{}: {e}", path.display())))?
                .into(),
            SpirvKind::Binary(bytes) => bytes,
        };
        let words = crate::ash::util::read_spv(&mut Cursor::new(bytes))
            .map_err(|e| DriverError::InvalidModule(e.to_string()))?;
        if words.len() < SPIRV_HEADER_WORDS {
            return Err(DriverError::InvalidModule(
                "shorter than a SPIR-V header".into(),
            ));
        }
        let unsupported = for_each_instruction(&words, |_, op, operands| {
            let capability = (op == OP_CAPABILITY).then(|| operands[0])?;
            self.device
                .unsupported_capabilities
                .iter()
                .find(|&&(c, _)| c == capability)
                .map(|&(_, feature)| feature)
        })?;
        if let Some(feature) = unsupported {
            return Err(DriverError::MissingFeature(feature));
        }
        Ok(Arc::new(GpuModule {
            ctx: self.clone(),
            words,
        }))
    }
}

/// Finds a compute entry point's result id.
fn find_entry_point(words: &[u32], name: &str) -> Result<Option<u32>, DriverError> {
    for_each_instruction(words, |_, op, operands| {
        (op == OP_ENTRY_POINT
            && operands[0] == EXECUTION_MODEL_GL_COMPUTE
            && decode_literal(&operands[2..]) == name)
            .then(|| operands[1])
    })
}

/// Visits instructions up to the first function body, which is everything that
/// declares entry points and execution modes, until `visit` returns a value. `visit`
/// gets each instruction's index in `words`, its opcode and its operands.
fn for_each_instruction<R>(
    words: &[u32],
    mut visit: impl FnMut(usize, u32, &[u32]) -> Option<R>,
) -> Result<Option<R>, DriverError> {
    let mut i = SPIRV_HEADER_WORDS;
    while i < words.len() {
        let (op, count) = (words[i] & 0xffff, (words[i] >> 16) as usize);
        if count == 0 || i + count > words.len() {
            return Err(DriverError::InvalidModule("malformed instruction".into()));
        }
        if op == OP_FUNCTION {
            break;
        }
        // The fewest operands each instruction read below can have, so `visit` can index them.
        let min_operands = match op {
            OP_CAPABILITY => 1,
            OP_EXECUTION_MODE => 2,
            OP_ENTRY_POINT => 3,
            _ => 0,
        };
        if count - 1 < min_operands {
            return Err(DriverError::InvalidModule(
                "instruction missing operands".into(),
            ));
        }
        if let Some(found) = visit(i, op, &words[i + 1..i + count]) {
            return Ok(Some(found));
        }
        i += count;
    }
    Ok(None)
}

fn decode_literal(words: &[u32]) -> String {
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Rewrites an entry point's declared workgroup size to `block_dim`.
///
/// CUDA picks the block size at launch, but a shader declares it at compile time. So
/// kernels can declare any size, such as `[numthreads(1, 1, 1)]`, and this applies the
/// launch's. Shaders must not read the size from a compile-time constant, because that
/// isn't patched. Compute indices from `SV_DispatchThreadID` instead.
fn with_block_dim(
    words: &[u32],
    entry: u32,
    block_dim: (u32, u32, u32),
) -> Result<Vec<u32>, DriverError> {
    let local_size = for_each_instruction(words, |i, op, operands| {
        (op == OP_EXECUTION_MODE
            && operands.len() == 5
            && operands[0] == entry
            && operands[1] == EXECUTION_MODE_LOCAL_SIZE)
            .then_some(i)
    })?;
    let mut patched = words.to_vec();
    if let Some(i) = local_size {
        patched[i + 3..i + 6].copy_from_slice(&[block_dim.0, block_dim.1, block_dim.2]);
    }
    Ok(patched)
}

impl GpuModule {
    /// Gets a kernel by its entry point name.
    pub fn load_function(self: &Arc<Self>, name: &str) -> Result<GpuFunction, DriverError> {
        let entry = find_entry_point(&self.words, name)?
            .ok_or_else(|| DriverError::FunctionNotFound(name.to_owned()))?;
        let device = self.ctx.device.clone();
        // A name with a NUL byte can't have matched above.
        let c_name = CString::new(name).expect("a matched name has no NUL");
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(device.properties.limits.max_push_constants_size)];
        let info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);
        let layout = unsafe { device.device.create_pipeline_layout(&info, None) }?;
        device.set_name(layout, format_args!("{name}"));
        Ok(GpuFunction {
            inner: Arc::new(FunctionInner {
                device,
                layout,
                words: self.words.clone(),
                entry,
                name: c_name,
                pipelines: Mutex::default(),
            }),
        })
    }
}

#[derive(Clone)]
pub struct GpuFunction {
    inner: Arc<FunctionInner>,
}

impl GpuFunction {
    /// What the driver reports about this kernel compiled for `block_dim`, such as registers,
    /// spills and code size, like `ptxas -v`. Names and meanings depend on the driver.
    /// Compiles a separate pipeline to collect them, so it costs a compile and doesn't touch launches.
    pub fn statistics(
        &self,
        block_dim: (u32, u32, u32),
    ) -> Result<Vec<KernelStatistic>, DriverError> {
        let inner = &self.inner;
        let device = &inner.device;
        if !device
            .extensions
            .contains(&khr::pipeline_executable_properties::NAME)
        {
            return Err(DriverError::MissingFeature("pipelineExecutableInfo"));
        }
        let pipeline =
            inner.create_pipeline(block_dim, vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR)?;
        let functions = load_ext!(
            khr::pipeline_executable_properties::Device,
            &device.instance,
            &device.device
        );
        let fp = functions.fp();
        let result = unsafe {
            let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
            read_all(|count, data| {
                (fp.get_pipeline_executable_properties_khr)(
                    device.device.handle(),
                    &info,
                    count,
                    data,
                )
            })
            .and_then(|executables| {
                let mut all = Vec::new();
                for index in 0..executables.len() as u32 {
                    let info = vk::PipelineExecutableInfoKHR::default()
                        .pipeline(pipeline)
                        .executable_index(index);
                    let statistics: Vec<vk::PipelineExecutableStatisticKHR> =
                        read_all(|count, data| {
                            (fp.get_pipeline_executable_statistics_khr)(
                                device.device.handle(),
                                &info,
                                count,
                                data,
                            )
                        })?;
                    all.extend(statistics.iter().map(KernelStatistic::from));
                }
                Ok(all)
            })
        };
        unsafe { device.device.destroy_pipeline(pipeline, None) };
        Ok(result?)
    }
}

/// One statistic from [`GpuFunction::statistics`].
#[derive(Clone, Debug, PartialEq)]
pub struct KernelStatistic {
    pub name: String,
    pub description: String,
    pub value: StatisticValue,
}

/// A [`KernelStatistic`]'s value, in the format the driver reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StatisticValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
}

impl From<&vk::PipelineExecutableStatisticKHR<'_>> for KernelStatistic {
    fn from(statistic: &vk::PipelineExecutableStatisticKHR<'_>) -> Self {
        let text = |chars: &[std::ffi::c_char]| {
            let bytes: Vec<u8> = chars
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };
        // SAFETY: `format` says which union field the driver wrote.
        let value = unsafe {
            match statistic.format {
                vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                    StatisticValue::Bool(statistic.value.b32 != 0)
                }
                vk::PipelineExecutableStatisticFormatKHR::INT64 => {
                    StatisticValue::I64(statistic.value.i64)
                }
                vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => {
                    StatisticValue::F64(statistic.value.f64)
                }
                _ => StatisticValue::U64(statistic.value.u64),
            }
        };
        Self {
            name: text(&statistic.name),
            description: text(&statistic.description),
            value,
        }
    }
}

/// Vulkan's two-call pattern: ask for the count, then fill that many default structs.
unsafe fn read_all<T: Default + Clone>(
    mut call: impl FnMut(*mut u32, *mut T) -> vk::Result,
) -> Result<Vec<T>, vk::Result> {
    let mut count = 0;
    call(&mut count, std::ptr::null_mut()).result()?;
    let mut items = vec![T::default(); count as usize];
    call(&mut count, items.as_mut_ptr()).result()?;
    items.truncate(count as usize);
    Ok(items)
}

impl std::fmt::Debug for GpuFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuFunction")
            .field("name", &self.inner.name)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for GpuModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuModule")
            .field("ctx", &self.ctx)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for LaunchArgs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaunchArgs")
            .field("function", self.function)
            .field("arg_bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

struct FunctionInner {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
    words: Vec<u32>,
    entry: u32,
    name: CString,
    pipelines: Mutex<HashMap<(u32, u32, u32), vk::Pipeline>>,
}

impl FunctionInner {
    fn pipeline(&self, block_dim: (u32, u32, u32)) -> Result<vk::Pipeline, DriverError> {
        let mut pipelines = self.pipelines.lock().unwrap();
        if let Some(&pipeline) = pipelines.get(&block_dim) {
            return Ok(pipeline);
        }
        let pipeline = self.create_pipeline(block_dim, vk::PipelineCreateFlags::empty())?;
        pipelines.insert(block_dim, pipeline);
        Ok(pipeline)
    }

    fn create_pipeline(
        &self,
        block_dim: (u32, u32, u32),
        flags: vk::PipelineCreateFlags,
    ) -> Result<vk::Pipeline, DriverError> {
        let device = &self.device.device;
        let code = with_block_dim(&self.words, self.entry, block_dim)?;
        let name = self.name.to_string_lossy();
        let mut code_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let pipeline = unsafe {
            // maintenance5: the pipeline takes the code itself, without a module object.
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .name(&self.name)
                .push_next(&mut code_info);
            let info = [vk::ComputePipelineCreateInfo::default()
                .flags(flags)
                .stage(stage)
                .layout(self.layout)];
            let created = device.create_compute_pipelines(vk::PipelineCache::null(), &info, None);
            let pipeline = created.map_err(|(_, e)| e)?[0];
            self.device
                .set_name(pipeline, format_args!("{name} block {block_dim:?}"));
            pipeline
        };
        Ok(pipeline)
    }
}

impl Drop for FunctionInner {
    fn drop(&mut self) {
        // Submissions hold a reference until they finish, so the GPU is done here.
        let pipelines = self.pipelines.get_mut().unwrap_or_else(|e| e.into_inner());
        unsafe {
            for &pipeline in pipelines.values() {
                self.device.device.destroy_pipeline(pipeline, None);
            }
            self.device
                .device
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    /// (width, height, depth) of the grid, in blocks.
    pub grid_dim: (u32, u32, u32),
    /// (x, y, z) dimensions of each block, in threads.
    pub block_dim: (u32, u32, u32),
    /// Ignored, because a shader declares its shared memory at compile time.
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// A launch with at least `n` threads, in blocks of 1024.
    pub fn for_num_elems(n: u32) -> Self {
        const NUM_THREADS: u32 = 1024;
        Self {
            grid_dim: (n.div_ceil(NUM_THREADS), 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// A kernel launch being set up, like cudarc's `LaunchArgs`. Arguments are laid out
/// in push constants in the order they're passed, like C function arguments: slices
/// and views as 64-bit GPU addresses, and everything else by value.
pub struct LaunchArgs<'a> {
    stream: &'a GpuStream,
    function: &'a GpuFunction,
    bytes: Vec<u8>,
    reads: Vec<Arc<Buffer>>,
    writes: Vec<Arc<Buffer>>,
}

impl GpuStream {
    /// Starts setting up a kernel launch on this stream.
    pub fn launch_builder<'a>(&'a self, function: &'a GpuFunction) -> LaunchArgs<'a> {
        LaunchArgs {
            stream: self,
            function,
            bytes: Vec::new(),
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }
}

impl LaunchArgs<'_> {
    fn push_bytes(&mut self, bytes: &[u8], align: usize) {
        self.bytes
            .resize(self.bytes.len().next_multiple_of(align), 0);
        self.bytes.extend_from_slice(bytes);
    }

    fn push_read(&mut self, range: RawRange<'_>) -> &mut Self {
        self.reads.push(range.buffer.clone());
        self.push_address(range)
    }

    fn push_write(&mut self, range: RawRange<'_>) -> &mut Self {
        self.writes.push(range.buffer.clone());
        self.push_address(range)
    }

    fn push_address(&mut self, range: RawRange<'_>) -> &mut Self {
        let address = range.buffer.address + range.offset;
        self.push_bytes(&address.to_ne_bytes(), align_of::<u64>());
        self
    }

    /// Queues the kernel on the stream.
    ///
    /// # Safety
    /// As with any kernel launch, the kernel's arguments must match what's been passed,
    /// and it must stay in bounds and be free of data races.
    pub unsafe fn launch(&mut self, config: LaunchConfig) -> Result<(), DriverError> {
        let function = &self.function.inner;
        let limits = &function.device.properties.limits;
        let (bx, by, bz) = config.block_dim;
        let (gx, gy, gz) = config.grid_dim;
        let max_size = limits.max_compute_work_group_size;
        let max_count = limits.max_compute_work_group_count;
        let invocations = (bx as u64) * (by as u64) * (bz as u64);
        if invocations == 0
            || invocations > limits.max_compute_work_group_invocations as u64
            || bx > max_size[0]
            || by > max_size[1]
            || bz > max_size[2]
            || gx > max_count[0]
            || gy > max_count[1]
            || gz > max_count[2]
        {
            return Err(DriverError::LaunchLimit(format!(
                "block {:?} in grid {:?}, but the device allows blocks up to {max_size:?} with \
                 at most {} threads, and grids up to {max_count:?}",
                config.block_dim, config.grid_dim, limits.max_compute_work_group_invocations
            )));
        }
        if self.bytes.len() > limits.max_push_constants_size as usize {
            return Err(DriverError::LaunchLimit(format!(
                "arguments take {} bytes, but the device allows {}",
                self.bytes.len(),
                limits.max_push_constants_size
            )));
        }

        let pipeline = function.pipeline(config.block_dim)?;
        let layout = function.layout;
        let bytes = &self.bytes;
        let reads: Vec<_> = self.reads.iter().collect();
        let writes: Vec<_> = self.writes.iter().collect();
        self.stream.core.submit(
            format_args!(
                "launch {} grid {:?} block {:?}",
                function.name.to_string_lossy(),
                config.grid_dim,
                config.block_dim
            ),
            &reads,
            &writes,
            vec![function.clone()],
            |device, commands| unsafe {
                device.cmd_bind_pipeline(commands, vk::PipelineBindPoint::COMPUTE, pipeline);
                if !bytes.is_empty() {
                    let stages = vk::ShaderStageFlags::COMPUTE;
                    device.cmd_push_constants(commands, layout, stages, 0, bytes);
                }
                device.cmd_dispatch(commands, gx, gy, gz);
            },
        )
    }
}

/// Things that can be passed to a kernel with `arg`, like cudarc's `PushKernelArg`.
///
/// # Safety
/// Implementations must describe their memory accurately to the launch.
pub unsafe trait PushKernelArg<T> {
    fn arg(&mut self, arg: T) -> &mut Self;
}

unsafe impl<'a, 'b: 'a, T: DeviceRepr> PushKernelArg<&'b T> for LaunchArgs<'a> {
    fn arg(&mut self, arg: &'b T) -> &mut Self {
        // SAFETY: `T: DeviceRepr` is plain data.
        let bytes = unsafe {
            std::slice::from_raw_parts(std::ptr::from_ref(arg).cast::<u8>(), size_of::<T>())
        };
        self.push_bytes(bytes, align_of::<T>());
        self
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b GpuSlice<T>> for LaunchArgs<'a> {
    fn arg(&mut self, arg: &'b GpuSlice<T>) -> &mut Self {
        self.push_read(arg.raw())
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b mut GpuSlice<T>> for LaunchArgs<'a> {
    fn arg(&mut self, arg: &'b mut GpuSlice<T>) -> &mut Self {
        self.push_write(arg.raw_mut())
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b GpuView<'c, T>> for LaunchArgs<'a> {
    fn arg(&mut self, arg: &'b GpuView<'c, T>) -> &mut Self {
        self.push_read(arg.raw())
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b mut GpuViewMut<'c, T>> for LaunchArgs<'a> {
    fn arg(&mut self, arg: &'b mut GpuViewMut<'c, T>) -> &mut Self {
        self.push_write(arg.raw_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Header, `OpCapability Shader`, then `OpEntryPoint GLCompute %7 "main"` and
    /// `OpExecutionMode %7 LocalSize 1 2 3`.
    fn module() -> Vec<u32> {
        let name = u32::from_le_bytes(*b"main");
        vec![
            0x0723_0203,
            0x0001_0300,
            0,
            8,
            0,
            (2 << 16) | OP_CAPABILITY, // one operand, which every real module starts with
            1,
            (5 << 16) | OP_ENTRY_POINT,
            EXECUTION_MODEL_GL_COMPUTE,
            7,
            name,
            0,
            (6 << 16) | OP_EXECUTION_MODE,
            7,
            EXECUTION_MODE_LOCAL_SIZE,
            1,
            2,
            3,
        ]
    }

    #[test]
    fn entry_points_are_found_by_name() {
        assert_eq!(find_entry_point(&module(), "main").unwrap(), Some(7));
        assert_eq!(find_entry_point(&module(), "other").unwrap(), None);
    }

    #[test]
    fn block_dim_only_patches_the_chosen_entry_point() {
        assert_eq!(
            with_block_dim(&module(), 7, (9, 8, 4)).unwrap()[15..],
            [9, 8, 4]
        );
        assert_eq!(
            with_block_dim(&module(), 6, (9, 8, 4)).unwrap()[15..],
            [1, 2, 3]
        );
    }

    #[test]
    fn capabilities_the_gpu_lacks_are_errors() {
        let ctx = GpuContext::new(0).unwrap();
        let load = |words: &[u32]| {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            ctx.load_module(Spirv::from_binary(bytes))
        };
        let mut words = module();
        assert!(load(&words).is_ok());
        let Some(&(capability, feature)) = ctx.device.unsupported_capabilities.first() else {
            return; // a GPU with every feature has nothing to refuse
        };
        words[6] = capability;
        assert!(matches!(load(&words), Err(DriverError::MissingFeature(f)) if f == feature));
    }

    /// A real kernel has statistics where the driver reports them, and the pipeline used for
    /// them isn't kept for launches.
    #[test]
    fn statistics_describe_a_kernel_or_name_the_missing_feature() {
        let ctx = GpuContext::new(0).unwrap();
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/double_all.spv"));
        let module = ctx.load_module(Spirv::from_binary(&spirv[..])).unwrap();
        let function = module.load_function("double_all").unwrap();
        match function.statistics((64, 1, 1)) {
            Ok(statistics) => {
                assert!(!statistics.is_empty());
                assert!(statistics.iter().all(|s| !s.name.is_empty()));
            }
            Err(error) => assert!(matches!(
                error,
                DriverError::MissingFeature("pipelineExecutableInfo")
            )),
        }
        assert!(function.inner.pipelines.lock().unwrap().is_empty());
    }

    #[test]
    fn malformed_modules_are_errors() {
        let mut words = module();
        words[7] = 15; // an instruction claiming zero length
        assert!(find_entry_point(&words, "main").is_err());
        words[7] = (99 << 16) | OP_ENTRY_POINT; // one running past the end
        assert!(find_entry_point(&words, "main").is_err());

        words[7] = (1 << 16) | OP_ENTRY_POINT; // one without operands
        assert!(find_entry_point(&words, "main").is_err());
        let mut words = module();
        words[5] = (1 << 16) | OP_CAPABILITY; // `OpCapability` with no capability,
        words[6] = 1 << 16; // then `OpNop`
        assert!(find_entry_point(&words, "main").is_err());

        // The entry point is fine, and the block size instruction after it is cut off.
        let mut words = module();
        words.truncate(17);
        assert_eq!(find_entry_point(&words, "main").unwrap(), Some(7));
        assert!(with_block_dim(&words, 7, (9, 8, 4)).is_err());
    }
}
