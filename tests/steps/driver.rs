//! Steps for the Vulkan-only `cudash::driver` features.

use std::sync::Arc;

use cucumber::{given, then, when};
use cudash::{
    ash::vk::Handle as _,
    driver::{self, LaunchConfig, PushKernelArg, Spirv},
};

use super::world::{
    DBuf, Element, Scalar, Ty, World, alloc_driver, expect_by_index, expect_every, expect_values,
    on_buf, parse_list, upload_driver, with_ty, wrap,
};

/// A 300-byte argument, for the push-constant limit scenario.
#[repr(C)]
#[derive(Clone, Copy)]
struct Blob([u8; 300]);

// SAFETY: plain bytes, no padding that matters.
unsafe impl driver::DeviceRepr for Blob {}

/// Loads the Slang kernel `tests/kernels/<name>.slang`, which `build.rs` compiled. Its entry point
/// has the same name.
fn load_kernel(w: &mut World, name: &str) {
    let path = std::path::Path::new(env!("OUT_DIR")).join(format!("{name}.spv"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let module = w
        .ctx()
        .load_module(Spirv::from_binary(bytes.clone()))
        .expect("the kernel loads as a module");
    let function = module
        .load_function(name)
        .expect("the kernel has its entry point");
    w.kernels.insert(name.to_owned(), Arc::new(function));
    w.spirv.insert(name.to_owned(), bytes);
}

/// Launch configuration: `for_num_elems`, or an explicit block size with enough blocks.
fn config(n: u32, block: Option<u32>, shared: u32) -> LaunchConfig {
    match block {
        None => LaunchConfig {
            shared_mem_bytes: shared,
            ..LaunchConfig::for_num_elems(n)
        },
        Some(b) => LaunchConfig {
            grid_dim: (n.div_ceil(b.max(1)), 1, 1),
            block_dim: (b, 1, 1),
            shared_mem_bytes: shared,
        },
    }
}

/// Launches a kernel with the shape `(buffer, n)` on `view` of `slice`.
fn launch_shape<T: Element>(
    stream: &Arc<driver::CudaStream>,
    function: &driver::CudaFunction,
    slice: &mut driver::CudaSlice<T>,
    range: (usize, usize),
    n: i32,
    cfg: LaunchConfig,
    extra: bool,
) -> Result<(), String> {
    let (start, count) = range;
    let mut view = slice.slice_mut(start..start + count);
    let blob = Blob([0; 300]);
    let mut builder = stream.launch_builder(function);
    builder.arg(&mut view).arg(&n);
    if extra {
        builder.arg(&blob);
    }
    unsafe { builder.launch(cfg) }.map_err(|e| format!("{e:?}"))
}

/// Runs a `(buffer, n)` kernel on a named buffer, optionally on a sub-range.
#[allow(clippy::too_many_arguments)]
fn launch_on(
    w: &mut World,
    kernel: &str,
    buffer: &str,
    range: Option<(usize, usize)>,
    n: u32,
    block: Option<u32>,
    shared: u32,
    extra: bool,
) {
    let stream = w.default_stream();
    let function = w
        .kernels
        .get(kernel)
        .unwrap_or_else(|| panic!("kernel {kernel:?} is loaded"))
        .clone();
    let cfg = config(n, block, shared);
    let dbuf = w
        .dbufs
        .get_mut(buffer)
        .unwrap_or_else(|| panic!("no driver buffer {buffer:?}"));
    let len = dbuf.len();
    let range = range.unwrap_or((0, len));
    let outcome =
        on_buf!(dbuf, s => launch_shape(&stream, &function, s, range, n as i32, cfg, extra));
    let _ = w.record(outcome);
}

/// Runs `axpy` on two f32 buffers: y = a * x + y.
fn launch_axpy(w: &mut World, y_name: &str, x_name: &str, a: f32, n: u32) {
    let stream = w.default_stream();
    let function = w.kernels["axpy"].clone();
    let mut y = match w.dbufs.remove(y_name) {
        Some(DBuf::F32(s)) => s,
        _ => panic!("{y_name:?} must be an f32 driver buffer"),
    };
    let x = match w.dbufs.get(x_name) {
        Some(DBuf::F32(s)) => s,
        _ => panic!("{x_name:?} must be an f32 driver buffer"),
    };
    let n_arg = n as i32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&mut y).arg(x).arg(&a).arg(&n_arg);
    let outcome =
        unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }.map_err(|e| format!("{e:?}"));
    w.dbufs.insert(y_name.to_owned(), DBuf::F32(y));
    let _ = w.record(outcome);
}

/// Runs `add_i64` on an i64 buffer: data[i] += k.
fn launch_add_i64(w: &mut World, name: &str, k: i64, n: u32) {
    let stream = w.default_stream();
    let function = w.kernels["add_i64"].clone();
    let data = match w.dbufs.get_mut(name) {
        Some(DBuf::I64(s)) => s,
        _ => panic!("{name:?} must be an i64 driver buffer"),
    };
    let n_arg = n as i32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(data).arg(&k).arg(&n_arg);
    let outcome =
        unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }.map_err(|e| format!("{e:?}"));
    let _ = w.record(outcome);
}

/// Allocates `len` zeroed elements of `ty` as a named driver buffer on `stream_name`.
fn new_zeroed(w: &mut World, name: &str, len: usize, ty: Ty, stream_name: &str) {
    let stream = w.stream(stream_name);
    let outcome = with_ty!(ty, T => alloc_driver::<T>(&stream, len).map(T::wrap_driver));
    if let Some(buf) = w.record(outcome) {
        w.dbufs.insert(name.to_owned(), buf);
    }
}

/// Uploads `values` as a named driver buffer on `stream_name`.
fn new_uploaded(w: &mut World, name: &str, ty: Ty, values: &[Scalar], stream_name: &str) {
    let stream = w.stream(stream_name);
    let outcome = with_ty!(ty, T => upload_driver::<T>(&stream, values).map(T::wrap_driver));
    let buf = w.record(outcome).expect("upload succeeds");
    w.dbufs.insert(name.to_owned(), buf);
}

/// Reads `buffer` back to the host on `stream`, as the result.
fn read_into_result(w: &mut World, buffer: &str, stream: &Arc<driver::CudaStream>) {
    let (values, ty) = {
        let buf = w.dbuf(buffer);
        let values = buf.read_on(stream).expect("read succeeds");
        (values, buf.ty())
    };
    w.result = values;
    w.result_ty = Some(ty);
}

// ---- Givens: context, streams, buffers ------------------------------------------------

#[given(regex = r"^a context on the first Vulkan 1\.3 GPU$")]
fn context_on_first_gpu(w: &mut World) {
    let ctx = driver::CudaContext::new(0).expect("a Vulkan 1.3 GPU exists");
    w.streams.insert("default".to_owned(), ctx.default_stream());
    w.drv = Some(ctx);
}

#[given(regex = r"^a default stream$")]
fn default_stream_exists(w: &mut World) {
    assert!(
        w.streams.contains_key("default"),
        "the context created a default stream"
    );
}

#[given(regex = r"^at least one Vulkan 1\.3 GPU$")]
fn at_least_one_gpu(_w: &mut World) {
    let count = driver::CudaContext::device_count().expect("Vulkan can be queried");
    assert!(count >= 1, "this scenario needs a Vulkan 1.3 GPU");
}

#[given(regex = r#"^a second stream from the same context as "(\w+)"$"#)]
fn second_stream(w: &mut World, name: String) {
    let stream = w.ctx().new_stream().expect("a second stream");
    w.streams.insert(name, stream);
}

#[given(regex = r#"^a zeroed buffer "(\w+)" of (\d+) (\w+) elements$"#)]
fn given_zeroed(w: &mut World, name: String, len: usize, ty: String) {
    new_zeroed(w, &name, len, Ty::parse(&ty), "default");
}

#[given(regex = r#"^an empty buffer "(\w+)" of (\w+)$"#)]
fn given_empty(w: &mut World, name: String, ty: String) {
    new_zeroed(w, &name, 0, Ty::parse(&ty), "default");
}

#[given(regex = r#"^a buffer "(\w+)" of (\d+) (\w+) elements holding (.+)$"#)]
fn given_holding(w: &mut World, name: String, len: usize, ty: String, values: String) {
    let values = parse_list(&values);
    assert_eq!(values.len(), len, "the list gives one value per element");
    new_uploaded(w, &name, Ty::parse(&ty), &values, "default");
}

#[given(regex = r#"^a buffer "(\w+)" of (\d+) (\w+) elements all set to (\S+)$"#)]
fn given_all_set(w: &mut World, name: String, len: usize, ty: String, value: String) {
    let values = vec![Scalar::parse(&value); len];
    new_uploaded(w, &name, Ty::parse(&ty), &values, "default");
}

#[given(regex = r#"^a buffer "(\w+)" of (\d+) (\w+) elements counting up from (\d+)$"#)]
fn given_counting(w: &mut World, name: String, len: usize, ty: String, start: i128) {
    let values: Vec<Scalar> = (0..len).map(|i| Scalar::Int(start + i as i128)).collect();
    new_uploaded(w, &name, Ty::parse(&ty), &values, "default");
}

#[given(regex = r"^host data (.+) of type (\w+)$")]
fn given_host(w: &mut World, values: String, ty: String) {
    w.host = parse_list(&values);
    w.host_ty = Some(Ty::parse(&ty));
}

#[given(regex = r"^host data of (\d+) (\w+) values counting up by (\d+)$")]
fn given_host_counting(w: &mut World, len: usize, ty: String, increment: i128) {
    w.host = (0..len)
        .map(|i| Scalar::Int(i as i128 * increment))
        .collect();
    w.host_ty = Some(Ty::parse(&ty));
}

#[given(regex = r#"^the kernel "(\w+)"$"#)]
fn given_kernel(w: &mut World, name: String) {
    load_kernel(w, &name);
}

// ---- Context description -------------------------------------------------------------

#[when(regex = r"^I create a context on ordinal (\d+)$")]
fn when_context_on(w: &mut World, ordinal: usize) {
    let outcome = driver::CudaContext::new(ordinal);
    w.drv = w.record(outcome);
}

#[when(regex = r"^I create a context on the ordinal after the last GPU$")]
fn when_context_after_last(w: &mut World) {
    let count = driver::CudaContext::device_count().expect("Vulkan can be queried");
    when_context_on(w, count as usize);
}

#[then(regex = r"^the device count is at least (\d+)$")]
fn then_device_count(_w: &mut World, least: i32) {
    assert!(driver::CudaContext::device_count().unwrap() >= least);
}

#[then(regex = r"^the context ordinal is (\d+)$")]
fn then_context_ordinal(w: &mut World, ordinal: usize) {
    assert_eq!(w.ctx().ordinal(), ordinal);
}

#[then(regex = r"^the context name is not empty$")]
fn then_context_name(w: &mut World) {
    assert!(!w.ctx().name().unwrap().is_empty());
}

#[then(regex = r"^the context UUID is not all zeros$")]
fn then_context_uuid(w: &mut World) {
    assert!(w.ctx().uuid().unwrap().iter().any(|&b| b != 0));
}

#[then(regex = r"^the context total memory is greater than 0$")]
fn then_context_memory(w: &mut World) {
    assert!(w.ctx().total_mem().unwrap() > 0);
}

#[then(regex = r#"^the buffer "(\w+)" reads the values of "(\w+)"$"#)]
fn then_reads_values_of(w: &mut World, left: String, right: String) {
    then_buffers_equal(w, left, right);
}

#[when(regex = r#"^I load a SPIR-V module from "([^"]+)"$"#)]
fn when_load_file(w: &mut World, path: String) {
    let outcome = w.ctx().load_module(Spirv::from_file(path)).map(|_| ());
    let _ = w.record(outcome);
}

#[when(regex = r"^I load a SPIR-V module from the bytes (\d+), (\d+) and (\d+)$")]
fn when_load_bytes(w: &mut World, a: u8, b: u8, c: u8) {
    let outcome = w
        .ctx()
        .load_module(Spirv::from_binary(vec![a, b, c]))
        .map(|_| ());
    let _ = w.record(outcome);
}

#[when(regex = r#"^I look up the function "(\w+)" in the kernel "(\w+)"$"#)]
fn when_look_up(w: &mut World, function: String, kernel: String) {
    let bytes = w.spirv[&kernel].clone();
    let outcome = w
        .ctx()
        .load_module(Spirv::from_binary(bytes))
        .and_then(|module| module.load_function(&function))
        .map(|_| ());
    let _ = w.record(outcome);
}

// ---- Whens: allocation -----------------------------------------------------------------

#[when(regex = r#"^I allocate (\d+) zeroed (\w+) elements as "(\w+)"$"#)]
fn when_allocate(w: &mut World, len: usize, ty: String, name: String) {
    new_zeroed(w, &name, len, Ty::parse(&ty), "default");
}

#[when(regex = r#"^I allocate (\d+) zeroed (\w+) elements as "(\w+)" on the "(\w+)" stream$"#)]
fn when_allocate_on(w: &mut World, len: usize, ty: String, name: String, stream: String) {
    new_zeroed(w, &name, len, Ty::parse(&ty), &stream);
}

#[when(regex = r#"^I allocate (\d+) uninitialized (\w+) elements as "(\w+)"$"#)]
fn when_allocate_uninit(w: &mut World, len: usize, ty: String, name: String) {
    let stream = w.default_stream();
    let ty = Ty::parse(&ty);
    let outcome = with_ty!(ty, T => unsafe { stream.alloc::<T>(len) }.map(T::wrap_driver));
    let buf = w.record(outcome).expect("alloc succeeds");
    w.dbufs.insert(name, buf);
}

#[when(regex = r"^I try to allocate more bytes than the device has$")]
fn when_allocate_too_much(w: &mut World) {
    let stream = w.default_stream();
    let total = w.ctx().total_mem().expect("total memory is reported");
    // Not `total + 1`: on a unified-memory GPU the driver spills that into system RAM, and filling
    // it with zeros exhausted 15 GB and got the desktop OOM-killed. Four times the device-local heap
    // is beyond VRAM plus GTT, so the kernel refuses it up front.
    let outcome = stream.alloc_zeros::<u8>(total.saturating_mul(4));
    let _ = w.record(outcome.map(|_| ()));
}

#[when(regex = r"^I create and drop (\d+) buffers of (\d+) KiB each$")]
fn when_create_drop(w: &mut World, count: usize, kib: usize) {
    let stream = w.default_stream();
    for _ in 0..count {
        let outcome = stream.alloc_zeros::<u8>(kib * 1024);
        if let Err(e) = outcome {
            w.error = Some(format!("{e:?}"));
            return;
        }
    }
    w.error = None;
}

#[when(regex = r"^I then hold (\d+) buffers of (\d+) MiB at the same time$")]
fn when_hold(w: &mut World, count: usize, mib: usize) {
    let stream = w.default_stream();
    let held: Result<Vec<_>, _> = (0..count)
        .map(|_| stream.alloc_zeros::<u8>(mib << 20))
        .collect();
    w.error = held.err().map(|e| format!("{e:?}"));
}

#[when(regex = r"^I drop the stream and the context$")]
fn when_drop_context(w: &mut World) {
    w.streams.clear();
    w.drv = None;
}

// ---- Whens: host data and copies -------------------------------------------------------

#[when(regex = r#"^I copy the host data to the device as "(\w+)"$"#)]
fn when_host_to_device(w: &mut World, name: String) {
    let ty = w.host_ty.expect("host data has a type");
    let values = w.host.clone();
    new_uploaded(w, &name, ty, &values, "default");
}

#[when(regex = r"^I copy the host data to the device and back$")]
fn when_host_round_trip(w: &mut World) {
    let ty = w.host_ty.expect("host data has a type");
    let stream = w.default_stream();
    let values = w.host.clone();
    let outcome = with_ty!(ty, T => {
        upload_driver::<T>(&stream, &values)
            .and_then(|buf| stream.clone_dtoh(&buf).map_err(|e| format!("{e:?}")))
            .map(|host| host.into_iter().map(Element::to_scalar).collect::<Vec<_>>())
    });
    w.result = outcome.expect("round trip succeeds");
    w.result_ty = Some(ty);
}

#[when(regex = r#"^I write (.+) into "(\w+)"$"#)]
fn when_write(w: &mut World, values: String, name: String) {
    let values = parse_list(&values);
    let stream = w.default_stream();
    let buf = w
        .dbufs
        .get_mut(&name)
        .unwrap_or_else(|| panic!("no buffer {name:?}"));
    let outcome = buf.write_on(&stream, &values);
    let _ = w.record(outcome);
}

#[when(regex = r#"^I write (.+) into "(\w+)" starting at index (\d+)$"#)]
fn when_write_at(w: &mut World, values: String, name: String, start: usize) {
    let values = parse_list(&values);
    let stream = w.default_stream();
    let buf = w
        .dbufs
        .get_mut(&name)
        .unwrap_or_else(|| panic!("no buffer {name:?}"));
    let outcome = buf.write_at_on(&stream, start, &values);
    let _ = w.record(outcome);
}

#[when(regex = r#"^I write (.+) into a new buffer "(\w+)" on the "(\w+)" stream$"#)]
fn when_write_new_buffer(w: &mut World, values: String, name: String, stream: String) {
    let values = parse_list(&values);
    new_uploaded(w, &name, Ty::U8, &values, &stream);
}

#[when(regex = r#"^I fill "(\w+)" with (\S+) on the default stream$"#)]
fn when_fill_default(w: &mut World, name: String, value: String) {
    fill(w, &name, &value, "default");
}

#[when(regex = r#"^I fill "(\w+)" with (\S+) on the "(\w+)" stream$"#)]
fn when_fill_on(w: &mut World, name: String, value: String, stream: String) {
    fill(w, &name, &value, &stream);
}

fn fill(w: &mut World, name: &str, value: &str, stream_name: &str) {
    let stream = w.stream(stream_name);
    let buf = w
        .dbufs
        .get_mut(name)
        .unwrap_or_else(|| panic!("no buffer {name:?}"));
    let len = buf.len();
    let values = vec![Scalar::parse(value); len];
    let outcome = buf.write_on(&stream, &values);
    let _ = w.record(outcome);
}

#[when(regex = r#"^I copy an empty host slice into "(\w+)"$"#)]
fn when_copy_empty(w: &mut World, name: String) {
    let stream = w.default_stream();
    let buf = w
        .dbufs
        .get_mut(&name)
        .unwrap_or_else(|| panic!("no buffer {name:?}"));
    let outcome = buf.write_on(&stream, &[]);
    let _ = w.record(outcome);
}

#[when(regex = r#"^I read "(\w+)" back to the host$"#)]
fn when_read_back(w: &mut World, name: String) {
    let (values, ty) = w.read_named(&name);
    w.result = values;
    w.result_ty = Some(ty);
}

#[when(regex = r#"^I read "(\w+)" back on the "(\w+)" stream$"#)]
fn when_read_back_on(w: &mut World, name: String, stream: String) {
    let stream = w.stream(&stream);
    read_into_result(w, &name, &stream);
}

#[when(regex = r#"^I read "(\w+)" back on the default stream$"#)]
fn when_read_back_default(w: &mut World, name: String) {
    let stream = w.default_stream();
    read_into_result(w, &name, &stream);
}

#[when(regex = r#"^I copy "(\w+)" into a host array of (\d+) (\w+)$"#)]
fn when_copy_to_array(w: &mut World, name: String, len: usize, ty: String) {
    let stream = w.default_stream();
    let buf = w.dbuf(&name);
    let want = Ty::parse(&ty);
    assert_eq!(buf.ty(), want, "the host array has the buffer's type");
    let mut host = vec![Scalar::Int(0); len];
    let outcome = download_into(buf, &stream, &mut host);
    let _ = w.record(outcome);
    w.result = host;
    w.result_ty = Some(want);
}

/// Copies a driver buffer into a host array of the same type, using `memcpy_dtoh`.
fn download_into(
    buf: &DBuf,
    stream: &Arc<driver::CudaStream>,
    host: &mut [Scalar],
) -> Result<(), String> {
    on_buf!(buf, s => download_typed(s, stream, host))
}

fn download_typed<T: Element>(
    slice: &driver::CudaSlice<T>,
    stream: &Arc<driver::CudaStream>,
    host: &mut [Scalar],
) -> Result<(), String> {
    let mut typed: Vec<T> = host.iter().map(|v| T::from_scalar(*v)).collect();
    stream
        .memcpy_dtoh(slice, &mut typed)
        .map_err(|e| format!("{e:?}"))?;
    for (dst, src) in host.iter_mut().zip(typed) {
        *dst = src.to_scalar();
    }
    Ok(())
}

#[when(regex = r#"^I copy "(\w+)" into a host slice of (\d+) (\w+) elements$"#)]
fn when_copy_panics(w: &mut World, name: String, len: usize, _ty: String) {
    let stream = w.default_stream();
    let buf = w.dbuf(&name);
    let mut host = vec![Scalar::Int(0); len];
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = download_into(buf, &stream, &mut host);
    }))
    .is_err();
    w.panicked = panicked;
}

#[when(
    regex = r#"^I copy the (\d+) elements starting at index (\d+) of "(\w+)" into the (\d+) elements starting at index (\d+) of "(\w+)"$"#
)]
fn when_copy_views(
    w: &mut World,
    count: usize,
    from: usize,
    src: String,
    count_dst: usize,
    to: usize,
    dst: String,
) {
    assert_eq!(
        count, count_dst,
        "both sides copy the same number of elements"
    );
    let stream = w.default_stream();
    let mut dst_buf = w.dbufs.remove(&dst).expect("destination exists");
    let outcome = {
        let src_buf = w.dbufs.get(&src).expect("source exists");
        copy_views(src_buf, from, &mut dst_buf, to, count, &stream)
    };
    w.dbufs.insert(dst, dst_buf);
    let _ = w.record(outcome);
}

/// Device-to-device copy between views of two buffers of the same type.
fn copy_views(
    src: &DBuf,
    from: usize,
    dst: &mut DBuf,
    to: usize,
    count: usize,
    stream: &Arc<driver::CudaStream>,
) -> Result<(), String> {
    macro_rules! pair {
        ($($v:ident),*) => {
            match (src, dst) {
                $((DBuf::$v(a), DBuf::$v(b)) => {
                    let view_src = a.slice(from..from + count);
                    let mut view_dst = b.slice_mut(to..to + count);
                    stream.memcpy_dtod(&view_src, &mut view_dst).map_err(|e| format!("{e:?}"))
                })*
                _ => Err("buffers have different element types".to_owned()),
            }
        };
    }
    pair!(U8, U16, U32, U64, I32, I64, F32, F64)
}

#[when(regex = r#"^I make a device copy of "(\w+)" as "(\w+)"$"#)]
fn when_clone_dtod(w: &mut World, src: String, dst: String) {
    let stream = w.default_stream();
    let buf = w.dbuf(&src);
    let copy = on_buf!(buf, s => clone_typed(&stream, s));
    let copy = w.record(copy).expect("clone succeeds");
    w.dbufs.insert(dst, copy);
}

#[when(
    regex = r#"^I zero the (\d+) elements starting at index (\d+) of "(\w+)" with memset_zeros$"#
)]
fn when_memset(w: &mut World, count: usize, from: usize, name: String) {
    let stream = w.default_stream();
    let buf = w.dbuf_mut(&name);
    let outcome = on_buf!(buf, s => {
        let mut view = s.slice_mut(from..from + count);
        stream.memset_zeros(&mut view).map_err(|e| format!("{e:?}"))
    });
    let _ = w.record(outcome);
}

#[when(regex = r#"^I copy "(\w+)" to a new zeroed buffer "(\w+)" on the "(\w+)" stream$"#)]
fn when_copy_new_on(w: &mut World, src: String, dst: String, stream: String) {
    let stream = w.stream(&stream);
    copy_to_new(w, &src, &dst, &stream, false);
}

#[when(regex = r#"^I copy "(\w+)" to a new zeroed buffer "(\w+)" and synchronize the stream$"#)]
fn when_copy_new_sync(w: &mut World, src: String, dst: String) {
    let stream = w.default_stream();
    copy_to_new(w, &src, &dst, &stream, true);
}

fn copy_to_new(w: &mut World, src: &str, dst: &str, stream: &Arc<driver::CudaStream>, sync: bool) {
    let buf = w.dbuf(src);
    let copy = on_buf!(buf, s => copy_new_typed(stream, s));
    if let Some(copy) = w.record(copy) {
        w.dbufs.insert(dst.to_owned(), copy);
    }
    if sync {
        let outcome = stream.synchronize();
        let _ = w.record(outcome);
    }
}

/// A new buffer holding a device copy of `src`.
fn clone_typed<T: Element>(
    stream: &Arc<driver::CudaStream>,
    src: &driver::CudaSlice<T>,
) -> Result<DBuf, String> {
    stream
        .clone_dtod(src)
        .map(wrap)
        .map_err(|e| format!("{e:?}"))
}

/// A new zeroed buffer, filled by a device copy of `src`.
fn copy_new_typed<T: Element>(
    stream: &Arc<driver::CudaStream>,
    src: &driver::CudaSlice<T>,
) -> Result<DBuf, String> {
    let mut out = stream
        .alloc_zeros::<T>(src.len())
        .map_err(|e| format!("{e:?}"))?;
    stream
        .memcpy_dtod(src, &mut out)
        .map_err(|e| format!("{e:?}"))?;
    Ok(wrap(out))
}

// ---- Whens: launches -------------------------------------------------------------------

#[when(regex = r#"^I launch "(\w+)" on "(\w+)" with n (\d+)$"#)]
fn when_launch(w: &mut World, kernel: String, buffer: String, n: u32) {
    launch_on(w, &kernel, &buffer, None, n, None, 0, false);
}

#[when(regex = r#"^I launch "(\w+)" on "(\w+)" with n (\d+) and block \((\d+), 1, 1\)$"#)]
fn when_launch_block(w: &mut World, kernel: String, buffer: String, n: u32, block: u32) {
    launch_on(w, &kernel, &buffer, None, n, Some(block), 0, false);
}

#[when(regex = r#"^I launch "(\w+)" on "(\w+)" with n (\d+) and shared memory (\d+) bytes$"#)]
fn when_launch_shared(w: &mut World, kernel: String, buffer: String, n: u32, shared: u32) {
    launch_on(w, &kernel, &buffer, None, n, None, shared, false);
}

#[when(regex = r#"^I launch "(\w+)" on "(\w+)" with n (\d+) and 300 bytes of extra arguments$"#)]
fn when_launch_extra(w: &mut World, kernel: String, buffer: String, n: u32) {
    launch_on(w, &kernel, &buffer, None, n, None, 0, true);
}

#[when(
    regex = r#"^I launch "(\w+)" on the (\d+) elements starting at index (\d+) of "(\w+)" with n (\d+)$"#
)]
fn when_launch_view(
    w: &mut World,
    kernel: String,
    count: usize,
    start: usize,
    buffer: String,
    n: u32,
) {
    launch_on(w, &kernel, &buffer, Some((start, count)), n, None, 0, false);
}

#[when(regex = r#"^I launch "(\w+)" (\d+) times in a row on "(\w+)" with n (\d+)$"#)]
fn when_launch_repeat(w: &mut World, kernel: String, times: usize, buffer: String, n: u32) {
    for _ in 0..times {
        launch_on(w, &kernel, &buffer, None, n, None, 0, false);
    }
}

#[when(
    regex = r#"^I launch "(\w+)" on "(\w+)" with n (\d+) and then drop "(\w+)" before the stream finishes$"#
)]
fn when_launch_then_drop(w: &mut World, kernel: String, buffer: String, n: u32, dropped: String) {
    assert_eq!(buffer, dropped, "the launched buffer is the dropped one");
    launch_on(w, &kernel, &buffer, None, n, None, 0, false);
    w.dbufs.remove(&dropped);
    let stream = w.default_stream();
    let outcome = stream.synchronize();
    let _ = w.record(outcome);
}

#[when(regex = r#"^I launch "axpy" with y "(\w+)", x "(\w+)", a (\S+) and n (\d+)$"#)]
fn when_axpy(w: &mut World, y: String, x: String, a: String, n: u32) {
    let a: f32 = a.parse().expect("a float");
    launch_axpy(w, &y, &x, a, n);
}

#[when(regex = r#"^I launch "add_i64" on "(\w+)" with k (-?\d+) and n (\d+)$"#)]
fn when_add_i64(w: &mut World, name: String, k: i64, n: u32) {
    launch_add_i64(w, &name, k, n);
}

#[when(
    regex = r#"^I repeat (\d+) times: launch "double_all" on "(\w+)" with n (\d+), copy "(\w+)" to "(\w+)", then launch "double_all" on "(\w+)" with n (\d+)$"#
)]
#[allow(clippy::too_many_arguments)]
fn when_repeat_rounds(
    w: &mut World,
    rounds: usize,
    a: String,
    n: u32,
    copy_src: String,
    copy_dst: String,
    b: String,
    _n2: u32,
) {
    assert_eq!(a, copy_src);
    assert_eq!(b, copy_dst);
    let stream = w.default_stream();
    for _ in 0..rounds {
        launch_on(w, "double_all", &a, None, n, None, 0, false);
        copy_into(w, &a, &b, &stream);
        launch_on(w, "double_all", &b, None, n, None, 0, false);
    }
}

/// Device-to-device copy of a whole buffer into an existing buffer of the same type.
fn copy_into(w: &mut World, src: &str, dst: &str, stream: &Arc<driver::CudaStream>) {
    let mut target = w.dbufs.remove(dst).expect("destination exists");
    let outcome = {
        let source = w.dbufs.get(src).expect("source exists");
        let len = source.len();
        copy_views(source, 0, &mut target, 0, len, stream)
    };
    w.dbufs.insert(dst.to_owned(), target);
    let _ = w.record(outcome);
}

#[when(regex = r"^eight threads each allocate, copy and read back their own buffer$")]
fn when_threads(w: &mut World) {
    let ctx = w.ctx().clone();
    let ok: Vec<bool> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8u32)
            .map(|t| {
                let ctx = ctx.clone();
                scope.spawn(move || {
                    let stream = ctx.new_stream().expect("a stream");
                    let values = vec![t; 256];
                    let buf = stream.clone_htod(&values).expect("upload");
                    stream.clone_dtoh(&buf).expect("read") == values
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("thread ran"))
            .collect()
    });
    w.all_ok = ok.iter().all(|&b| b);
}

#[then(regex = r"^every thread reads back the values it wrote$")]
fn then_threads_ok(w: &mut World) {
    assert!(w.all_ok, "some thread read back different values");
}

#[when(regex = r"^I queue a copy between two buffers and drop both, (\d+) times in a loop$")]
fn when_queue_and_drop(w: &mut World, times: usize) {
    let stream = w.default_stream();
    for _ in 0..times {
        let a = match upload_driver::<u32>(&stream, &vec![Scalar::Int(1); 256]) {
            Ok(a) => a,
            Err(e) => {
                w.error = Some(e);
                return;
            }
        };
        if let Err(e) = stream.clone_dtod(&a) {
            w.error = Some(format!("{e:?}"));
            return;
        }
    }
    w.error = None;
}

// ---- Whens: streams --------------------------------------------------------------------

#[when(regex = r#"^I fork the default stream as "(\w+)"$"#)]
fn when_fork(w: &mut World, name: String) {
    let parent = w.default_stream();
    let fork = parent.fork().expect("fork succeeds");
    w.streams.insert(name, fork);
}

#[then(regex = r#"^the result equals the values of "(\w+)"$"#)]
fn then_result_equals_buffer(w: &mut World, name: String) {
    let (values, _) = w.read_named(&name);
    assert_eq!(w.result, values, "the result matches {name:?}");
}

// ---- Thens: buffers --------------------------------------------------------------------

#[then(regex = r#"^the buffer "(\w+)" has length (\d+)$"#)]
fn then_length(w: &mut World, name: String, len: usize) {
    assert_eq!(w.dbuf(&name).len(), len);
}

#[then(regex = r#"^the buffer "(\w+)" has byte size (\d+)$"#)]
fn then_byte_size(w: &mut World, name: String, bytes: usize) {
    assert_eq!(w.dbuf(&name).num_bytes(), bytes);
}

#[then(regex = r#"^the buffer "(\w+)" is empty$"#)]
fn then_empty(w: &mut World, name: String) {
    assert_eq!(w.dbuf(&name).len(), 0);
}

#[then(regex = r#"^the buffer "(\w+)" has (\d+) zero elements$"#)]
fn then_zeros(w: &mut World, name: String, count: usize) {
    let (values, ty) = w.read_named(&name);
    assert_eq!(values.len(), count);
    expect_every(&values, ty, Scalar::Int(0)).expect("every element is zero");
}

#[then(regex = r#"^the buffer "(\w+)" reads ([-\d.].*)$"#)]
fn then_reads(w: &mut World, name: String, expected: String) {
    let (values, ty) = w.read_named(&name);
    expect_values(&values, ty, &parse_list(&expected)).expect("values match");
}

#[then(regex = r#"^the buffer "(\w+)" reads the host data$"#)]
fn then_reads_host(w: &mut World, name: String) {
    let (values, ty) = w.read_named(&name);
    let host = w.host.clone();
    expect_values(&values, ty, &host).expect("values match the host data");
}

#[then(regex = r#"^the buffer "(\w+)" holds (\S+) in every element$"#)]
fn then_holds(w: &mut World, name: String, value: String) {
    let (values, ty) = w.read_named(&name);
    expect_every(&values, ty, Scalar::parse(&value)).expect("every element matches");
}

#[then(regex = r#"^the buffer "(\w+)" holds its own index in every element$"#)]
fn then_holds_index(w: &mut World, name: String) {
    let (values, ty) = w.read_named(&name);
    expect_by_index(&values, ty, |i| Scalar::Int(i as i128)).expect("each element holds its index");
}

#[then(regex = r#"^the buffer "(\w+)" equals ([\d.]+) times the index plus ([\d.]+)$"#)]
fn then_index_formula(w: &mut World, name: String, a: String, b: String) {
    let (values, ty) = w.read_named(&name);
    let (a, b) = (Scalar::parse(&a).as_f64(), Scalar::parse(&b).as_f64());
    expect_by_index(&values, ty, |i| Scalar::Float(a * i as f64 + b)).expect("formula holds");
}

#[then(regex = r#"^the buffer "(\w+)" equals the buffer "(\w+)"$"#)]
fn then_buffers_equal(w: &mut World, left: String, right: String) {
    let (a, ty) = w.read_named(&left);
    let (b, _) = w.read_named(&right);
    expect_values(&a, ty, &b).expect("the buffers match");
}

#[then(regex = r#"^the buffers "(\w+)" and "(\w+)" have different Vulkan buffer handles$"#)]
fn then_different_buffer_handles(w: &mut World, a: String, b: String) {
    let (x, y) = (w.dbuf(&a).vk_buffer(), w.dbuf(&b).vk_buffer());
    assert_ne!(x, y, "buffers {a:?} and {b:?} share a handle");
    assert!(!x.is_null() && !y.is_null(), "a handle is null");
}

#[then(
    regex = r#"^the elements (\d+) to (\d+) of "(\w+)" equal ([\d.]+) times the index plus ([\d.]+)$"#
)]
fn then_range_formula(w: &mut World, from: usize, to: usize, name: String, a: String, b: String) {
    let (values, ty) = w.read_named(&name);
    let (a, b) = (Scalar::parse(&a).as_f64(), Scalar::parse(&b).as_f64());
    expect_by_index(&values[from..=to], ty, |i| {
        Scalar::Float(a * (from + i) as f64 + b)
    })
    .expect("range formula holds");
}

#[then(regex = r#"^the elements (\d+) to (\d+) of "(\w+)" still hold (\S+)$"#)]
fn then_range_holds(w: &mut World, from: usize, to: usize, name: String, value: String) {
    let (values, ty) = w.read_named(&name);
    expect_every(&values[from..=to], ty, Scalar::parse(&value)).expect("range holds the value");
}

// ---- Thens: results and errors ---------------------------------------------------------

#[then(regex = r"^the result reads ([-\d.].*)$")]
fn then_result_reads(w: &mut World, expected: String) {
    let ty = w.result_ty.expect("a result was read");
    let result = w.result.clone();
    expect_values(&result, ty, &parse_list(&expected)).expect("result matches");
}

#[then(regex = r"^the result holds (\S+) in every element$")]
fn then_result_holds(w: &mut World, value: String) {
    let ty = w.result_ty.expect("a result was read");
    let result = w.result.clone();
    expect_every(&result, ty, Scalar::parse(&value)).expect("every element matches");
}

#[then(regex = r"^the result equals the host data$")]
fn then_result_equals_host(w: &mut World) {
    let ty = w.result_ty.or(w.host_ty).expect("a type is known");
    let result = w.result.clone();
    let host = w.host.clone();
    expect_values(&result, ty, &host).expect("result matches the host data");
}

#[then(regex = r"^the result is empty$")]
fn then_result_empty(w: &mut World) {
    assert!(
        w.result.is_empty(),
        "the result has {} elements",
        w.result.len()
    );
}

#[then(regex = r"^all allocations succeed$")]
#[then(regex = r"^no error occurs$")]
#[then(regex = r"^the stream finishes without error$")]
#[then(regex = r"^the result is no error$")]
fn then_no_error(w: &mut World) {
    assert!(w.error.is_none(), "unexpected error: {:?}", w.error);
}

#[then(regex = r"^the result is an? (\w+) error$")]
fn then_error_is(w: &mut World, variant: String) {
    let error = w.error.clone().expect("an error was returned");
    assert!(
        error.contains(&variant),
        "expected a {variant} error, got {error}"
    );
}

#[then(regex = r"^the result is an? (\w+) error for (\w+)$")]
fn then_error_for(w: &mut World, variant: String, flag: String) {
    let error = w.error.clone().expect("an error was returned");
    assert!(
        error.contains(&variant) && error.contains(&flag),
        "expected a {variant} error for {flag}, got {error}"
    );
}

#[then(regex = r"^the program panics$")]
fn then_panics(w: &mut World) {
    assert!(w.panicked, "the copy should have panicked");
}
