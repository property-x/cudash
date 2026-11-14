//! Steps for `features/driver_graphs.feature`: stream capture and `CudaGraph` replay.

use std::sync::Arc;

use cucumber::{given, then, when};
use cudash::driver::{
    LaunchConfig, PushKernelArg,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode, CUstreamCaptureStatus},
};

use super::world::World;

const RELAXED: CUstreamCaptureMode = CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;
const FLAGS: CUgraphInstantiate_flags = CUgraphInstantiate_flags(0);

#[given(regex = r"^pinned host memory of 4 f32 values holding (\S+)$")]
fn given_pinned(w: &mut World, value: f32) {
    let mut pinned = unsafe { w.ctx().alloc_pinned::<f32>(4) }.expect("pinned allocation");
    pinned.as_mut_slice().unwrap().fill(value);
    w.pinned = Some(pinned);
}

#[when(regex = r"^I fill the pinned host memory with (\S+)$")]
fn when_refill_pinned(w: &mut World, value: f32) {
    w.pinned
        .as_mut()
        .unwrap()
        .as_mut_slice()
        .unwrap()
        .fill(value);
}

#[when(regex = r"^I begin capturing on the default stream$")]
fn when_begin(w: &mut World) {
    let r = w.default_stream().begin_capture(RELAXED);
    let _ = w.record(r);
}

/// Runs one call on the default stream. `b` is a 4-element f32 buffer, `a` another one.
#[when(regex = r#"^I call "(\w+)" on "(\w+)"$"#)]
fn when_call(w: &mut World, call: String, name: String) {
    let stream = w.default_stream();
    let mut b = match w.dbufs.remove(&name) {
        Some(cudash_buf) => cudash_buf,
        None => panic!("no buffer {name:?}"),
    };
    let super::world::DBuf::F32(b_slice) = &mut b else {
        panic!("{name:?} must be f32")
    };
    let a = match w.dbufs.get("a") {
        Some(super::world::DBuf::F32(a)) => Some(a),
        _ => None,
    };
    let outcome = match call.as_str() {
        "memset_zeros" => stream.memset_zeros(b_slice),
        "memcpy_dtod" => stream.memcpy_dtod(a.expect("buffer a"), b_slice),
        "clone_dtod" => stream.clone_dtod(&*b_slice).map(drop),
        "alloc_zeros" => stream.alloc_zeros::<f32>(4).map(drop),
        "device_ptr" => {
            use cudash::driver::DevicePtr as _;
            let _ = b_slice.device_ptr(&stream);
            Ok(())
        }
        "launch" => {
            let double = w.kernels["double_all"].clone();
            let n = 4i32;
            let mut builder = stream.launch_builder(&double);
            builder.arg(b_slice).arg(&n);
            unsafe { builder.launch(LaunchConfig::for_num_elems(4)) }
        }
        "memcpy_htod_pinned" => stream.memcpy_htod(w.pinned.as_ref().unwrap(), b_slice),
        "image_copy" => stream.clone_to_image(&*b_slice, 2, 2).map(drop),
        "memcpy_htod_pageable" => stream.memcpy_htod(&[0f32; 4][..], b_slice),
        "memcpy_dtoh" => stream.memcpy_dtoh(&*b_slice, &mut [0f32; 4][..]),
        "clone_dtoh" => stream.clone_dtoh(&*b_slice).map(drop),
        "synchronize" => stream.synchronize(),
        "frame" => stream.frame().map(drop),
        "stop_frame" => stream.stop_frame(),
        other => panic!("unknown call {other:?}"),
    };
    w.dbufs.insert(name, b);
    let _ = w.record(outcome);
}

#[then(regex = r"^the capture status is (\w+)$")]
fn then_status(w: &mut World, status: String) {
    let got = w.default_stream().capture_status().unwrap();
    let want = match status.as_str() {
        "NONE" => CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
        "ACTIVE" => CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE,
        "INVALIDATED" => CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED,
        other => panic!("unknown status {other:?}"),
    };
    assert_eq!(got, want);
}

#[when(regex = r#"^I end the capture as "(\w+)"$"#)]
fn when_end(w: &mut World, name: String) {
    let graph = w
        .default_stream()
        .end_capture(FLAGS)
        .expect("end_capture")
        .expect("the capture was valid, so it gives a graph");
    w.graphs.insert(name, Arc::new(graph));
}

#[then(regex = r"^ending the capture gives no graph$")]
fn then_no_graph(w: &mut World) {
    assert!(w.default_stream().end_capture(FLAGS).unwrap().is_none());
}

#[when(regex = r#"^I launch the graph "(\w+)" (\d+) times?$"#)]
fn when_launch(w: &mut World, name: String, times: usize) {
    for _ in 0..times {
        w.graphs[&name].launch().expect("graph launch");
    }
}

#[when(regex = r#"^I launch the graph "(\w+)" and drop it at once$"#)]
fn when_launch_drop(w: &mut World, name: String) {
    let graph = w.graphs.remove(&name).unwrap();
    graph.launch().expect("graph launch");
    drop(graph);
}

#[when(regex = r#"^two threads each launch the graph "(\w+)" (\d+) times$"#)]
fn when_threads(w: &mut World, name: String, times: usize) {
    let graph = w.graphs[&name].clone();
    std::thread::scope(|s| {
        for _ in 0..2 {
            s.spawn(|| (0..times).for_each(|_| graph.launch().expect("graph launch")));
        }
    });
}

#[when(regex = r#"^I begin capturing on a new stream "(\w+)" and drop it$"#)]
fn when_drop_capturing(w: &mut World, name: String) {
    let stream = w.ctx().new_stream().unwrap();
    stream.begin_capture(RELAXED).unwrap();
    w.streams.insert(name.clone(), stream);
    w.streams.remove(&name);
}

#[then(regex = r"^a new stream on the same context still works$")]
fn then_new_stream(w: &mut World) {
    let stream = w.ctx().new_stream().unwrap();
    assert_eq!(
        stream
            .clone_dtoh(&stream.clone_htod(&[3u8]).unwrap())
            .unwrap(),
        [3]
    );
}

#[when(regex = r#"^I double "(\w+)" in the capture$"#)]
fn when_double(w: &mut World, name: String) {
    let stream = w.default_stream();
    let double = w.kernels["double_all"].clone();
    let Some(super::world::DBuf::F32(s)) = w.dbufs.get_mut(&name) else {
        panic!("f32 buffer")
    };
    let n = s.len() as i32;
    let mut builder = stream.launch_builder(&double);
    builder.arg(s).arg(&n);
    unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.unwrap();
}
