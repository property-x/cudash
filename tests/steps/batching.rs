//! Steps for `features/driver_batching.feature`: stream batching (`experimental` feature).
//!
//! A missed flush must fail, not hang or pass by luck. Every blocking call runs on a thread that
//! `within` abandons after a timeout. And to see the GPU run work that nothing waits for, the
//! copy goes to pinned memory whose pointer was taken before the work was queued: the accessors
//! wait (and so flush), the raw pointer doesn't.

use std::{
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

use cucumber::{given, then, when};
use cudash::driver::{
    GpuContext, GpuGraph, GpuSlice, GpuStream, PinnedHostSlice,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};

use super::world::World;

const N: usize = 64;
const RELAXED: CUstreamCaptureMode = CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;
const FLAGS: CUgraphInstantiate_flags = CUgraphInstantiate_flags(0);
const TIMEOUT: Duration = Duration::from_secs(10);

/// Runs `f` on a thread and panics if it hasn't finished after `TIMEOUT`. The thread is left
/// behind, which is fine: the test run exits soon after.
fn within<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(TIMEOUT)
        .unwrap_or_else(|_| panic!("{what} did not finish within {TIMEOUT:?}: a flush is missing"))
}

struct SendPtr(*const u32);
// SAFETY: the pinned memory outlives every poll (the Probe owns it) and is only read.
unsafe impl Send for SendPtr {}

/// A batching stream with `7`s copied into `dev` and then into pinned memory, all still queued.
struct Probe {
    stream: Option<Arc<GpuStream>>,
    /// Another stream, which owns the buffers, so the batching stream can be dropped.
    owner: Arc<GpuStream>,
    dev: GpuSlice<u32>,
    pinned: PinnedHostSlice<u32>,
    ptr: SendPtr,
    graph: Option<GpuGraph>,
}

impl Probe {
    fn new(ctx: &Arc<GpuContext>) -> Probe {
        let owner = ctx.new_stream().unwrap();
        let mut dev = owner.alloc_zeros::<u32>(N).unwrap();
        // SAFETY: filled before it is read.
        let mut pinned = unsafe { ctx.alloc_pinned::<u32>(N) }.unwrap();
        pinned.as_mut_slice().unwrap().fill(0);
        let ptr = SendPtr(pinned.as_ptr().unwrap());
        owner.synchronize().unwrap();
        let stream = ctx.new_stream().unwrap();
        stream.set_batching(true).unwrap();
        stream.memcpy_htod(&[7u32; N], &mut dev).unwrap();
        stream.memcpy_dtoh(&dev, &mut pinned).unwrap();
        let probe = Probe {
            stream: Some(stream),
            owner,
            dev,
            pinned,
            ptr,
            graph: None,
        };
        // Nothing may have run: the batch is still open.
        thread::sleep(Duration::from_millis(100));
        assert_eq!(probe.seen(), 0, "the GPU ran work that was never submitted");
        probe
    }

    fn seen(&self) -> u32 {
        // SAFETY: valid pinned memory of N u32; the GPU may write it, so read volatile.
        unsafe { std::ptr::read_volatile(self.ptr.0.add(N - 1)) }
    }

    fn stream(&self) -> Arc<GpuStream> {
        self.stream.clone().expect("the stream is still alive")
    }

    /// Waits for the GPU to write the 7s, without calling anything that flushes.
    fn done(&self) -> bool {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(5) {
            if self.seen() == 7 {
                return true;
            }
            thread::sleep(Duration::from_millis(5));
        }
        false
    }
}

fn probes(w: &mut World) -> &mut Vec<Probe> {
    w.extra
        .get_or_insert_with(|| Box::new(Vec::<Probe>::new()))
        .downcast_mut()
        .expect("batching state")
}

#[given(regex = r"^(\d+) batching streams? each with a copy queued to pinned host memory$")]
fn given_probes(w: &mut World, count: usize) {
    let ctx = w.ctx().clone();
    for _ in 0..count {
        let probe = Probe::new(&ctx);
        probes(w).push(probe);
    }
}

#[when(regex = r#"^I use the flush point "([^"]+)"$"#)]
fn when_flush_point(w: &mut World, point: String) {
    let ctx = w.ctx().clone();
    let probe = &mut probes(w)[0];
    let stream = probe.stream();
    match point.as_str() {
        "flush" => within(&point, move || stream.flush().unwrap()),
        "set_batching(false)" => within(&point, move || stream.set_batching(false).unwrap()),
        "synchronize" => within(&point, move || stream.synchronize().unwrap()),
        "ctx.synchronize" => within(&point, move || ctx.synchronize().unwrap()),
        "fork" => {
            within(&point, move || drop(stream.fork().unwrap()));
        }
        "begin_capture" => within(&point, move || stream.begin_capture(RELAXED).unwrap()),
        "drop" => {
            drop(stream);
            probe.stream = None;
        }
        "300 operations" => {
            let mut scratch = probe.owner.alloc_zeros::<u32>(N).unwrap();
            within(&point, move || {
                for _ in 0..300 {
                    stream.memset_zeros(&mut scratch).unwrap();
                }
            });
        }
        "frame" => within(&point, move || {
            stream.frame().unwrap();
            stream.stop_frame().unwrap();
        }),
        other => panic!("unknown flush point {other:?}"),
    }
}

#[then(regex = r"^the GPU finishes every queued copy without anything waiting for it$")]
fn then_finished(w: &mut World) {
    let all = probes(w);
    let ok: Vec<bool> = all.iter().map(Probe::done).collect();
    // Clean up before failing, so the device is idle for the next scenario.
    for p in all.iter() {
        if let Some(s) = &p.stream {
            let s = s.clone();
            let _ = within("cleanup", move || {
                let _ = s.synchronize();
                let _ = s.end_capture(FLAGS);
            });
        }
    }
    assert!(ok.iter().all(|&b| b), "queued copies finished: {ok:?}");
}

/// What waiting on the queued data does, for each kind of waiter.
#[then(regex = r#"^"([^"]+)" returns the queued data within 10 seconds$"#)]
fn then_waiter(w: &mut World, waiter: String) {
    let ctx = w.ctx().clone();
    let probe = &mut probes(w)[0];
    let stream = probe.stream();
    let owner = probe.owner.clone();
    let dev = &probe.dev;
    let got: Vec<u32> = match waiter.as_str() {
        "clone_dtoh on the batching stream" => {
            let (s, d) = (stream.clone(), dev as *const GpuSlice<u32> as usize);
            // SAFETY: `dev` outlives the scoped wait below.
            within(&waiter, move || s.clone_dtoh(unsafe { &*(d as *const GpuSlice<u32>) }).unwrap())
        }
        "a read on another stream" => {
            let d = dev as *const GpuSlice<u32> as usize;
            within(&waiter, move || {
                let other = ctx.new_stream().unwrap();
                // SAFETY: as above.
                other.clone_dtoh(unsafe { &*(d as *const GpuSlice<u32>) }).unwrap()
            })
        }
        "a read on the buffer's own stream" => {
            let d = dev as *const GpuSlice<u32> as usize;
            // SAFETY: as above.
            within(&waiter, move || owner.clone_dtoh(unsafe { &*(d as *const GpuSlice<u32>) }).unwrap())
        }
        "the pinned slice" => {
            let p = &probe.pinned as *const PinnedHostSlice<u32> as usize;
            // SAFETY: as above.
            within(&waiter, move || {
                unsafe { &*(p as *const PinnedHostSlice<u32>) }.as_slice().unwrap().to_vec()
            })
        }
        "a read after the stream is dropped" => {
            drop(stream);
            probe.stream = None;
            let d = dev as *const GpuSlice<u32> as usize;
            // SAFETY: as above.
            within(&waiter, move || owner.clone_dtoh(unsafe { &*(d as *const GpuSlice<u32>) }).unwrap())
        }
        other => panic!("unknown waiter {other:?}"),
    };
    assert_eq!(got, vec![7u32; N]);
}

#[given(regex = r"^a batching stream and an unbatched one$")]
fn given_pair(w: &mut World) {
    let ctx = w.ctx().clone();
    let a = ctx.new_stream().unwrap();
    let b = ctx.new_stream().unwrap();
    a.set_batching(true).unwrap();
    w.streams.insert("batching".into(), a);
    w.streams.insert("plain".into(), b);
}

#[when(regex = r"^two batching streams read each other's buffers in both directions, 300 times each$")]
fn when_deadlock(w: &mut World) {
    let ctx = w.ctx().clone();
    within("the exchange", move || {
        let (s1, s2) = (ctx.new_stream().unwrap(), ctx.new_stream().unwrap());
        s1.set_batching(true).unwrap();
        s2.set_batching(true).unwrap();
        // x is written by s1 and read by s2; y the other way round.
        let x = Mutex::new(s1.alloc_zeros::<u32>(N).unwrap());
        let y = Mutex::new(s2.alloc_zeros::<u32>(N).unwrap());
        let mut z1 = s1.alloc_zeros::<u32>(N).unwrap();
        let mut z2 = s2.alloc_zeros::<u32>(N).unwrap();
        thread::scope(|scope| {
            scope.spawn(|| {
                for i in 0..300u32 {
                    s1.memcpy_htod(&[i; N], &mut *x.lock().unwrap()).unwrap();
                    s1.memcpy_dtod(&*y.lock().unwrap(), &mut z1).unwrap();
                }
            });
            scope.spawn(|| {
                for i in 0..300u32 {
                    s2.memcpy_htod(&[i; N], &mut *y.lock().unwrap()).unwrap();
                    s2.memcpy_dtod(&*x.lock().unwrap(), &mut z2).unwrap();
                }
            });
        });
        s1.synchronize().unwrap();
        s2.synchronize().unwrap();
        assert_eq!(s1.clone_dtoh(&*x.lock().unwrap()).unwrap(), [299; N]);
        assert_eq!(s2.clone_dtoh(&*y.lock().unwrap()).unwrap(), [299; N]);
        // Each copy saw some value its writer had queued, never garbage.
        assert!(s1.clone_dtoh(&z1).unwrap().iter().all(|&v| v < 300));
        assert!(s2.clone_dtoh(&z2).unwrap().iter().all(|&v| v < 300));
    });
}

#[then(regex = r"^the exchange completed with the final values$")]
fn then_exchange(_w: &mut World) {}

#[when(regex = r"^I queue (\d+) copies into consecutive elements, each after a write of its index$")]
fn when_ordered(w: &mut World, count: usize) {
    let ctx = w.ctx().clone();
    let out = within("the ordered batch", move || {
        let s = ctx.new_stream().unwrap();
        s.set_batching(true).unwrap();
        let mut tmp = s.alloc_zeros::<u32>(1).unwrap();
        let mut out = s.alloc_zeros::<u32>(count).unwrap();
        for i in 0..count {
            s.memcpy_htod(&[i as u32], &mut tmp).unwrap();
            s.memcpy_dtod(&tmp, &mut out.slice_mut(i..i + 1)).unwrap();
        }
        s.clone_dtoh(&out).unwrap()
    });
    w.extra = Some(Box::new(out));
}

#[then(regex = r"^every element holds its own index$")]
fn then_indices(w: &mut World) {
    let out = w.extra.take().unwrap().downcast::<Vec<u32>>().unwrap();
    let bad = out.iter().enumerate().find(|&(i, &v)| v != i as u32);
    assert!(bad.is_none(), "first element out of order: {bad:?}");
}

#[when(regex = r"^I capture a copy from b to a on a batching stream, queue a write to b, and launch the graph$")]
fn when_graph(w: &mut World) {
    let ctx = w.ctx().clone();
    let (a, graph_ran_after) = within("the graph", move || {
        let own = ctx.new_stream().unwrap();
        let mut b = own.clone_htod(&[9u32; N]).unwrap();
        let mut a = own.alloc_zeros::<u32>(N).unwrap();
        own.synchronize().unwrap();
        let s = ctx.new_stream().unwrap();
        s.set_batching(true).unwrap();
        s.begin_capture(RELAXED).unwrap();
        s.memcpy_dtod(&b, &mut a).unwrap();
        let graph = s.end_capture(FLAGS).unwrap().expect("a graph");
        s.memcpy_htod(&[4u32; N], &mut b).unwrap();
        graph.launch().unwrap();
        (own.clone_dtoh(&a).unwrap(), true)
    });
    assert!(graph_ran_after);
    w.extra = Some(Box::new(a));
}

#[then(regex = r"^a holds the value written before the launch$")]
fn then_graph_data(w: &mut World) {
    let a = w.extra.take().unwrap().downcast::<Vec<u32>>().unwrap();
    assert_eq!(*a, [4u32; N], "the graph ran before the loose batched work");
}

#[when(regex = r"^I write 1, turn batching off, write 2, turn it on, write 3 and read back$")]
fn when_toggle(w: &mut World) {
    let ctx = w.ctx().clone();
    let got = within("toggling", move || {
        let s = ctx.new_stream().unwrap();
        let mut a = s.alloc_zeros::<u32>(N).unwrap();
        let mut seen = Vec::new();
        s.set_batching(true).unwrap();
        s.memcpy_htod(&[1u32; N], &mut a).unwrap();
        s.set_batching(false).unwrap();
        seen.push(s.clone_dtoh(&a).unwrap()[0]);
        s.memcpy_htod(&[2u32; N], &mut a).unwrap();
        s.set_batching(true).unwrap();
        s.memcpy_htod(&[3u32; N], &mut a).unwrap();
        seen.push(s.clone_dtoh(&a).unwrap()[0]);
        seen
    });
    w.extra = Some(Box::new(got));
}

#[then(regex = r"^the reads saw 1 and 3$")]
fn then_toggle(w: &mut World) {
    let got = w.extra.take().unwrap().downcast::<Vec<u32>>().unwrap();
    assert_eq!(*got, [1, 3]);
}
