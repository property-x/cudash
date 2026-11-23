# Streams

A `GpuStream` is ordered GPU work, as in cudarc: everything queued on it runs in call order, and most calls return
before the GPU has finished.

```rust
# use cudash::driver::*;
# fn main() -> Result<(), DriverError> {
let ctx = GpuContext::new(0)?;
let stream = ctx.default_stream();   // shared, like cudarc's default stream
let other = ctx.new_stream()?;       // independent of `stream`
let after = stream.fork()?;          // starts after work already on `stream`

let mut data = stream.alloc_zeros::<u32>(4)?;
other.memcpy_htod(&[1u32; 4], &mut data)?;   // waits for the zeroing on `stream`
assert_eq!(after.clone_dtoh(&data)?, [1; 4]);
stream.synchronize()?;
ctx.synchronize()?;
# Ok(())
# }
```

| Call | What it does |
|---|---|
| `ctx.default_stream()` | The context's default stream |
| `ctx.new_stream()` | A new stream, unordered against the others except through shared buffers |
| `stream.fork()` | A new stream whose work starts after the work already queued on `stream` |
| `stream.synchronize()` | Blocks until everything queued on the stream has finished |
| `ctx.synchronize()` | Blocks until the whole device is idle |

## Ordering across streams

cudarc orders a slice's uses across streams with events. cudash does the same with each stream's Vulkan timeline
semaphore: every buffer remembers its last write and the reads since, and work on another stream waits for them.
Using one slice from several streams is safe without manual synchronization.

A slice belongs to the stream that allocated it, but any stream on the same context can use it.

## How it maps to Vulkan

Each stream is a timeline semaphore. Every operation is one queue submission that waits for the previous value and
for the buffers' earlier uses, then signals the next value. All streams share one Vulkan queue, so a stream that's
waiting can hold up the others, which CUDA streams don't do.

## Batching (experimental)

Each operation costs one queue submission, about 9 µs of CPU time on RADV and 2 µs on NVIDIA, and on RADV each
submission that waits on the one before adds about 20 µs on the GPU. A burst of 16 launches then takes about 550 µs
on RADV and 160 µs on a GTX 1650 Ti, against about 50 µs when they go in one submission.

With the `experimental` feature, `stream.set_batching(true)` records the stream's operations into one command
buffer instead, and submits it when something waits for them:

```rust
stream.set_batching(true)?;
for _ in 0..16 {
    unsafe { stream.launch_builder(&f).arg(&mut data).launch(cfg) }?;
}
let result = stream.clone_dtoh(&data)?; // submits the batch, then waits
```

The batch is submitted by `synchronize` (on the stream or the context), copies to the host, host slices, another
stream using one of its buffers, `fork`, `begin_capture` and graph launches, `stop_frame`, `set_batching(false)`,
dropping the stream, every 256 operations, and `stream.flush()`. Until then the GPU doesn't start on it, so call
`flush()` when nothing will wait for a while, and before submitting your own Vulkan work that depends on the stream.

On a batching stream, errors from submitting, such as a lost device, come from the call that submits rather than
the operation, and memory of dropped buffers is freed after the batch is submitted.

If the same operations run every frame, a [graph](graphs.md) is cheaper still: it's recorded once.

