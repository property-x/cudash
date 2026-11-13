# Streams

A `CudaStream` is ordered GPU work, as in cudarc: everything queued on it runs in call order, and most calls return
before the GPU has finished.

```rust
# use cudash::driver::*;
# fn main() -> Result<(), DriverError> {
let ctx = CudaContext::new(0)?;
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
