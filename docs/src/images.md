# Images

cudarc has no images. cudash has them so you can show results: in your own Vulkan shaders, or in RenderDoc's texture
viewer.

```rust
# use cudash::driver::*;
# fn main() -> Result<(), DriverError> {
# let stream = CudaContext::new(0)?.default_stream();
let mut pixels = stream.clone_htod(&vec![[255u8, 0, 0, 255]; 64 * 32])?;
let mut image = pixels.to_image(64, 32)?;   // a copy, R8G8B8A8_UNORM
stream.memcpy_htod(&vec![[0u8, 0, 255, 255]; 64 * 32], &mut pixels)?;
stream.imgcpy_dtod(&pixels, &mut image)?;   // refreshes it
println!("{:?} {}x{}", image.vk_format(), image.width(), image.height());
# Ok(())
# }
```

| Call | What it does |
|---|---|
| `slice.to_image(width, height)` | Copies the slice into a new `CudaImage<T>`, on the slice's stream |
| `stream.clone_to_image(&slice, width, height)` | The same, on a stream of your choice |
| `stream.imgcpy_dtod(&slice, &mut image)` | Copies a slice into an existing image |
| `slice.as_image(width, height)` | Copies nothing: the image is a view of the slice's memory, so later writes to the slice show in it |

The slice holds `width * height` elements in rows, without padding. The calls panic if the length doesn't match.

## Formats

The format comes from the element type, through the `ImageFormat` trait:

| `T` | Vulkan format |
|---|---|
| `f32`, `[f32; 2]`, `[f32; 4]` | `R32_SFLOAT`, `R32G32_SFLOAT`, `R32G32B32A32_SFLOAT` |
| `u8`, `[u8; 2]`, `[u8; 4]` | `R8_UNORM`, `R8G8_UNORM`, `R8G8B8A8_UNORM` |
| `u16` | `R16_UNORM` |
| `u32`, `i32` | `R32_UINT`, `R32_SINT` |

## Using an image

`vk_image()`, `vk_image_view()` and `vk_format()` give the Vulkan objects. The layout is always `GENERAL`, so an
image can be sampled or used as a storage image without a transition. Ordering your own Vulkan work against cudash
streams is up to you, as with `vk_buffer()`.

## Limits

- **Kernels can't take images.** Their arguments are push constants, and images need descriptor sets.
- **`as_image` doesn't work everywhere.** It needs a linear image over the slice's memory, which some GPUs don't
  allow for small or unaligned slices. It then returns `DriverError::InvalidImage`, and `to_image` still works.
  `imgcpy_dtod` into an `as_image` view is an error: write to the slice instead.
- **Images aren't tracked across streams** like slices are. Only copies on the same stream are ordered.
