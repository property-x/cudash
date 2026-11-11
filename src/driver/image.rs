//! Images: copies of slices as Vulkan images, for the caller's own shaders and for RenderDoc.
//! cudash-only, since cudarc has no images. Kernels can't take them.

use std::{
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use ash::vk;

use super::{
    DriverError,
    device::{Buffer, Device},
    slice::{DevicePtr, DeviceRepr},
    stream::CudaStream,
};

/// Element types that are one texel of a Vulkan format.
///
/// # Safety
/// `FORMAT`'s texel must have the size and layout of `Self`.
pub unsafe trait ImageFormat: DeviceRepr {
    const FORMAT: vk::Format;
}

macro_rules! image_formats {
    ($($t:ty => $format:ident),* $(,)?) => {$(
        unsafe impl ImageFormat for $t {
            const FORMAT: vk::Format = vk::Format::$format;
        }
    )*};
}
image_formats!(
    f32 => R32_SFLOAT,
    [f32; 2] => R32G32_SFLOAT,
    [f32; 4] => R32G32B32A32_SFLOAT,
    u8 => R8_UNORM,
    [u8; 2] => R8G8_UNORM,
    [u8; 4] => R8G8B8A8_UNORM,
    u16 => R16_UNORM,
    u32 => R32_UINT,
    i32 => R32_SINT,
);

/// A 2D Vulkan image holding `width * height` elements of `T`, made by
/// [`CudaStream::clone_to_image`] or [`DevicePtr::to_image`] (copies) or [`DevicePtr::as_image`] (a view of the slice's
/// memory, so later writes to the slice show in the image). Its layout is `GENERAL`, so your
/// own pipelines can sample it or use it as a storage image. Order that work against the
/// slice's stream yourself, as for [`CudaSlice::vk_buffer`](super::CudaSlice::vk_buffer).
pub struct CudaImage<T> {
    inner: Arc<ImageInner>,
    width: u32,
    height: u32,
    marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for CudaImage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl<T: ImageFormat> CudaImage<T> {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn vk_image(&self) -> vk::Image {
        self.inner.image
    }

    /// A 2D color view of the whole image.
    pub fn vk_image_view(&self) -> vk::ImageView {
        self.inner.view
    }

    pub fn vk_format(&self) -> vk::Format {
        T::FORMAT
    }
}

/// Freed once submissions using it have finished, since they hold a reference.
struct ImageInner {
    device: Arc<Device>,
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    /// For a view, the buffer that owns `memory`, kept alive by this image.
    source: Option<Arc<Buffer>>,
}

impl Drop for ImageInner {
    fn drop(&mut self) {
        let device = &self.device.device;
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            if self.source.is_none() {
                device.free_memory(self.memory, None);
            }
        }
    }
}

impl ImageInner {
    fn new(
        device: &Arc<Device>,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<Self, DriverError> {
        let max = device.properties.limits.max_image_dimension2_d;
        if width == 0 || height == 0 || width > max || height > max {
            return Err(DriverError::InvalidImage(format!(
                "{width}x{height}, but each side must be 1 to {max}"
            )));
        }
        let vk_device = &device.device;
        let features = unsafe {
            device
                .instance
                .get_physical_device_format_properties(device.physical_device, format)
        }
        .optimal_tiling_features;
        if !features.contains(vk::FormatFeatureFlags::TRANSFER_DST) {
            return Err(DriverError::InvalidImage(format!(
                "the GPU can't copy into {format:?} images"
            )));
        }
        let mut usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
        if features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        unsafe {
            let image = vk_device.create_image(&info, None)?;
            // Destroys what was made so far if a later step fails.
            let mut inner = Self {
                device: device.clone(),
                image,
                view: vk::ImageView::null(),
                memory: vk::DeviceMemory::null(),
                source: None,
            };
            let requirements = vk_device.get_image_memory_requirements(image);
            let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let memory_type = device.memory_type(
                requirements.memory_type_bits,
                local,
                vk::MemoryPropertyFlags::empty(),
            )?;
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            inner.memory = vk_device.allocate_memory(&allocate, None)?;
            vk_device.bind_image_memory(image, inner.memory, 0)?;
            let view = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(COLOR);
            inner.view = vk_device.create_image_view(&view, None)?;
            Ok(inner)
        }
    }
}

impl ImageInner {
    /// A linear image bound to `buffer`'s memory at `offset`, holding the bytes as they are.
    fn over(
        buffer: &Arc<Buffer>,
        offset: u64,
        format: vk::Format,
        texel: usize,
        width: u32,
        height: u32,
    ) -> Result<Self, DriverError> {
        let invalid = |why: String| Err(DriverError::InvalidImage(why));
        let device = &buffer.device;
        let vk_device = &device.device;
        if width == 0 || height == 0 {
            return invalid(format!(
                "{width}x{height}, but each side must be at least 1"
            ));
        }
        if buffer.export.is_some_and(|dedicated| dedicated.0) {
            return invalid(
                "this slice has a dedicated allocation, which an image can't share".into(),
            );
        }
        let features = unsafe {
            device
                .instance
                .get_physical_device_format_properties(device.physical_device, format)
        }
        .linear_tiling_features;
        let mut usage = vk::ImageUsageFlags::empty();
        if features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if usage.is_empty() {
            return invalid(format!(
                "the GPU can't sample or store to linear {format:?} images"
            ));
        }
        let limits = unsafe {
            device.instance.get_physical_device_image_format_properties(
                device.physical_device,
                format,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::LINEAR,
                usage,
                vk::ImageCreateFlags::empty(),
            )
        }?
        .max_extent;
        if width > limits.width || height > limits.height {
            return invalid(format!(
                "{width}x{height}, but linear images are at most {}x{}",
                limits.width, limits.height
            ));
        }
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::PREINITIALIZED);
        unsafe {
            let image = vk_device.create_image(&info, None)?;
            // Destroys what was made so far if a later step fails. `source` is set, so the
            // buffer's memory isn't freed with it.
            let mut inner = Self {
                device: device.clone(),
                image,
                view: vk::ImageView::null(),
                memory: buffer.memory,
                source: Some(buffer.clone()),
            };
            let requirements = vk_device.get_image_memory_requirements(image);
            if requirements.memory_type_bits & (1 << buffer.memory_type) == 0 {
                return invalid("the slice's memory type can't hold images".into());
            }
            if !offset.is_multiple_of(requirements.alignment) {
                return invalid(format!(
                    "the slice starts {offset} bytes in, but images need {}",
                    requirements.alignment
                ));
            }
            if offset + requirements.size > buffer.allocation_size {
                return invalid("the image needs more memory than the slice has".into());
            }
            let layout = vk_device.get_image_subresource_layout(
                image,
                vk::ImageSubresource {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    array_layer: 0,
                },
            );
            if layout.offset != 0 || layout.row_pitch != (width as usize * texel) as u64 {
                return invalid(format!(
                    "the GPU pads rows of linear images to {} bytes, not {}",
                    layout.row_pitch,
                    width as usize * texel
                ));
            }
            vk_device.bind_image_memory(image, buffer.memory, offset)?;
            let view = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(COLOR);
            inner.view = vk_device.create_image_view(&view, None)?;
            Ok(inner)
        }
    }
}

/// [`DevicePtr::as_image`].
pub(super) fn image_over<T: ImageFormat, Src: DevicePtr<T>>(
    src: &Src,
    width: u32,
    height: u32,
) -> Result<CudaImage<T>, DriverError> {
    assert_eq!(src.len(), width as usize * height as usize);
    let range = src.raw();
    let inner = Arc::new(ImageInner::over(
        range.buffer,
        range.offset,
        T::FORMAT,
        size_of::<T>(),
        width,
        height,
    )?);
    let (device, image) = (&inner.device, inner.image);
    let name = format_args!(
        "CudaImage<{}> view {width}x{height}",
        std::any::type_name::<T>()
    );
    device.set_name(inner.image, name);
    device.set_name(inner.view, name);
    // The contents stay as they are, only the layout changes. Reading the buffer makes it wait
    // for the slice's earlier writes.
    src.stream().core.submit(
        format_args!("as_image {width}x{height}"),
        &[range.buffer],
        &[],
        vec![inner.clone()],
        |device, commands| unsafe {
            let to_general = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::PREINITIALIZED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(COLOR);
            let dependency = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&to_general));
            device.cmd_pipeline_barrier2(commands, &dependency);
        },
    )?;
    Ok(CudaImage {
        inner,
        width,
        height,
        marker: PhantomData,
    })
}

const COLOR: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

impl CudaStream {
    /// Copies `src`, `width * height` elements in rows, into a new image, in stream order.
    ///
    /// # Panics
    /// If `src` doesn't hold exactly `width * height` elements.
    pub fn clone_to_image<T: ImageFormat, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
        width: u32,
        height: u32,
    ) -> Result<CudaImage<T>, DriverError> {
        let inner = ImageInner::new(self.device(), T::FORMAT, width, height)?;
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let id = COUNT.fetch_add(1, Ordering::Relaxed);
        let ty = std::any::type_name::<T>();
        let name = format_args!("CudaImage<{ty}> {id} {width}x{height}");
        self.device().set_name(inner.image, name);
        self.device().set_name(inner.view, name);
        self.device().set_name(inner.memory, name);
        let mut image = CudaImage {
            inner: Arc::new(inner),
            width,
            height,
            marker: PhantomData,
        };
        self.imgcpy_dtod(src, &mut image)?;
        Ok(image)
    }

    /// Copies `src` into `dst`, replacing all of it, in stream order. `dst` is in the
    /// `GENERAL` layout afterwards.
    ///
    /// shortcut: images aren't tracked like slices, so only copies on this stream are ordered
    /// before this one. Track them if images get used across streams.
    ///
    /// # Panics
    /// If `src` doesn't hold exactly `dst.width() * dst.height()` elements.
    pub fn imgcpy_dtod<T: ImageFormat, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut CudaImage<T>,
    ) -> Result<(), DriverError> {
        if dst.inner.source.is_some() {
            return Err(DriverError::InvalidImage(
                "this image is a view of a slice, so write to the slice instead".into(),
            ));
        }
        let (width, height) = (dst.width, dst.height);
        assert_eq!(src.len(), width as usize * height as usize);
        let range = src.raw();
        let (buffer, offset, image) = (range.buffer.buffer, range.offset, dst.inner.image);
        let keep: Vec<super::stream::Keep> = vec![dst.inner.clone()];
        self.core.submit(
            format_args!("imgcpy_dtod {width}x{height}"),
            &[range.buffer],
            &[],
            keep,
            |device, commands| unsafe {
                // The whole image is replaced, so its old contents can be discarded.
                let to_copy = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(image)
                    .subresource_range(COLOR);
                let dependency = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&to_copy));
                device.cmd_pipeline_barrier2(commands, &dependency);
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });
                device.cmd_copy_buffer_to_image(
                    commands,
                    buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
                let to_general = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(image)
                    .subresource_range(COLOR);
                let dependency = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&to_general));
                device.cmd_pipeline_barrier2(commands, &dependency);
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{CudaContext, DevicePtrMut};

    /// Reads the image back through a buffer, which is how a user's shader would see it.
    fn read_back<T: ImageFormat + Default + Clone + crate::driver::ValidAsZeroBits>(
        stream: &Arc<CudaStream>,
        image: &CudaImage<T>,
    ) -> Vec<T> {
        let len = (image.width * image.height) as usize;
        let mut out = stream.alloc_zeros::<T>(len).unwrap();
        let (width, height, vk_image) = (image.width, image.height, image.inner.image);
        let range = out.raw_mut();
        let buffer = range.buffer.buffer;
        stream
            .core
            .submit(
                format_args!("test read back"),
                &[],
                &[range.buffer],
                vec![image.inner.clone()],
                |device, commands| unsafe {
                    let region = vk::BufferImageCopy::default()
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        });
                    device.cmd_copy_image_to_buffer(
                        commands,
                        vk_image,
                        vk::ImageLayout::GENERAL,
                        buffer,
                        &[region],
                    );
                },
            )
            .unwrap();
        stream.clone_dtoh(&out).unwrap()
    }

    #[test]
    fn images_hold_the_slice_in_rows_and_can_be_overwritten() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let data: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let slice = stream.clone_htod(&data).unwrap();
        let mut image = stream.clone_to_image(&slice, 5, 3).unwrap();
        assert_eq!((image.width(), image.height()), (5, 3));
        assert_eq!(image.vk_format(), vk::Format::R32_SFLOAT);
        assert_eq!(read_back(&stream, &image), data);

        let doubled: Vec<f32> = data.iter().map(|x| x * 2.0).collect();
        let other = stream.clone_htod(&doubled).unwrap();
        stream.imgcpy_dtod(&other, &mut image).unwrap();
        assert_eq!(read_back(&stream, &image), doubled);
    }

    #[test]
    fn views_copy_from_their_offset() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let pixels: Vec<[u8; 4]> = (0..8u8).map(|i| [i, i + 1, i + 2, 255]).collect();
        let slice = stream.clone_htod(&pixels).unwrap();
        let image = stream.clone_to_image(&slice.slice(4..8), 2, 2).unwrap();
        assert_eq!(read_back(&stream, &image), pixels[4..]);
    }

    #[test]
    fn as_image_sees_the_slice_without_copying() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let data: Vec<f32> = (0..64 * 64).map(|i| i as f32).collect();
        let mut slice = stream.clone_htod(&data).unwrap();
        let image = match slice.as_image(64, 64) {
            Ok(image) => image,
            // Linear images are optional in practice, and this GPU can't have one.
            Err(DriverError::InvalidImage(why)) => return eprintln!("skipped: {why}"),
            Err(e) => panic!("{e}"),
        };
        assert_eq!(read_back(&stream, &image), data);
        // A write through the slice shows in the image that was made before it.
        stream.memset_zeros(&mut slice).unwrap();
        assert_eq!(read_back(&stream, &image), vec![0.0; 64 * 64]);
        let other = stream.clone_htod(&data).unwrap();
        let mut copy = stream.clone_to_image(&other, 64, 64).unwrap();
        assert!(stream.imgcpy_dtod(&slice, &mut copy).is_ok());
        let mut view = slice.as_image(64, 64).unwrap();
        assert!(matches!(
            stream.imgcpy_dtod(&other, &mut view),
            Err(DriverError::InvalidImage(_))
        ));
    }

    #[test]
    fn bad_sizes_are_errors_or_panics() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let slice = stream.alloc_zeros::<f32>(4).unwrap();
        assert!(matches!(
            stream.clone_to_image(&slice.slice(0..0), 0, 0),
            Err(DriverError::InvalidImage(_))
        ));
        let wrong = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stream.clone_to_image(&slice, 3, 1)
        }));
        assert!(wrong.is_err());
    }
}
