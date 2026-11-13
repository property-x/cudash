//! Steps for the CUDA–Vulkan `cudash::interop` features. Only the Vulkan-only scenario has steps so
//! far; the `@nvidia` ones can't run on this machine.

use cucumber::{given, when};
use cudash::{driver, interop};

use super::world::World;

const NVIDIA: u32 = 0x10de;

#[given(regex = r"^a cudash context on a GPU from a vendor other than NVIDIA$")]
fn given_other_vendor(w: &mut World) {
    let count = driver::CudaContext::device_count().expect("Vulkan can be queried");
    let ctx = (0..count as usize)
        .filter_map(|ordinal| driver::CudaContext::new(ordinal).ok())
        .find(|ctx| {
            let props = unsafe {
                ctx.vk_instance()
                    .get_physical_device_properties(ctx.vk_physical_device())
            };
            props.vendor_id != NVIDIA
        })
        .expect("this scenario needs a non-NVIDIA Vulkan 1.3 GPU");
    w.streams.insert("default".to_owned(), ctx.default_stream());
    w.drv = Some(ctx);
}

#[when(regex = r"^I call as_cuda on its (context|default stream)$")]
fn when_as_cuda_on(w: &mut World, what: String) {
    let outcome = match what.as_str() {
        "context" => w.ctx().as_cuda().map(|_| ()),
        _ => w.default_stream().as_cuda().map(|_| ()),
    };
    let _ = w.record(outcome);
}

#[when(regex = r"^I lend a new slice of 4 u32 elements with (as_cuda|as_cuda_mut)$")]
fn when_as_cuda(w: &mut World, method: String) {
    use interop::AsCuda;
    let mut slice = w.default_stream().alloc_zeros::<u32>(4).expect("alloc");
    let outcome = match method.as_str() {
        "as_cuda" => slice.as_cuda().map(|_| ()),
        _ => slice.as_cuda_mut().map(|_| ()),
    };
    let _ = w.record(outcome);
}
