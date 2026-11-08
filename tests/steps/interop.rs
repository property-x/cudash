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

#[when(regex = r"^I call cuda_context_for on it$")]
fn when_cuda_context_for(w: &mut World) {
    let outcome = interop::cuda_context_for(w.ctx()).map(|_| ());
    let _ = w.record(outcome);
}
