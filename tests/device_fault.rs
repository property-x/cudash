//! A kernel that writes to an address outside every allocation loses the device, and
//! `DriverError::DeviceLost` carries the fault address where the driver has `VK_EXT_device_fault`.
//!
//! Skipped unless `CUDASH_FAULT_TESTS=1`: a real fault can reset the GPU and kill the desktop session
//! on AMD, and on llvmpipe the bad write is a CPU segfault in this process. Never run in CI.
//! One test per binary, because the device is gone afterwards.
//!
//!     CUDASH_FAULT_TESTS=1 cargo test --test device_fault -- --nocapture

use cudash::driver::*;

const WRITE_TO_ADDRESS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/write_to_address.spv"));
const BAD_ADDRESS: u64 = 0xdead_0000_0000;

#[test]
fn write_outside_every_allocation_loses_the_device() {
    if std::env::var("CUDASH_FAULT_TESTS").as_deref() != Ok("1") {
        eprintln!("skipped: set CUDASH_FAULT_TESTS=1 (can reset the GPU)");
        return;
    }
    let ctx = GpuContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let f = ctx
        .load_module(Spirv::from_binary(WRITE_TO_ADDRESS.to_vec()))
        .unwrap()
        .load_function("write_to_address")
        .unwrap();
    let mut builder = stream.launch_builder(&f);
    builder.arg(&BAD_ADDRESS);
    unsafe { builder.launch(LaunchConfig::for_num_elems(1)) }.unwrap();

    let err = ctx
        .synchronize()
        .expect_err("a wild write must lose the device");
    eprintln!("synchronize: {err}");
    let DriverError::DeviceLost(fault) = err else {
        panic!("expected DeviceLost, got {err:?}");
    };
    match fault {
        None => eprintln!("driver reported no fault info"),
        Some(fault) => {
            eprintln!("{fault:#?}");
            // Precision can be coarse (a page or more), and some drivers report only instruction addresses.
            if let Some(a) = fault.addresses.iter().find(|a| {
                a.address <= BAD_ADDRESS
                    && BAD_ADDRESS < a.address.saturating_add(a.precision.max(1))
            }) {
                eprintln!("fault range covers the bad address: {a:?}");
            } else {
                eprintln!("no reported address range covers {BAD_ADDRESS:#x}");
            }
        }
    }

    // The context stays failed: no hang, no panic.
    assert!(matches!(ctx.synchronize(), Err(DriverError::DeviceLost(_))));
    assert!(matches!(
        stream.synchronize(),
        Err(DriverError::DeviceLost(_))
    ));
    let again = unsafe {
        stream
            .launch_builder(&f)
            .arg(&BAD_ADDRESS)
            .launch(LaunchConfig::for_num_elems(1))
    };
    assert!(again.is_err() || matches!(ctx.synchronize(), Err(DriverError::DeviceLost(_))));
}
