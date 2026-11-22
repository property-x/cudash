//! Porting from cudarc: code with cudarc's names runs through `cudarc_compat`, and the prelude
//! has the `Gpu*` names always and the `Cuda*` names with the `cuda` feature.

mod cudarc_names {
    use cudash::cudarc_compat::*;

    #[test]
    fn cudarc_code_compiles_and_runs() -> Result<(), DriverError> {
        let ctx: std::sync::Arc<CudaContext> = CudaContext::new(0)?;
        let stream: std::sync::Arc<CudaStream> = ctx.default_stream();
        let on_device: CudaSlice<f32> = stream.clone_htod(&[1.0, 2.0, 3.0])?;
        assert_eq!(stream.clone_dtoh(&on_device)?, [1.0, 2.0, 3.0]);
        let _view: CudaView<f32> = on_device.slice(1..);
        // the names are aliases: the same types as the Gpu names
        let _: &GpuSlice<f32> = &on_device;
        Ok(())
    }
}

mod prelude {
    use cudash::prelude::*;

    #[test]
    fn gpu_names_are_always_there() -> Result<(), DriverError> {
        let ctx: std::sync::Arc<GpuContext> = GpuContext::new(0)?;
        let stream: std::sync::Arc<GpuStream> = ctx.default_stream();
        let s: GpuSlice<u8> = stream.clone_htod(&[7u8; 4])?;
        assert_eq!(stream.clone_dtoh(&s)?, [7u8; 4]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_names_come_with_the_feature() -> Result<(), DriverError> {
        let ctx: std::sync::Arc<CudaContext> = CudaContext::new(0)?;
        let s: CudaSlice<u8> = ctx.default_stream().clone_htod(&[1u8; 2])?;
        let _: &GpuSlice<u8> = &s;
        Ok(())
    }
}
