// One kernel for both backends. build.rs compiles it to SPIR-V for cudash, and PTX for cudarc.
// Arguments must be `uniform` parameters: on CUDA, global buffers aren't launch arguments.

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uniform float* data, uniform int n, uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < n) data[tid.x] = float(tid.x) * 2.0 + 1.0;
}
