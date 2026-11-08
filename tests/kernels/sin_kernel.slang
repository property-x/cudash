// cudarc's examples/sin.cu, in Slang.
[shader("compute")]
[numthreads(1, 1, 1)]
void sin_kernel(uniform float* out_, uniform Ptr<float, Access.Read> inp, uniform int numel, uint3 id : SV_DispatchThreadID)
{
    if (id.x < numel) out_[id.x] = sin(inp[id.x]);
}
