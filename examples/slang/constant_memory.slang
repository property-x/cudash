// cudarc's examples/constant_memory.cu, in Slang, with the coefficients as an argument.
[shader("compute")]
[numthreads(1, 1, 1)]
void polynomial_kernel(
    uniform float* out_,
    uniform Ptr<float, Access.Read> inp,
    uniform int numel,
    uniform Ptr<float, Access.Read> coefficients,
    uint3 id : SV_DispatchThreadID)
{
    if (id.x < numel)
    {
        float x = inp[id.x];
        out_[id.x] = coefficients[0] +
                     coefficients[1] * x +
                     coefficients[2] * x * x +
                     coefficients[3] * x * x * x;
    }
}
