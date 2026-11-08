[shader("compute")]
[numthreads(1, 1, 1)]
void double_all(uniform float* x, uniform int n, uint3 id : SV_DispatchThreadID)
{
    if (id.x < n) x[id.x] *= 2.0;
}
