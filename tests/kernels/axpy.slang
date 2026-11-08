// y[i] = a * x[i] + y[i] for i < n. Two pointers, then a float scalar and an int.
[shader("compute")]
[numthreads(1, 1, 1)]
void axpy(uniform float* y, uniform float* x, uniform float a, uniform int n, uint3 id : SV_DispatchThreadID)
{
    if (id.x < n) y[id.x] = a * x[id.x] + y[id.x];
}
