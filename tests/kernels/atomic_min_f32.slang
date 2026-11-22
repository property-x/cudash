// x[0] = min(x[0], id) from each of n threads. Needs AtomicFloat32MinMaxEXT.
[shader("compute")]
[numthreads(64, 1, 1)]
void atomic_min_f32(uniform Atomic<float>* x, uniform int n, uint3 id : SV_DispatchThreadID)
{
    if (id.x < n) x[0].min(float(id.x));
}
