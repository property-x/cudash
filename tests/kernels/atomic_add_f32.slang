// x[0] += 1.0 from each of n threads, as CUDA's atomicAdd on a float. Needs AtomicFloat32AddEXT.
[shader("compute")]
[numthreads(64, 1, 1)]
void atomic_add_f32(uniform Atomic<float>* x, uniform int n, uint3 id : SV_DispatchThreadID)
{
    if (id.x < n) x[0].add(1.0f);
}
