// Needs AtomicFloat16AddEXT, which RADV lacks: loading it must fail with MissingFeature there.
[shader("compute")]
[numthreads(64, 1, 1)]
void atomic_add_f16(uniform Atomic<half>* x, uniform int n, uint3 id : SV_DispatchThreadID)
{
    if (id.x < n) x[0].add(half(1.0));
}
