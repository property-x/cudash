[shader("compute")]
[numthreads(1, 1, 1)]
void fill_index(uniform uint64_t* out, uniform int n, uint3 id : SV_DispatchThreadID)
{
    if (id.x < n) out[id.x] = uint64_t(id.x);
}