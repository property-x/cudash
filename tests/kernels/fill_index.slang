// out[i] = i for i < n, so every element shows which thread wrote it.
[shader("compute")]
[numthreads(1, 1, 1)]
void fill_index(uniform uint64_t* out_, uniform int n, uint3 id : SV_DispatchThreadID)
{
    if (id.x < n) out_[id.x] = uint64_t(id.x);
}
