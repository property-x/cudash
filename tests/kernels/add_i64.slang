// data[i] += k for i < n. A 64-bit signed scalar argument between a pointer and an int.
[shader("compute")]
[numthreads(1, 1, 1)]
void add_i64(uniform int64_t* data, uniform int64_t k, uniform int n, uint3 id : SV_DispatchThreadID)
{
    if (id.x < n) data[id.x] += k;
}
