// x[i] = 1 if two reads of the device clock are nonzero and not decreasing, else 0.
// Needs ShaderClockKHR.
[shader("compute")]
[numthreads(64, 1, 1)]
void read_clock(uniform uint* x, uniform int n, uint3 id : SV_DispatchThreadID)
{
    uint2 a = getRealtimeClock();
    uint2 b = getRealtimeClock();
    uint64_t first = uint64_t(a.x) | (uint64_t(a.y) << 32);
    uint64_t second = uint64_t(b.x) | (uint64_t(b.y) << 32);
    if (id.x < n) x[id.x] = (first != 0 && second >= first) ? 1 : 0;
}
