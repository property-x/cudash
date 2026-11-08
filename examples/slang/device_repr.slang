// The kernel from cudarc's examples/05-device-repr.rs, in Slang.
struct MyCoolStruct
{
    float a;
    double b;
    uint c;
    uint64_t d;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void my_custom_kernel(uniform uint* ok, uniform MyCoolStruct thing, uint3 id : SV_DispatchThreadID)
{
    if (id.x == 0)
        ok[0] = (thing.a == 1.0 && thing.b == 2.34l && thing.c == 57 && thing.d == 420) ? 1 : 0;
}
