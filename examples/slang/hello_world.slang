// The kernel from cudarc's examples/06-threading.rs. It writes `i` rather than printing it.
[shader("compute")]
[numthreads(1, 1, 1)]
void hello_world(uniform int* out_, uniform int i, uint3 id : SV_DispatchThreadID)
{
    if (id.x == 0) out_[0] = i;
}
