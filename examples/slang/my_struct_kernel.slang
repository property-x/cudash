// cudarc's examples/07-build-workflow/src/cuda/kernels/my_struct_kernel.cu, in Slang.
struct MyStruct
{
    float data[4];
};

[shader("compute")]
[numthreads(1, 1, 1)]
void my_struct_kernel(uniform MyStruct* my_structs, uniform uint64_t n, uint3 id : SV_DispatchThreadID)
{
    if (id.x < n)
    {
        my_structs[id.x].data[0] += 1;
        my_structs[id.x].data[1] += 1;
        my_structs[id.x].data[2] += 1;
        my_structs[id.x].data[3] += 1;
    }
}
