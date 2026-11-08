// Takes the struct from cudarc's examples/05-device-repr.rs by value.
struct MyCoolStruct
{
    float a;
    double b;
    uint c;
    uint64_t d;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void write_struct(uniform double* out_, uniform MyCoolStruct thing)
{
    out_[0] = thing.a;
    out_[1] = thing.b;
    out_[2] = thing.c;
    out_[3] = double(thing.d);
}
