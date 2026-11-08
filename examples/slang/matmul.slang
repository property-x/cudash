// The kernel from cudarc's examples/matmul-kernel.rs, in Slang.
[shader("compute")]
[numthreads(1, 1, 1)]
void matmul(
    uniform Ptr<float, Access.Read> A,
    uniform Ptr<float, Access.Read> B,
    uniform float* C,
    uniform int N,
    uint3 id : SV_DispatchThreadID)
{
    int ROW = id.y;
    int COL = id.x;

    float tmpSum = 0;

    if (ROW < N && COL < N)
    {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++)
        {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
        C[ROW * N + COL] = tmpSum;
    }
}
