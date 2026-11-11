// A 3x3 box blur over one `f32` frame, one thread per pixel. Edge pixels average the neighbours
// that exist.

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(
    uniform Ptr<float, Access.Read> src,
    uniform float* dst,
    uniform int width,
    uniform int height,
    uint3 tid : SV_DispatchThreadID)
{
    int x = int(tid.x), y = int(tid.y);
    if (x >= width || y >= height) return;
    float sum = 0.0;
    int count = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                sum += src[ny * width + nx];
                count++;
            }
        }
    dst[y * width + x] = sum / float(count);
}
