// Compute port of iced_shader's fragment shader (itself from the 21st.dev "shader-animation"
// component): concentric rings per colour channel, broken up by a diagonal stripe pattern.
// One thread per pixel, writing RGBA8 with the same bytes as the fragment's opaque output.

float glsl_mod(float x, float y)
{
    return x - y * floor(x / y);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(
    uniform uint* dst,
    uniform int width,
    uniform int height,
    uniform float time,
    uniform float line_width,
    uniform int lines,
    uniform float ring_scale,
    uniform float mosaic,
    uniform float color_spread,
    uint3 tid : SV_DispatchThreadID)
{
    int x = int(tid.x), y = int(tid.y);
    if (x >= width || y >= height) return;

    // The fragment's `in.ndc` is -1..1 with y up; pixel rows here run downwards.
    float2 resolution = float2(width, height);
    float2 ndc = float2((float(x) + 0.5) * 2.0 / resolution.x - 1.0,
                        1.0 - (float(y) + 0.5) * 2.0 / resolution.y);
    float2 uv = ndc * resolution / min(resolution.x, resolution.y);
    float t = time * 0.05;

    float3 color = float3(0.0);
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < lines; i++)
        {
            color[j] += line_width * float(i * i)
                / abs(frac(t - color_spread * float(j) + float(i) * 0.01) * ring_scale
                      - length(uv) + glsl_mod(uv.x + uv.y, mosaic));
        }
    }
    uint3 rgb = uint3(saturate(color) * 255.0 + 0.5);
    dst[y * width + x] = rgb.r | (rgb.g << 8) | (rgb.b << 16) | (255u << 24);
}
