// x[i] = 1 if rotating the lane index by one gives the next lane's index, wrapping at the subgroup
// size, else 0. Needs GroupNonUniformRotateKHR. The rotate is outside the bounds check so no lane
// is inactive.
[shader("compute")]
[numthreads(64, 1, 1)]
void rotate_lanes(uniform uint* x, uniform int n, uint3 id : SV_DispatchThreadID)
{
    uint lane = WaveGetLaneIndex();
    uint rotated = WaveRotate(lane, 1);
    if (id.x < n) x[id.x] = rotated == (lane + 1) % WaveGetLaneCount() ? 1 : 0;
}
