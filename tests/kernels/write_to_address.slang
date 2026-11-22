// Writes 1 to a raw 64-bit device address. Used to fault on purpose (tests/device_fault.rs).
[shader("compute")]
[numthreads(1, 1, 1)]
void write_to_address(uniform uint64_t address)
{
    uint32_t* p = (uint32_t*)address;
    *p = 1;
}
