Feature: Creating shared buffers with the interop API
  `cudash::interop` makes buffers on a `cudash::driver` context's own Vulkan device that both
  Vulkan and real cudarc can see. Only the scenario that runs without NVIDIA is here. Buffer
  creation rules (zero length, overflow, zeroed contents, handles, usage) are in
  `tests/interop_memory.rs` and `tests/interop_roundtrip.rs`, which need an NVIDIA GPU.

  @vulkan
  Scenario: A Vulkan GPU from another vendor is not paired with CUDA
    Given a cudash context on a GPU from a vendor other than NVIDIA
    When I call cuda_context_for on it
    Then the result is a NoMatchingCudaDevice error
