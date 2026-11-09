Feature: Using cudash interop without CUDA
  `cudash::interop` lends cudash slices to real cudarc calls. Only the scenarios that run
  without NVIDIA are here. Lending rules (empty slices, devices, ordering, paired streams) are
  in `tests/interop_memory.rs` and `tests/interop_roundtrip.rs`, which need an NVIDIA GPU.

  @vulkan
  Scenario: A Vulkan GPU from another vendor is not paired with CUDA
    Given a cudash context on a GPU from a vendor other than NVIDIA
    When I call cuda_context_for on it
    Then the result is a NoMatchingCudaDevice error

  @vulkan
  Scenario Outline: Lending without CUDA is an error, not a panic
    Given a cudash context on a GPU from a vendor other than NVIDIA
    When I lend a new slice of 4 u32 elements with <method>
    Then the result is a NoMatchingCudaDevice error

    Examples:
      | method      |
      | as_cuda     |
      | as_cuda_mut |
