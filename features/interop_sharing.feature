@nvidia
Feature: Sharing one buffer between CUDA and Vulkan
  A shared buffer is one allocation that CUDA and Vulkan both see. A timeline semaphore
  orders the two APIs. Neither API knows about the other's work, so the order is the
  caller's responsibility.

  Background:
    Given an NVIDIA GPU with CUDA
    And a timeline semaphore shared between them

  # CUDA-writes-Vulkan-reads and the reverse are `cuda_writes_vulkan_reads` and
  # `vulkan_writes_cuda_reads` in `tests/interop_roundtrip.rs`.

  @untested
  Scenario: CUDA and Vulkan take turns on one buffer
    Given a cudash slice "a" of 2 u32 elements
    And a cudash slice "b" of 2 u32 elements
    And a cudash slice "c" of 2 u32 elements
    When CUDA writes 41 and 42 into a lent "a"
    And Vulkan copies "a" to "b"
    Then the slice "b" reads 41 and 42
    When CUDA writes 1 and 1 into "b"
    And Vulkan copies "b" to "c"
    Then the slice "c" reads 1 and 1

  @untested
  Scenario: A Vulkan wait with no signal times out and returns false
    When Vulkan waits on a semaphore value that nothing signals, with a 10 ms timeout
    Then the wait returns false

  @open-question
  Scenario: CUDA waits for a timeline value that Vulkan has not signaled yet
    Given a semaphore value that Vulkan will signal later
    When CUDA waits for that value
    Then CUDA waits and then continues once Vulkan signals
    # Unverified. See notes/open-questions.md item 2. cudash avoids this case by blocking
    # on the Vulkan side first (see cuda_wait usage in the README).

  @two-gpus @untested
  Scenario: Pairing fails when the CUDA stream is on a different GPU
    Given two NVIDIA GPUs, and a Vulkan device on the first
    When I pair the Vulkan device with a CUDA stream on the second
    Then the result is a DeviceMismatch error
