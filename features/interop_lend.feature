@cudarc @nvidia @untested
Feature: Lending a cudash buffer to cudarc calls
  `AsCuda::as_cuda` and `as_cuda_mut` (plan/api.md, "Interop") give a cudarc call temporary
  access to a cudash slice, on the CUDA stream paired with the slice's cudash stream. Borrow
  rules and the wait on Vulkan work are checked for us. The Rust tests in
  `tests/interop_*.rs` cover most of this; no steps exist, so these stay `@untested`.

  Background:
    Given an NVIDIA GPU paired with a Vulkan device
    And a cudash slice of 100 f32 elements, allocated on the cudash stream

  Scenario: A lent slice works as a cudarc copy destination
    When I lend the slice mutably and copy host values into it with cudarc's memcpy_htod
    Then after the lend ends, cudash reads the same values back

  Scenario: A lent slice works as a cudarc kernel argument
    When I lend the slice mutably and pass it to a cudarc PTX kernel with builder.arg
    Then after the lend ends, cudash reads the kernel's output

  Scenario: Lending waits for pending cudash work on the slice
    Given cudash has queued a fill of the slice that has not finished
    When I lend the slice to cudarc
    Then cudarc sees the filled values

  Scenario: Lending a slice that was not allocated for export is an error
    Given a slice allocated in a way that cannot be exported
    When I lend it
    Then the result is an error

  Scenario: Lends from one cudash stream share one CUDA stream
    Given two slices allocated on the same cudash stream
    When I lend both
    Then both lends report the same CUDA stream

  Scenario: Lends from different cudash streams use different CUDA streams
    Given two slices allocated on different cudash streams
    When I lend both
    Then the lends report different CUDA streams

  Scenario: The paired CUDA stream is not cudarc's default stream
    When I lend the slice
    Then the lend's CUDA stream is not the CUDA context's default stream

  Scenario: Lending on a stream the caller chose uses that stream
    Given a CUDA stream created by the caller
    When I lend the slice with as_cuda_with on that stream
    Then the lend reports that stream

  Scenario: Lending on a stream of another GPU is an error
    Given a CUDA stream on a different GPU than the slice
    When I lend the slice with as_cuda_with on that stream
    Then the result is a DeviceMismatch error
    # Needs two NVIDIA GPUs.

  Scenario: Using the slice from cudash while it is lent does not compile
    When the code uses the slice from cudash inside a lend scope
    Then the compiler rejects the program
    # Compile-fail test. Needs a trybuild-style check.

  @open-question
  Scenario: Lending twice in a row reuses the first import
    When I lend the slice, end the lend, and lend it again
    Then the second lend does not import the memory a second time
    # Whether to cache imports is undecided. This scenario states the goal, and measurement
    # decides it. See plan/api.md, "Import cost".
