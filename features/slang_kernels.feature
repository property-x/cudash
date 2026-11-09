Feature: One Slang kernel on both backends
  `examples/slang/kernel.slang` is compiled by `build.rs` to SPIR-V for cudash and to PTX for
  cudarc. The Rust code differs only in the module type and the path. The PTX scenarios also need NVIDIA.

  Background:
    Given a context on the first Vulkan 1.3 GPU
    And a default stream

  @vulkan
  Scenario: The SPIR-V kernel writes 2i + 1 into element i
    Given the Slang SPIR-V build of "kernel" as "slang" with entry point "computeMain"
    And a zeroed buffer "out" of 256 f32 elements
    When I launch "slang" on "out" with n 256
    Then the buffer "out" equals 2 times the index plus 1

  # The PTX kernel running through cudarc is `same_slang_source_runs_on_cudarc` in
  # `tests/interop_slang.rs`.

  @untested @nvidia @vulkan
  Scenario: Both backends produce the same bytes for the same input
    Given an NVIDIA GPU with CUDA
    And the Slang SPIR-V build of "kernel" as "slang" with entry point "computeMain"
    And the Slang PTX build of "kernel" as "slang_ptx" with entry point "computeMain"
    And a zeroed buffer "out" of 256 f32 elements
    And a zeroed buffer "outp" of 256 f32 elements
    When I launch "slang" on "out" with n 256
    And I launch "slang_ptx" on "outp" with n 256 through cudarc on a lend of "outp"
    Then the buffer "out" equals the buffer "outp"

  @vulkan
  Scenario: The entry point keeps its source name
    Given the Slang SPIR-V build of "kernel" as "slang" with entry point "computeMain"
    Then the result is no error
    # Slang renames entry points to "main" unless -fvk-use-entrypoint-name is passed.
    # build.rs passes it, so the name "computeMain" is found.

  @vulkan
  Scenario: The kernel's bounds check keeps threads outside n from writing
    Given the Slang SPIR-V build of "kernel" as "slang" with entry point "computeMain"
    And a buffer "out" of 300 f32 elements all set to 7.0
    When I launch "slang" on "out" with n 256
    Then the elements 0 to 255 of "out" equal 2 times the index plus 1
    And the elements 256 to 299 of "out" still hold 7.0

  @vulkan
  Scenario: A struct passed by value arrives with the same field values
    Given the Slang SPIR-V build of "write_struct" as "write_struct" with entry point "write_struct"
    And a zeroed buffer "out" of 4 f64 elements
    When I launch "write_struct" on "out" with a struct of a float 1.0, a double 2.34, a uint 57 and a size_t 420
    Then the buffer "out" reads 1.0, 2.34, 57 and 420
