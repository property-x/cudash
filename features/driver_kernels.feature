@vulkan
Feature: Launching SPIR-V kernels with cudash::driver
  Kernels are SPIR-V compute shaders. Buffers and slices are passed as 64-bit device
  addresses in push constants, and scalars by value, in the order they're passed to
  `arg`. The block size comes from the launch, not from the shader.

  The kernels are Slang files in tests/kernels, compiled to SPIR-V by build.rs. Each kernel's
  entry point has the same name as its file.

  Background:
    Given a context on the first Vulkan 1.3 GPU
    And a default stream

  Scenario: A kernel with a buffer and an int argument doubles every element
    Given a buffer "b" of 2 f32 elements holding 1.0 and 2.0
    And the kernel "double_all"
    When I launch "double_all" on "b" with n 2
    Then the buffer "b" reads 2.0 and 4.0

  Scenario: A kernel with a float scalar argument
    Given a buffer "x" of 1000 f32 elements counting up from 0
    And a buffer "y" of 1000 f32 elements all set to 1.0
    And the kernel "axpy"
    When I launch "axpy" with y "y", x "x", a 2.5 and n 1000
    Then the buffer "y" equals 2.5 times the index plus 1.0

  Scenario: A kernel with a 64-bit signed scalar argument
    Given a buffer "d" of 3 i64 elements holding -5, 0 and 7
    And the kernel "add_i64"
    When I launch "add_i64" on "d" with k -10 and n 3
    Then the buffer "d" reads -15, -10 and -3

  Scenario: One thread per element over a million elements
    Given a zeroed buffer "o" of 1000000 u64 elements
    And the kernel "fill_index"
    When I launch "fill_index" on "o" with n 1000000
    Then the buffer "o" holds its own index in every element

  Scenario: An explicit block size with more threads than elements is bounded by the kernel
    Given a zeroed buffer "o" of 1000 u64 elements
    And the kernel "fill_index"
    When I launch "fill_index" on "o" with n 1000 and block (256, 1, 1)
    Then the buffer "o" holds its own index in every element

  Scenario: A kernel writes through a view, and only the view changes
    Given a buffer "b" of 5 f32 elements holding 1.0, 2.0, 3.0, 4.0 and 5.0
    And the kernel "double_all"
    When I launch "double_all" on the 3 elements starting at index 1 of "b" with n 3
    Then the buffer "b" reads 1.0, 4.0, 6.0, 8.0 and 5.0

  Scenario: Consecutive launches on one stream run in order
    Given a buffer "b" of 2 f32 elements holding 1.0 and 2.0
    And the kernel "double_all"
    When I launch "double_all" 3 times in a row on "b" with n 2
    Then the buffer "b" reads 8.0 and 16.0

  Scenario: The same kernel launched with two block sizes gives the same answer
    Given a buffer "b" of 1000 f32 elements all set to 1.0
    And the kernel "double_all"
    When I launch "double_all" on "b" with n 1000 and block (64, 1, 1)
    And I launch "double_all" on "b" with n 1000 and block (1024, 1, 1)
    Then the buffer "b" holds 4.0 in every element

  Scenario: A block size larger than one thread still runs the kernel on every element
    Given a buffer "b" of 1000 f32 elements all set to 1.0
    And the kernel "double_all"
    When I launch "double_all" on "b" with n 1000 and block (1024, 1, 1)
    Then the buffer "b" holds 2.0 in every element

  Scenario: Arguments larger than the device's push constant limit are rejected
    Given a buffer "b" of 2 f32 elements holding 1.0 and 2.0
    And the kernel "double_all"
    When I launch "double_all" on "b" with n 2 and 300 bytes of extra arguments
    Then the result is a LaunchLimit error

  Scenario: A block larger than the device allows is rejected
    Given a buffer "b" of 1000 f32 elements all set to 1.0
    And the kernel "double_all"
    When I launch "double_all" on "b" with n 1000 and block (4294967295, 1, 1)
    Then the result is a LaunchLimit error

  Scenario: A zero-sized block is rejected
    Given a buffer "b" of 2 f32 elements holding 1.0 and 2.0
    And the kernel "double_all"
    When I launch "double_all" on "b" with n 2 and block (0, 1, 1)
    Then the result is a LaunchLimit error

  Scenario: Shared memory size in the launch config is ignored
    Given a buffer "b" of 2 f32 elements holding 1.0 and 2.0
    And the kernel "double_all"
    When I launch "double_all" on "b" with n 2 and shared memory 4096 bytes
    Then the result is no error
    And the buffer "b" reads 2.0 and 4.0

  Scenario: A module file that does not exist is an InvalidModule error
    When I load a SPIR-V module from "features/kernels/does_not_exist.spv"
    Then the result is an InvalidModule error

  Scenario: Bytes that are not SPIR-V are an InvalidModule error
    When I load a SPIR-V module from the bytes 1, 2 and 3
    Then the result is an InvalidModule error

  Scenario: Looking up an entry point that does not exist is a FunctionNotFound error
    Given the kernel "double_all"
    When I look up the function "no_such_entry" in the kernel "double_all"
    Then the result is a FunctionNotFound error
