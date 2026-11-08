@vulkan
Feature: Allocating device memory with cudash::driver
  The driver API allocates device memory through Vulkan. Any Vulkan 1.3 GPU works, so
  these scenarios run on AMD and Intel as well as NVIDIA.

  Background:
    Given a context on the first Vulkan 1.3 GPU
    And a default stream

  Scenario Outline: A zeroed allocation reads back as zeros
    When I allocate <count> zeroed <type> elements as "b"
    Then the buffer "b" has <count> zero elements

    Examples:
      | type | count |
      | u8   | 17    |
      | u32  | 4     |
      | i64  | 3     |
      | f64  | 5     |

  Scenario: An uninitialized allocation is usable once every element is written
    When I allocate 4 uninitialized u32 elements as "b"
    And I write 7, 8, 9 and 10 into "b"
    Then the buffer "b" reads 7, 8, 9 and 10

  Scenario: Copying host data makes a buffer with the same length and values
    Given host data of 300 u16 values counting up by 3
    When I copy the host data to the device as "b"
    Then the buffer "b" has length 300
    And the buffer "b" reads the host data

  Scenario: A zero-length allocation is accepted
    When I allocate 0 zeroed u32 elements as "b"
    Then the buffer "b" is empty

  Scenario: The byte size follows the element type
    When I allocate 10 zeroed u16 elements as "b"
    Then the buffer "b" has length 10
    And the buffer "b" has byte size 20

  Scenario: Each buffer gets its own Vulkan buffer handle
    When I allocate 8 zeroed u32 elements as "a"
    And I allocate 8 zeroed u32 elements as "b"
    Then the buffers "a" and "b" have different Vulkan buffer handles

  @untested
  Scenario: Each buffer gets its own Vulkan memory handle
    When I allocate 8 zeroed u32 elements as "a"
    And I allocate 8 zeroed u32 elements as "b"
    Then the buffers "a" and "b" have different Vulkan memory handles

  Scenario: A buffer outlives the stream and context handles that made it
    Given a buffer "b" of 2 i32 elements holding 1 and 2
    When I drop the stream and the context
    Then the buffer "b" reads 1 and 2

  Scenario: Creating and dropping many buffers in a loop does not fail
    When I create and drop 500 buffers of 64 KiB each
    And I then hold 16 buffers of 1 MiB at the same time
    Then all allocations succeed

  @known-gap
  Scenario: Allocating more bytes than the device has is reported as an error
    When I try to allocate more bytes than the device has
    Then the result is a Vulkan error
