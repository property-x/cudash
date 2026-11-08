@vulkan
Feature: Moving data with cudash::driver
  Copies follow cudarc's names and rules. Host copies are staged, and device copies run on
  the Vulkan queue. The stream orders all of them.

  Background:
    Given a context on the first Vulkan 1.3 GPU
    And a default stream

  Scenario Outline: Host data survives a round trip through the device
    Given host data <values> of type <type>
    When I copy the host data to the device and back
    Then the result equals the host data

    Examples:
      | type | values                                |
      | u8   | 1, 255 and 0                          |
      | u16  | 65535, 0 and 7                        |
      | u32  | 4294967295 and 3237937677             |
      | u64  | 18446744073709551615 and 1099511627776 |
      | i32  | -2147483648, -1 and 2147483647        |
      | i64  | -9223372036854775808 and -42          |
      | f32  | 1.5, -2.25 and 3.4028235e38           |
      | f64  | 2.2250738585072014e-308 and -0.5      |

  Scenario: Copying host data into a view changes only the view
    Given a zeroed buffer "b" of 8 u32 elements
    When I write 1, 2 and 3 into "b" starting at index 2
    Then the buffer "b" reads 0, 0, 1, 2, 3, 0, 0 and 0

  Scenario: Copying to host memory fills a slice that already exists
    Given a buffer "b" of 3 u8 elements holding 4, 5 and 6
    When I copy "b" into a host array of 3 u8
    Then the result reads 4, 5 and 6

  Scenario: A device-to-device copy between views at different offsets
    Given a buffer "a" of 4 u32 elements holding 10, 11, 12 and 13
    And a zeroed buffer "b" of 6 u32 elements
    When I copy the 2 elements starting at index 1 of "a" into the 2 elements starting at index 3 of "b"
    Then the buffer "b" reads 0, 0, 0, 11, 12 and 0

  Scenario: A copy made with clone_dtod does not change when the source changes
    Given a buffer "a" of 3 u32 elements holding 1, 2 and 3
    When I make a device copy of "a" as "b"
    And I write 9, 9 and 9 into "b"
    Then the buffer "a" reads 1, 2 and 3
    And the buffer "b" reads 9, 9 and 9

  Scenario: memset_zeros clears only the range it is given
    Given a buffer "b" of 7 u8 elements holding 1, 1, 1, 1, 1, 1 and 1
    When I zero the 3 elements starting at index 1 of "b" with memset_zeros
    Then the buffer "b" reads 1, 0, 0, 0, 1, 1 and 1

  Scenario: Empty copies are allowed
    Given an empty buffer "e" of u32
    When I copy an empty host slice into "e"
    And I read "e" back to the host
    Then the result is empty

  Scenario: A host destination that is too short panics, as in cudarc
    Given a buffer "b" of 4 u32 elements holding 1, 2, 3 and 4
    When I copy "b" into a host slice of 2 u32 elements
    Then the program panics
