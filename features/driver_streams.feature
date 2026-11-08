@vulkan
Feature: Streams, ordering and synchronization in cudash::driver
  Work on one stream runs in the order it was queued, whether it's a copy or a kernel.
  Forked and new streams are independent, but reads wait for writes made on other streams.

  Background:
    Given a context on the first Vulkan 1.3 GPU
    And a default stream

  Scenario: Work on the default stream runs in order without any manual waits
    Given a buffer "a" of 4096 f32 elements all set to 1.0
    And a zeroed buffer "b" of 4096 f32 elements
    And the kernel "double_all"
    When I repeat 20 times: launch "double_all" on "a" with n 4096, copy "a" to "b", then launch "double_all" on "b" with n 4096
    Then the buffer "b" holds 2097152.0 in every element
    # After round n, b is 2 to the power n+1. Twenty rounds give 2 to the power 21.

  Scenario: A forked stream starts only after the work already queued on its parent
    Given a buffer "a" of 1024 u32 elements counting up from 0
    When I fork the default stream as "fork"
    And I copy "a" to a new zeroed buffer "b" on the "fork" stream
    And I read "b" back on the "fork" stream
    Then the result equals the values of "a"

  Scenario: A forked stream can allocate its own buffers
    Given a buffer "a" of 1024 u32 elements counting up from 0
    When I fork the default stream as "fork"
    And I allocate 1024 zeroed u32 elements as "b" on the "fork" stream
    Then the buffer "b" has length 1024

  Scenario: A read on one stream waits for a write made on another stream
    Given a zeroed buffer "b" of 1024 u32 elements
    And a second stream from the same context as "second"
    When I fill "b" with 5 on the default stream
    And I read "b" back on the "second" stream
    Then the result holds 5 in every element

  Scenario: A write on one stream waits for reads still in flight on another stream
    Given a buffer "b" of 1024 f32 elements all set to 1.0
    And a second stream from the same context as "second"
    And the kernel "double_all"
    When I launch "double_all" on "b" with n 1024
    And I fill "b" with 9 on the "second" stream
    And I read "b" back on the default stream
    Then the result holds 9 in every element

  Scenario: Buffers allocated on a second stream work
    Given a second stream from the same context as "second"
    When I write 3 and 4 into a new buffer "b" on the "second" stream
    And I read "b" back on the "second" stream
    Then the result reads 3 and 4

  Scenario: Synchronize waits for all queued work on the stream
    Given a buffer "a" of 1024 u32 elements counting up from 0
    When I copy "a" to a new zeroed buffer "b" and synchronize the stream
    Then the buffer "b" reads the values of "a"

  Scenario: Many threads can use one context
    When eight threads each allocate, copy and read back their own buffer
    Then every thread reads back the values it wrote

  Scenario: Dropping buffers right after queuing work is safe
    When I queue a copy between two buffers and drop both, 100 times in a loop
    Then no error occurs

  Scenario: A buffer can be dropped while kernel work on it is still queued
    Given a buffer "b" of 1024 f32 elements all set to 1.0
    And the kernel "double_all"
    When I launch "double_all" on "b" with n 1024 and then drop "b" before the stream finishes
    Then the stream finishes without error
