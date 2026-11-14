@vulkan
Feature: Stream capture and graphs in cudash::driver
  A capturing stream records its work instead of running it. `CudaGraph::launch` replays it,
  as many times as asked, in stream order. See "Graphs" in plan/api.md.

  Background:
    Given a context on the first Vulkan 1.3 GPU
    And a default stream
    And the kernel "double_all"
    And a buffer "a" of 4 f32 elements holding 5, 5, 5 and 5
    And a buffer "b" of 4 f32 elements holding 1, 2, 3 and 4
    And pinned host memory of 4 f32 values holding 7

  Scenario Outline: A call that can be captured is recorded, not run, and replays
    When I begin capturing on the default stream
    And I call "<call>" on "b"
    Then the result is no error
    When I end the capture as "g"
    Then the buffer "b" reads 1, 2, 3 and 4
    When I launch the graph "g" 1 time
    Then the buffer "b" reads <after>

    Examples:
      | call               | after           |
      | memset_zeros       | 0, 0, 0 and 0   |
      | memcpy_dtod        | 5, 5, 5 and 5   |
      | launch             | 2, 4, 6 and 8   |
      | memcpy_htod_pinned | 7, 7, 7 and 7   |
      | clone_dtod         | 1, 2, 3 and 4   |
      | alloc_zeros        | 1, 2, 3 and 4   |
      | device_ptr         | 1, 2, 3 and 4   |
      | image_copy         | 1, 2, 3 and 4   |

  Scenario Outline: A call that can't be captured fails and invalidates the capture
    When I begin capturing on the default stream
    And I call "<call>" on "b"
    Then the result is a StreamCapture error
    And the capture status is INVALIDATED
    And ending the capture gives no graph
    And the capture status is NONE
    And a new stream on the same context still works

    Examples:
      | call                 |
      | memcpy_htod_pageable |
      | memcpy_dtoh          |
      | clone_dtoh           |
      | synchronize          |
      | frame                |
      | stop_frame           |

  Scenario: A second begin_capture is refused and the first capture goes on
    When I begin capturing on the default stream
    And I begin capturing on the default stream
    Then the result is a StreamCapture error
    And the capture status is ACTIVE

  Scenario: Each replay repeats the captured work
    When I begin capturing on the default stream
    And I call "launch" on "b"
    And I end the capture as "g"
    And I launch the graph "g" 3 times
    Then the buffer "b" reads 8, 16, 24 and 32

  Scenario: A replay of a pinned copy reads the pinned memory as it is then
    When I begin capturing on the default stream
    And I call "memcpy_htod_pinned" on "b"
    And I end the capture as "g"
    And I fill the pinned host memory with 9
    And I launch the graph "g" 1 time
    Then the buffer "b" reads 9, 9, 9 and 9

  Scenario: Another stream sees the result of a replay
    Given a second stream from the same context as "second"
    When I begin capturing on the default stream
    And I call "memset_zeros" on "b"
    And I end the capture as "g"
    And I launch the graph "g" 1 time
    And I read "b" back on the "second" stream
    Then the result holds 0 in every element

  Scenario: Dropping a stream that is still capturing leaves the context usable
    When I begin capturing on a new stream "doomed" and drop it
    Then a new stream on the same context still works

  Scenario: A graph dropped right after launch still finishes its work
    When I begin capturing on the default stream
    And I call "launch" on "b"
    And I end the capture as "g"
    And I launch the graph "g" and drop it at once
    Then the buffer "b" reads 2, 4, 6 and 8

  Scenario: Queued replays of a large kernel don't overlap
    Given a buffer "big" of 4194304 f32 elements all set to 1.0
    When I begin capturing on the default stream
    And I double "big" in the capture
    And I end the capture as "g"
    And I launch the graph "g" 2 times
    Then the buffer "big" holds 4.0 in every element

  Scenario: Two threads launching one graph run their replays one after the other
    Given a buffer "big" of 65536 f32 elements all set to 1.0
    When I begin capturing on the default stream
    And I double "big" in the capture
    And I end the capture as "g"
    And two threads each launch the graph "g" 10 times
    Then the buffer "big" holds 1048576.0 in every element
