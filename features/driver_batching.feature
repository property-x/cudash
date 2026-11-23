@vulkan @experimental
Feature: Stream batching in cudash::driver (experimental)
  A batching stream records operations and submits them at flush points. Each scenario here queues work
  that nothing has waited for, uses one flush point, and checks that the GPU runs the work and the data is
  right. Every wait has a timeout, so a missing flush fails instead of hanging. Needs the `experimental`
  Cargo feature. The GPU is watched through pinned memory read by raw pointer, which doesn't wait.

  Background:
    Given a context on the first Vulkan 1.3 GPU

  Scenario Outline: A flush point submits the open batch without anything waiting for its result
    Given 1 batching stream each with a copy queued to pinned host memory
    When I use the flush point "<point>"
    Then the GPU finishes every queued copy without anything waiting for it

    Examples:
      | point                |
      | flush                |
      | set_batching(false)  |
      | synchronize          |
      | ctx.synchronize      |
      | fork                 |
      | begin_capture        |
      | drop                 |
      | 300 operations       |

  @gpu-has-frame-boundary
  Scenario: stop_frame submits the open batch
    Given 1 batching stream each with a copy queued to pinned host memory
    When I use the flush point "frame"
    Then the GPU finishes every queued copy without anything waiting for it

  Scenario: ctx.synchronize flushes every batching stream
    Given 2 batching streams each with a copy queued to pinned host memory
    When I use the flush point "ctx.synchronize"
    Then the GPU finishes every queued copy without anything waiting for it

  Scenario Outline: Waiting for batched work flushes it
    Given 1 batching stream each with a copy queued to pinned host memory
    Then "<waiter>" returns the queued data within 10 seconds

    Examples:
      | waiter                              |
      | clone_dtoh on the batching stream   |
      | a read on another stream            |
      | a read on the buffer's own stream   |
      | the pinned slice                    |
      | a read after the stream is dropped  |

  Scenario: Two batching streams that use each other's buffers don't deadlock
    When two batching streams read each other's buffers in both directions, 300 times each
    Then the exchange completed with the final values

  Scenario: Operations keep their order across the 256-operation flushes
    When I queue 1000 copies into consecutive elements, each after a write of its index
    Then every element holds its own index

  Scenario: A graph launch runs after the loose work batched before it
    When I capture a copy from b to a on a batching stream, queue a write to b, and launch the graph
    Then a holds the value written before the launch

  Scenario: Turning batching off and on in the middle of a stream keeps the order
    When I write 1, turn batching off, write 2, turn it on, write 3 and read back
    Then the reads saw 1 and 3
