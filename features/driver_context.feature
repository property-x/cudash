Feature: Choosing and describing a GPU with cudash::driver
  A context is one GPU, picked by ordinal from the Vulkan 1.3 GPUs on the machine. Discrete
  GPUs come before integrated ones.

  @vulkan
  Scenario: The context reports its device
    Given at least one Vulkan 1.3 GPU
    When I create a context on ordinal 0
    Then the device count is at least 1
    And the context ordinal is 0
    And the context name is not empty
    And the context UUID is not all zeros
    And the context total memory is greater than 0

  @vulkan
  Scenario: Asking for a GPU that does not exist is an error
    Given at least one Vulkan 1.3 GPU
    When I create a context on the ordinal after the last GPU
    Then the result is a NoSuchDevice error

  @two-gpus @untested
  Scenario: Discrete GPUs are numbered before integrated GPUs
    Given a machine with both a discrete and an integrated Vulkan 1.3 GPU
    When I create a context on ordinal 0
    Then the context is on the discrete GPU

  @no-vulkan @untested
  Scenario: A machine without Vulkan reports an error instead of crashing
    Given no Vulkan loader is installed
    When I create a context on ordinal 0
    Then the result is a NoVulkan error
