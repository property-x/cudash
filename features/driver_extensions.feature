Feature: Optional shader extensions in cudash::driver
  cudash enables every optional shader extension the GPU has, as CUDA has everything its hardware
  has. A module declaring a SPIR-V capability the GPU lacks is refused at load with a
  MissingFeature error naming the capability. The list is in src/driver/extensions.rs.
  A scenario tagged gpu-has-X or gpu-lacks-X skips on a GPU that differs.

  Background:
    Given a context on the first Vulkan 1.3 GPU
    And a default stream

  # ---- Kernels that use an enabled extension -------------------------------------------

  @vulkan @gpu-has-float32-atomic-add
  Scenario: Many threads add to one float
    Given the Slang SPIR-V build of "atomic_add_f32" as "add" with entry point "atomic_add_f32"
    And a buffer "sum" of 1 f32 elements all set to 0
    When I launch "add" on "sum" with n 100000
    Then the buffer "sum" reads 100000

  @vulkan @gpu-has-float32-atomic-add
  Scenario: Float atomics keep adding across launches
    Given the Slang SPIR-V build of "atomic_add_f32" as "add" with entry point "atomic_add_f32"
    And a buffer "sum" of 1 f32 elements all set to 0.5
    When I launch "add" 3 times in a row on "sum" with n 4096
    Then the buffer "sum" reads 12288.5

  @vulkan @gpu-has-float32-atomic-min-max
  Scenario: Many threads take the minimum of one float
    Given the Slang SPIR-V build of "atomic_min_f32" as "min" with entry point "atomic_min_f32"
    And a buffer "low" of 1 f32 elements all set to 1000000
    When I launch "min" on "low" with n 5000
    Then the buffer "low" reads 0

  @vulkan @gpu-has-subgroup-rotate
  Scenario: Rotating a value across a subgroup moves it one lane
    Given the Slang SPIR-V build of "rotate_lanes" as "rotate" with entry point "rotate_lanes"
    And a zeroed buffer "ok" of 512 u32 elements
    When I launch "rotate" on "ok" with n 512
    Then the buffer "ok" holds 1 in every element

  @vulkan @gpu-has-shader-clock
  Scenario: A kernel reads a device clock that does not run backwards
    Given the Slang SPIR-V build of "read_clock" as "clock" with entry point "read_clock"
    And a zeroed buffer "ok" of 128 u32 elements
    When I launch "clock" on "ok" with n 128
    Then the buffer "ok" holds 1 in every element

  # ---- Modules declaring a capability the GPU lacks ------------------------------------

  @vulkan @gpu-lacks-float64-atomic-add
  Scenario: A kernel using fp64 atomic add is refused, naming the capability
    When I load the Slang SPIR-V build of "atomic_add_f64"
    Then the result is a MissingFeature error for AtomicFloat64AddEXT

  @vulkan @gpu-lacks-float16-atomic-add
  Scenario: A kernel using fp16 atomic add is refused, naming the capability
    When I load the Slang SPIR-V build of "atomic_add_f16"
    Then the result is a MissingFeature error for AtomicFloat16AddEXT

  @vulkan @gpu-lacks-float64-atomic-add
  Scenario: A refused module does not stop the context working
    When I load the Slang SPIR-V build of "atomic_add_f64"
    Then the result is a MissingFeature error
    When I load the Slang SPIR-V build of "double_all"
    Then the result is no error

  @vulkan @gpu-lacks-float64-atomic-add
  Scenario: Refusing a module again gives the same error
    When I load the Slang SPIR-V build of "atomic_add_f64"
    And I load the Slang SPIR-V build of "atomic_add_f64"
    Then the result is a MissingFeature error for AtomicFloat64AddEXT

  @vulkan @gpu-lacks-float64-atomic-add
  Scenario: One unsupported capability among supported ones refuses the module
    When I load a module declaring the capabilities 1, 6034 and 11
    Then the result is a MissingFeature error for AtomicFloat64AddEXT

  @vulkan @gpu-lacks-float64-atomic-add
  Scenario: A capability declared after the memory model is still checked
    When I load a module declaring Shader, then its memory model, then the capability 6034
    Then the result is a MissingFeature error for AtomicFloat64AddEXT

  # Capability 5404 is AtomicFloat16VectorNV, from an NVIDIA-only extension.
  @vulkan @gpu-lacks-float16-vector-atomics
  Scenario: An NVIDIA-only capability is refused on a GPU without it
    When I load a module declaring the capabilities 5404
    Then the result is a MissingFeature error for AtomicFloat16VectorNV

  @vulkan @gpu-has-float16-vector-atomics
  Scenario: An NVIDIA-only capability is accepted on a GPU with it
    When I load a module declaring the capabilities 1 and 5404
    Then the result is no error

  # ---- Modules declaring only supported capabilities -----------------------------------

  @vulkan
  Scenario: The same supported capability declared three times is accepted
    When I load a module declaring the capabilities 1, 1 and 1
    Then the result is no error

  @vulkan @gpu-has-float32-atomic-add
  Scenario: A module declaring several supported capabilities loads
    When I load a module declaring the capabilities 1, 11 and 6033
    Then the result is no error

  @vulkan @gpu-has-quad-control
  Scenario: Quad control is accepted
    When I load a module declaring the capabilities 1 and 5087
    Then the result is no error

  @vulkan @gpu-has-explicit-layout
  Scenario: Workgroup memory explicit layout is accepted
    When I load a module declaring the capabilities 1 and 4428
    Then the result is no error

  @vulkan @gpu-has-cooperative-matrix
  Scenario: Cooperative matrix is accepted
    When I load a module declaring the capabilities 1 and 6022
    Then the result is no error

  # ---- Malformed modules -----------------------------------------------------------------

  @vulkan
  Scenario: An OpCapability with no operand is an InvalidModule error, not a crash
    When I load a module whose OpCapability has no operand
    Then the result is an InvalidModule error
