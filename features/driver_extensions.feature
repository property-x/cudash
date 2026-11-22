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

  # ---- Extensions that need the ash_latest feature ---------------------------------------
  # Each capability is checked on hand-made modules, accepted where the GPU has the feature bit
  # and refused where it lacks it. Slang can't emit any of them, so no kernel runs. Without the
  # ash_latest Cargo feature these scenarios are skipped.

  @vulkan @ash-latest @gpu-has-bfloat16-type
  Scenario: A GPU with the feature accepts BFloat16TypeKHR
    When I load a module declaring the capabilities 1 and 5116
    Then the result is no error

  @vulkan @ash-latest @gpu-has-bfloat16-dot-product
  Scenario: A GPU with the feature accepts BFloat16DotProductKHR
    When I load a module declaring the capabilities 1 and 5117
    Then the result is no error

  @vulkan @ash-latest @gpu-has-bfloat16-cooperative-matrix
  Scenario: A GPU with the feature accepts BFloat16CooperativeMatrixKHR
    When I load a module declaring the capabilities 1 and 5118
    Then the result is no error

  @vulkan @ash-latest @gpu-has-float8
  Scenario: A GPU with the feature accepts Float8EXT
    When I load a module declaring the capabilities 1 and 4212
    Then the result is no error

  @vulkan @ash-latest @gpu-has-float8-cooperative-matrix
  Scenario: A GPU with the feature accepts Float8CooperativeMatrixEXT
    When I load a module declaring the capabilities 1 and 4213
    Then the result is no error

  @vulkan @ash-latest @gpu-has-untyped-pointers
  Scenario: A GPU with the feature accepts UntypedPointersKHR
    When I load a module declaring the capabilities 1 and 4473
    Then the result is no error

  @vulkan @ash-latest @gpu-has-replicated-composites
  Scenario: A GPU with the feature accepts ReplicatedCompositesEXT
    When I load a module declaring the capabilities 1 and 6024
    Then the result is no error

  @vulkan @ash-latest @gpu-has-cooperative-matrix-reductions
  Scenario: A GPU with the feature accepts CooperativeMatrixReductionsNV
    When I load a module declaring the capabilities 1 and 5430
    Then the result is no error

  @vulkan @ash-latest @gpu-has-cooperative-matrix-conversions
  Scenario: A GPU with the feature accepts CooperativeMatrixConversionsNV
    When I load a module declaring the capabilities 1 and 5431
    Then the result is no error

  @vulkan @ash-latest @gpu-has-cooperative-matrix-per-element-operations
  Scenario: A GPU with the feature accepts CooperativeMatrixPerElementOperationsNV
    When I load a module declaring the capabilities 1 and 5432
    Then the result is no error

  @vulkan @ash-latest @gpu-has-cooperative-matrix-tensor-addressing
  Scenario: A GPU with the feature accepts CooperativeMatrixTensorAddressingNV
    When I load a module declaring the capabilities 1 and 5433
    Then the result is no error

  @vulkan @ash-latest @gpu-has-cooperative-matrix-block-loads
  Scenario: A GPU with the feature accepts CooperativeMatrixBlockLoadsNV
    When I load a module declaring the capabilities 1 and 5434
    Then the result is no error

  @vulkan @ash-latest @gpu-has-cooperative-matrix-tensor-addressing
  Scenario: A GPU with the feature accepts TensorAddressingNV
    When I load a module declaring the capabilities 1 and 5439
    Then the result is no error

  @vulkan @ash-latest @gpu-has-cooperative-vector
  Scenario: A GPU with the feature accepts CooperativeVectorNV
    When I load a module declaring the capabilities 1 and 5394
    Then the result is no error

  @vulkan @ash-latest @gpu-has-cooperative-vector-training
  Scenario: A GPU with the feature accepts CooperativeVectorTrainingNV
    When I load a module declaring the capabilities 1 and 5435
    Then the result is no error

  @vulkan @ash-latest @gpu-lacks-bfloat16-type
  Scenario: A GPU without the feature refuses BFloat16TypeKHR
    When I load a module declaring the capabilities 1 and 5116
    Then the result is a MissingFeature error for BFloat16TypeKHR

  @vulkan @ash-latest @gpu-lacks-bfloat16-dot-product
  Scenario: A GPU without the feature refuses BFloat16DotProductKHR
    When I load a module declaring the capabilities 1 and 5117
    Then the result is a MissingFeature error for BFloat16DotProductKHR

  @vulkan @ash-latest @gpu-lacks-bfloat16-cooperative-matrix
  Scenario: A GPU without the feature refuses BFloat16CooperativeMatrixKHR
    When I load a module declaring the capabilities 1 and 5118
    Then the result is a MissingFeature error for BFloat16CooperativeMatrixKHR

  @vulkan @ash-latest @gpu-lacks-float8
  Scenario: A GPU without the feature refuses Float8EXT
    When I load a module declaring the capabilities 1 and 4212
    Then the result is a MissingFeature error for Float8EXT

  @vulkan @ash-latest @gpu-lacks-float8-cooperative-matrix
  Scenario: A GPU without the feature refuses Float8CooperativeMatrixEXT
    When I load a module declaring the capabilities 1 and 4213
    Then the result is a MissingFeature error for Float8CooperativeMatrixEXT

  @vulkan @ash-latest @gpu-lacks-untyped-pointers
  Scenario: A GPU without the feature refuses UntypedPointersKHR
    When I load a module declaring the capabilities 1 and 4473
    Then the result is a MissingFeature error for UntypedPointersKHR

  @vulkan @ash-latest @gpu-lacks-replicated-composites
  Scenario: A GPU without the feature refuses ReplicatedCompositesEXT
    When I load a module declaring the capabilities 1 and 6024
    Then the result is a MissingFeature error for ReplicatedCompositesEXT

  @vulkan @ash-latest @gpu-lacks-cooperative-matrix-reductions
  Scenario: A GPU without the feature refuses CooperativeMatrixReductionsNV
    When I load a module declaring the capabilities 1 and 5430
    Then the result is a MissingFeature error for CooperativeMatrixReductionsNV

  @vulkan @ash-latest @gpu-lacks-cooperative-matrix-conversions
  Scenario: A GPU without the feature refuses CooperativeMatrixConversionsNV
    When I load a module declaring the capabilities 1 and 5431
    Then the result is a MissingFeature error for CooperativeMatrixConversionsNV

  @vulkan @ash-latest @gpu-lacks-cooperative-matrix-per-element-operations
  Scenario: A GPU without the feature refuses CooperativeMatrixPerElementOperationsNV
    When I load a module declaring the capabilities 1 and 5432
    Then the result is a MissingFeature error for CooperativeMatrixPerElementOperationsNV

  @vulkan @ash-latest @gpu-lacks-cooperative-matrix-tensor-addressing
  Scenario: A GPU without the feature refuses CooperativeMatrixTensorAddressingNV
    When I load a module declaring the capabilities 1 and 5433
    Then the result is a MissingFeature error for CooperativeMatrixTensorAddressingNV

  @vulkan @ash-latest @gpu-lacks-cooperative-matrix-block-loads
  Scenario: A GPU without the feature refuses CooperativeMatrixBlockLoadsNV
    When I load a module declaring the capabilities 1 and 5434
    Then the result is a MissingFeature error for CooperativeMatrixBlockLoadsNV

  @vulkan @ash-latest @gpu-lacks-cooperative-matrix-tensor-addressing
  Scenario: A GPU without the feature refuses TensorAddressingNV
    When I load a module declaring the capabilities 1 and 5439
    Then the result is a MissingFeature error for TensorAddressingNV

  @vulkan @ash-latest @gpu-lacks-cooperative-vector
  Scenario: A GPU without the feature refuses CooperativeVectorNV
    When I load a module declaring the capabilities 1 and 5394
    Then the result is a MissingFeature error for CooperativeVectorNV

  @vulkan @ash-latest @gpu-lacks-cooperative-vector-training
  Scenario: A GPU without the feature refuses CooperativeVectorTrainingNV
    When I load a module declaring the capabilities 1 and 5435
    Then the result is a MissingFeature error for CooperativeVectorTrainingNV

  # VK_KHR_shader_relaxed_extended_instruction has no capability. It makes the NonSemantic debug
  # info that `slangc -g` writes valid, so the scenario loads and runs such a kernel.
  @vulkan @ash-latest @gpu-has-relaxed-extended-instruction
  Scenario: A kernel built with debug info loads and runs
    Given the Slang SPIR-V build of "double_all" with debug info as "double" with entry point "double_all"
    And a buffer "a" of 1024 f32 elements all set to 1.5
    When I launch "double" on "a" with n 1024
    Then the buffer "a" holds 3 in every element

  # ---- What the context reports ----------------------------------------------------------

  @vulkan
  Scenario: Enabled extensions are ones the GPU reports, with no duplicates
    Then every enabled extension is one the GPU reports, once

  @vulkan
  Scenario: Every optional extension the GPU reports is enabled when its requirements are
    Then every optional extension the GPU reports with its requirements is enabled

  @vulkan
  Scenario: An enabled extension has its requirement enabled
    Then every enabled extension has its requirement enabled

  @vulkan
  Scenario: A GPU feature a tag relies on means its extension is enabled
    Then the feature tags agree with the enabled extensions

  @vulkan
  Scenario: The caps example runs
    Then the caps example exits successfully

  # ---- Malformed modules -----------------------------------------------------------------

  @vulkan
  Scenario: An OpCapability with no operand is an InvalidModule error, not a crash
    When I load a module whose OpCapability has no operand
    Then the result is an InvalidModule error
