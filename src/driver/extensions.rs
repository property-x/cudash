// Optional shader extensions cudash enables when the GPU has them, included by `device.rs`.
//
// <extension> => <ash feature struct> {  enabled with every bit the GPU reports
//     requires: [<extension>, ...],  optional; the entry is dropped when one of them isn't enabled.
//         Only what Vulkan 1.3 doesn't already include, from `depends` in Khronos's `vk.xml`.
//     capabilities: [<SPIR-V capability>(<number>): [<feature bit>, ...]],
//         `load_module` refuses a kernel declaring the capability unless one of the bits is on.
//         Numbers are from Khronos's `spirv.core.grammar.json`; a test in `device.rs` checks them.
//     off: [<feature bit>, ...],  bits never enabled
// }
//
// The comment above each entry says what it gives a kernel, and what enabling it costs kernels
// that don't use it. Costs are expected, not measured: these features only allow SPIR-V
// instructions, and a driver compiles each kernel for the instructions it actually uses.
//
// Extensions cudash's own code depends on, and the core 1.1-1.3 features, are in `device.rs`.

optional_extensions! {
    // Float atomic add on buffers, shared memory and images, as CUDA's atomicAdd on float. (cost: none)
    c"VK_EXT_shader_atomic_float" => PhysicalDeviceShaderAtomicFloatFeaturesEXT {
        capabilities: [
            AtomicFloat32AddEXT(6033): [
                shader_buffer_float32_atomic_add,
                shader_shared_float32_atomic_add,
                shader_image_float32_atomic_add,
            ],
            AtomicFloat64AddEXT(6034): [
                shader_buffer_float64_atomic_add,
                shader_shared_float64_atomic_add,
            ],
        ],
        off: [],
    },

    // fp16 atomic add, and float atomic min/max, as CUDA's atomicMin/atomicMax on floats. (cost: none)
    c"VK_EXT_shader_atomic_float2" => PhysicalDeviceShaderAtomicFloat2FeaturesEXT {
        requires: [c"VK_EXT_shader_atomic_float"],
        capabilities: [
            AtomicFloat16AddEXT(6095): [
                shader_buffer_float16_atomic_add,
                shader_shared_float16_atomic_add,
            ],
            AtomicFloat32MinMaxEXT(5612): [
                shader_buffer_float32_atomic_min_max,
                shader_shared_float32_atomic_min_max,
                shader_image_float32_atomic_min_max,
            ],
            AtomicFloat64MinMaxEXT(5613): [
                shader_buffer_float64_atomic_min_max,
                shader_shared_float64_atomic_min_max,
            ],
            AtomicFloat16MinMaxEXT(5616): [
                shader_buffer_float16_atomic_min_max,
                shader_shared_float16_atomic_min_max,
            ],
        ],
        off: [],
    },

    // Matrix multiply on tensor cores or WMMA units, as CUDA's wmma. (cost: none, robust access stays off)
    c"VK_KHR_cooperative_matrix" => PhysicalDeviceCooperativeMatrixFeaturesKHR {
        capabilities: [CooperativeMatrixKHR(6022): [cooperative_matrix]],
        off: [cooperative_matrix_robust_buffer_access], // robustness slows everything down
    },

    // Reading a GPU timer inside a kernel, as CUDA's clock64. (cost: none)
    c"VK_KHR_shader_clock" => PhysicalDeviceShaderClockFeaturesKHR {
        capabilities: [ShaderClockKHR(5055): [shader_subgroup_clock, shader_device_clock]],
        off: [],
    },

    // Lets a kernel ask that diverged threads rejoin as early as possible, as on CUDA.
    // (cost: none, only kernels asking for it change)
    c"VK_KHR_shader_maximal_reconvergence" => PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR {
        capabilities: [],
        off: [],
    },

    // Rotating values across the lanes of a subgroup, as a wrapping __shfl_sync. (cost: none)
    c"VK_KHR_shader_subgroup_rotate" => PhysicalDeviceShaderSubgroupRotateFeaturesKHR {
        capabilities: [GroupNonUniformRotateKHR(6026): [shader_subgroup_rotate]],
        off: [],
    },

    // Fast-math or strict float handling chosen per instruction, not per kernel. (cost: none)
    c"VK_KHR_shader_float_controls2" => PhysicalDeviceShaderFloatControls2FeaturesKHR {
        capabilities: [FloatControls2(6029): [shader_float_controls2]],
        off: [],
    },

    // Branch and value hints for the compiler, as __builtin_expect and __builtin_assume. (cost: none)
    c"VK_KHR_shader_expect_assume" => PhysicalDeviceShaderExpectAssumeFeaturesKHR {
        capabilities: [ExpectAssumeKHR(5629): [shader_expect_assume]],
        off: [],
    },

    // Reading the same shared memory as different types, as extern __shared__ and unions. (cost: none)
    c"VK_KHR_workgroup_memory_explicit_layout" => PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR {
        capabilities: [
            WorkgroupMemoryExplicitLayoutKHR(4428): [workgroup_memory_explicit_layout],
            WorkgroupMemoryExplicitLayout8BitAccessKHR(4429): [workgroup_memory_explicit_layout8_bit_access],
            WorkgroupMemoryExplicitLayout16BitAccessKHR(4430): [workgroup_memory_explicit_layout16_bit_access],
        ],
        off: [],
    },

    // Operations across the 4 threads of a quad, and branches they take together. (cost: none)
    c"VK_KHR_shader_quad_control" => PhysicalDeviceShaderQuadControlFeaturesKHR {
        requires: [c"VK_KHR_shader_maximal_reconvergence"],
        capabilities: [QuadControlKHR(5087): [shader_quad_control]],
        off: [],
    },

    // Atomic add, min and max on __half2 vectors, as CUDA's atomicAdd(__half2*). NVIDIA only. (cost: none)
    c"VK_NV_shader_atomic_float16_vector" => PhysicalDeviceShaderAtomicFloat16VectorFeaturesNV {
        capabilities: [AtomicFloat16VectorNV(5404): [shader_float16_vector_atomics]],
        off: [],
    },

    // The SM and warp a thread runs on, and their counts, as %smid and %warpid. NVIDIA only. (cost: none)
    c"VK_NV_shader_sm_builtins" => PhysicalDeviceShaderSMBuiltinsFeaturesNV {
        capabilities: [ShaderSMBuiltinsNV(5373): [shader_sm_builtins]],
        off: [],
    },

    // bfloat16 values, dot products and cooperative matrices, as CUDA's __nv_bfloat16. (cost: none)
    #[cfg(feature = "ash_latest")]
    c"VK_KHR_shader_bfloat16" => PhysicalDeviceShaderBfloat16FeaturesKHR {
        capabilities: [
            BFloat16TypeKHR(5116): [shader_b_float16_type],
            BFloat16DotProductKHR(5117): [shader_b_float16_dot_product],
            BFloat16CooperativeMatrixKHR(5118): [shader_b_float16_cooperative_matrix],
        ],
        off: [],
    },

    // fp8 (e4m3, e5m2) values and cooperative matrices, as CUDA's __nv_fp8_e4m3. (cost: none)
    #[cfg(feature = "ash_latest")]
    c"VK_EXT_shader_float8" => PhysicalDeviceShaderFloat8FeaturesEXT {
        capabilities: [
            Float8EXT(4212): [shader_float8],
            Float8CooperativeMatrixEXT(4213): [shader_float8_cooperative_matrix],
        ],
        off: [],
    },

    // Raw pointers that can be cast between types, as CUDA's void*. (cost: none)
    #[cfg(feature = "ash_latest")]
    c"VK_KHR_shader_untyped_pointers" => PhysicalDeviceShaderUntypedPointersFeaturesKHR {
        capabilities: [UntypedPointersKHR(4473): [shader_untyped_pointers]],
        off: [],
    },

    // Makes the debug info of `slangc -g` valid SPIR-V, for Slang source in RenderDoc. (cost: none)
    #[cfg(feature = "ash_latest")]
    c"VK_KHR_shader_relaxed_extended_instruction" => PhysicalDeviceShaderRelaxedExtendedInstructionFeaturesKHR {
        capabilities: [],
        off: [],
    },

    // Filling a vector or matrix with one value in a single instruction. (cost: none)
    #[cfg(feature = "ash_latest")]
    c"VK_EXT_shader_replicated_composites" => PhysicalDeviceShaderReplicatedCompositesFeaturesEXT {
        capabilities: [ReplicatedCompositesEXT(6024): [shader_replicated_composites]],
        off: [],
    },

    // Workgroup-wide and flexibly sized cooperative matrices, with reductions, conversions and
    // tensor loads. NVIDIA only. (cost: none)
    #[cfg(feature = "ash_latest")]
    c"VK_NV_cooperative_matrix2" => PhysicalDeviceCooperativeMatrix2FeaturesNV {
        requires: [c"VK_KHR_cooperative_matrix"],
        capabilities: [
            CooperativeMatrixReductionsNV(5430): [cooperative_matrix_reductions],
            CooperativeMatrixConversionsNV(5431): [cooperative_matrix_conversions],
            CooperativeMatrixPerElementOperationsNV(5432): [cooperative_matrix_per_element_operations],
            CooperativeMatrixTensorAddressingNV(5433): [cooperative_matrix_tensor_addressing],
            CooperativeMatrixBlockLoadsNV(5434): [cooperative_matrix_block_loads],
            TensorAddressingNV(5439): [cooperative_matrix_tensor_addressing],
        ],
        off: [],
    },

    // Matrix-vector products for small neural networks, on tensor cores. NVIDIA only. (cost: none)
    #[cfg(feature = "ash_latest")]
    c"VK_NV_cooperative_vector" => PhysicalDeviceCooperativeVectorFeaturesNV {
        capabilities: [
            CooperativeVectorNV(5394): [cooperative_vector],
            CooperativeVectorTrainingNV(5435): [cooperative_vector_training],
        ],
        off: [],
    },
}
