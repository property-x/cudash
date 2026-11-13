# Getting started

## Requirements

- **A Vulkan 1.3 GPU and driver.** Discrete GPUs are picked first, then integrated, virtual, other and CPU devices,
  so `CudaContext::new(0)` is the fastest GPU. Software drivers such as llvmpipe work, but slowly.
- **Linux.** Interop shares memory through Unix file descriptors. The `driver` module itself has no Unix-only code,
  but has only run on Linux.
- **For interop, an NVIDIA GPU with CUDA.** cudash finds the CUDA device with the same UUID as the Vulkan one.

## Features

| Feature | Default | What it does |
|---|---|---|
| `cudarc` | On | Adds `cudash::interop` and the cudarc dependency. Without CUDA at run time, interop calls return errors; the rest works |
| `kernel_debug` | On | Compiles the crate's own kernels with `slangc -g`, so RenderDoc shows their source. This embeds the source in the binary, so turn it off for builds you ship |

With `--no-default-features`, cudash has no cudarc dependency at all.

## Building

The repository uses [just](https://just.systems). Run `just` to list every command. The main ones:

| Command | What it does |
|---|---|
| `just setup` | Installs the tools: Rust, `slangc`, NVRTC, Vulkan tools, mdBook |
| `just doctor` | Shows the tools and GPUs this machine has |
| `just test` | Runs the tests. `just test graph` runs only those matching `graph` |
| `just validate` | Runs the tests under the Vulkan validation layers, and fails on any error |
| `just ci` | Runs everything that works without an NVIDIA GPU |
| `just test-nvidia` | Runs the CUDA tests too. Needs an NVIDIA GPU |
| `just docs` | Builds this book into `target/book` |

`build.rs` compiles the Slang kernels in `tests/kernels` and `examples/slang` into `OUT_DIR`, as SPIR-V and, when
NVRTC is installed, PTX. `SLANGC` and `NVRTC_PATH` override where those tools are.

## Examples

`examples/` has a version of each of cudarc's examples, under the same name. Run one with
`just example 01-allocate`. The README lists how each differs from cudarc's.
