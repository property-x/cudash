# cudash developer commands. Currently assumes `ubuntu`.

# List all commands
default:
    @just --list --unsorted

# Install everything needed to build and test. Safe to re-run
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v rustup >/dev/null; then
        echo "Install Rust first: https://rustup.rs" >&2
        exit 1
    fi
    rustup component add rustfmt clippy
    rustup toolchain install nightly --component rustfmt
    # libnvrtc12 is in Ubuntu's multiverse repository, and doesn't need an NVIDIA GPU.
    sudo apt-get update
    sudo apt-get install -y build-essential slang-compiler libnvrtc12 mesa-vulkan-drivers \
        vulkan-tools vulkan-validationlayers spirv-tools
    # Slang looks for the unversioned library name, which the package doesn't provide.
    sudo ln -sf libnvrtc.so.12 /usr/lib/x86_64-linux-gnu/libnvrtc.so
    sudo ldconfig
    # RenderDoc is unpacked by hand in /opt. Its Vulkan layer manifest names the build machine's path, so point it at /opt.
    # The layer only loads when ENABLE_VULKAN_RENDERDOC_CAPTURE is set, so registering it is harmless otherwise.
    if [ -f /opt/renderdoc_1.47/etc/vulkan/implicit_layer.d/renderdoc_capture.json ]; then
        sudo install -d /etc/vulkan/implicit_layer.d
        sed -E 's#"library_path" *: *"[^"]*"#"library_path": "/opt/renderdoc_1.47/lib/librenderdoc.so"#' \
            /opt/renderdoc_1.47/etc/vulkan/implicit_layer.d/renderdoc_capture.json |
            sudo tee /etc/vulkan/implicit_layer.d/renderdoc_capture.json >/dev/null
    fi

# Show the tools and GPUs this machine has
doctor:
    #!/usr/bin/env bash
    for tool in cargo cc slangc spirv-val vulkaninfo nvidia-smi renderdoccmd; do
        printf '%-12s %s\n' "$tool" "$(command -v "$tool" || echo MISSING)"
    done
    printf '%-12s %s\n' nvrtc "$(ldconfig -p | grep -m1 'libnvrtc\.so ' | sed 's/.*=> //' || true)"
    printf '%-12s %s\n' renderdoc-layer "$(test -f /etc/vulkan/implicit_layer.d/renderdoc_capture.json && echo registered || echo MISSING)"
    vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|apiVersion' || echo "no Vulkan GPU found"

# Capture one frame of examples/frame_capture in RenderDoc, then open it in qrenderdoc. Needs `just setup`
renderdoc-example name:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --example {{name}}
    marker=$(mktemp)
    trap 'rm -f "$marker"' EXIT
    renderdoccmd capture -w -c "$PWD/target/renderdoc/frame" target/debug/examples/{{name}}
    capture=$(find target/renderdoc -name '*.rdc' -newer "$marker" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$capture" ]; then
        echo "RenderDoc saved no capture" >&2
        exit 1
    fi
    echo "$PWD/$capture"
    if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        qrenderdoc "$PWD/$capture" >/dev/null 2>&1 &
    fi

# Build the library
build:
    cargo build

# Build everything, and check the library still builds without cudarc
check:
    cargo check --all-targets
    cargo check --no-default-features

# Run the tests, optionally only those matching a filter. Needs a Vulkan 1.3 GPU
test *filter:
    cargo test {{filter}}

# Run the tests under Vulkan validation, and fail on any validation error
validate:
    #!/usr/bin/env bash
    set -euo pipefail
    log=$(VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_VALIDATE_SYNC=1 \
        cargo test -- --test-threads=1 --nocapture)
    errors=$(grep -c 'Validation Error' <<<"$log" || true)
    if [ "$errors" != 0 ]; then
        grep -A8 'Validation Error' <<<"$log"
        echo "$errors validation errors" >&2
        exit 1
    fi
    echo "no validation errors"

# Run everything that needs real CUDA, failing rather than skipping. Needs an NVIDIA GPU
test-nvidia:
    CUDASH_REQUIRE_NVIDIA=1 cargo test -- --nocapture
    cargo run --example slang_cudarc

# Rebuild the Slang kernels and list what was generated
kernels:
    #!/usr/bin/env bash
    set -euo pipefail
    marker=$(mktemp)
    trap 'rm -f "$marker"' EXIT
    # touching build.rs left stale kernels, so drop cudash's build output and force the build script to run
    cargo clean -p cudash
    cargo build
    kernels=$(find target/debug/build -path '*/out/*' \( -name '*.spv' -o -name '*.ptx' \) -newer "$marker")
    if [ -z "$kernels" ]; then
        echo "build script wrote no kernels" >&2
        exit 1
    fi
    echo "$kernels"

# Run an example, such as slang_spirv
example name:
    cargo run --example {{name}}

# Run a benchmark: launch or transfer on any Vulkan 1.3 GPU, memory on NVIDIA only
bench name:
    cargo bench --bench {{name}}

# Format the code and insisting on imports_granularity = "Crate" so we need nightly.
fmt:
    cargo +nightly fmt

# Lint everything, including interop, treating warnings as errors
lint:
    cargo +nightly fmt --check
    cargo +nightly clippy --all-targets -- -D warnings

# Build the API docs and open them
doc:
    cargo doc --no-deps --open

# Run all the checks that work without an NVIDIA GPU. `validate` runs the whole test suite
ci: lint check validate

# Delete build output
clean:
    cargo clean
