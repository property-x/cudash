# cudash developer commands. Currently assumes `ubuntu`.

# List all commands
default:
    @just --list --unsorted

# Install everything needed to build and test. Safe to re-run. Runs without sudo as root, as in the Containerfile
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    sudo=sudo
    [ "$(id -u)" -ne 0 ] || sudo=
    if ! command -v rustup >/dev/null; then
        echo "Install Rust first: https://rustup.rs" >&2
        exit 1
    fi
    rustup component add rustfmt clippy
    rustup toolchain install nightly --component rustfmt
    # Ubuntu's mdbook is 0.4.x, and 0.5 is current. cargo skips the build when this version is already installed.
    cargo install --locked mdbook --version 0.5.4
    # libnvrtc12 is in Ubuntu's multiverse repository, and doesn't need an NVIDIA GPU.
    $sudo apt-get update
    $sudo apt-get install -y build-essential slang-compiler libnvrtc12 mesa-vulkan-drivers \
        vulkan-tools vulkan-validationlayers spirv-tools
    # Slang looks for the unversioned library name, which the package doesn't provide.
    $sudo ln -sf libnvrtc.so.12 /usr/lib/x86_64-linux-gnu/libnvrtc.so
    $sudo ldconfig
    # RenderDoc is unpacked by hand in /opt. Its Vulkan layer manifest names the build machine's path, so point it at /opt.
    # The layer only loads when ENABLE_VULKAN_RENDERDOC_CAPTURE is set, so registering it is harmless otherwise.
    if [ -f /opt/renderdoc_1.47/etc/vulkan/implicit_layer.d/renderdoc_capture.json ]; then
        $sudo install -d /etc/vulkan/implicit_layer.d
        sed -E 's#"library_path" *: *"[^"]*"#"library_path": "/opt/renderdoc_1.47/lib/librenderdoc.so"#' \
            /opt/renderdoc_1.47/etc/vulkan/implicit_layer.d/renderdoc_capture.json |
            $sudo tee /etc/vulkan/implicit_layer.d/renderdoc_capture.json >/dev/null
    fi
    # distrobox --nvidia mounts the host's NVIDIA manifest, which names Fedora's library path. Add a copy with Ubuntu's
    # path, only when the mounted one points at a missing file and Ubuntu's library exists. Otherwise do nothing.
    icd=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
    lib=$(sed -nE 's/.*"library_path" *: *"([^"]*)".*/\1/p' "$icd" 2>/dev/null || true)
    ubuntu_lib=/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
    if [ -n "$lib" ] && [ ! -e "$lib" ] && [ -e "$ubuntu_lib" ]; then
        $sudo install -d /etc/vulkan/icd.d
        sed "s#\"$lib\"#\"$ubuntu_lib\"#" "$icd" | $sudo tee /etc/vulkan/icd.d/nvidia_icd.json >/dev/null
    fi

# Show the tools and GPUs this machine has
doctor:
    #!/usr/bin/env bash
    for tool in cargo cc slangc spirv-val vulkaninfo nvidia-smi renderdoccmd mdbook; do
        printf '%-12s %s\n' "$tool" "$(command -v "$tool" || echo MISSING)"
    done
    printf '%-12s %s\n' nvrtc "$(ldconfig -p | grep -m1 'libnvrtc\.so ' | sed 's/.*=> //' || true)"
    printf '%-12s %s\n' renderdoc-layer "$(if [ -f /etc/vulkan/implicit_layer.d/renderdoc_capture.json ] || [ -f "$HOME/.local/share/vulkan/implicit_layer.d/renderdoc_capture.json" ]; then echo registered; else echo MISSING; fi)"
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

# Capture the off-screen frame_capture example in Nsight Graphics without opening the GUI.
nsight-capture-example name:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --example {{name}}
    executable="$PWD/target/debug/examples/{{name}}"
    nsight_graphics="${NSIGHT_GRAPHICS:-ngfx-ui-for-linux}"
    if ! command -v "$nsight_graphics" >/dev/null 2>&1; then
        echo "Nsight Graphics executable not found: $nsight_graphics" >&2
        exit 1
    fi
    nsight_capture="${NSIGHT_GRAPHICS_CAPTURE:-$(dirname "$(readlink -f "$(command -v "$nsight_graphics")")")/ngfx-capture}"
    if [ ! -x "$nsight_capture" ]; then
        echo "Nsight Graphics capture CLI not found: $nsight_capture" >&2
        exit 1
    fi
    mkdir -p target/nsight
    capture="$PWD/target/nsight/{{name}}-$(date +%s%N).ngfx-capture"
    frame_count="${NSIGHT_FRAME_COUNT:-1}"
    if [ "{{name}}" = "shader_animation" ]; then
        frame_count="${NSIGHT_FRAME_COUNT:-60}"
    fi
    capture_status=0
    if "$nsight_capture" \
        --exe "$executable" \
        --working-dir "$PWD" \
        --capture-countdown-timer 10 \
        --frame-count "$frame_count" \
        --delimiter-vk-frame-boundary-ext \
        --no-block-on-first-incompatibility \
        --no-hud \
        --output-file "$capture"; then
        :
    else
        capture_status=$?
    fi
    if [ ! -s "$capture" ]; then
        echo "Nsight Graphics wrote no capture; the GPU must support VK_EXT_frame_boundary" >&2
        exit 1
    fi
    if [ "$capture_status" -ne 0 ]; then
        echo "Nsight Graphics exited with status $capture_status after saving the capture" >&2
    fi
    printf 'Nsight Graphics capture: %s\n' "$capture"

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

# Build the Rust API reference (cargo doc) and open it in a browser
doc:
    cargo doc --no-deps --open

# Build the mdBook guide from docs/ into target/book, and print the path to its index.html
docs:
    #!/usr/bin/env bash
    mdbook build docs --dest-dir "$PWD/target/book"
    echo "$PWD/target/book/index.html"

# Serve the mdBook guide on localhost:3000 and rebuild on change
docs-serve:
    mdbook serve docs --dest-dir "$PWD/target/book" --open

# Run all the checks that work without an NVIDIA GPU. `validate` runs the whole test suite
ci: lint check validate

# Delete build output
clean:
    cargo clean
