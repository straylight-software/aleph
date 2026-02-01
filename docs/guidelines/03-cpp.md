```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                                           // straylight // cpp
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "The sky above the port was the color of television, tuned to a dead
     channel."

                                                               — Neuromancer
```

# `// straylight // cpp`

## `// strategy // motivation`

We use C++ in situations where we need to do something extreme along one or more dimensions: we are
in a regime where no compromise is possible. Typically we do this by having low-friction access to
efficient, ergonomic implementations of best-in-class algorithms. Sometimes, we have the opportunity
to do something best-in-class ourselves; we consider such proposals with open minds and healthy
skepticism. Our C++ codebase and the investment represented by maintaining it is the optionality
premium on these degrees of freedom.

Much if not most excellent modern C++ code is proprietary because worthwhile C++ code is expensive
and most contemporary projects don't need it. This leads to a situation where it is difficult to
learn well outside of an elite technology or finance company. For non-commercial examples of extreme
requirements, consider people working at the frontiers of human knowledge: CERN has excellent code
because they operate in regimes that would be daunting for any company.

This document is aimed at three audiences:

- Experienced C++ programmers who have missed recent developments
- Programmers new to serious C++ who want to skip learning curve friction
- Agents with extensive informational resources who need clear guidelines

## `// basic // guidelines`

- We fully qualify *everything*, "What do you mean Norman?", ["EVERYTHING!"](https://www.youtube.com/watch?v=74BzSTQCl_c)
- We don't say `cuda` by choice, we use the same abbreviation that NVIDIA does: `nv`.
- The `straylight::` namespace contains general-purpose utilities.
- The `s4::` namespace contains most of our `nv`-adjacent code.
- The `libmodern-cpp` libraries are preferred over equivalent alternatives.
-

## `// economics // agent-heavy // development`

**In a codebase with heavy agent contribution, traditional economics invert:**

- Code is written once by agents in seconds
- Code is read hundreds of times by humans and agents
- Code is debugged when you're under pressure by tired humans
- Code is modified by agents who lack the original context

**Every ambiguity compounds exponentially.**

### `// fundamental // principle`

```cpp

// this costs an agent 0.1 seconds to write, a human 10 seconds to debug:
auto e = edge{};
if (e.p > 0) process(e);

// this costs an agent 0.2 seconds to write, saves hours of cumulative confusion:
auto inference_configuration = s4::inference::config::engine{};
if (inference_configuration.batch_size > 0) {
  initialize_inference_engine(inference_configuration);
}
```

**Optimize for disambiguation, not brevity.**

### `// config // parsing // sacred`

Configuration parsing is the most critical code in any system because:

1. **Multiplication Effect**: One config bug affects every component
1. **Trust Boundary**: External input that everything else trusts implicitly
1. **Silent Corruption**: Config errors manifest as business logic failures
1. **Audit Trail**: In regulated environments, you must prove correct configuration

Config parsing should be **human-written**, **brutally simple**, and **fail-fast**.

## `// high-level // choices`

1. **Explicit Types over `AAA`** (for agents) — disambiguation beats brevity
1. **Fully qualified names** — no `using namespace`, absolute clarity
1. **C++23 features** — use modern constructs maximally
1. **Measure, don't guess** — data-driven optimization
1. **Name for `grep`** — every identifier must be globally searchable

## `// mandatory // compiler // flags`

The straylight build system enforces these flags via `aleph.build.toolchain.cxx`.
They are **non-negotiable** and cannot be overridden by individual targets.

See `nix/modules/flake/build/options.nix` for the authoritative source.

### `// c // flags`

```
# ── optimization ───────────────────────────────────────────────

-O2                           # icache pressure > microbenchmark gains
-g3                           # maximum debug info
-gdwarf-5                     # modern debug format

# ── frame // pointers ──────────────────────────────────────────

-fno-omit-frame-pointer       # essential for profiling
-mno-omit-leaf-frame-pointer  # keep even in leaf functions

# ── reproducibility ────────────────────────────────────────────

-fdebug-prefix-map=/build/source=.
-ffile-prefix-map=/build/source=.
-fmacro-prefix-map=/build/source=.
-Wno-builtin-macro-redefined
-D__DATE__="redacted"
-D__TIMESTAMP__="redacted"
-D__TIME__="redacted"

# ── ub // mitigation ───────────────────────────────────────────

-fno-strict-aliasing          # routinely violated in practice
-fwrapv                       # signed overflow wraps (formal verification)
-fno-delete-null-pointer-checks

# ── security // disabled ───────────────────────────────────────

-U_FORTIFY_SOURCE             # interferes with verification
-D_FORTIFY_SOURCE=0
-fno-stack-protector          # override per-target if needed

# ── standard ───────────────────────────────────────────────────

-std=c23                      # no extensions

# ── warnings ───────────────────────────────────────────────────

-Wall
-Wextra
-Wpedantic
-Wshadow                      # variable shadowing
-Wcast-align                  # misaligned casts
-Wunused                      # unused anything
-Wconversion                  # narrowing conversions
-Wsign-conversion
-Wnull-dereference
-Wdouble-promotion            # float→double promotion
-Wformat=2                    # format string checking
-Wimplicit-fallthrough
-Wstrict-prototypes           # K&R style declarations
-Wmissing-prototypes

# ── codegen ────────────────────────────────────────────────────

-fdiagnostics-color=always
-fPIC                         # position-independent code
-fvisibility=hidden           # explicit exports only
```

### `// cxx // flags`

C++ flags include everything above (except C-specific warnings) plus:

```
# ── standard ───────────────────────────────────────────────────

-std=c++23                    # no extensions

# ── warnings // cpp-specific ───────────────────────────────────

-Wnon-virtual-dtor            # classic C++ footgun
-Wold-style-cast              # C-style casts hide intent
-Woverloaded-virtual          # virtual function hiding
-Wextra-semi                  # extra semicolons
-Wc++20-compat                # compatibility warnings
-Wc++23-extensions            # extensions beyond standard

# ── diagnostics ────────────────────────────────────────────────

-fdiagnostics-show-template-tree  # readable template errors
```

### `// rationale`

**`-O2` over `-O3`**: icache pressure and TLB thrashing are real problems that
vendors systematically underweight when tuning for microbenchmarks. This matters
more over time as memory hierarchies deepen. Per-target override to `-O3` is
fine when you've measured it helps.

**UB mitigation flags**: strict aliasing is routinely violated in practice.
Signed overflow wrapping and null pointer check preservation are required for
formal verification work where you need defined semantics.

**Security hardening disabled**: `_FORTIFY_SOURCE` and stack protector add
overhead and complexity that interferes with verification. Override per-target
with `-D_FORTIFY_SOURCE=3` and `-fstack-protector-strong` if needed.

**Visibility hidden**: symbols are hidden by default, explicit exports only via
visibility attributes. This produces smaller binaries and faster load times.

## `// naming // conventions`

### `// disambiguation // imperative`

In an agent-heavy codebase, names must be:

- **Globally unique** within their semantic domain
- **Self-documenting** without context
- **Searchable** with basic tools

```cpp
// BAD: Will create confusion at scale
class parser;
auto config = load();
int process(data& d);

// GOOD: Unambiguous even with 100 agents contributing
class tokenizer_engine;
auto inference_configuration = load_inference_configuration();
int process_tensor_batch(tensor_batch_data& batch);
```

### `// core // naming // rules`

- **snake_case** for everything: `tensor_batch`, `model_weights`, `execute_inference()`
- **Full words** over abbreviations: `configuration` not `config`, `connection` not `conn`
- **Domain prefixes** for common concepts: `nv_stream`, `device_memory`, `host_memory`
- **member_suffix\_** for members: `tensor_shape_`, `latency_us_`, `device_id_`
- **Preserve acronyms**: `NVFP4_quantizer` not `Nvfp4Quantizer`

### `// three-letter // rule`

If an abbreviation is less than 4 characters, it's too short:

```cpp
// BAD
auto cfg = load_cfg();
auto conn = db.get_conn();
auto res = process(req);

// GOOD
auto configuration = load_configuration();
auto connection = database.get_connection();
auto result = process_request(request);
```

### `// standard // abbreviations`

Only when the full name would be absurd:

- `idx/jdx/kdx` - index (prefer descriptive names like `row_index`)
- `rxbuf/txbuf` - receive/transmit buffer (domain-specific)
- `ctx` - context (only when type makes it unambiguous)

## `// code // organization`

### `// directory // structure`

```
s4/
├── core/           # Foundation utilities (exceptions, hash, workspace, nvtx)
│   ├── exceptions.h
│   ├── exceptions.cpp
│   ├── generator.h
│   └── workspace.h
├── nv/           # NV primitives and utilities
│   ├── nvfp4/
│   │   ├── nvfp4.h
│   │   ├── nvfp4.cuh
│   │   └── nvfp4.cu
│   └── cccl_standard.h
├── attention/      # Attention mechanisms and kernels
│   ├── sage_attention_plugin.h
│   ├── sage_attention_plugin.cu
│   └── score_correction.h
├── tensor/         # Tensor abstractions
│   ├── device_tensor.h
│   └── view.h
├── dtypes/         # Data type system
│   ├── dtype.h
│   ├── nv_types.h
│   └── dispatch.h
└── trt/            # TensorRT integration
  ├── affine_unary_plugin.h
  └── affine_unary_plugin.cu
```

- **Headers and implementations are adjacent** - `foo.h` and `foo.cpp` live together
- Test files live in separate `tests/` directory: `tests/unit/test_*.cpp`
- Property tests: `tests/property/test_*_properties.cpp`
- Python hypothesis tests: `tests/python/test_*_hypothesis.py`
- NV device code uses `.cu` extension, device-only headers use `.cuh`

### `// headers`

```cpp
#pragma once

#include <chrono>
#include <memory>
#include <span>

#include "s4/core/exceptions.h"
#include "s4/dtypes/dtype.h"
#include "s4/tensor/device_tensor.h"

namespace s4::inference {

class engine {  // Full descriptive names
public:
  engine();

  // full words in function names
  auto initialize_from_configuration(std::string configuration_path) noexcept
    -> s4::core::status;

  auto run_inference(std::span<const float> input_tensor) noexcept
    -> s4::core::result<tensor_batch>;

private:
  // clear member names with units where applicable
  std::unique_ptr<model_executor> executor_;
  std::chrono::microseconds inference_timeout_us_;
  int device_id_;
};

}  // namespace s4::inference
```

### `// implementation`

```cpp
#include "s4/inference/engine.h"

#include <format>

#include "s4/core/logging.h"
#include "s4/nv/device.h"

namespace s4::inference {

auto engine::initialize_from_configuration(
  std::string configuration_path) noexcept -> s4::core::status {

  // Descriptive variable names throughout
  auto configuration_result = s4::core::fs::read_file_to_string(configuration_path);

  if (!configuration_result) {
    return s4::core::fail(
      std::format("[s4] [inference] [engine] failed to read configuration: {}",
                  configuration_result.error().what()));
  }

  auto parsed_configuration = parse_inference_configuration(configuration_result.value());
  // ...

  return s4::core::ok();
}

}  // namespace s4::inference
```

## `// modern // cpp23 // patterns`

### `// core // hardware // realities`

Modern GPUs and CPUs are not the abstraction models from your CS courses, they're not even the ones you worked with a few years ago:

- **Cache lines are 64 bytes** - This is the unit of CPU memory transfer. On a GPU it's usually more.
- **Branches are heinously expensive** - A mispredicted branch costs 15-20 cycles on modern CPUs
- **The prefetcher is your friend** - Linear access patterns let it work magic
- **The compiler is your best optimizer** - With `-O3 -march=native`, it knows tricks you don't
- **This is even more true of Myelin** - When attempting to go fast on a GPU, you will almost never outsmart Myelin except when it has a pathological failure.

### `// performance // anti-patterns`

**Write simple, clear loops. The compiler will optimize them:**

```cpp
// BAD: Hand-rolled "optimization" that confuses compiler and humans
for (; data_index + 8 <= data_length; data_index += 8) {
  auto chunk = *reinterpret_cast<const uint64_t*>(data + data_index);
  // Complex bit manipulation
}

// GOOD: Clear intent, compiler optimizes perfectly
for (std::size_t data_index = 0; data_index < data_length; ++data_index) {
  if (data[data_index] == target_value) {
    match_count++;
  }
}
```

### `// error // handling // philosophy`

We don't throw exceptions. We use `straylight::result<T>` and when something is truly unrecoverable:

```cpp
// when failure is recoverable - return result
auto parse_configuration(std::string_view configuration_json) noexcept
  -> straylight::core::result<s4::tritonserver::configuration> {

  if (configuration_json.empty()) {
    return straylight::fail<s4::tritonserver::configuration>("empty configuration string");
  }

  // parse...
  return straylight::ok(s4::tritonserver::configuration{...});
}

// when failure is unrecoverable - fatal and we do the postmortem...
if (!critical_resource_handle) {
  straylight::fatal("[s4] [tritonserver] critical resource unavailable: {}", resource_name);
}
```

### `// error // handling // patterns`

```cpp
// DO: Use specific fail overloads
if (size > max_size) {
  return straylight::fail<buffer>("buffer size {} exceeds maximum {}", size, max_size);
}

if (::listen(socket_fd, backlog) < 0) {
  return straylight::fail_errno<socket>("[s4] [models] failed to listen on socket");
}

// DON'T: build error messages manually when avoidable, 
if (size > max_size) {
  return straylight::fail<thrust::host_vector>(
      std::format("buffer size {} exceeds maximum {}", size, max_size));
}
```

### `// result // type // usage`

```cpp
// prefer explicit type parameters for `fail` - aids readability...
auto parse_config(std::string_view json) 
  -> straylight::result<s4::tritonserver::configuration> {

  if (json.empty()) {
    return straylight::fail<s4::tritonserver::configuration>(
        "empty configuration string");
  }

  // ...
}

// for functions returning status, the type parameter can be omitted
auto validate_connection() -> s4::core::status {

  if (!is_connected()) {
    return straylight::fail("not connected");  // T defaults to monostate
  }

  return straylight::ok();
}
```

### `// const // correctness`

```cpp
// DO: mark everything const that can be...
auto process_batch(const tensor_batch& batch_data) const noexcept -> straylight::status;

// DO: use const for local variables that don't change...
const auto configuration = load_configuration();
const auto batch_count = batches.size();

// DON'T: forget const on method that doesn't modify state...
auto get_status() -> status_code;  // n.b. should be const, often [[nodiscard]]...
```

### `// span // usage`

```cpp
// DO: use `span` or `mdspan` for non-owning array views...
auto process_batch(std::span<const inference_request> requests) -> straylight::status;

// DON'T: use raw pointer + size
auto process_batch(const inference_request* requests, std::size_t count) -> straylight::status;

// DO: use span for fixed-size buffers...
auto read_into(std::span<std::byte> buffer) -> straylight::result<std::size_t>;
```

## `// nv // gpu // patterns`

### `// cccl // modern // nv`

We use [NV C++ Core Libraries (CCCL)](https://nvidia.github.io/cccl/) for modern, standards-compliant NV code. As of March 2024, CCCL unifies Thrust, CUB, and libnvcxx.

**Key principle**: Always prefer `nv::std::` over `std::` - it works in both host and device code, works with NVRTC, and is tested for NV.

```cpp
#include <nv/std/span>
#include <nv/std/array>
#include <nv/stream_ref>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// DO: Use nv::std:: entities (not std::) for device compatibility
__global__ void process_kernel(nv::std::span<float> input_data,
                         nv::std::span<float> output_data) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < input_data.size()) {
    output_data[thread_id] = input_data[thread_id] * 2.0f;
  }
}

// DO: Use nv::stream_ref for stream management
auto launch_inference_kernel(nv::stream_ref stream,
    std::span<const float> device_input) -> straylight::status {

  constexpr auto threads_per_block = 256;
  auto block_count = (device_input.size() + threads_per_block - 1) / threads_per_block;

  process_kernel<<<block_count, threads_per_block, 0, stream>>>(
      nv::std::span{device_input.data(), device_input.size()},
      // ...
      );

  return s4::nv::check_last_error();
}
```

### `// thrust // vectors`

[Thrust](https://nvidia.github.io/cccl/thrust/) provides STL-like containers for host and device memory:

```cpp
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>
#include <thrust/async/copy.h>

// DO: Use thrust::device_vector for device-side data
auto prepare_inference_batch(std::span<const float> host_data)
  -> straylight::result<thrust::device_vector<float>> {

  // Host vector with STL-like interface
  auto host_batch = thrust::host_vector<float>(host_data.begin(), host_data.end());

  // Transfer to device (synchronous) - type deduced
  auto device_batch = host_batch;

  return straylight::ok(std::move(device_batch));
}

// DO: Use thrust::async for non-blocking operations
auto prepare_batch_async(std::span<const float> host_data,
                   nvStream_t stream)
  -> thrust::device_future<thrust::device_vector<float>> {

  auto host_batch = thrust::host_vector<float>(host_data.begin(), host_data.end());
  auto device_batch = thrust::device_vector<float>(host_batch.size());

  // Asynchronous copy
  return thrust::async::copy(thrust::device.on(stream),
                       host_batch.begin(), host_batch.end(),
                       device_batch.begin());
}

// DO: Use thrust::universal_vector for unified memory scenarios
// Accessible by both host and device without explicit transfers

auto shared_buffer = thrust::universal_vector<float>(batch_size);

// DON'T: Access individual device_vector elements in loops
// Each access requires nvMemcpy!

for (auto idx = 0; idx < device_vec.size(); ++idx) {
  auto value = device_vec[idx];  // BAD: N nvMemcpy calls
}

// DO: Transfer once, process in bulk
auto host_copy = device_vec;  // One transfer, type deduced
for (auto idx = 0; idx < host_copy.size(); ++idx) {
  auto value = host_copy[idx];  // GOOD: Local memory access
}
```

### `// mdspan // multidimensional`

[mdspan](https://github.com/kokkos/mdspan) provides non-owning views of multidimensional arrays. NV support is available via [Kokkos implementation](https://github.com/kokkos/mdspan):

```cpp
#include <mdspan>

// DO: Use mdspan for type-safe multidimensional indexing
template<typename T>
using matrix_view = std::mdspan<T, std::dextents<size_t, 2>>;

template<typename T>
using tensor3d_view = std::mdspan<T, std::dextents<size_t, 3>>;

// DO: Express tensor operations with clear dimensionality
auto quantize_weight_matrix(matrix_view<const float> weights_fp32,
                      matrix_view<uint8_t> weights_nvfp4,
                      float scale_factor) -> s4::core::status {

  if (weights_fp32.extent(0) != weights_nvfp4.extent(0) ||
      weights_fp32.extent(1) != weights_nvfp4.extent(1)) {

    return straylight::fail("dimension mismatch: fp32[{},{}] vs nvfp4[{},{}]",
        weights_fp32.extent(0), weights_fp32.extent(1),
        weights_nvfp4.extent(0), weights_nvfp4.extent(1));
  }

  // C++23 bracket operator for multidimensional access
  // with `cute-mdspan` this can tile and swizzle...

  for (auto idx = 0; idx < weights_fp32.extent(0); ++idx) {
    for (auto jdx = 0; jdx < weights_fp32.extent(1); ++jdx) {
      weights_nvfp4[idx, jdx] = quantize_value(weights_fp32[idx, jdx], scale_factor);
    }
  }

  return straylight::ok();
}

// DO: Use mdspan for batch tensor layouts (N, C, H, W)
auto process_image_batch(s4::tensor3d_view<const float> batch,  // [batch, height, width]
                         std::size_t channels) -> s4::core::status {

  auto batch_size = batch.extent(0);
  auto height = batch.extent(1);
  auto width = batch.extent(2);

  straylight::info("[s4] [tensor] processing batch shape=[{},{},{}] channels={}",
      batch_size, height, width, channels);

  // clear dimensional semantics
  return straylight::ok();
}
```

### `// cutlass // cute::tensor`

[CUTLASS cute::Tensor](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html) provides layout-aware tensor abstractions for high-performance kernels:

```cpp
#include <cute/tensor.hpp>

using namespace cute;

// DO: Use cute::Tensor for layout-aware kernel code
// DO: Consider `cute-mdspan` where available

template<class T, class Layout>
__global__ void gemm_kernel(Tensor<T, Layout> const& A,
                     Tensor<T, Layout> const& B,
                     Tensor<T, Layout>& C) {

  // cute::Tensor provides hierarchical operations
  auto tile_shape = make_shape(Int<16>{}, Int<16>{});

  // Access with logical coordinates
  for (auto idx = 0; idx < size<0>(A); ++idx) {
    for (auto jdx = 0; jdx < size<1>(B); ++jdx) {
      C(idx, jdx) = A(idx, 0) * B(0, jdx);  // Simplified GEMM
    }
  }
}

// DO: Create tensors with explicit layout control
auto create_row_major_tensor(float* device_ptr, std::size_t rows, std::size_t cols) {

  auto shape = make_shape(rows, cols);
  auto stride = make_stride(cols, Int<1>{});  // row-major: stride by cols
  auto layout = make_layout(shape, stride);

  return make_tensor(device_ptr, layout);
}

// DO: Use cute for copy algorithms with optimal layouts
template<class TA, class ALayout, class TB, class BLayout>
__global__ void copy_kernel(Tensor<TA, ALayout> const& src,
                     Tensor<TB, BLayout>& dst) {

  // ceneric copy that respects layout
  for (auto idx = 0; idx < size(src); ++idx) {
    dst(idx) = src(idx);
  }
}

// DO: Integrate with PyTorch via dlpack (Python API, 2025)
// Python: cute_tensor = cute.from_dlpack(torch_tensor)
// Access shape, stride, memspace, element_type attributes
```

### `// nvfp4 // quantization`

NVFP4 (4-bit floating point) requires careful handling for optimal inference performance:

```cpp
namespace s4::quantization {

// explicit quantization configuration
struct nvfp4_config {
  float scale_factor;
  float zero_point;
  bool use_symmetric_quantization;
  std::size_t block_size;  // quantization block size in elements
};

// DO: Make quantization operations explicit and verifiable
auto quantize_tensor_to_nvfp4(nv::std::span<const float> input_fp32,
                              nv::std::span<uint8_t> output_nvfp4,
                              const s4::nvfp4_config& config,
                              nv::stream_ref stream)
  -> s4::core::result<quantization_metadata> {

  if (input_fp32.size() * 4 / 8 != output_nvfp4.size()) {
    return straylight::fail<s4::quantization_metadata>(
      "output buffer size mismatch: expected {} bytes, got {}",
      input_fp32.size() / 2, output_nvfp4.size());
  }

  // Launch quantization kernel with explicit block size
  constexpr auto threads_per_block = 256;
  auto block_count = (input_fp32.size() + config.block_size - 1) / config.block_size;

  nvfp4_quantize_kernel<<<block_count, threads_per_block, 0, stream>>>(
    input_fp32, output_nvfp4, config);

  if (auto error = s4::nv::check_last_error(); !error) {
    return straylight::fail<quantization_metadata>("quantization kernel failed: {}",
        error.error().what());
  }

  return straylight::ok(s4::quantization_metadata{config.scale_factor, config.zero_point});
}

}  // namespace s4::quantization
```

### `// myelin // tactics`

[TensorRT Myelin](https://docs.nvidia.com/deeplearning/tensorrt/) tactics for fused kernel generation:

```cpp
namespace s4::tensorrt {

// DO: Wrap Myelin tactics in type-safe interfaces
struct myelin_tactic_config {
  std::string tactic_name;
  std::vector<size_t> input_shapes;
  data_type precision;  // FP32, FP16, INT8, NVFP4
  std::size_t workspace_size_bytes;
};

// DO: Make tactic selection explicit and logged
auto select_myelin_tactic(const model_layer& layer,
                   const execution_context& context)
  -> s4::core::result<s4::myelin_tactic_config> {

  auto available_tactics = query_available_tactics(layer, context);

  if (available_tactics.empty()) {
    return s4::fail<myelin_tactic_config>(
      "no myelin tactics available for layer: {}", layer.name);
  }

  // select based on measured performance
  auto selected_tactic = profile_and_select_best(available_tactics, context);

  s4::info("[s4] [tensorrt] [myelin] selected tactic `{}` for layer `{}` "
      "(workspace: {} MB, precision: {})",
      selected_tactic.tactic_name, layer.name,
      selected_tactic.workspace_size_bytes / (1024 * 1024),
      to_string(selected_tactic.precision));

  return s4::ok(selected_tactic);
}

}  // namespace s4::tensorrt
```

### `// stream // management`

```cpp
namespace s4::nv {

// DO: Use RAII for stream management
struct scoped_stream {

  scoped_stream() {
    if (auto result = create_stream(); !result) {
      s4::fatal("failed to create NV stream: {}", result.error().what());
    }
  }

  ~scoped_stream() noexcept {
    if (stream_handle_) {
      nvStreamDestroy(stream_handle_);
    }
  }

  // Non-copyable, movable
  scoped_stream(const scoped_stream&) = delete;
  scoped_stream(scoped_stream&& other) noexcept
  : stream_handle_(std::exchange(other.stream_handle_, nullptr)) {}

  auto get() const noexcept -> nvStream_t { return stream_handle_; }
  auto ref() const noexcept -> nv::stream_ref { return nv::stream_ref{stream_handle_}; }

  nvStream_t stream_handle_ = nullptr;
};

// DO: Use stream ordering for complex pipelines
auto execute_inference_pipeline(const s4::model& model_instance,
                         std::span<const float> input_data)
  -> s4::core::result<tensor_batch> {

  s4::scoped_stream preprocessing_stream;
  s4::scoped_stream inference_stream;
  s4::scoped_stream postprocessing_stream;

  // launch preprocessing (independent)
  preprocess_input_async(input_data, preprocessing_stream.ref());

  // synchronize and launch inference
  nvStreamWaitEvent(inference_stream.get(), preprocessing_done_event);
  run_inference_async(model_instance, inference_stream.ref());

  // synchronize and launch postprocessing
  nvStreamWaitEvent(postprocessing_stream.get(), inference_done_event);
  postprocess_output_async(postprocessing_stream.ref());

  return straylight::ok(/* result */);
}
}  // namespace s4::nv
```

### `// device // memory // management`

```cpp
namespace s4::nv {

// DO: Use typed wrappers for device memory
template<typename T>
class device_buffer {
public:
  explicit device_buffer(size_t element_count) : count_(element_count) {
    auto alloc_result = allocate_device_memory(element_count * sizeof(T));
    if (!alloc_result) {
      s4::fatal("failed to allocate device memory: {}", alloc_result.error().what());
    }
    data_ = static_cast<T*>(alloc_result.value());
  }

  ~device_buffer() noexcept {
    if (data_) {
      nvFree(data_);
    }
  }

  // Non-copyable, movable
  device_buffer(const device_buffer&) = delete;
  device_buffer(device_buffer&& other) noexcept
  : data_(std::exchange(other.data_, nullptr))
  , count_(std::exchange(other.count_, 0)) {}

  auto data() noexcept -> T* { return data_; }
  auto data() const noexcept -> const T* { return data_; }
  auto size() const noexcept { return count_; }
  auto size_bytes() const noexcept { return count_ * sizeof(T); }

  auto span() noexcept -> nv::std::span<T> { return {data_, count_}; }
  auto span() const noexcept -> nv::std::span<const T> { return {data_, count_}; }

private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

// DO: Make host-device transfers explicit
auto copy_to_device_async(std::span<const float> host_data,
                   device_buffer<float>& device_buffer,
                   nv::stream_ref stream) -> s4::core::status {

  if (host_data.size() != device_buffer.size()) {
    return s4::fail("size mismatch: host {} elements, device {} elements",
                    host_data.size(), device_buffer.size());
  }

  auto result = nvMemcpyAsync(device_buffer.data(),
                                host_data.data(),
                                device_buffer.size_bytes(),
                                nvMemcpyHostToDevice,
                                stream);

  if (result != nvSuccess) {
    return s4::fail_errno<void>("nvMemcpyAsync failed");
  }

  return s4::ok();
}

}  // namespace s4::nv
```

### `// nv // error // handling`

```cpp
namespace s4::nv {

// DO: Check every NV call
auto check_nv_error(nvError_t error, std::string_view operation) -> s4::core::status {
  if (error != nvSuccess) {
    return s4::fail("NV operation '{}' failed: {} (code: {})",
                    operation, nvGetErrorString(error), static_cast<int>(error));
  }
  return s4::ok();
}

// DO: Macro for inline error checking (use sparingly)
#define S4_NV_CHECK(call) \
  do { \
    if (auto _error = (call); _error != nvSuccess) { \
      return s4::fail("NV call '" #call "' failed: {} at {}:{}", \
                      nvGetErrorString(_error), __FILE__, __LINE__); \
    } \
  } while (0)

// DO: Check for asynchronous errors after kernel launches
auto check_last_error() -> s4::core::status {
  if (auto error = nvGetLastError(); error != nvSuccess) {
    return s4::fail("NV kernel launch failed: {}", nvGetErrorString(error));
  }
  return s4::ok();
}

}  // namespace s4::nv
```

### `// kernel // launch`

```cpp
// DO: Document kernel launch parameters
namespace s4::kernels {

struct launch_config {
  dim3 grid_dimensions;       // Number of blocks
  dim3 block_dimensions;      // Threads per block
  size_t shared_memory_bytes; // Dynamic shared memory
  nvStream_t stream;
};

// DO: Provide clear launch configuration calculators
auto calculate_1d_launch_config(size_t total_elements,
                          size_t threads_per_block = 256)
  -> launch_config {

  auto block_count = (total_elements + threads_per_block - 1) / threads_per_block;

  return launch_config{
    .grid_dimensions = dim3(block_count),
    .block_dimensions = dim3(threads_per_block),
    .shared_memory_bytes = 0,
    .stream = nullptr
  };
}

// DO: Log kernel launches in debug builds
template<typename KernelFunc, typename... Args>
auto launch_kernel(const char* kernel_name,
            const launch_config& config,
            KernelFunc kernel,
            Args&&... args) -> s4::core::status {

#ifndef NDEBUG
  s4::debug("[s4] [nv] [kernel] launching '{}' with grid({},{},{}) block({},{},{})",
       kernel_name,
       config.grid_dimensions.x, config.grid_dimensions.y, config.grid_dimensions.z,
       config.block_dimensions.x, config.block_dimensions.y, config.block_dimensions.z);
#endif

  kernel<<<config.grid_dimensions, config.block_dimensions,
           config.shared_memory_bytes, config.stream>>>(
    std::forward<Args>(args)...);

  return check_last_error();
}

}  // namespace s4::kernels
```

## `// agent-human // collaboration`

### `// critical // path // marking`

Identify code requiring human review:

```cpp
// CRITICAL PATH: Model quantization - human review required
namespace s4::quantization {
  // Config parsing errors here corrupt inference results
  auto parse_quantization_config(std::string_view config_json)
  -> s4::core::result<quantization_config> {
  // Human-written parser with aggressive validation
  }
}

// AUXILIARY: Metrics collection - agent generation acceptable
namespace s4::metrics {
  // Agent can generate this boilerplate
}
```

### `// legacy // apis`

When core APIs can't be changed without breaking everything:

1. **Add better-named aliases** alongside existing functions
1. **Use the new names in new code** to model good patterns
1. **Document the preferred style** in comments
1. **Gradually migrate** during other refactoring

```cpp
// Example: result.h evolution
// Old API (keep for compatibility):
auto ok(T value) -> result<T>;
auto fail(string msg) -> result<T>;

// New aliases (use in new code):
auto make_success(T value) -> result<T>;
auto make_error(string message) -> result<T>;
```

## `// testing // philosophy`

### `// five-minute // rule`

If you can't understand what agent-generated code does in 5 minutes, regenerate it with better
structure.

### `// property-based // testing`

Agents generate thorough unit tests but miss semantic invariants:

```cpp
// Agent-generated test - thorough but mechanical

TEST_CASE("tokenizer handles empty input") {
  auto tokenize_result = tokenize_input("");
  REQUIRE(!tokenize_result.has_value());
}

// Human-written property test - catches semantic violations
TEST_CASE("quantizer preserves tensor shape") {
  check_property([](const tensor_fp32& input_tensor) {
    auto quantized_tensor = quantize_to_nvfp4(input_tensor);
    if (!quantized_tensor) return true;

    return quantized_tensor->shape == input_tensor.shape &&
           quantized_tensor->rank == input_tensor.rank;
  });
}
```

### `// testing // error // handling`

```cpp
// Check error content
REQUIRE(!result.has_value());
CHECK(!result.error().what().empty());
CHECK_THAT(result.error().what(), ContainsSubstring("expected text"));

// Check error codes
if (auto code = result.error().code()) {
  CHECK(code->value() == ENOENT);
}

// Check formatted errors work
auto error = s4::fail<int>("failed at position {}", 42);
CHECK_THAT(error.error().what(), ContainsSubstring("failed at position 42"));
```

### `// fuzz // testing`

```cpp
// Add fuzz tests for any parser handling external input
FUZZ_TEST(configuration_parser, random_input) {
  auto result = parse_configuration(fuzz_input);
  // Should never crash, only return error
  if (result) {
    validate_configuration_invariants(*result);
  }
}
```

## `// debugging // patterns`

### `// grep // test`

Every function should be globally unique and searchable:

```bash
# BAD: Too many results
grep -r "process(" .          # 500 matches
grep -r "handler::" .         # 200 matches

# GOOD: Finds exactly what you need
grep -r "process_tensor_batch(" .     # 3 relevant matches
grep -r "quantization_handler::" .    # 10 specific matches
```

### `// state // machine // clarity`

Make states explicit for debugging:

```cpp
// BAD: Implicit state machines become agent debugging nightmares
if (flags & 0x04 && !error_flag && counter > threshold) {
  // What state is this?
}

// GOOD: Self-documenting states
enum class connection_state {
  disconnected,
  connecting,
  authenticated,
  active,
  draining
};

if (current_state == connection_state::authenticated &&
  error_count == 0 &&
  retry_counter > max_retries) {
  transition_to_state(connection_state::draining);
}
```

## `// performance // guidelines`

1. **Start with clear, simple code** - The compiler optimizes clarity
1. **Measure with production flags**: `-O3 -march=native`
1. **Small types belong in registers** - pass by value
1. **Profile before optimizing** - Data always surprises

```cpp
// Let the compiler work
for (const auto& request : pending_requests) {
  process_inference_request(request);
}

// Not this cleverness
for (auto idx = 0; idx < pending_requests.size(); idx += 4) {
  // Unrolled loop that's probably slower
}
```

### `// constexpr // usage`

```cpp
// DO: use constexpr for compile-time constants
constexpr size_t max_batch_size = 1024;
constexpr std::string_view model_architecture = "transformer";

// DO: mark functions constexpr when possible

constexpr auto calculate_tensor_size(std::uint64_t batch, 
                                     std::uint64_t seq_len, 
                                     std::uint64_t hidden_dim) 
    -> uint64_t {

  return batch * seq_len * hidden_dim;
}

// DON'T: force constexpr when it complicates implementation
constexpr auto complex_quantization() {  // Requires contortions
  // ...
}
```

## `// logging`

Hierarchical tagging for structured logs:

```cpp
straylight::info("[s4] [inference] [engine] [batch] executing batch id={} device={}",
   batch_id, device_id);
straylight::error("[s4] [inference] [engine] [error] inference failed: {}",
    error_description);
```

Format: `[project] [system] [component] [detail] message`

## `// configuration // philosophy`

### `// parse // upfront`

```cpp
// Parse and validate entire config at startup
auto load_system_configuration(std::string_view config_path)
  -> s4::core::result<s4::system_configuration> {

  auto file_content = s4::core::fs::read_file_to_string(config_path);
  if (!file_content) {
    s4::fatal("Cannot read configuration file: {}", config_path);
  }

  auto parsed_config = s4::util::parse_toml_configuration(file_content.value());
  if (!parsed_config) {
    s4::fatal("Invalid configuration: {}", parsed_config.error().what());
  }

  auto validation_result = validate_configuration(parsed_config.value());
  if (!validation_result) {
    straylight::fatal("[s4] [init] configuration validation failed: {}",
                      validation_result.error().what());
  }

  return straylight::ok(parsed_config.value());
}
```

### `// configuration // errors // fatal`

If configuration is wrong, nothing else can be trusted:

```cpp
if (!model_config.has_valid_weights_path()) {
  straylight::fatal("[s4] [models] model configuration missing weights path");
}

if (inference_config.max_batch_size <= 0) {
  straylight::fatal("[s4] [gemm] invalid max_batch_size: {}", inference_config.max_batch_size);
}
```

## `// api // evolution`

When core APIs need updates:

1. **Start with backwards compatibility** - Keep old functions working
1. **Fix fundamental issues** - Like string lifetime problems
1. **Add better alternatives** - New overloads following style guide
1. **Constexpr where reasonable** - Don't force it if it complicates
1. **Document breaking changes** - Even minor ones like `error_code()` → `code()`

### `// incremental // improvement`

For widely-used modules like `s4::core::result`:

1. **Never break existing code** - Aliases are cheap
1. **Model better patterns** in new functions
1. **Update documentation** to prefer new patterns
1. **Consider `[[deprecated]]`** only after wide adoption

## `// anti-patterns`

### `// abbreviation // cascade`

```cpp
// Starts innocent...
auto cfg = load_config();

// Spreads like a virus...
auto conn = create_conn(cfg);
auto mgr = conn_mgr(conn);
auto proc = mgr.get_proc();

// Ends in debugging hell
if (!proc.is_valid()) {  // What is proc again?
  // ...
}
```

### `// context-dependent // names`

```cpp
// BAD: "decoder" means different things in different places
namespace tokenizer {
  class decoder;  // Decodes tokens
}
namespace model {
  class decoder;  // Transformer decoder layer
}

// GOOD: Names carry their domain
namespace tokenizer {
  class token_decoder;
}
namespace model {
  class transformer_decoder_layer;
}
```

### `// implicit // state // machines`

```cpp
// BAD: State spread across booleans
bool is_connected;
bool is_authenticated;
bool is_active;
bool has_error;

// GOOD: Explicit state
enum class session_state {
  disconnected,
  connected_unauthenticated,
  authenticated_inactive,
  active,
  error_recovery
};
```

## `// summary`

In an agent-heavy codebase:

1. **Every name must be globally unambiguous**
1. **Every abbreviation creates exponential confusion**
1. **Every implicit assumption becomes a debugging nightmare**
1. **Every configuration error multiplies across the system**

Write code as if 100 agents will be pattern-matching against it tomorrow, and a tired human will be
debugging it at 3am next month. Because both will happen.

The Unix authors optimized for scarce memory. We optimize for scarce human comprehension. In 1970,
every character cost bytes. In 2025, every ambiguity costs hours.

## `// required // reading`

### `// performance`

- [CppCon 2017: Carl Cook "When a Microsecond Is an Eternity"](https://www.youtube.com/watch?v=NH1Tta7purM)
- [Cliff Click: "A Lock-Free Hash Table"](https://www.youtube.com/watch?v=HJ-719EGIts)
- [Andrei Alexandrescu: "Optimization Tips"](https://www.youtube.com/watch?v=Qq_WaiwzOtI)

### `// modern // cpp`

- [GotW #94: "AAA Style (Almost Always Auto)"](https://herbsutter.com/2013/08/12/gotw-94-solution-aaa-style-almost-always-auto/)
- [Abseil: "The Danger of Atomic Operations"](https://abseil.io/docs/cpp/atomic_danger)

## `// living // list // great // code`

**Tier 1** (Perfection - Study every line)

- [simdjson](https://github.com/simdjson/simdjson) - SIMD JSON parsing, exemplary modern C++
- [Abseil](https://github.com/abseil/abseil-cpp) - Google's foundation library, production-hardened
- [fmt](https://github.com/fmtlib/fmt) - The formatting library that became std::format

**Tier 2** (Domain Excellence - Best-in-class for their problem space)

- [DuckDB](https://github.com/duckdb/duckdb) - Analytical database, zero dependencies, clean
  architecture
- [RocksDB](https://github.com/facebook/rocksdb) - LSM storage engine, battle-tested at scale
- [DPDK](https://github.com/DPDK/dpdk) - Kernel bypass networking, when microseconds matter
- [ClickHouse](https://github.com/ClickHouse/ClickHouse) - Columnar database, SIMD everywhere

**Tier 3** (Specific Excellence - Outstanding implementations of focused problems)

- [parallel-hashmap](https://github.com/greg7mdp/parallel-hashmap) - Swiss tables with parallel
  access
- [concurrentqueue](https://github.com/cameron314/concurrentqueue) - Lock-free queue that actually
  works
- [mimalloc](https://github.com/microsoft/mimalloc) - Microsoft's superb allocator
- [liburing](https://github.com/axboe/liburing) - io_uring done right (see kernel code too)

**Study Specific Files/Techniques**

- Facebook's [F14](https://github.com/facebook/folly/blob/main/folly/container/F14.md) - Vector
  instructions in hash tables
- Google's [SwissTable](https://abseil.io/about/design/swisstables) - The hash table design that
  conquered all
- Lemire's [streamvbyte](https://github.com/lemire/streamvbyte) - SIMD integer compression
- [Aeron](https://github.com/real-logic/aeron) - Reliable UDP messaging, mechanical sympathy
  exemplar

**Controversial but Instructive**

- [Seastar](https://github.com/scylladb/seastar) - Futures done differently, polarizing but
  educational
- [EASTL](https://github.com/electronicarts/EASTL) - EA's STL replacement, different tradeoffs
- [Boost.Asio](https://github.com/boostorg/asio) - The async model that influenced networking TS

**Required Reading (Papers/Docs)**

- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)
  \- Drepper's classic
- [Can Seqlocks Get Along With Programming Language Memory Models?](https://www.hpl.hp.com/techreports/2012/HPL-2012-68.pdf)
  \- Hans Boehm on the hard stuff
- [There is No Fork](https://www.microsoft.com/en-us/research/uploads/prod/2019/04/fork-hotos19.pdf)
  \- Microsoft Research on process creation

**What Makes Code "Great" for This List**

1. **Clarity despite complexity** - Solving hard problems with readable code
1. **Performance without compromise** - Fast but not at the expense of correctness
1. **Teaching value** - You become a better programmer by reading it
1. **Battle-tested** - Used in production at serious scale
1. **Influential** - Changed how we think about the problem

**What Doesn't Belong**

- Clever for cleverness' sake
- Template metaprogramming gymnastics without purpose
- "Look how few lines!" code golf
- Abandoned experiments (unless historically important)
