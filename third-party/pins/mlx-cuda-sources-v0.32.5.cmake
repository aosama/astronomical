# Immutable pins for the sources MLX v0.32.5's CUDA backend declares in
# mlx/backend/cuda/CMakeLists.txt. A Linux build with the CUDA backend
# (issue #2699) provisions them like every other native archive, and the
# native CMake build declares them first under MLX's own dependency names, so
# MLX's FetchContent requests are satisfied from verified local archives and
# configure never fetches from the network. The versions are exactly the
# ones MLX v0.32.5 declares; bump them together with the MLX pin.

# CCCL (libcu++, CUB, Thrust): MLX compiles against it and installs its
# headers for its run-time CUDA compiler. MLX fetches the release asset, whose
# flattened include/ the build relies on, so the pin is that asset.
set(ASTRONOMICAL_CCCL_VERSION "3.1.3")
set(ASTRONOMICAL_CCCL_SOURCE_ARCHIVE_URL "https://github.com/NVIDIA/cccl/releases/download/v3.1.3/cccl-v3.1.3.zip")
set(ASTRONOMICAL_CCCL_SOURCE_ARCHIVE_SHA256 "30f388ef784eb691d7de9d2cf918d53ab33464672500a17945b99d16af136cd6")
set(ASTRONOMICAL_CCCL_SOURCE_ARCHIVE_FILE_NAME "cccl-v3.1.3.zip")

# NVTX: MLX's range annotations (header-only, its c/ folder).
set(ASTRONOMICAL_NVTX_VERSION "3.1.1")
set(ASTRONOMICAL_NVTX_GIT_COMMIT "6230bdf710bc94f44d433acceba735aaa9090ba5")
set(ASTRONOMICAL_NVTX_SOURCE_ARCHIVE_URL "https://github.com/NVIDIA/NVTX/archive/6230bdf710bc94f44d433acceba735aaa9090ba5.tar.gz")
set(ASTRONOMICAL_NVTX_SOURCE_ARCHIVE_SHA256 "1c515cf0e6c05597b62f3b3a190f7a759446597aa88a27afdc08d1730a905e29")
set(ASTRONOMICAL_NVTX_SOURCE_ARCHIVE_FILE_NAME "nvtx-3.1.1-6230bdf710bc94f44d433acceba735aaa9090ba5.tar.gz")

# cuDNN frontend: the header-only C++ API MLX's attention and convolutions
# call cuDNN through (the cuDNN libraries themselves come from the system).
set(ASTRONOMICAL_CUDNN_FRONTEND_VERSION "1.16.0")
set(ASTRONOMICAL_CUDNN_FRONTEND_GIT_COMMIT "be6c079be8aaffa0fc079fcf039887e637c289c7")
set(ASTRONOMICAL_CUDNN_FRONTEND_SOURCE_ARCHIVE_URL "https://github.com/NVIDIA/cudnn-frontend/archive/be6c079be8aaffa0fc079fcf039887e637c289c7.tar.gz")
set(ASTRONOMICAL_CUDNN_FRONTEND_SOURCE_ARCHIVE_SHA256 "bf3eef8b07394a40f73a510f6ce1d1deb11de253c5cff46a71adc9ab6fee3b00")
set(ASTRONOMICAL_CUDNN_FRONTEND_SOURCE_ARCHIVE_FILE_NAME "cudnn-frontend-1.16.0-be6c079be8aaffa0fc079fcf039887e637c289c7.tar.gz")

# CUTLASS and CuTe: MLX's quantized and grouped matrix multiplies, compiled
# ahead of time and at run time (qmm_naive), so its headers ship with the
# native image for MLX's run-time CUDA compiler.
set(ASTRONOMICAL_CUTLASS_VERSION "4.4.2")
set(ASTRONOMICAL_CUTLASS_GIT_COMMIT "da5e086dab31d63815acafdac9a9c5893b1c69e2")
set(ASTRONOMICAL_CUTLASS_SOURCE_ARCHIVE_URL "https://github.com/NVIDIA/cutlass/archive/da5e086dab31d63815acafdac9a9c5893b1c69e2.tar.gz")
set(ASTRONOMICAL_CUTLASS_SOURCE_ARCHIVE_SHA256 "7f735e3e8cf03ef3d898edf7a7329f92b5cf7bc5bb9049a0299f524d43d37919")
set(ASTRONOMICAL_CUTLASS_SOURCE_ARCHIVE_FILE_NAME "cutlass-4.4.2-da5e086dab31d63815acafdac9a9c5893b1c69e2.tar.gz")
