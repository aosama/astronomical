# Two facts a Linux CUDA build records for the Rust side, which links the
# static MLX image itself and so cannot rely on CMake's link or on MLX finding
# its headers next to its own binary:
#
#   cuda/link-libraries.txt: the CUDA and cuDNN libraries MLX's CUDA backend
#     calls, one linker input per line. MLX's installed package records them
#     only as CMake targets (CUDA::cublasLt, CUDNN::cudnn_all), which have no
#     file a build script can link, so they are resolved here, where CMake
#     knows each file.
#   cuda/toolkit-include-directory.txt: the CUDA runtime headers MLX's
#     run-time compiler (NVRTC) needs, which the runtime hands MLX through
#     mlx_cuda_set_jit_header_directories instead of an environment variable.
find_package(CUDAToolkit REQUIRED)
# The cuDNN file MLX's own FindCUDNN chose, read from the CMake cache entry
# its find_library leaves (its CUDNN::cudnn target is local to MLX's folder).
# Searching again here could pick another file, or none where only
# libcudnn.so.<major> is installed, which MLX's search accepts.
set(astronomical_cudnn_library "$CACHE{cudnn_LIBRARY}")
if(NOT astronomical_cudnn_library)
    message(FATAL_ERROR "MLX's FindCUDNN recorded no cuDNN library (cudnn_LIBRARY)")
endif()

# Exactly the libraries mlx/backend/cuda/CMakeLists.txt links (cuBLASLt,
# cuFFT, cuSOLVER, NVRTC, the driver) plus the ones they and MLX's own CUDA
# objects need at link time: cuBLAS and the CUDA runtime. Revisit with every
# MLX bump.
set(astronomical_cuda_link_lines "")
foreach(cuda_library_target CUDA::cublasLt CUDA::cublas CUDA::cufft CUDA::cusolver CUDA::nvrtc CUDA::cudart)
    get_target_property(cuda_library_file ${cuda_library_target} IMPORTED_LOCATION)
    if(NOT cuda_library_file)
        message(FATAL_ERROR "CUDAToolkit found no library file for ${cuda_library_target}")
    endif()
    string(APPEND astronomical_cuda_link_lines "${cuda_library_file}\n")
endforeach()
string(APPEND astronomical_cuda_link_lines "${astronomical_cudnn_library}\n")
# The driver library ships with the NVIDIA driver, not the toolkit; the
# toolkit only carries a link stub, which must never become a run path. So it
# is named, and the loader finds the driver's own copy.
string(APPEND astronomical_cuda_link_lines "-lcuda\n")
file(WRITE "${CMAKE_BINARY_DIR}/cuda/link-libraries.txt" "${astronomical_cuda_link_lines}")

list(GET CUDAToolkit_INCLUDE_DIRS 0 astronomical_cuda_toolkit_include_directory)
file(WRITE "${CMAKE_BINARY_DIR}/cuda/toolkit-include-directory.txt" "${astronomical_cuda_toolkit_include_directory}\n")
