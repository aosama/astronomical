# Immutable pin for the MLX performance profiler, a measurement library that
# compiles against the pinned, patched MLX. It is linked only into measurement
# tests (crates/tooling/mlx-performance-profiler), never into the serving
# image, and is provisioned and digest-verified like every other archive.

set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_VERSION "0.3.0")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_GIT_REPOSITORY "https://github.com/aosama/mlx-performance-profiler.git")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_GIT_COMMIT "1af6418d803bf10cdc3eba14640a48ac18570a80")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_URL "https://github.com/aosama/mlx-performance-profiler/releases/download/v0.3.0/mlx-performance-profiler-0.3.0.tar.gz")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_SHA256 "f8c0b7695e3925fd9be0355cecc7b45ed8f5473f27725853d54d977f7f579cc4")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_FILE_NAME "mlx-performance-profiler-0.3.0.tar.gz")
