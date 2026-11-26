# Immutable pin for the MLX performance profiler, a measurement library that
# compiles against the pinned, patched MLX. It is linked only into measurement
# tests (crates/tooling/mlx-performance-profiler), never into the serving
# image, and is provisioned and digest-verified like every other archive.

set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_VERSION "0.4.0")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_GIT_REPOSITORY "https://github.com/aosama/mlx-performance-profiler.git")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_GIT_COMMIT "de3395de6f11f748b6e2c748c35c4b6c9dbbe804")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_URL "https://github.com/aosama/mlx-performance-profiler/releases/download/v0.4.0/mlx-performance-profiler-0.4.0.tar.gz")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_SHA256 "7dfa2a65d80454cf8b3bd393f01b07449d5c86b547267730ae27d1c8e818874c")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_FILE_NAME "mlx-performance-profiler-0.4.0.tar.gz")
