# Immutable pin for the MLX performance profiler, a measurement library that
# compiles against the pinned, patched MLX. It is linked only into measurement
# tests (crates/tooling/mlx-performance-profiler), never into the serving
# image, and is provisioned and digest-verified like every other archive.

set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_VERSION "0.5.0")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_GIT_REPOSITORY "https://github.com/aosama/mlx-performance-profiler.git")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_GIT_COMMIT "e00c8c036f647b4f4b787bff3fc9710d419c4b5a")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_URL "https://github.com/aosama/mlx-performance-profiler/releases/download/v0.5.0/mlx-performance-profiler-0.5.0.tar.gz")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_SHA256 "cc2acc16e9bf62c321c45898c52852eb0d4917bd9ee68a7678fcdee96eb3e080")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_FILE_NAME "mlx-performance-profiler-0.5.0.tar.gz")
