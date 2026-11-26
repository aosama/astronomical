# Immutable pin for the MLX performance profiler, a measurement library that
# compiles against the pinned, patched MLX. It is linked only into measurement
# tests (crates/tooling/mlx-performance-profiler), never into the serving
# image, and is provisioned and digest-verified like every other archive.

set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_VERSION "0.2.0")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_GIT_REPOSITORY "https://github.com/aosama/mlx-performance-profiler.git")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_GIT_COMMIT "126db3a3c68571c0824b2963cc715529aa019a3f")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_URL "https://github.com/aosama/mlx-performance-profiler/releases/download/v0.2.0/mlx-performance-profiler-0.2.0.tar.gz")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_SHA256 "d8ad4c93a449abee130a1bdac6f759542ae9dd297694104959aa3abd27d04fe5")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_FILE_NAME "mlx-performance-profiler-0.2.0.tar.gz")
