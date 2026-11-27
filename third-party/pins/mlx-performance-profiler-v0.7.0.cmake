# Immutable pin for the MLX performance profiler, a measurement library that
# compiles against the pinned, patched MLX. It is linked only into measurement
# tests (crates/tooling/mlx-performance-profiler), never into the serving
# image, and is provisioned and digest-verified like every other archive.

set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_VERSION "0.7.0")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_GIT_REPOSITORY "https://github.com/aosama/mlx-performance-profiler.git")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_GIT_COMMIT "0425dc05e3a13f3989f72d854f836348ee169303")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_URL "https://github.com/aosama/mlx-performance-profiler/releases/download/v0.7.0/mlx-performance-profiler-0.7.0.tar.gz")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_SHA256 "3466770490611f4131cda49e1568c58ed73fffd2162548c251adffdaf7040233")
set(ASTRONOMICAL_MLX_PERFORMANCE_PROFILER_SOURCE_ARCHIVE_FILE_NAME "mlx-performance-profiler-0.7.0.tar.gz")
