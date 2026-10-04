# Immutable official MLX C dependency pin for the permanent Rust runtime boundary.
# Callers provision this exact archive; CMake verifies it before extraction and
# never resolves the upstream repository during an Astronomical build.

set(ASTRONOMICAL_MLX_C_VERSION "0.7.0")
set(ASTRONOMICAL_MLX_C_GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git")
set(ASTRONOMICAL_MLX_C_GIT_COMMIT "a341b4925024b88b2c593468f16e12f5e17315da")
set(ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_URL "https://github.com/ml-explore/mlx-c/archive/a341b4925024b88b2c593468f16e12f5e17315da.tar.gz")
set(ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_SHA256 "4424dd3f6225708d111b691be4041bba9bc6da09719ae8d32e6900744852c7f6")
set(ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_FILE_NAME "mlx-c-0.7.0-a341b4925024b88b2c593468f16e12f5e17315da.tar.gz")
