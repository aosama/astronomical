# Immutable official MLX C dependency pin for the permanent Rust runtime boundary.
# Callers provision this exact archive; CMake verifies it before extraction and
# never resolves the upstream repository during an Astronomical build.

set(ASTRONOMICAL_MLX_C_VERSION "0.6.0")
set(ASTRONOMICAL_MLX_C_GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git")
set(ASTRONOMICAL_MLX_C_GIT_COMMIT "0726ca922fc902c4c61ef9c27d94132be418e945")
set(ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_URL "https://github.com/ml-explore/mlx-c/archive/0726ca922fc902c4c61ef9c27d94132be418e945.tar.gz")
set(ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_SHA256 "fb8412ce060e0584873d3398c8ad0dd3785b03236cec2488023a7f94451d559c")
set(ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_FILE_NAME "mlx-c-0.6.0-0726ca922fc902c4c61ef9c27d94132be418e945.tar.gz")
