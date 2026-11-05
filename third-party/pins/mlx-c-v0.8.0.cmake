# Immutable official MLX C dependency pin for the permanent Rust runtime boundary.
# Callers provision this exact archive; CMake verifies it before extraction and
# never resolves the upstream repository during an Astronomical build.

set(ASTRONOMICAL_MLX_C_VERSION "0.8.0")
set(ASTRONOMICAL_MLX_C_GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git")
set(ASTRONOMICAL_MLX_C_GIT_COMMIT "a26f2d584a14ec619ae8db9a09be03becd47598a")
set(ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_URL "https://github.com/ml-explore/mlx-c/archive/a26f2d584a14ec619ae8db9a09be03becd47598a.tar.gz")
set(ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_SHA256 "b2cd55408fb15e5cf44c622355ff3445b58fb835f691db07d620047dff24548c")
set(ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_FILE_NAME "mlx-c-0.8.0-a26f2d584a14ec619ae8db9a09be03becd47598a.tar.gz")
