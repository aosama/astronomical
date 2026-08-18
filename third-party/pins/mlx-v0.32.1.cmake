# Immutable native dependency pin for Astronomical's MLX runtime.
# The CMake build consumes caller-provisioned archives with pinned SHA-256 values
# instead of fetching source during configure, so native builds remain auditable.

set(ASTRONOMICAL_MLX_PROJECT_NAME "mlx")
set(ASTRONOMICAL_MLX_VERSION "0.32.1")
set(ASTRONOMICAL_MLX_GIT_REPOSITORY "https://github.com/ml-explore/mlx.git")
set(ASTRONOMICAL_MLX_GIT_COMMIT "3a6219917e4535575ce5bce2fc2ba27a483a709b")
set(ASTRONOMICAL_MLX_MACOS_DEPLOYMENT_TARGET "26.2")
set(ASTRONOMICAL_MLX_SOURCE_ARCHIVE_URL "https://github.com/ml-explore/mlx/archive/3a6219917e4535575ce5bce2fc2ba27a483a709b.tar.gz")
set(ASTRONOMICAL_MLX_SOURCE_ARCHIVE_SHA256 "31014524e37bf790891511114ccc0866c2338d3717d4db30f37ce7a48bb11616")
set(ASTRONOMICAL_MLX_SOURCE_ARCHIVE_FILE_NAME "mlx-0.32.1-3a6219917e4535575ce5bce2fc2ba27a483a709b.tar.gz")

# MLX v0.32.1 declares these transitive source dependencies in its root CMake
# project. Astronomical requires their archives to be provisioned explicitly and
# verifies each digest before MLX configuration, preventing nested live fetches.
set(ASTRONOMICAL_METAL_CPP_VERSION "26")
set(ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_URL "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip")
set(ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_SHA256 "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4")
set(ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_FILE_NAME "metal-cpp-26.zip")

set(ASTRONOMICAL_NLOHMANN_JSON_VERSION "3.11.3")
set(ASTRONOMICAL_NLOHMANN_JSON_SOURCE_ARCHIVE_URL "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz")
set(ASTRONOMICAL_NLOHMANN_JSON_SOURCE_ARCHIVE_SHA256 "d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d")
set(ASTRONOMICAL_NLOHMANN_JSON_SOURCE_ARCHIVE_FILE_NAME "json-3.11.3.tar.xz")

set(ASTRONOMICAL_FMT_VERSION "12.1.0")
set(ASTRONOMICAL_FMT_SOURCE_ARCHIVE_URL "https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz")
set(ASTRONOMICAL_FMT_SOURCE_ARCHIVE_SHA256 "ea7de4299689e12b6dddd392f9896f08fb0777ac7168897a244a6d6085043fea")
set(ASTRONOMICAL_FMT_SOURCE_ARCHIVE_FILE_NAME "fmt-12.1.0.tar.gz")
