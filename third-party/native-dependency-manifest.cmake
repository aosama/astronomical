cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED ASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH)
    message(FATAL_ERROR "ASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH is required")
endif()
if(NOT IS_ABSOLUTE "${ASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH}")
    message(FATAL_ERROR "ASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH must be an absolute path")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/pins/mlx-v0.32.1.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/pins/mlx-c-v0.6.0.cmake")

function(append_native_dependency_manifest_entry archive_file_name archive_url archive_sha256 dependency_description)
    file(
        APPEND
        "${ASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH}"
        "${archive_file_name}|${archive_url}|${archive_sha256}|${dependency_description}\n"
    )
endfunction()

file(WRITE "${ASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH}" "")
append_native_dependency_manifest_entry(
    "${ASTRONOMICAL_MLX_SOURCE_ARCHIVE_FILE_NAME}"
    "${ASTRONOMICAL_MLX_SOURCE_ARCHIVE_URL}"
    "${ASTRONOMICAL_MLX_SOURCE_ARCHIVE_SHA256}"
    "MLX ${ASTRONOMICAL_MLX_VERSION}"
)
append_native_dependency_manifest_entry(
    "${ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_FILE_NAME}"
    "${ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_URL}"
    "${ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_SHA256}"
    "MLX C ${ASTRONOMICAL_MLX_C_VERSION}"
)
append_native_dependency_manifest_entry(
    "${ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_FILE_NAME}"
    "${ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_URL}"
    "${ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_SHA256}"
    "metal-cpp ${ASTRONOMICAL_METAL_CPP_VERSION}"
)
append_native_dependency_manifest_entry(
    "${ASTRONOMICAL_NLOHMANN_JSON_SOURCE_ARCHIVE_FILE_NAME}"
    "${ASTRONOMICAL_NLOHMANN_JSON_SOURCE_ARCHIVE_URL}"
    "${ASTRONOMICAL_NLOHMANN_JSON_SOURCE_ARCHIVE_SHA256}"
    "nlohmann/json ${ASTRONOMICAL_NLOHMANN_JSON_VERSION}"
)
append_native_dependency_manifest_entry(
    "${ASTRONOMICAL_FMT_SOURCE_ARCHIVE_FILE_NAME}"
    "${ASTRONOMICAL_FMT_SOURCE_ARCHIVE_URL}"
    "${ASTRONOMICAL_FMT_SOURCE_ARCHIVE_SHA256}"
    "fmt ${ASTRONOMICAL_FMT_VERSION}"
)
