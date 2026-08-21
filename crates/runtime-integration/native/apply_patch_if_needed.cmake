if(NOT DEFINED PATCH_EXECUTABLE OR "${PATCH_EXECUTABLE}" STREQUAL "")
    message(FATAL_ERROR "PATCH_EXECUTABLE is required")
endif()
if(NOT DEFINED PATCH_FILE OR "${PATCH_FILE}" STREQUAL "")
    message(FATAL_ERROR "PATCH_FILE is required")
endif()

file(SHA256 "${PATCH_FILE}" patch_sha256)
set(patch_marker_file ".astronomical-applied-patch-${patch_sha256}")
if(EXISTS "${patch_marker_file}")
    message(STATUS "Native dependency patch is already applied: ${PATCH_FILE}")
    return()
endif()

execute_process(
    COMMAND "${PATCH_EXECUTABLE}" -p1 --forward --batch -i "${PATCH_FILE}"
    RESULT_VARIABLE patch_application_status
)
if(NOT patch_application_status EQUAL 0)
    message(FATAL_ERROR "Native dependency patch conflicts with generated source state: ${PATCH_FILE}. Remove only the affected staged native build and rebuild from the verified archive.")
endif()
file(TOUCH "${patch_marker_file}")
