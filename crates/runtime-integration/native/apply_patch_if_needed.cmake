# Applies one repository-owned patch to an already-extracted native
# dependency, idempotently.
#
# Why a marker file: FetchContent re-runs PATCH_COMMAND on reconfiguration,
# but `patch` refuses to apply an already-applied diff. Hashing the patch
# file into the marker name means editing a patch re-applies it (the marker
# no longer matches), while reconfiguring with the same patch set is free.
#
# Why --forward --batch: never interactive, never reverse-apply. A genuine
# conflict means the extracted archive no longer matches the patch's
# expectations, and the failure message tells the operator to discard the
# staged build rather than hand-merge generated source state.
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
