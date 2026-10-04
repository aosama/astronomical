#!/usr/bin/env sh

# Sourced pipeline library for scripts/provision-bindgen-headers.sh; not an
# entry point. Defines the extraction, patching, and verification functions:
# it extracts the pinned MLX and MLX-C archives from the verified native
# dependency cache and applies the same patch pipeline as the native build.

if [ "${0##*/}" = "bindgen-headers-pipeline.sh" ]; then
    printf '%s\n' "Error: scripts/internal/bindgen-headers-pipeline.sh is a sourced library, not an entry point; run scripts/provision-bindgen-headers.sh" >&2
    exit 2
fi

readonly HEADERS_CACHE_SUBDIRECTORY="bindgen-headers"
readonly NATIVE_STORE_SCHEMA_DIRECTORY_NAME="v1"
readonly DEFAULT_PROFILE="core"
readonly EXTRACTION_TREE_MLX="mlx-src"
readonly EXTRACTION_TREE_MLX_C="mlx_c-src"
readonly COMPLETION_MARKER_FILE_NAME="complete"

NATIVE_DEPENDENCY_CACHE_DIRECTORY=""
NATIVE_BUILD_STORE_DIRECTORY=""
EXPLICIT_MANIFEST_PATH=""
EXPLICIT_PATCH_LIST_PATH=""
NATIVE_BUILD_PROFILE="$DEFAULT_PROFILE"
GENERATED_MANIFEST_PATH=""
GENERATED_PATCH_LIST_PATH=""
EFFECTIVE_MANIFEST_PATH=""
EFFECTIVE_PATCH_LIST_PATH=""
TEMPORARY_DIRECTORY=""
TEMPORARY_STAGING_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

log_line() {
    printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*"
}

step_started_at_seconds=0

start_step() {
    step_name="$1"
    step_started_at_seconds="$(date +%s)"
    printf '%s step=%s status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name"
}

finish_step() {
    step_name="$1"
    step_status="$2"
    step_finished_at_seconds="$(date +%s)"
    step_elapsed_seconds=$((step_finished_at_seconds - step_started_at_seconds))
    printf '%s step=%s status=%s elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name" "$step_status" "$step_elapsed_seconds"
}

remove_owned_path() {
    owned_path="$1"
    [ -n "$owned_path" ] || return 0
    case "$owned_path" in
        /|.|..)
            print_error "refusing to remove unsafe temporary path: $owned_path"
            return 0
            ;;
    esac
    if [ -d "$owned_path" ]; then
        rm -rf "$owned_path"
    elif [ -f "$owned_path" ]; then
        rm -f -- "$owned_path"
    fi
}

cleanup() {
    remove_owned_path "$GENERATED_MANIFEST_PATH"
    remove_owned_path "$GENERATED_PATCH_LIST_PATH"
    remove_owned_path "$TEMPORARY_DIRECTORY"
    remove_owned_path "$TEMPORARY_STAGING_DIRECTORY"
}

resolve_repository_root() {
    REPOSITORY_ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
}

validate_manifest_field() {
    manifest_field_name="$1"
    manifest_field_text="$2"
    case "$manifest_field_text" in
        ''|*'|'*|*'\n'*|*'\r'*)
            print_error "pinned archive manifest has an invalid $manifest_field_name"
            exit 1
            ;;
    esac
}

# Reads the pinned archive manifest (file-name|url|sha256|description) and
# prints the two entries this script consumes, MLX first. The MLX C entry
# must be tested before the MLX entry because "MLX" is a strict prefix of
# "MLX C". metal-cpp, nlohmann/json, and fmt are build-time dependencies of
# the MLX C++ build and contribute no bindgen-relevant headers, so they are
# skipped here.
select_pinned_header_archives() {
    manifest_path="$1"
    mlx_entry=""
    mlx_c_entry=""
    while IFS='|' read -r archive_file_name archive_url archive_sha256 dependency_description; do
        validate_manifest_field "archive file name" "$archive_file_name"
        validate_manifest_field "archive URL" "$archive_url"
        validate_manifest_field "archive SHA-256" "$archive_sha256"
        validate_manifest_field "dependency description" "$dependency_description"
        case "$archive_file_name" in
            *[!A-Za-z0-9._-]*|'')
                print_error "pinned archive manifest has an unsafe archive file name"
                exit 1
                ;;
        esac
        case "$archive_url" in
            https://*) ;;
            *)
                print_error "pinned archive manifest requires an HTTPS archive URL"
                exit 1
                ;;
        esac
        case "$archive_sha256" in
            *[!0123456789abcdef]*|'')
                print_error "pinned archive manifest has a non-lowercase SHA-256"
                exit 1
                ;;
        esac
        if [ "${#archive_sha256}" -ne 64 ]; then
            print_error "pinned archive manifest has an invalid SHA-256 length"
            exit 1
        fi
        case "$dependency_description" in
            "MLX C "*)
                if [ -n "$mlx_c_entry" ]; then
                    print_error "pinned archive manifest declares the MLX C archive twice"
                    exit 1
                fi
                mlx_c_entry="${archive_file_name}|${archive_sha256}"
                ;;
            "MLX "*)
                if [ -n "$mlx_entry" ]; then
                    print_error "pinned archive manifest declares the MLX archive twice"
                    exit 1
                fi
                mlx_entry="${archive_file_name}|${archive_sha256}"
                ;;
        esac
    done < "$manifest_path"
    if [ -z "$mlx_entry" ] || [ -z "$mlx_c_entry" ]; then
        print_error "pinned archive manifest must declare both the MLX and the MLX C archives"
        exit 1
    fi
    printf '%s\n' "$mlx_entry"
    printf '%s\n' "$mlx_c_entry"
}

# Parses the two FetchContent patch pipelines out of the native CMakeLists so
# this script and the CMake build share one definition of the patch list.
# Parsing instead of duplicating the list keeps the native build identity
# untouched by this script. The contract test pins this parser against the
# real CMakeLists so a restructure fails loudly instead of silently
# extracting unpatched headers.
parse_patch_pipeline() {
    GENERATED_PATCH_LIST_PATH="${TEMPORARY_DIRECTORY}/parsed-patch-list"
    awk '
        in_block && /^\)/ { in_block = 0; target = ""; next }
        !in_block && /^FetchContent_Declare\(/ { in_block = 1; target = ""; next }
        in_block && target == "" {
            candidate = $0
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", candidate)
            if (candidate == "mlx" || candidate == "mlx_c") { target = candidate }
            next
        }
        in_block && target != "" && match($0, /PATCH_FILE=\$\{CMAKE_CURRENT_LIST_DIR\}\/\.\.\/\.\.\/\.\.\//) {
            patch_path = substr($0, RSTART + RLENGTH)
            gsub(/".*$/, "", patch_path)
            printf "%s\t%s\n", target, patch_path
        }
    ' "${REPOSITORY_ROOT}/crates/runtime-integration/native/CMakeLists.txt" > "$GENERATED_PATCH_LIST_PATH"
    while IFS="$(printf '\t')" read -r patch_dependency patch_relative_path; do
        resolve_patch_path "$patch_relative_path"
        [ -f "$resolved_patch_path" ] || {
            print_error "patch pipeline references a missing patch file: $patch_relative_path"
            exit 1
        }
    done < "$GENERATED_PATCH_LIST_PATH"
    EFFECTIVE_PATCH_LIST_PATH="$GENERATED_PATCH_LIST_PATH"
}

# Patch paths are repository-relative when parsed from the native CMakeLists
# and may be absolute when supplied through --patch-list by tooling and
# contract tests.
resolve_patch_path() {
    patch_relative_path="$1"
    case "$patch_relative_path" in
        /*) resolved_patch_path="$patch_relative_path" ;;
        *) resolved_patch_path="${REPOSITORY_ROOT}/${patch_relative_path}" ;;
    esac
}

# The extraction directory key is the source-only identity: pins, patches,
# patch pipeline, and store schema, with toolchain probes excluded. Headers
# do not depend on the toolchain, so a compiler update must not invalidate a
# valid extraction; a pin or patch edit does. --verify-headers, in contrast,
# must resolve the full identity (profile plus toolchain) because it looks up
# an entry in the native build store, whose entries are keyed by the full
# identity the build actually used.
resolve_source_identity() {
    source_identity="$(
        "${REPOSITORY_ROOT}/scripts/native-build-cache-fingerprint.sh" \
            --source-only --profile core "$REPOSITORY_ROOT"
    )"
    validate_identity_text "$source_identity" "source identity fingerprint"
    printf '%s\n' "$source_identity"
}

validate_identity_text() {
    identity_text="$1"
    identity_description="$2"
    case "$identity_text" in
        *[!0-9a-f]*|'')
            print_error "$identity_description returned an invalid identity"
            exit 1
            ;;
    esac
    if [ "${#identity_text}" -ne 64 ]; then
        print_error "$identity_description returned an invalid identity length"
        exit 1
    fi
}

sha256_matches() (
    candidate_archive_path="$1"
    expected_sha256="$2"
    if [ ! -f "$candidate_archive_path" ] || [ -L "$candidate_archive_path" ]; then
        return 1
    fi
    actual_sha256_line="$(shasum -a 256 "$candidate_archive_path")"
    actual_sha256="${actual_sha256_line%% *}"
    [ "$actual_sha256" = "$expected_sha256" ]
)

# Extracts one verified archive into the staging directory and applies the
# dependency's patch pipeline inside the extracted tree root, mirroring how
# FetchContent stages and patches sources for the native build: the archive's
# single top-level directory is stripped, and patch markers are written at
# the tree root, outside the published header directory.
extract_and_patch_archive() {
    archive_path="$1"
    expected_archive_sha256="$2"
    dependency_description="$3"
    extraction_tree_name="$4"
    patch_dependency="$5"

    if ! sha256_matches "$archive_path" "$expected_archive_sha256"; then
        print_error "$dependency_description archive is missing or fails its pinned SHA-256: $archive_path"
        print_error "run scripts/bootstrap-native-dependencies.sh to provision the verified cache"
        exit 1
    fi

    archive_staging_directory="${TEMPORARY_STAGING_DIRECTORY}/${extraction_tree_name}.archive"
    tree_directory="${TEMPORARY_STAGING_DIRECTORY}/${extraction_tree_name}"
    mkdir -p "$archive_staging_directory"
    if ! tar -xzf "$archive_path" -C "$archive_staging_directory"; then
        print_error "failed to extract the $dependency_description archive"
        exit 1
    fi

    archive_top_level_count=0
    archive_top_level_directory=""
    for archive_entry in "$archive_staging_directory"/* "$archive_staging_directory"/.[!.]*; do
        [ -e "$archive_entry" ] || continue
        archive_top_level_count=$((archive_top_level_count + 1))
        archive_top_level_directory="$archive_entry"
    done
    if [ "$archive_top_level_count" -ne 1 ] || [ ! -d "$archive_top_level_directory" ]; then
        print_error "$dependency_description archive must contain exactly one top-level directory"
        exit 1
    fi
    mv "$archive_top_level_directory" "$tree_directory"

    patch_application_count=0
    while IFS="$(printf '\t')" read -r patch_dependency_name patch_relative_path; do
        [ "$patch_dependency_name" = "$patch_dependency" ] || continue
        resolve_patch_path "$patch_relative_path"
        (cd "$tree_directory" && cmake \
            "-DPATCH_EXECUTABLE=$(command -v patch)" \
            "-DPATCH_FILE=${resolved_patch_path}" \
            -P "${REPOSITORY_ROOT}/crates/runtime-integration/native/apply_patch_if_needed.cmake" >/dev/null)
        patch_application_count=$((patch_application_count + 1))
    done < "$EFFECTIVE_PATCH_LIST_PATH"
    if [ "$patch_application_count" -eq 0 ]; then
        print_error "patch pipeline declares no patches for $patch_dependency"
        exit 1
    fi
    extracted_file_count="$(find "$tree_directory" -type f | wc -l | tr -d ' ')"
    log_line "dependency=$patch_dependency patches=$patch_application_count files=$extracted_file_count"
}

write_hash_manifest() {
    tree_directory="$1"
    manifest_output_path="$2"
    (cd "$tree_directory" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256) \
        > "$manifest_output_path"
}

hash_manifests_match() {
    tree_directory="$1"
    recorded_manifest_path="$2"
    recomputed_manifest_path="$3"
    write_hash_manifest "$tree_directory" "$recomputed_manifest_path"
    cmp -s "$recomputed_manifest_path" "$recorded_manifest_path"
}

remove_extraction_directory() {
    extraction_directory="$1"
    case "$extraction_directory" in
        /|.|..)
            print_error "refusing to remove unsafe extraction directory: $extraction_directory"
            exit 1
            ;;
    esac
    if [ -L "$extraction_directory" ]; then
        print_error "refusing to remove a symlinked extraction directory: $extraction_directory"
        exit 1
    fi
    rm -rf "$extraction_directory"
}

# Runs one provisioning pass: reuse the existing extraction when its recorded
# hash manifests still match, otherwise re-extract from the verified archives
# and swap the directory in one move. The completion marker is written last so
# a partially written extraction never validates as complete.
provision_headers() {
    headers_root_directory="${NATIVE_DEPENDENCY_CACHE_DIRECTORY}/${HEADERS_CACHE_SUBDIRECTORY}"
    mkdir -p "$headers_root_directory"
    chmod 700 "$headers_root_directory"

    start_step "read-pinned-manifest"
    if [ -n "$EXPLICIT_MANIFEST_PATH" ]; then
        EFFECTIVE_MANIFEST_PATH="$EXPLICIT_MANIFEST_PATH"
    else
        GENERATED_MANIFEST_PATH="${TEMPORARY_DIRECTORY}/generated-archive-manifest"
        cmake \
            "-DASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH=${GENERATED_MANIFEST_PATH}" \
            -P "${REPOSITORY_ROOT}/third-party/native-dependency-manifest.cmake" >/dev/null
        EFFECTIVE_MANIFEST_PATH="$GENERATED_MANIFEST_PATH"
    fi
    select_pinned_header_archives "$EFFECTIVE_MANIFEST_PATH" > "${TEMPORARY_DIRECTORY}/selected-archives"
    finish_step "read-pinned-manifest" "success"

    start_step "resolve-source-identity"
    source_identity="$(resolve_source_identity)"
    extraction_directory="${headers_root_directory}/${source_identity}"
    finish_step "resolve-source-identity" "success"
    log_line "extraction_directory=$extraction_directory"

    start_step "parse-patch-pipeline"
    if [ -n "$EXPLICIT_PATCH_LIST_PATH" ]; then
        EFFECTIVE_PATCH_LIST_PATH="$EXPLICIT_PATCH_LIST_PATH"
    else
        parse_patch_pipeline
    fi
    finish_step "parse-patch-pipeline" "success"

    mlx_archive_file_name="$(awk -F'|' 'NR == 1 { print $1 }' "${TEMPORARY_DIRECTORY}/selected-archives")"
    mlx_archive_sha256="$(awk -F'|' 'NR == 1 { print $2 }' "${TEMPORARY_DIRECTORY}/selected-archives")"
    mlx_c_archive_file_name="$(awk -F'|' 'NR == 2 { print $1 }' "${TEMPORARY_DIRECTORY}/selected-archives")"
    mlx_c_archive_sha256="$(awk -F'|' 'NR == 2 { print $2 }' "${TEMPORARY_DIRECTORY}/selected-archives")"

    start_step "verify-existing-extraction"
    if [ -f "${extraction_directory}/${COMPLETION_MARKER_FILE_NAME}" ] \
        && hash_manifests_match \
            "${extraction_directory}/${EXTRACTION_TREE_MLX}" \
            "${extraction_directory}/mlx.manifest" \
            "${TEMPORARY_DIRECTORY}/recomputed-mlx.manifest" \
        && hash_manifests_match \
            "${extraction_directory}/${EXTRACTION_TREE_MLX_C}" \
            "${extraction_directory}/mlx_c.manifest" \
            "${TEMPORARY_DIRECTORY}/recomputed-mlx_c.manifest"; then
        log_line "extraction_directory=$extraction_directory status=verified-cached"
        finish_step "verify-existing-extraction" "success"
        return 0
    fi
    if [ -e "$extraction_directory" ]; then
        log_line "extraction_directory=$extraction_directory status=re-extracting reason=missing-or-modified"
    fi
    finish_step "verify-existing-extraction" "success"

    TEMPORARY_STAGING_DIRECTORY="$(mktemp -d "${headers_root_directory}/.staging.XXXXXX")"
    start_step "extract-and-patch-archives"
    extract_and_patch_archive \
        "${NATIVE_DEPENDENCY_CACHE_DIRECTORY}/${mlx_archive_file_name}" \
        "$mlx_archive_sha256" \
        "MLX" \
        "$EXTRACTION_TREE_MLX" \
        "mlx"
    extract_and_patch_archive \
        "${NATIVE_DEPENDENCY_CACHE_DIRECTORY}/${mlx_c_archive_file_name}" \
        "$mlx_c_archive_sha256" \
        "MLX C" \
        "$EXTRACTION_TREE_MLX_C" \
        "mlx_c"
    finish_step "extract-and-patch-archives" "success"

    start_step "publish-extraction"
    mlx_file_count="$(find "${TEMPORARY_STAGING_DIRECTORY}/${EXTRACTION_TREE_MLX}" -type f | wc -l | tr -d ' ')"
    mlx_c_file_count="$(find "${TEMPORARY_STAGING_DIRECTORY}/${EXTRACTION_TREE_MLX_C}" -type f | wc -l | tr -d ' ')"
    write_hash_manifest \
        "${TEMPORARY_STAGING_DIRECTORY}/${EXTRACTION_TREE_MLX}" \
        "${TEMPORARY_STAGING_DIRECTORY}/mlx.manifest"
    write_hash_manifest \
        "${TEMPORARY_STAGING_DIRECTORY}/${EXTRACTION_TREE_MLX_C}" \
        "${TEMPORARY_STAGING_DIRECTORY}/mlx_c.manifest"
    {
        printf 'source_identity=%s\n' "$source_identity"
        printf 'mlx_archive=%s\n' "$mlx_archive_file_name"
        printf 'mlx_archive_sha256=%s\n' "$mlx_archive_sha256"
        printf 'mlx_c_archive=%s\n' "$mlx_c_archive_file_name"
        printf 'mlx_c_archive_sha256=%s\n' "$mlx_c_archive_sha256"
        printf 'mlx_file_count=%s\n' "$mlx_file_count"
        printf 'mlx_c_file_count=%s\n' "$mlx_c_file_count"
    } > "${TEMPORARY_STAGING_DIRECTORY}/${COMPLETION_MARKER_FILE_NAME}"
    remove_extraction_directory "$extraction_directory"
    mv "$TEMPORARY_STAGING_DIRECTORY" "$extraction_directory"
    TEMPORARY_STAGING_DIRECTORY=""
    finish_step "publish-extraction" "success"
    log_line "extraction_directory=$extraction_directory status=provisioned mlx_files=$mlx_file_count mlx_c_files=$mlx_c_file_count"
}

# The build store keys its entries by the full identity, which honors the
# same toolchain override variables the native build honors
# (ASTRONOMICAL_NATIVE_IDENTITY_*, TARGET); passing them through here is
# intentional so verification can address an entry built under an override.
resolve_built_identity() {
    if [ -z "${TARGET:-}" ]; then
        TARGET="$(rustc -vV | sed -n 's/^host: //p')"
    fi
    [ -n "$TARGET" ] || {
        print_error "TARGET or ASTRONOMICAL_NATIVE_IDENTITY_TARGET is required to resolve the built identity"
        exit 1
    }
    export TARGET
    built_identity="$(
        "${REPOSITORY_ROOT}/scripts/native-build-cache-fingerprint.sh" \
            --profile "$NATIVE_BUILD_PROFILE" "$REPOSITORY_ROOT"
    )"
    validate_identity_text "$built_identity" "built identity fingerprint"
    printf '%s\n' "$built_identity"
}

verify_headers_against_native_build() {
    start_step "resolve-built-identity"
    built_identity="$(resolve_built_identity)"
    built_entry_directory="${NATIVE_BUILD_STORE_DIRECTORY}/${NATIVE_STORE_SCHEMA_DIRECTORY_NAME}/entries/${built_identity}"
    if [ ! -f "${built_entry_directory}/complete" ]; then
        print_error "no completed native build for identity ${built_identity}: ${built_entry_directory}"
        print_error "run scripts/prewarm-native-build.sh --profile $NATIVE_BUILD_PROFILE (or pass --profile for a profile that is already built)"
        exit 1
    fi
    built_headers_directory="${built_entry_directory}/include/mlx"
    finish_step "resolve-built-identity" "success"

    source_identity="$(resolve_source_identity)"
    extraction_directory="${NATIVE_DEPENDENCY_CACHE_DIRECTORY}/${HEADERS_CACHE_SUBDIRECTORY}/${source_identity}"
    extracted_headers_directory="${extraction_directory}/${EXTRACTION_TREE_MLX_C}/mlx"
    if [ ! -f "${extraction_directory}/${COMPLETION_MARKER_FILE_NAME}" ]; then
        print_error "no provisioned bindgen headers for identity ${source_identity}; run this script without --verify-headers first"
        exit 1
    fi

    start_step "compare-headers"
    write_hash_manifest "$extracted_headers_directory" "${TEMPORARY_DIRECTORY}/extracted-headers.manifest"
    write_hash_manifest "$built_headers_directory" "${TEMPORARY_DIRECTORY}/built-headers.manifest"
    if cmp -s "${TEMPORARY_DIRECTORY}/extracted-headers.manifest" "${TEMPORARY_DIRECTORY}/built-headers.manifest"; then
        matched_file_count="$(wc -l < "${TEMPORARY_DIRECTORY}/extracted-headers.manifest" | tr -d ' ')"
        log_line "headers_match=yes files=$matched_file_count"
        log_line "extracted_headers=$extracted_headers_directory"
        log_line "built_headers=$built_headers_directory"
        finish_step "compare-headers" "success"
        return 0
    fi
    log_line "headers_match=no"
    log_line "extracted_headers=$extracted_headers_directory"
    log_line "built_headers=$built_headers_directory"
    awk -F'  ' 'NR == FNR { extracted[$2] = $1; next } $2 in extracted { if (extracted[$2] != $1) print "modified " $2 } !($2 in extracted) { print "built-only " $2 }' \
        "${TEMPORARY_DIRECTORY}/extracted-headers.manifest" "${TEMPORARY_DIRECTORY}/built-headers.manifest" \
        > "${TEMPORARY_DIRECTORY}/built-side-differences"
    awk -F'  ' 'NR == FNR { built[$2] = $1; next } !($2 in built) { print "extracted-only " $2 }' \
        "${TEMPORARY_DIRECTORY}/built-headers.manifest" "${TEMPORARY_DIRECTORY}/extracted-headers.manifest" \
        > "${TEMPORARY_DIRECTORY}/extracted-side-differences"
    differing_paths="$(cat \
        "${TEMPORARY_DIRECTORY}/built-side-differences" \
        "${TEMPORARY_DIRECTORY}/extracted-side-differences")"
    log_line "differing_paths=$(printf '%s\n' "$differing_paths" | head -10 | tr '\n' ' ')"
    finish_step "compare-headers" "mismatch"
    exit 1
}
