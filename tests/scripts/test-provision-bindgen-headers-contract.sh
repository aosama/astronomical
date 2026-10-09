#!/usr/bin/env sh

# Proves scripts/provision-bindgen-headers.sh extracts and patches bindgen
# headers from verified archives under a pin-keyed directory, reuses a valid
# extraction, self-heals a damaged one, refuses tampered or missing archives
# without any network access, and compares extracted MLX-C headers byte for
# byte against a completed native build's staged headers. Fixture archives,
# manifests, patches, and a fixture native build store keep every contract
# hermetic; one contract pins the CMakeLists patch-pipeline parser so a
# restructure fails loudly.

set -eu

SANDBOX_DIRECTORY=""
PROVISION_SCRIPT_PATH=""
PROVISION_OUTPUT_FILE=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe provision-test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

require_command() {
    command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || {
        print_error "required command is unavailable: ${command_name}"
        exit 2
    }
}

assert_contains() {
    haystack_file="$1"
    needle="$2"
    expectation_description="$3"
    if ! grep -q "$needle" "$haystack_file"; then
        print_error "$expectation_description"
        print_error "missing text: $needle"
        cat "$haystack_file" >&2
        exit 1
    fi
}

assert_file_content_contains() {
    haystack_file="$1"
    needle="$2"
    expectation_description="$3"
    if ! grep -q "$needle" "$haystack_file"; then
        print_error "$expectation_description"
        print_error "missing text: $needle in $haystack_file"
        exit 1
    fi
}

assert_run_succeeds() {
    run_description="$1"
    shift
    if ! timeout 120 sh "$PROVISION_SCRIPT_PATH" "$@" >"$PROVISION_OUTPUT_FILE" 2>&1; then
        print_error "$run_description should succeed"
        cat "$PROVISION_OUTPUT_FILE" >&2
        exit 1
    fi
}

assert_run_fails() {
    run_description="$1"
    shift
    if timeout 120 sh "$PROVISION_SCRIPT_PATH" "$@" >"$PROVISION_OUTPUT_FILE" 2>&1; then
        print_error "$run_description should fail"
        cat "$PROVISION_OUTPUT_FILE" >&2
        exit 1
    fi
}

repository_root="$(cd "$(dirname "$0")/../.." && pwd)"
PROVISION_SCRIPT_PATH="${repository_root}/scripts/provision-bindgen-headers.sh"

require_command shellcheck
require_command timeout
require_command cmake
require_command shasum
require_command patch
require_command rustc

shellcheck "$PROVISION_SCRIPT_PATH" "${repository_root}/scripts/internal/bindgen-headers-pipeline.sh"

SANDBOX_DIRECTORY="$(mktemp -d)"
PROVISION_OUTPUT_FILE="${SANDBOX_DIRECTORY}/provision-output.txt"
CACHE_DIRECTORY="${SANDBOX_DIRECTORY}/cache"
HEADERS_ROOT_DIRECTORY="${CACHE_DIRECTORY}/bindgen-headers"
FIXTURE_DIRECTORY="${SANDBOX_DIRECTORY}/fixtures"
FIXTURE_PATCH_LIST="${FIXTURE_DIRECTORY}/patch-list"
FIXTURE_MANIFEST="${FIXTURE_DIRECTORY}/manifest"
FIXTURE_STORE_DIRECTORY="${SANDBOX_DIRECTORY}/store"
FIXTURE_TARGET="aarch64-apple-darwin"
mkdir -p "$CACHE_DIRECTORY" "$FIXTURE_DIRECTORY"

export TARGET="$FIXTURE_TARGET"
export ASTRONOMICAL_NATIVE_IDENTITY_TARGET="$FIXTURE_TARGET"

# Builds one fixture archive: a pristine tree, a modified tree whose single
# header gains one line, and the patch that transforms one into the other.
build_fixture_dependency() {
    fixture_archive_stem="$1"
    fixture_header_directory="$2"
    fixture_header_name="$3"
    fixture_added_line="$4"

    fixture_pristine="${FIXTURE_DIRECTORY}/${fixture_archive_stem}/pristine"
    fixture_modified="${FIXTURE_DIRECTORY}/${fixture_archive_stem}/modified"
    mkdir -p \
        "${fixture_pristine}/${fixture_header_directory}" \
        "${fixture_modified}/${fixture_header_directory}"
    printf 'placeholder header for the provision contract fixture\n' \
        > "${fixture_pristine}/${fixture_header_directory}/${fixture_header_name}"
    cp "${fixture_pristine}/${fixture_header_directory}/${fixture_header_name}" \
        "${fixture_modified}/${fixture_header_directory}/${fixture_header_name}"
    printf '%s\n' "$fixture_added_line" \
        >> "${fixture_modified}/${fixture_header_directory}/${fixture_header_name}"
    # diff exits 1 when the trees differ, which is exactly the fixture's intent.
    diff -u \
        --label "a/${fixture_header_directory}/${fixture_header_name}" \
        --label "b/${fixture_header_directory}/${fixture_header_name}" \
        "${fixture_pristine}/${fixture_header_directory}/${fixture_header_name}" \
        "${fixture_modified}/${fixture_header_directory}/${fixture_header_name}" \
        > "${FIXTURE_DIRECTORY}/${fixture_archive_stem}.patch" || true
    (cd "${FIXTURE_DIRECTORY}/${fixture_archive_stem}" && tar -czf "${CACHE_DIRECTORY}/${fixture_archive_stem}.tar.gz" pristine)
    fixture_archive_sha256="$(shasum -a 256 "${CACHE_DIRECTORY}/${fixture_archive_stem}.tar.gz")"
    printf '%s\n' "${fixture_archive_sha256%% *}" > "${FIXTURE_DIRECTORY}/${fixture_archive_stem}.sha256"
}

build_fixture_dependency "mlx-fake-1.2.3" "mlx" "array.h" "astronomical fixture patch applied"
build_fixture_dependency "mlx-c-fake-0.1" "mlx/c" "mlx.h" "astronomical fixture patch applied"

{
    printf 'mlx-fake-1.2.3.tar.gz|https://fixture.invalid/mlx-fake-1.2.3.tar.gz|%s|MLX 1.2.3\n' \
        "$(cat "${FIXTURE_DIRECTORY}/mlx-fake-1.2.3.sha256")"
    printf 'mlx-c-fake-0.1.tar.gz|https://fixture.invalid/mlx-c-fake-0.1.tar.gz|%s|MLX C 0.1\n' \
        "$(cat "${FIXTURE_DIRECTORY}/mlx-c-fake-0.1.sha256")"
} > "$FIXTURE_MANIFEST"

printf 'mlx\t%s\nmlx_c\t%s\n' \
    "${FIXTURE_DIRECTORY}/mlx-fake-1.2.3.patch" \
    "${FIXTURE_DIRECTORY}/mlx-c-fake-0.1.patch" > "$FIXTURE_PATCH_LIST"

provision() {
    assert_run_succeeds "provisioning run" \
        --cache-dir "$CACHE_DIRECTORY" \
        --manifest "$FIXTURE_MANIFEST" \
        --patch-list "$FIXTURE_PATCH_LIST"
}

extraction_identity="$(
    timeout 120 sh "${repository_root}/scripts/native-build-cache-fingerprint.sh" \
        --source-only --profile core "$repository_root"
)"
EXTRACTION_DIRECTORY="${HEADERS_ROOT_DIRECTORY}/${extraction_identity}"

mlx_patch_marker=".astronomical-applied-patch-$(shasum -a 256 "${FIXTURE_DIRECTORY}/mlx-fake-1.2.3.patch" | awk '{print $1}')"
mlx_c_patch_marker=".astronomical-applied-patch-$(shasum -a 256 "${FIXTURE_DIRECTORY}/mlx-c-fake-0.1.patch" | awk '{print $1}')"

# Contract 1: a fresh run extracts both pinned archives, strips the single
# top-level directory, applies the dependency's patch pipeline at the tree
# root, and publishes the pin-keyed extraction with hash manifests and a
# completion marker.
provision
assert_contains "$PROVISION_OUTPUT_FILE" "status=provisioned" "the fresh run should close with a provisioned summary"
[ -f "${EXTRACTION_DIRECTORY}/mlx-src/mlx/array.h" ] || {
    print_error "the MLX tree should be extracted with its top-level directory stripped"
    exit 1
}
[ -f "${EXTRACTION_DIRECTORY}/mlx_c-src/mlx/c/mlx.h" ] || {
    print_error "the MLX C tree should be extracted with its top-level directory stripped"
    exit 1
}
assert_file_content_contains "${EXTRACTION_DIRECTORY}/mlx-src/mlx/array.h" "astronomical fixture patch applied" \
    "the MLX patch should modify the extracted header"
assert_file_content_contains "${EXTRACTION_DIRECTORY}/mlx_c-src/mlx/c/mlx.h" "astronomical fixture patch applied" \
    "the MLX C patch should modify the extracted header"
[ -f "${EXTRACTION_DIRECTORY}/mlx-src/${mlx_patch_marker}" ] || {
    print_error "the MLX patch marker should be written at the tree root like FetchContent does"
    exit 1
}
[ -f "${EXTRACTION_DIRECTORY}/mlx_c-src/${mlx_c_patch_marker}" ] || {
    print_error "the MLX C patch marker should be written at the tree root like FetchContent does"
    exit 1
}
[ -f "${EXTRACTION_DIRECTORY}/complete" ] || {
    print_error "the extraction should publish a completion marker"
    exit 1
}
[ -f "${EXTRACTION_DIRECTORY}/mlx.manifest" ] && [ -f "${EXTRACTION_DIRECTORY}/mlx_c.manifest" ] || {
    print_error "the extraction should publish hash manifests for both trees"
    exit 1
}
assert_contains "$PROVISION_OUTPUT_FILE" "extraction_directory=${EXTRACTION_DIRECTORY}" \
    "the extraction should live under the pin-keyed directory"

# Contract 2: a second run verifies the existing extraction without
# re-extracting, leaving the published tree byte for byte unchanged.
tree_state_before_provisioned="$(find "$EXTRACTION_DIRECTORY" -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256)"
provision
assert_contains "$PROVISION_OUTPUT_FILE" "status=verified-cached" "the second run should reuse the existing extraction"
tree_state_after_reuse="$(find "$EXTRACTION_DIRECTORY" -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256)"
if [ "$tree_state_before_provisioned" != "$tree_state_after_reuse" ]; then
    print_error "a reused extraction must not change a single published file"
    exit 1
fi

# Contract 3: a damaged extraction self-heals from the verified archives on
# the next run.
rm "${EXTRACTION_DIRECTORY}/mlx_c-src/mlx/c/mlx.h"
printf 'corrupted\n' >> "${EXTRACTION_DIRECTORY}/mlx-src/mlx/array.h"
provision
assert_contains "$PROVISION_OUTPUT_FILE" "status=re-extracting" "a damaged extraction should trigger re-extraction"
assert_file_content_contains "${EXTRACTION_DIRECTORY}/mlx_c-src/mlx/c/mlx.h" "astronomical fixture patch applied" \
    "the self-heal should restore the deleted header with its patch applied"
assert_file_content_contains "${EXTRACTION_DIRECTORY}/mlx-src/mlx/array.h" "astronomical fixture patch applied" \
    "the self-heal should restore the corrupted header with its patch applied"

# Contract 4: an archive whose bytes no longer match the pinned SHA-256 is
# refused without extraction, and the error points at the bootstrap script.
cp "${CACHE_DIRECTORY}/mlx-c-fake-0.1.tar.gz" "${SANDBOX_DIRECTORY}/mlx-c-fake-0.1.tar.gz.pristine"
printf 'tampered\n' >> "${CACHE_DIRECTORY}/mlx-c-fake-0.1.tar.gz"
rm -rf "$EXTRACTION_DIRECTORY"
assert_run_fails "a tampered archive" \
    --cache-dir "$CACHE_DIRECTORY" \
    --manifest "$FIXTURE_MANIFEST" \
    --patch-list "$FIXTURE_PATCH_LIST"
assert_contains "$PROVISION_OUTPUT_FILE" "fails its pinned SHA-256" "a tampered archive must be refused"
assert_contains "$PROVISION_OUTPUT_FILE" "bootstrap-native-dependencies.sh" \
    "the tampered-archive error should point at the bootstrap script"
[ ! -e "$EXTRACTION_DIRECTORY" ] || {
    print_error "a refused run must not publish an extraction"
    exit 1
}
mv "${SANDBOX_DIRECTORY}/mlx-c-fake-0.1.tar.gz.pristine" "${CACHE_DIRECTORY}/mlx-c-fake-0.1.tar.gz"

# Contract 5: a cache missing a pinned archive fails offline with an
# actionable message; the fixture URLs are intentionally unreachable, so any
# download attempt would surface as a different failure.
EMPTY_CACHE_DIRECTORY="${SANDBOX_DIRECTORY}/empty-cache"
mkdir -p "$EMPTY_CACHE_DIRECTORY"
assert_run_fails "a cache missing the pinned archives" \
    --cache-dir "$EMPTY_CACHE_DIRECTORY" \
    --manifest "$FIXTURE_MANIFEST" \
    --patch-list "$FIXTURE_PATCH_LIST"
assert_contains "$PROVISION_OUTPUT_FILE" "bootstrap-native-dependencies.sh" \
    "a missing archive should point at the bootstrap script"

# Contract 6: --verify-headers proves the extracted MLX-C headers are
# byte-identical to a completed native build's staged headers, and detects a
# one-byte difference.
provision
built_identity="$(
    timeout 120 sh "${repository_root}/scripts/native-build-cache-fingerprint.sh" \
        --profile core "$repository_root"
)"
FIXTURE_ENTRY_DIRECTORY="${FIXTURE_STORE_DIRECTORY}/v1/entries/${built_identity}"
mkdir -p "${FIXTURE_ENTRY_DIRECTORY}/include"
cp -R "${EXTRACTION_DIRECTORY}/mlx_c-src/mlx" "${FIXTURE_ENTRY_DIRECTORY}/include/mlx"
printf '%s\n' "$built_identity" > "${FIXTURE_ENTRY_DIRECTORY}/complete"

assert_run_succeeds "header verification against a matching build" \
    --cache-dir "$CACHE_DIRECTORY" \
    --native-build-store-dir "$FIXTURE_STORE_DIRECTORY" \
    --verify-headers
assert_contains "$PROVISION_OUTPUT_FILE" "headers_match=yes" "verification should match an identical staged tree"

printf '\n' >> "${FIXTURE_ENTRY_DIRECTORY}/include/mlx/c/mlx.h"
assert_run_fails "header verification against a modified build" \
    --cache-dir "$CACHE_DIRECTORY" \
    --native-build-store-dir "$FIXTURE_STORE_DIRECTORY" \
    --verify-headers
assert_contains "$PROVISION_OUTPUT_FILE" "headers_match=no" "verification must detect a one-byte difference"
assert_contains "$PROVISION_OUTPUT_FILE" "modified ./c/mlx.h" "the differing path should be reported"

rm -rf "${FIXTURE_STORE_DIRECTORY}/v1/entries/${built_identity}"
assert_run_fails "header verification without a completed build" \
    --cache-dir "$CACHE_DIRECTORY" \
    --native-build-store-dir "$FIXTURE_STORE_DIRECTORY" \
    --verify-headers
assert_contains "$PROVISION_OUTPUT_FILE" "no completed native build" \
    "a missing build entry should be reported with guidance"

# Contract 7: the patch-pipeline parser reads the real native CMakeLists and
# yields existing repository-relative patch files for both dependencies.
assert_run_succeeds "printing the parsed patch pipeline" --print-patch-list
PARSED_PATCH_LIST="${SANDBOX_DIRECTORY}/parsed-patch-list"
awk -F'\t' '$1 == "mlx" || $1 == "mlx_c"' "$PROVISION_OUTPUT_FILE" > "$PARSED_PATCH_LIST"
parsed_mlx_patches="$(awk -F'\t' '$1 == "mlx" { count++ } END { print count + 0 }' "$PARSED_PATCH_LIST")"
parsed_mlx_c_patches="$(awk -F'\t' '$1 == "mlx_c" { count++ } END { print count + 0 }' "$PARSED_PATCH_LIST")"
if [ "$parsed_mlx_patches" -lt 1 ] || [ "$parsed_mlx_c_patches" -lt 1 ]; then
    print_error "the parser must find patches for both the mlx and mlx_c dependencies in the real CMakeLists"
    cat "$PROVISION_OUTPUT_FILE" >&2
    exit 1
fi
while IFS="$(printf '\t')" read -r parsed_dependency parsed_patch_path; do
    [ -f "${repository_root}/${parsed_patch_path}" ] || {
        print_error "the parser must emit existing repository-relative patch paths: ${parsed_patch_path}"
        exit 1
    }
    case "$parsed_dependency:$parsed_patch_path" in
        mlx:third-party/patches/mlx-*.patch|mlx_c:third-party/patches/mlx-c-*.patch) ;;
        *)
            print_error "parsed patch entries must pair each dependency with its own patch prefix: ${parsed_dependency} ${parsed_patch_path}"
            exit 1
            ;;
    esac
done < "$PARSED_PATCH_LIST"

printf '%s\n' "provision-bindgen-headers contract tests passed"
