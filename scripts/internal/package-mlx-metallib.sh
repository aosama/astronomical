# Packs the MLX AOT metallib into an app bundle. Sourced by build-macos-app.sh,
# which provides print_error.

# Cache directories are purgeable. Copy the build-produced metallib into the bundle
# so a shipped app can start after ~/Library/Caches is deleted.
resolve_packaged_mlx_metallib_source() {
    repository_root="$1"
    native_target_triple="$2"
    if [ -n "${ASTRONOMICAL_MLX_METALLIB_PATH:-}" ]; then
        printf '%s\n' "$ASTRONOMICAL_MLX_METALLIB_PATH"
        return 0
    fi
    schema_version_file="${repository_root}/crates/runtime-integration/native-build-store-schema-version"
    [ -s "$schema_version_file" ] || {
        print_error "native build store schema version is unavailable: ${schema_version_file}"
        return 1
    }
    schema_version="$(tr -d '[:space:]' < "$schema_version_file")"
    case "$schema_version" in
        ''|*[!0-9]*)
            print_error "native build store schema version must be an unsigned integer"
            return 1
            ;;
    esac
    fingerprint_script="${repository_root}/scripts/native-build-cache-fingerprint.sh"
    [ -x "$fingerprint_script" ] || {
        print_error "native identity script is unavailable: ${fingerprint_script}"
        return 1
    }
    native_identity="$(TARGET="$native_target_triple" "$fingerprint_script" --profile core "$repository_root")" || return 1
    case "$native_identity" in
        ''|*[!0-9a-f]*)
            print_error "native build identity must be lowercase hexadecimal"
            return 1
            ;;
    esac
    if [ "${#native_identity}" -ne 64 ]; then
        print_error "native build identity must be 64 lowercase hexadecimal characters"
        return 1
    fi
    native_store_root="${ASTRONOMICAL_NATIVE_BUILD_STORE_DIR:-}"
    if [ -z "$native_store_root" ]; then
        case "${HOME:-}" in
            /*) ;;
            *)
                print_error "HOME or ASTRONOMICAL_NATIVE_BUILD_STORE_DIR is required to locate mlx.metallib"
                return 1
                ;;
        esac
        native_store_root="${HOME}/Library/Caches/Astronomical/native-builds"
    fi
    printf '%s\n' "${native_store_root}/v${schema_version}/entries/${native_identity}/share/mlx/mlx.metallib"
}

package_mlx_metallib() {
    repository_root="$1"
    app_bundle_path="$2"
    native_target_triple="$3"
    case "$native_target_triple" in
        ''|*[!A-Za-z0-9_.-]*)
            print_error "native target triple is required to locate mlx.metallib"
            exit 1
            ;;
    esac
    packaging_started_at="$(date +%s)"
    printf '%s operation=package-mlx-metallib status=start\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    source_metallib="$(resolve_packaged_mlx_metallib_source "$repository_root" "$native_target_triple")" || exit 1
    if [ -L "$source_metallib" ] || [ ! -f "$source_metallib" ]; then
        print_error "required MLX AOT metallib is unavailable: ${source_metallib}"
        exit 1
    fi
    bundled_metallib_directory="${app_bundle_path}/Contents/Resources/share/mlx"
    bundled_metallib_path="${bundled_metallib_directory}/mlx.metallib"
    mkdir -p "$bundled_metallib_directory"
    cp "$source_metallib" "$bundled_metallib_path"
    if [ -L "$bundled_metallib_path" ] || [ ! -s "$bundled_metallib_path" ]; then
        print_error "packaged MLX AOT metallib is unavailable: ${bundled_metallib_path}"
        exit 1
    fi
    printf '%s operation=package-mlx-metallib status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$(( $(date +%s) - packaging_started_at ))"
}
