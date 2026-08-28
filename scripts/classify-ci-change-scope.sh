#!/usr/bin/env sh

# Owns the event-specific CI authority policy so pull-request verification and
# default-branch cache publication cannot drift across embedded workflow code.

set -eu

readonly ZERO_GIT_SHA='0000000000000000000000000000000000000000'
CHANGED_FILES_PATH=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${CHANGED_FILES_PATH:-}" ] && [ -f "$CHANGED_FILES_PATH" ]; then
        rm -f "$CHANGED_FILES_PATH"
    fi
}
trap cleanup 0

require_environment_value() {
    environment_name="$1"
    environment_value="$2"
    [ -n "$environment_value" ] || {
        print_error "${environment_name} is required"
        exit 2
    }
}

resolve_repository_root() {
    if [ -n "${REPOSITORY_ROOT:-}" ]; then
        repository_root_candidate="$REPOSITORY_ROOT"
    else
        repository_root_candidate="$(dirname -- "$0")/.."
    fi
    repository_root="$(CDPATH='' cd -- "$repository_root_candidate" && pwd -P)" || {
        print_error "repository root is unavailable: ${repository_root_candidate}"
        exit 2
    }
    git -C "$repository_root" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        print_error "repository root is not a Git worktree: ${repository_root}"
        exit 2
    }
}

publish_change_scope() {
    code_changed="$1"
    native_inputs_changed="$2"
    macos_verification_required="$3"
    classification="$4"
    changed_file_count="$5"
    {
        printf 'code_changed=%s\n' "$code_changed"
        printf 'native_inputs_changed=%s\n' "$native_inputs_changed"
        printf 'macos_verification_required=%s\n' "$macos_verification_required"
    } >> "$GITHUB_OUTPUT"
    printf '[ci-change-scope] status=complete event=%s classification=%s changed_files=%s code_changed=%s native_inputs_changed=%s macos_verification_required=%s elapsed_seconds=%s\n' \
        "$EVENT_NAME" "$classification" "$changed_file_count" "$code_changed" \
        "$native_inputs_changed" "$macos_verification_required" \
        "$(( $(date +%s) - started_at_seconds ))"
}

publish_fail_safe_scope() {
    classification="$1"
    publish_change_scope true true true "$classification" unknown
}

resolve_change_range() {
    case "$EVENT_NAME" in
        pull_request)
            base_sha="${PULL_REQUEST_BASE_SHA:-}"
            head_sha="${PULL_REQUEST_HEAD_SHA:-}"
            ;;
        push)
            base_sha="${PUSH_BEFORE_SHA:-}"
            head_sha="${CURRENT_SHA:-}"
            ;;
        *)
            print_error "unsupported CI event: ${EVENT_NAME}"
            exit 2
            ;;
    esac
}

has_resolvable_change_range() {
    [ -n "$base_sha" ] || return 1
    [ -n "$head_sha" ] || return 1
    [ "$base_sha" != "$ZERO_GIT_SHA" ] || return 1
    git -C "$repository_root" cat-file -e "${base_sha}^{commit}" >/dev/null 2>&1 || return 1
    git -C "$repository_root" cat-file -e "${head_sha}^{commit}" >/dev/null 2>&1 || return 1
}

classify_code_changes() {
    code_changed=false
    while IFS= read -r changed_path; do
        case "$changed_path" in
            site/*|*.md|.github/assets/*|.github/workflows/pages.yml) ;;
            *) code_changed=true; return ;;
        esac
    done < "$CHANGED_FILES_PATH"
}

classify_native_input_changes() {
    # The pathspecs mirror native source-identity ownership. The Rust toolchain
    # is added because it also participates in full hosted compatibility.
    if git -C "$repository_root" diff --quiet --no-renames "$base_sha" "$head_sha" -- \
        crates/runtime-integration/build.rs \
        crates/runtime-integration/build_bindings.rs \
        crates/runtime-integration/build_native_linking.rs \
        crates/runtime-integration/build_native_store.rs \
        crates/runtime-integration/build_native_store_manifest.rs \
        crates/runtime-integration/native-build-store-schema-version \
        crates/runtime-integration/native \
        scripts/native-build-cache-fingerprint.sh \
        third-party/native-dependency-manifest.cmake \
        third-party/pins \
        third-party/patches \
        rust-toolchain.toml
    then
        native_inputs_changed=false
        return
    else
        git_diff_status="$?"
    fi

    [ "$git_diff_status" -eq 1 ] || {
        print_error "native input comparison failed with status ${git_diff_status}"
        exit 1
    }
    native_inputs_changed=true
}

resolve_macos_verification_authority() {
    case "$EVENT_NAME" in
        # PRs and push events (PR merges) both gate on code_changed so that
        # default-branch pushes save native-build and sccache caches for future
        # PRs to restore.  Static-only changes (docs, site) still skip.
        pull_request) macos_verification_required="$code_changed" ;;
        push) macos_verification_required="$code_changed" ;;
        *)
            print_error "unsupported CI event: ${EVENT_NAME}"
            exit 2
            ;;
    esac
}

main() {
    if [ "$#" -ne 0 ]; then
        print_error "classify-ci-change-scope.sh does not accept arguments"
        exit 2
    fi
    require_environment_value EVENT_NAME "${EVENT_NAME:-}"
    require_environment_value GITHUB_OUTPUT "${GITHUB_OUTPUT:-}"
    resolve_repository_root
    started_at_seconds="$(date +%s)"
    printf '[ci-change-scope] status=start event=%s started_at=%s\n' \
        "$EVENT_NAME" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    if [ "$EVENT_NAME" = workflow_dispatch ]; then
        publish_change_scope true false true manual_run 0
        return
    fi

    resolve_change_range
    if ! has_resolvable_change_range; then
        publish_fail_safe_scope unknown_history
        return
    fi

    CHANGED_FILES_PATH="$(mktemp "${TMPDIR:-/tmp}/astronomical-ci-change-scope.XXXXXX")"
    git -C "$repository_root" diff --name-only --no-renames \
        "$base_sha" "$head_sha" > "$CHANGED_FILES_PATH"
    changed_file_count="$(wc -l < "$CHANGED_FILES_PATH" | tr -d '[:space:]')"
    classify_code_changes
    classify_native_input_changes
    resolve_macos_verification_authority

    if [ "$code_changed" = false ]; then
        classification=static_only
    elif [ "$native_inputs_changed" = true ]; then
        classification=native_inputs
    else
        classification=code_or_build
    fi
    publish_change_scope "$code_changed" "$native_inputs_changed" \
        "$macos_verification_required" "$classification" "$changed_file_count"
}

main "$@"
