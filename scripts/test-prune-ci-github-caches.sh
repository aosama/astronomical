#!/usr/bin/env sh

# Proves the cache-prune contract against a shimmed GitHub CLI: per-ref
# retention selection, dry-run safety, delete-failure tolerance, and skip
# behavior when the cache surface is unavailable.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=10
readonly FICTIONAL_REPOSITORY="fictional-owner/fictional-repo"

# Created-at timestamps ascend with id so retention must reorder internally:
# sccache keeps the newest entry per ref (1002 over 1001 on main), native-build
# keeps the two newest per ref (1004 and 1005 over 1003 on the merge ref), and
# cargo-downloads is never pruned.
readonly FIXTURE_CACHE_LIST_JSON='[
  {"id": 1001, "key": "astronomical-v2-sccache-macos-14-arm64-older", "ref": "refs/heads/main", "createdAt": "2024-05-01T10:00:00Z"},
  {"id": 1002, "key": "astronomical-v2-sccache-macos-14-arm64-newer", "ref": "refs/heads/main", "createdAt": "2024-05-02T10:00:00Z"},
  {"id": 1003, "key": "astronomical-v2-native-build-macos-14-arm64-oldest", "ref": "refs/pull/855/merge", "createdAt": "2024-05-03T10:00:00Z"},
  {"id": 1004, "key": "astronomical-v2-native-build-macos-14-arm64-middle", "ref": "refs/pull/855/merge", "createdAt": "2024-05-04T10:00:00Z"},
  {"id": 1005, "key": "astronomical-v2-native-build-macos-14-arm64-newest", "ref": "refs/pull/855/merge", "createdAt": "2024-05-05T10:00:00Z"},
  {"id": 1006, "key": "astronomical-v2-sccache-macos-14-arm64-merge", "ref": "refs/pull/855/merge", "createdAt": "2024-05-06T10:00:00Z"},
  {"id": 1007, "key": "astronomical-v2-cargo-downloads-linux-x64-fixture", "ref": "refs/heads/main", "createdAt": "2024-05-07T10:00:00Z"}
]'

SANDBOX_DIRECTORY=""
ORIGINAL_PATH="${PATH}"

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        rm -rf "$SANDBOX_DIRECTORY"
    fi
}
trap cleanup 0

write_gh_shim() {
    shim_path="$1"
    mkdir -p "$(dirname -- "$shim_path")"
    cat > "$shim_path" <<'GH_SHIM'
#!/bin/sh
# Contract-test shim for the GitHub CLI cache surface. Any invocation that
# deviates from the list/delete contract exits 64 so the subject either skips
# or the test assertions fail loudly.
printf '%s\n' "gh $*" >> "$SHIM_CALL_LOG"
if [ "$1" = "cache" ] && [ "$2" = "list" ]; then
    case "$*" in
        *" --json id,key,ref,createdAt"*) : ;;
        *)
            printf '%s\n' "shim: cache list must request id,key,ref,createdAt" >&2
            exit 64
            ;;
    esac
    case "$*" in
        *" -R ${SHIM_EXPECTED_REPOSITORY} "*) : ;;
        *)
            printf '%s\n' "shim: cache list must target ${SHIM_EXPECTED_REPOSITORY}" >&2
            exit 64
            ;;
    esac
    case "$*" in
        *" --limit 1000"*) : ;;
        *)
            printf '%s\n' "shim: cache list must bound its page size" >&2
            exit 64
            ;;
    esac
    if [ "${SHIM_LIST_FAILS:-0}" = "1" ]; then
        printf '%s\n' "shim: cache list unavailable" >&2
        exit 1
    fi
    printf '%s\n' "$SHIM_CACHE_LIST_JSON"
    exit 0
fi
if [ "$1" = "cache" ] && [ "$2" = "delete" ]; then
    case " ${SHIM_FAIL_DELETE_IDS:-} " in
        *" $5 "*)
            printf '%s\n' "shim: delete refused for $5" >&2
            exit 1
            ;;
    esac
    exit 0
fi
printf '%s\n' "shim: unsupported gh invocation: $*" >&2
exit 64
GH_SHIM
    chmod +x "$shim_path"
}

# Runs the prune subject in a per-case sandbox. Caller sets SHIM_LIST_FAILS,
# SHIM_FAIL_DELETE_IDS, and DRY_RUN_ARGUMENT before calling; results land in
# CASE_STATUS and CASE_OUTPUT.
run_prune_case() {
    case_name="$1"
    case_directory="${SANDBOX_DIRECTORY}/${case_name}"
    case_bin_directory="${case_directory}/bin"
    mkdir -p "$case_bin_directory"
    write_gh_shim "${case_bin_directory}/gh"
    SHIM_CALL_LOG="${case_directory}/gh-calls.log"
    : > "$SHIM_CALL_LOG"
    export SHIM_CALL_LOG SHIM_CACHE_LIST_JSON SHIM_EXPECTED_REPOSITORY SHIM_LIST_FAILS SHIM_FAIL_DELETE_IDS
    PATH="${case_bin_directory}:${ORIGINAL_PATH}"
    export PATH
    case_status=0
    case_output="$("$timeout_executable" --foreground -k 1s "${SUBJECT_TIMEOUT_SECONDS}s" \
        "$subject" $DRY_RUN_ARGUMENT 2>&1)" || case_status=$?
    PATH="${ORIGINAL_PATH}"
    export PATH
}

assert_equal() {
    [ "$1" = "$2" ] || {
        print_error "$3 (expected: $2, found: $1)"
        exit 1
    }
}

assert_output_contains() {
    printf '%s\n' "$case_output" | grep -qF -- "$1" || {
        print_error "$case_name output is missing: $1"
        printf '%s\n' "$case_output" >&2
        exit 1
    }
}

assert_no_delete_calls() {
    if grep -q 'gh cache delete' "$SHIM_CALL_LOG"; then
        print_error "$case_name must not delete caches"
        cat "$SHIM_CALL_LOG" >&2
        exit 1
    fi
}

main() {
    for required_command in mktemp ruby; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done
    if command -v timeout >/dev/null 2>&1; then
        timeout_executable="$(command -v timeout)"
    elif command -v gtimeout >/dev/null 2>&1; then
        timeout_executable="$(command -v gtimeout)"
    else
        print_error "GNU timeout is required; install Homebrew coreutils"
        exit 2
    fi

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    subject="${repository_root}/scripts/prune-ci-github-caches.sh"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-cache-prune-test.XXXXXX")"
    SHIM_CACHE_LIST_JSON="$FIXTURE_CACHE_LIST_JSON"
    SHIM_EXPECTED_REPOSITORY="$FICTIONAL_REPOSITORY"
    SHIM_LIST_FAILS=0
    SHIM_FAIL_DELETE_IDS=""
    DRY_RUN_ARGUMENT=""
    export GITHUB_REPOSITORY="$FICTIONAL_REPOSITORY"

    printf '%s\n' '[cache-prune-test] case=retention-keeps-newest-per-ref status=start'
    run_prune_case retention
    assert_equal "$case_status" 0 "retention case exit status"
    assert_output_contains "status=complete deleted=2 failed=0"
    assert_equal "$(grep -c 'gh cache delete' "$SHIM_CALL_LOG")" 2 "retention delete call count"
    assert_equal "$(grep -c 'gh cache delete -R fictional-owner/fictional-repo 1001$' "$SHIM_CALL_LOG")" 1 "oldest sccache deletion"
    assert_equal "$(grep -c 'gh cache delete -R fictional-owner/fictional-repo 1003$' "$SHIM_CALL_LOG")" 1 "oldest native-build deletion"
    printf '%s\n' '[cache-prune-test] case=retention-keeps-newest-per-ref status=success'

    printf '%s\n' '[cache-prune-test] case=dry-run-deletes-nothing status=start'
    DRY_RUN_ARGUMENT="--dry-run"
    run_prune_case dry-run
    assert_equal "$case_status" 0 "dry-run case exit status"
    assert_equal "$(printf '%s\n' "$case_output" | grep -cF 'status=would-delete')" 2 "dry-run would-delete count"
    assert_output_contains "would-delete cache_id=1001"
    assert_output_contains "would-delete cache_id=1003"
    assert_no_delete_calls
    DRY_RUN_ARGUMENT=""
    printf '%s\n' '[cache-prune-test] case=dry-run-deletes-nothing status=success'

    printf '%s\n' '[cache-prune-test] case=delete-failures-are-tolerated status=start'
    SHIM_FAIL_DELETE_IDS="1001 1003"
    run_prune_case delete-failures
    assert_equal "$case_status" 0 "delete-failure case exit status"
    assert_output_contains "status=complete deleted=0 failed=2"
    assert_equal "$(printf '%s\n' "$case_output" | grep -cF 'status=delete-failed')" 2 "delete-failure line count"
    SHIM_FAIL_DELETE_IDS=""
    printf '%s\n' '[cache-prune-test] case=delete-failures-are-tolerated status=success'

    printf '%s\n' '[cache-prune-test] case=unavailable-cache-list-skips status=start'
    SHIM_LIST_FAILS=1
    run_prune_case list-failure
    assert_equal "$case_status" 0 "list-failure case exit status"
    assert_output_contains "status=skipped reason=cache-list-unavailable"
    assert_no_delete_calls
    SHIM_LIST_FAILS=0
    printf '%s\n' '[cache-prune-test] case=unavailable-cache-list-skips status=success'

    printf '%s\n' '[cache-prune-test] case=unknown-argument-is-rejected status=start'
    DRY_RUN_ARGUMENT="--bogus"
    run_prune_case bad-args
    assert_equal "$case_status" 2 "bad-args case exit status"
    DRY_RUN_ARGUMENT=""
    printf '%s\n' '[cache-prune-test] case=unknown-argument-is-rejected status=success'

    printf '%s\n' '[cache-prune-test] case=missing-gh-cli-skips status=start'
    case_directory="${SANDBOX_DIRECTORY}/missing-gh"
    mkdir -p "$case_directory"
    SHIM_CALL_LOG="${case_directory}/gh-calls.log"
    : > "$SHIM_CALL_LOG"
    PATH="/usr/bin:/bin"
    export PATH
    missing_gh_status=0
    missing_gh_output="$("$timeout_executable" --foreground -k 1s "${SUBJECT_TIMEOUT_SECONDS}s" \
        "$subject" 2>&1)" || missing_gh_status=$?
    PATH="${ORIGINAL_PATH}"
    export PATH
    assert_equal "$missing_gh_status" 0 "missing-gh case exit status"
    printf '%s\n' "$missing_gh_output" | grep -qF 'status=skipped reason=gh-cli-unavailable' || {
        print_error "missing-gh case output is missing the skip reason"
        printf '%s\n' "$missing_gh_output" >&2
        exit 1
    }
    printf '%s\n' '[cache-prune-test] case=missing-gh-cli-skips status=success'

    printf '%s\n' '[cache-prune-test] case=step-summary-reports-prune status=start'
    step_summary_file="${SANDBOX_DIRECTORY}/step-summary.md"
    GITHUB_STEP_SUMMARY="$step_summary_file"
    export GITHUB_STEP_SUMMARY
    run_prune_case step-summary
    assert_equal "$case_status" 0 "step-summary case exit status"
    assert_equal "$(grep -c 'gh cache delete' "$SHIM_CALL_LOG")" 2 "step-summary delete call count"
    grep -qF 'Deleted caches: 2' "$step_summary_file" || {
        print_error "step summary is missing the deleted-cache count"
        cat "$step_summary_file" >&2
        exit 1
    }
    unset GITHUB_STEP_SUMMARY
    printf '%s\n' '[cache-prune-test] case=step-summary-reports-prune status=success'
}

main "$@"
