#!/usr/bin/env sh
# Contract for the bounded cargo test invocation lock.
#
# The lock protects real-model journeys from overlapping wired GPU memory.
# These cases pin its three behaviors against a sandboxed runner with a fake
# cargo so no compilation, model weights, or GPU time is involved:
# 1. A fresh clone or worktree without target/ acquires the lock.
# 2. A live owner pid is refused and the live lock remains in place.
# 3. A lock left by a provably dead owner pid is stolen and released on exit.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

SUBJECT_TIMEOUT_SECONDS=20

assert_path_is_absent() {
    asserted_path="$1"
    [ ! -e "$asserted_path" ] || {
        print_error "unexpected path exists: ${asserted_path}"
        exit 1
    }
}

run_subject() {
    timeout_executable="$1"
    shift
    "$timeout_executable" --foreground -k 1s "${SUBJECT_TIMEOUT_SECONDS}s" "$@"
}

main() {
    for required_command in mktemp cp chmod sh timeout; do
        if ! command -v "$required_command" >/dev/null 2>&1; then
            if [ "$required_command" = "timeout" ] && command -v gtimeout >/dev/null 2>&1; then
                continue
            fi
            print_error "required command is unavailable: ${required_command}"
            exit 2
        fi
    done
    if command -v timeout >/dev/null 2>&1; then
        timeout_executable="$(command -v timeout)"
    else
        timeout_executable="$(command -v gtimeout)"
    fi

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-bounded-lock.XXXXXX")"
    cleanup() {
        if [ -z "${SANDBOX_DIRECTORY:-}" ] || [ ! -d "$SANDBOX_DIRECTORY" ]; then
            return
        fi
        case "$SANDBOX_DIRECTORY" in
            /|.|..)
                print_error "refusing to remove unsafe bounded-lock test sandbox"
                ;;
            *)
                rm -rf "$SANDBOX_DIRECTORY"
                ;;
        esac
    }
    trap cleanup 0

    sandbox_repository="${SANDBOX_DIRECTORY}/repository"
    sandbox_scripts_directory="${sandbox_repository}/scripts"
    mkdir -p "$sandbox_scripts_directory"
    cp "${repository_root}/scripts/run-bounded-cargo-test.sh" \
        "${sandbox_scripts_directory}/run-bounded-cargo-test.sh"
    chmod +x "${sandbox_scripts_directory}/run-bounded-cargo-test.sh"
    lock_subject="${sandbox_scripts_directory}/run-bounded-cargo-test.sh"
    sandbox_lock_directory="${sandbox_repository}/target/bounded-cargo-test.lock"

    # A fake cargo completes every bounded phase instantly and records each
    # invocation so the cases can prove both phases ran.
    fake_command_directory="${SANDBOX_DIRECTORY}/fake-bin"
    invocation_record="${SANDBOX_DIRECTORY}/fake-cargo-invocations"
    mkdir -p "$fake_command_directory"
    cat > "${fake_command_directory}/cargo" <<CARGO
#!/usr/bin/env sh
set -eu
printf 'invoked\n' >> "$invocation_record"
exit 0
CARGO
    chmod +x "${fake_command_directory}/cargo"

    printf '%s\n' '[bounded-cargo-test-lock-test] case=fresh-worktree-lock-acquisition status=start'
    if [ -d "${sandbox_repository}/target" ]; then
        print_error "the lock contract needs a sandbox repository without a target directory"
        exit 1
    fi
    fresh_lock_output="${SANDBOX_DIRECTORY}/fresh-lock-output"
    if PATH="${fake_command_directory}:${PATH}" \
        run_subject "$timeout_executable" "$lock_subject" \
            cargo test --no-run -- --ignored \
            > "$fresh_lock_output" 2>&1
    then
        : # the fake cargo completes both phases once the lock is held
    else
        printf '%s\n' 'fresh-worktree lock acquisition output:' >&2
        cat "$fresh_lock_output" >&2
        print_error "the bounded runner must acquire its lock in a repository without a target directory"
        exit 1
    fi
    grep -q 'phase=compile status=start' "$fresh_lock_output" || {
        print_error "the bounded runner never reached the compile phase after lock acquisition"
        exit 1
    }
    if grep -q 'stealing a stale bounded-cargo-test lock' "$fresh_lock_output"; then
        printf '%s\n' 'fresh-worktree lock acquisition output:' >&2
        cat "$fresh_lock_output" >&2
        print_error "a missing target parent must not be misreported as a stale lock"
        exit 1
    fi
    [ "$(wc -l < "$invocation_record")" -eq 2 ] || {
        print_error "the fake cargo should run once per bounded phase, compile then test"
        exit 1
    }
    assert_path_is_absent "$sandbox_lock_directory"
    printf '%s\n' '[bounded-cargo-test-lock-test] case=fresh-worktree-lock-acquisition status=success'

    printf '%s\n' '[bounded-cargo-test-lock-test] case=live-lock-is-refused status=start'
    mkdir -p "$sandbox_lock_directory"
    # The contract script's own process id is alive for the whole case.
    printf '%s\n' "$$" > "${sandbox_lock_directory}/owner-pid"
    live_lock_output="${SANDBOX_DIRECTORY}/live-lock-output"
    if PATH="${fake_command_directory}:${PATH}" \
        run_subject "$timeout_executable" "$lock_subject" \
            cargo test --no-run -- --ignored \
            > "$live_lock_output" 2>&1
    then
        printf '%s\n' 'live-lock refusal output:' >&2
        cat "$live_lock_output" >&2
        print_error "a live invocation lock must refuse a second bounded runner"
        exit 1
    fi
    grep -q 'already running' "$live_lock_output" || {
        printf '%s\n' 'live-lock refusal output:' >&2
        cat "$live_lock_output" >&2
        print_error "the refusal must name the live owner instead of claiming a stale lock"
        exit 1
    }
    [ -d "$sandbox_lock_directory" ] || {
        print_error "a refused invocation must leave the live lock in place"
        exit 1
    }
    rm -rf "$sandbox_lock_directory"
    printf '%s\n' '[bounded-cargo-test-lock-test] case=live-lock-is-refused status=success'

    printf '%s\n' '[bounded-cargo-test-lock-test] case=dead-lock-is-stolen status=start'
    # A command substitution prints its own process identifier and exits, so by
    # the time the substitution returns the owner is reaped and provably dead.
    # A background job would race this script's EXIT trap and delete the sandbox.
    obtain_dead_owner_pid() {
        for steal_attempt in 1 2 3 4 5; do
            candidate_owner_pid="$(sh -c 'printf %s "$$"; exit 0')"
            if ! kill -0 "$candidate_owner_pid" 2>/dev/null; then
                printf '%s' "$candidate_owner_pid"
                return 0
            fi
            printf '%s\n' "candidate pid ${candidate_owner_pid} was recycled, retrying" >&2
        done
        print_error "could not obtain a provably dead owner pid for the steal contract"
        exit 1
    }
    dead_owner_pid="$(obtain_dead_owner_pid)"
    mkdir -p "$sandbox_lock_directory"
    printf '%s\n' "$dead_owner_pid" > "${sandbox_lock_directory}/owner-pid"
    dead_lock_output="${SANDBOX_DIRECTORY}/dead-lock-output"
    if PATH="${fake_command_directory}:${PATH}" \
        run_subject "$timeout_executable" "$lock_subject" \
            cargo test --no-run -- --ignored \
            > "$dead_lock_output" 2>&1
    then
        : # the provably dead owner is stolen and the phases run
    else
        printf '%s\n' 'dead-lock steal output:' >&2
        cat "$dead_lock_output" >&2
        print_error "a lock left by a provably dead pid must be stolen"
        exit 1
    fi
    grep -q 'phase=compile status=start' "$dead_lock_output" || {
        printf '%s\n' 'dead-lock steal output:' >&2
        cat "$dead_lock_output" >&2
        print_error "the stolen lock must proceed into the bounded phases"
        exit 1
    }
    assert_path_is_absent "$sandbox_lock_directory"
    printf '%s\n' '[bounded-cargo-test-lock-test] case=dead-lock-is-stolen status=success'

    printf '%s\n' '[bounded-cargo-test-lock-test] status=success'
}

main "$@"
