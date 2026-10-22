#!/usr/bin/env sh

# Proves the astronomical-memory-policy crate never reintroduces a MLX
# dependency: its Cargo.toml must not declare the direct MLX binding crate
# (astronomical-mlx-c-rust). Keeping the memory policy free of the
# graphics-processor bindings is the whole point of the crate — the policy
# crate builds and tests in the offline lane without ever pulling in the
# serving crate or its graphics-processor tests. A detector self-check with a
# synthetic fixture fails the contract if the scanner ever stops catching a
# reintroduced MLX dependency. A watchdog bounds the whole contract below the
# 120-second test ceiling.

set -eu

TEST_WATCHDOG_SECONDS=110
WATCHDOG_PID=""
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${WATCHDOG_PID:-}" ]; then
        kill -9 "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe layering-test sandbox" ;;
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

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"

require_command grep

# The MLX dependency the policy crate must never pull in: the direct MLX
# binding crate. Its presence would force the memory policy to rebuild the
# graphics-processor world, defeating the extraction.
MLX_DEPENDENCY_WORD="astronomical-mlx-c-rust"

# True when Cargo.toml text at $1 declares the MLX binding as a real
# dependency line (a `name = ...` declaration), not merely mentions it inside
# a comment — so a policy Cargo.toml that only talks about the binding in
# prose does not trip the sweep. Shared by the detector self-check and the
# repository sweep so the self-check exercises the exact code path the sweep
# runs, and detector rot fails loudly instead of passing vacuously.
mlx_dependency_is_declared() {
    grep -Eq "^[[:space:]]*${MLX_DEPENDENCY_WORD}[[:space:]]*=" "$1"
}

( sleep "$TEST_WATCHDOG_SECONDS" && kill -9 "$$" ) 2>/dev/null &
WATCHDOG_PID=$!

SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-memory-policy-layering.XXXXXX")"
FIXTURE_CARGO_TOML="${SANDBOX_DIRECTORY}/fixture-cargo-toml"

printf '%s\n' '[memory-policy-layering-contract] phase=detector-self-check status=start'
cat >"$FIXTURE_CARGO_TOML" <<'FIXTURE'
[dependencies]
astronomical-mlx-c-rust = { path = "../mlx-c-rust" }
FIXTURE

if mlx_dependency_is_declared "$FIXTURE_CARGO_TOML"; then
    printf '%s\n' '[memory-policy-layering-contract] phase=detector-self-check status=success'
else
    print_error "detector self-check failed: a known MLX dependency must be detected"
    exit 1
fi

printf '%s\n' '[memory-policy-layering-contract] phase=repository-sweep status=start'
policy_cargo_toml="${repository_root}/crates/astronomical-memory-policy/Cargo.toml"
if [ ! -f "$policy_cargo_toml" ]; then
    print_error "memory policy crate is missing: ${policy_cargo_toml}"
    exit 1
fi
if mlx_dependency_is_declared "$policy_cargo_toml"; then
    print_error "astronomical-memory-policy must not declare ${MLX_DEPENDENCY_WORD}: the policy crate must build and test in the offline lane without ever pulling in the serving crate or its graphics-processor tests"
    exit 1
fi

printf '%s\n' "[memory-policy-layering-contract] phase=repository-sweep status=success"
printf '%s\n' "memory-policy layering contract tests passed"
