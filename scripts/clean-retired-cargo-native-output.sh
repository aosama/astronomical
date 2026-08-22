#!/usr/bin/env sh

# Removes only retired complete native CMake trees from Cargo build-script
# generations while holding Cargo's target lock. Generated bindings, linked
# binaries, diagnostic symbols, and compatibility-keyed native products remain.

set -eu

ACTION=""
TARGET_DIRECTORY=""

print_usage() {
    printf '%s\n' "Usage: scripts/clean-retired-cargo-native-output.sh (--dry-run | --apply) [--target-directory PATH]"
}

print_error() {
    printf '%s\n' "Error: $1" >&2
}

require_command() {
    required_command_name="$1"
    command -v "$required_command_name" >/dev/null 2>&1 || {
        print_error "required command is unavailable: ${required_command_name}"
        exit 1
    }
}

parse_arguments() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --dry-run|--apply)
                [ -z "$ACTION" ] || {
                    print_error "choose exactly one of --dry-run or --apply"
                    exit 2
                }
                ACTION="${1#--}"
                shift
                ;;
            --target-directory)
                [ "$#" -ge 2 ] || {
                    print_error "--target-directory requires a path"
                    exit 2
                }
                TARGET_DIRECTORY="$2"
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                print_error "unknown argument: $1"
                print_usage >&2
                exit 2
                ;;
        esac
    done

    [ -n "$ACTION" ] || {
        print_error "choose --dry-run to inspect or --apply to remove retired native output"
        print_usage >&2
        exit 2
    }
}

main() {
    parse_arguments "$@"
    require_command python3

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    if [ -z "$TARGET_DIRECTORY" ]; then
        require_command cargo
        TARGET_DIRECTORY="$(
            CDPATH='' cd -- "$repository_root"
            cargo metadata --no-deps --format-version 1 \
                | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
        )"
    fi

    python3 -u - "$TARGET_DIRECTORY" "$ACTION" <<'PYTHON'
from __future__ import annotations

import concurrent.futures
import contextlib
import fcntl
import os
import pathlib
import shutil
import sys
import time


LEGACY_DIRECTORY_NAME = "mlx-c-runtime-build"
REMOVAL_DIRECTORY_PREFIX = f".{LEGACY_DIRECTORY_NAME}.astronomical-removing-"
RUNTIME_BUILD_PREFIX = "astronomical-runtime-integration-"
PROGRESS_INTERVAL_SECONDS = 5


def _run_with_progress(operation_name: str, operation):
    operation_started_at = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        operation_future = executor.submit(operation)
        while True:
            try:
                return operation_future.result(timeout=PROGRESS_INTERVAL_SECONDS)
            except concurrent.futures.TimeoutError:
                print(
                    "[legacy-native-output-cleanup] "
                    f"status=progress operation={operation_name} "
                    f"elapsed_ms={int((time.monotonic() - operation_started_at) * 1000)}"
                )


def _allocated_bytes(directory: pathlib.Path) -> int:
    allocated_bytes = directory.lstat().st_blocks * 512
    for descendant in directory.rglob("*"):
        allocated_bytes += descendant.lstat().st_blocks * 512
    return allocated_bytes


def _legacy_directories(target_directory: pathlib.Path) -> list[pathlib.Path]:
    candidate_patterns = (
        f"*/build/{RUNTIME_BUILD_PREFIX}*/out/{LEGACY_DIRECTORY_NAME}",
        f"*/*/build/{RUNTIME_BUILD_PREFIX}*/out/{LEGACY_DIRECTORY_NAME}",
        f"*/build/{RUNTIME_BUILD_PREFIX}*/out/{REMOVAL_DIRECTORY_PREFIX}*",
        f"*/*/build/{RUNTIME_BUILD_PREFIX}*/out/{REMOVAL_DIRECTORY_PREFIX}*",
    )
    candidates = {
        candidate
        for candidate_pattern in candidate_patterns
        for candidate in target_directory.glob(candidate_pattern)
    }
    return sorted(candidates)


def _validate_candidates(
    target_directory: pathlib.Path,
    candidates: list[pathlib.Path],
) -> None:
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(f"refusing unexpected legacy native output: {candidate}")
        if not candidate.resolve().is_relative_to(target_directory):
            raise ValueError(f"refusing legacy native output outside Cargo target: {candidate}")


def _profile_directory(candidate: pathlib.Path) -> pathlib.Path:
    if (
        candidate.parent.name != "out"
        or not candidate.parents[1].name.startswith(RUNTIME_BUILD_PREFIX)
        or candidate.parents[2].name != "build"
    ):
        raise ValueError(f"could not resolve Cargo profile ownership for {candidate}")
    profile_directory = candidate.parents[3]
    return profile_directory


def _acquire_profile_locks(
    lock_stack: contextlib.ExitStack,
    candidates: list[pathlib.Path],
) -> set[pathlib.Path]:
    profile_directories = {_profile_directory(candidate) for candidate in candidates}
    for profile_directory in sorted(profile_directories):
        lock_path = profile_directory / ".cargo-lock"
        if lock_path.is_symlink():
            raise ValueError(f"refusing symbolic-link Cargo profile lock: {lock_path}")
        target_lock = lock_stack.enter_context(lock_path.open("a+b"))
        try:
            # Cargo 1.97 holds this lock shared for every active profile; the
            # cleaner's exclusive lock therefore also excludes older Cargo.
            fcntl.flock(target_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Cargo profile is active; retry after its command finishes: {profile_directory}"
            ) from error
    return profile_directories


def _move_owned_candidate_for_removal(
    candidate: pathlib.Path,
    candidate_index: int,
) -> pathlib.Path:
    if candidate.name.startswith(REMOVAL_DIRECTORY_PREFIX):
        return candidate
    candidate_metadata = candidate.lstat()
    removal_path = candidate.with_name(
        f".{candidate.name}.astronomical-removing-{os.getpid()}-{candidate_index}"
    )
    if removal_path.exists() or removal_path.is_symlink():
        raise ValueError(f"refusing occupied cleanup staging path: {removal_path}")
    candidate.rename(removal_path)
    removal_metadata = removal_path.lstat()
    if (
        removal_path.is_symlink()
        or not removal_path.is_dir()
        or removal_metadata.st_dev != candidate_metadata.st_dev
        or removal_metadata.st_ino != candidate_metadata.st_ino
    ):
        raise ValueError(f"legacy native output changed during ownership handoff: {candidate}")
    return removal_path


def _clean_target(target_directory_argument: str, action: str) -> None:
    target_directory = pathlib.Path(target_directory_argument).expanduser().resolve(strict=True)
    if target_directory == pathlib.Path(target_directory.anchor):
        raise ValueError("refusing to inspect a filesystem root as a Cargo target")

    cargo_identity_path = target_directory / ".rustc_info.json"
    if action == "apply" and (cargo_identity_path.is_symlink() or not cargo_identity_path.is_file()):
        raise ValueError(
            f"refusing to modify a directory without Cargo target ownership evidence: {target_directory}"
        )

    with contextlib.ExitStack() as lock_stack:
        print(
            "[legacy-native-output-cleanup] "
            f"status=scan-start action={action} target_directory={target_directory}"
        )
        candidates = _legacy_directories(target_directory)
        _validate_candidates(target_directory, candidates)
        locked_profile_directories = _acquire_profile_locks(lock_stack, candidates)
        candidates_after_lock = _legacy_directories(target_directory)
        _validate_candidates(target_directory, candidates_after_lock)
        candidates = [
            candidate
            for candidate in candidates_after_lock
            if _profile_directory(candidate) in locked_profile_directories
        ]
        candidate_allocated_bytes = [
            _run_with_progress(
                f"measure-{candidate_index}-of-{len(candidates)}",
                lambda candidate=candidate: _allocated_bytes(candidate),
            )
            for candidate_index, candidate in enumerate(candidates, start=1)
        ]
        total_allocated_bytes = sum(candidate_allocated_bytes)
        print(
            "[legacy-native-output-cleanup] "
            f"status=start action={action} candidates={len(candidates)} "
            f"allocated_bytes={total_allocated_bytes}"
        )
        cleanup_started_at = time.monotonic()
        if action == "apply":
            for candidate_index, (candidate, allocated_bytes) in enumerate(
                zip(candidates, candidate_allocated_bytes),
                start=1,
            ):
                candidate_started_at = time.monotonic()
                print(
                    "[legacy-native-output-cleanup] "
                    f"status=removing candidate={candidate_index}/{len(candidates)} "
                    f"allocated_bytes={allocated_bytes} path={candidate}"
                )
                removal_path = _move_owned_candidate_for_removal(candidate, candidate_index)
                _run_with_progress(
                    f"remove-{candidate_index}-of-{len(candidates)}",
                    lambda removal_path=removal_path: shutil.rmtree(removal_path),
                )
                print(
                    "[legacy-native-output-cleanup] "
                    f"status=removed candidate={candidate_index}/{len(candidates)} "
                    f"elapsed_ms={int((time.monotonic() - candidate_started_at) * 1000)}"
                )
        print(
            "[legacy-native-output-cleanup] "
            f"status=success action={action} candidates={len(candidates)} "
            f"allocated_bytes={total_allocated_bytes} "
            f"elapsed_ms={int((time.monotonic() - cleanup_started_at) * 1000)}"
        )


def main() -> int:
    try:
        _clean_target(sys.argv[1], sys.argv[2])
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYTHON
}

main "$@"
