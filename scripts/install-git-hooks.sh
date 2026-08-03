#!/usr/bin/env sh

set -eu

REPOSITORY_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    printf '%s\n' '[git-hooks] status=failed reason=not-a-git-worktree' >&2
    exit 1
}
TRACKED_HOOK_PATH="${REPOSITORY_ROOT}/.githooks/pre-commit"

if [ ! -x "${TRACKED_HOOK_PATH}" ]; then
    printf '%s\n' "[git-hooks] status=failed reason=hook-not-executable path=${TRACKED_HOOK_PATH}" >&2
    exit 1
fi

printf '%s\n' "[git-hooks] status=start action=configure-hooks-path root=${REPOSITORY_ROOT}"
started_epoch_seconds="$(date +%s)"
git -C "${REPOSITORY_ROOT}" config --local core.hooksPath .githooks
CONFIGURED_HOOKS_PATH="$(git -C "${REPOSITORY_ROOT}" config --local --get core.hooksPath)"
if [ "${CONFIGURED_HOOKS_PATH}" != ".githooks" ]; then
    printf '%s\n' "[git-hooks] status=failed reason=unexpected-hooks-path value=${CONFIGURED_HOOKS_PATH}" >&2
    exit 1
fi
elapsed_seconds=$(( $(date +%s) - started_epoch_seconds ))
printf '%s\n' "[git-hooks] status=success hooks_path=${CONFIGURED_HOOKS_PATH} elapsed_seconds=${elapsed_seconds}"
