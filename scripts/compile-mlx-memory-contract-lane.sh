#!/usr/bin/env sh

# Compiles the feature-gated MLX memory-contract lane so API moves under its
# feature world fail at commit time instead of drifting silently. The lane is
# off every routine graph, which is exactly how its contract tests once broke
# without anything noticing.
#
# The lane's journeys allocate real MLX GPU arrays and stay behind #[ignore];
# this gate is deliberately compile-only. Run the journeys through
# scripts/test-mlx-memory-contracts.sh, which gives every process-global
# memory-limit world its own test process, when the native
# capacity-rejection contract itself needs re-proof.
#
# A cold native-build store makes Cargo run the probe-profile CMake build
# inside this step without live progress; the commit gate prewarms both
# profiles ahead of it, and outside the gate
# `scripts/prewarm-native-build.sh --profile core+memory-contract` streams
# that build live.

set -eu

readonly SCRIPT_PREFIX="[mlx-memory-contract-compile]"

if [ "$#" -ne 0 ]; then
    printf '%s\n' "Error: scripts/compile-mlx-memory-contract-lane.sh does not accept arguments" >&2
    exit 2
fi

started_at_seconds="$(date +%s)"
printf '%s\n' "${SCRIPT_PREFIX} status=start"

if cargo check \
    --package astronomical-runtime-integration \
    --features mlx-memory-contract-probe \
    --all-targets
then
    printf '%s\n' \
        "${SCRIPT_PREFIX} status=success elapsed_seconds=$(( $(date +%s) - started_at_seconds ))"
    exit 0
fi
printf '%s\n' \
    "${SCRIPT_PREFIX} status=failed elapsed_seconds=$(( $(date +%s) - started_at_seconds ))" >&2
exit 1
