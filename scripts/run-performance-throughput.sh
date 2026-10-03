#!/usr/bin/env sh

# Runs the single laptop-only performance-throughput journey — the #[ignore]d
# Rust test at apps/inference-worker/tests/performance_throughput/qwen3_5_moe.rs —
# through the repository's bounded cargo-test runner, which enforces serial
# execution, the machine-protection invocation lock, and a 120-second boundary
# per pass, in optimized (release) mode.
#
# The Rust test is the source of truth for the test case (a short ~1,000-token
# warmup, then a >=10,000-token Romeo and Juliet measured run of ~1,000 output
# tokens, SSD cache disabled); this script only invokes it. Journeys load real
# multi-gigabyte model weights into wired GPU memory, so they are not wired into
# CI and require an Apple-Silicon host.

set -eu

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
bounded_runner="${repository_root}/scripts/run-bounded-cargo-test.sh"

# The one ignored throughput journey, by its exact Rust test name.
test_name="performance_throughput::qwen3_5_moe::should_measure_resident_sparse_moe_prompt_processing_and_decode_throughput"

started_at_seconds="$(date +%s)"
printf '%s\n' "[performance-throughput] status=start name=${test_name} started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
if TEST_TIMEOUT_SECONDS=120 \
    "${bounded_runner}" cargo test --release \
        -p astronomical-inference-worker \
        --features astronomical-inference-worker/performance_throughput \
        --test performance_throughput_tests \
        "${test_name}" \
        -- --ignored --exact --nocapture; then
    status="success"
else
    status="failed"
fi
printf '%s\n' "[performance-throughput] status=${status} elapsed_seconds=$(( $(date +%s) - started_at_seconds ))"
if [ "$status" != "success" ]; then
    exit 1
fi

printf '%s\n' "[performance-throughput] status=complete finished_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
