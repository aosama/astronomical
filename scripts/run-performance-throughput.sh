#!/usr/bin/env sh

# Runs the laptop-only performance-throughput journeys — the #[ignore]d Rust
# tests under apps/inference-worker/tests/performance_throughput/ — through the
# repository's bounded cargo-test runner, which enforces serial execution, the
# machine-protection invocation lock, and a 120-second boundary per pass, in
# optimized (release) mode. The text journey and the vision journey run one
# after the other, never in parallel: each loads real multi-gigabyte model
# weights into wired GPU memory.
#
# The Rust tests are the source of truth for the test cases (a short ~1,000-
# token warmup, then a >=10,000-token measured run of ~1,000 output tokens —
# text-only for the text journey, plus a 9,216-visual-token image for the
# vision journey; SSD cache disabled); this script only invokes them. Journeys
# are not wired into CI and require an Apple-Silicon host.
#
# Command per journey (release profile, performance-throughput feature):
#   cargo test --release -p astronomical-inference-worker \
#     --features astronomical-inference-worker/performance_throughput \
#     --test performance_throughput_tests <TEST_NAME> -- --ignored --exact --nocapture
# with TEST_NAME = performance_throughput::qwen3_5_moe::should_measure_resident_sparse_moe_prompt_processing_and_decode_throughput
# or   TEST_NAME = performance_throughput::qwen3_5_moe_vision::should_measure_resident_sparse_moe_vision_prompt_processing_and_decode_throughput

set -eu

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
bounded_runner="${repository_root}/scripts/run-bounded-cargo-test.sh"

# The ignored throughput journeys, by their exact Rust test names.
text_journey_test_name="performance_throughput::qwen3_5_moe::should_measure_resident_sparse_moe_prompt_processing_and_decode_throughput"
vision_journey_test_name="performance_throughput::qwen3_5_moe_vision::should_measure_resident_sparse_moe_vision_prompt_processing_and_decode_throughput"

overall_status="success"
for test_name in "$text_journey_test_name" "$vision_journey_test_name"; do
    started_at_seconds="$(date +%s)"
    printf '%s\n' "[performance-throughput] status=start name=${test_name} started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    if TEST_TIMEOUT_SECONDS=120 \
        "${bounded_runner}" cargo test --release \
            -p astronomical-inference-worker \
            --features astronomical-inference-worker/performance_throughput \
            --test performance_throughput_tests \
            "${test_name}" \
            -- --ignored --exact --nocapture; then
        journey_status="success"
    else
        journey_status="failed"
        overall_status="failed"
    fi
    printf '%s\n' "[performance-throughput] status=${journey_status} elapsed_seconds=$(( $(date +%s) - started_at_seconds ))"
done

printf '%s\n' "[performance-throughput] status=${overall_status} finished_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
if [ "$overall_status" != "success" ]; then
    exit 1
fi
