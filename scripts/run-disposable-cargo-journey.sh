#!/usr/bin/env sh

# Maps named release and acceptance journeys to immutable Cargo commands.
# Their artifact ownership remains centralized in the disposable-target runner.

set -eu

print_usage() {
    printf '%s\n' "Usage: scripts/run-disposable-cargo-journey.sh JOURNEY"
    printf '%s\n' "       scripts/run-disposable-cargo-journey.sh --list"
}

print_journeys() {
    printf '%s\n' \
        accept-model-ssd-streaming \
        accept-serving \
        accept-cache-disabled-generation \
        accept-cached-reverse-model-swap \
        accept-tool-call-reuse \
        accept-laguna-family-swap \
        accept-thinking-seed \
        accept-hard-thinking-budget \
        accept-structured-output \
        accept-speculative-prefill \
        accept-prompt-cache \
        test-model-ssd-streaming-support \
        test-persistent-prompt-cache-performance-support \
        measure-model-ssd-streaming-summary \
        measure-persistent-prompt-cache-warmup \
        measure-persistent-prompt-cache-warmup-50k \
        measure-persistent-prompt-cache-warmup-100k \
        measure-model-ssd-streaming-cold-prefill-50k \
        measure-model-ssd-streaming-prefill-1024 \
        measure-model-ssd-streaming-prefill-2048 \
        measure-model-ssd-streaming-prefill-4096 \
        measure-model-ssd-streaming-prefill-8192 \
        measure-model-ssd-streaming-read-concurrency \
        measure-model-ssd-streaming-complete-expert-residency \
        measure-model-ssd-streaming-leftover-complete-layer-seating \
        measure-model-ssd-streaming-live-memory-ceiling-round-trip \
        measure-model-ssd-streaming-decode-expert-retention \
        measure-model-ssd-streaming-cached-suffix-streaming-prefill \
        measure-model-ssd-streaming-high-ram-cached-suffix-prefill \
        measure-model-ssd-streaming-large-sparse-moe-tight-ceiling-prefill \
        measure-model-ssd-streaming-prefill-memory-progress \
        measure-model-ssd-streaming-laguna-paging \
        measure-experimental-aligned-expert-packs-large-sparse-moe-generation \
        measure-experimental-aligned-expert-packs-large-sparse-moe-prompt-processing \
        measure-experimental-aligned-expert-packs-large-sparse-moe-data-plane
}

main() {
    if [ "$#" -eq 1 ] && [ "$1" = "--list" ]; then
        print_journeys
        return
    fi
    if [ "$#" -ne 1 ]; then
        print_usage >&2
        exit 2
    fi

    journey_name="$1"
    ignored_suite_name=""
    case "$journey_name" in
        accept-model-ssd-streaming)
            lane_name="model-ssd-streaming-acceptance"
            ignored_suite_name="memory-management"
            ;;
        accept-serving)
            lane_name="serving-acceptance"
            ignored_suite_name="serving"
            ;;
        accept-cache-disabled-generation)
            lane_name="cache-disabled-generation"
            set -- cargo test --release -p astronomical-model-serving --test prompt_cache_acceptance_tests --features direct-mlx should_generate_without_prompt_cache_storage_contract_work_when_cache_is_disabled -- --ignored --nocapture
            ;;
        accept-cached-reverse-model-swap)
            lane_name="cached-reverse-model-swap"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_complete_cached_reverse_swap_without_hidden_model_page_ins -- --ignored --nocapture
            ;;
        accept-tool-call-reuse)
            lane_name="deployed-rest-liveness"
            set -- cargo test --release -p astronomical-inference-worker --test serving_acceptance_tests --features serving-acceptance should_complete_a_tool_call_and_reuse_the_worker -- --ignored --nocapture
            ;;
        accept-laguna-family-swap)
            lane_name="laguna-family-model-swap"
            set -- cargo test --release -p astronomical-inference-worker --test serving_acceptance_tests --features serving-acceptance should_swap_qwen_then_laguna_xs_then_qwen_on_one_worker -- --ignored --nocapture
            ;;
        accept-thinking-seed)
            lane_name="qwen3-5-thinking-seed-rest"
            set -- cargo test --release -p astronomical-inference-worker --test serving_acceptance_tests --features serving-acceptance should_seed_the_first_reasoning_output_across_both_streaming_rest_apis -- --ignored --nocapture
            ;;
        accept-hard-thinking-budget)
            lane_name="small-dense-hard-thinking-budget-rest"
            set -- cargo test --release -p astronomical-inference-worker --test serving_acceptance_tests --features serving-acceptance should_commit_the_hard_thinking_budget_before_visible_answer_content -- --ignored --nocapture
            ;;
        accept-structured-output)
            lane_name="structured-output-rest"
            set -- cargo test --release -p astronomical-inference-worker --test serving_acceptance_tests --features serving-acceptance should_serve_structured_json_from_romeo_and_juliet_on_chat_and_responses -- --ignored --nocapture
            ;;
        accept-speculative-prefill)
            lane_name="speculative-prefill-rest"
            set -- cargo test --release -p astronomical-inference-worker --test serving_acceptance_tests --features serving-acceptance should_complete_the_cold_tool_journey_through_real_config_worker_and_rest_boundaries -- --ignored --nocapture
            ;;
        accept-prompt-cache)
            lane_name="persistent-prompt-cache-acceptance"
            ignored_suite_name="prompt-cache"
            ;;
        test-model-ssd-streaming-support)
            lane_name="model-ssd-streaming-support"
            set -- cargo test --no-fail-fast -p astronomical-inference-worker --test performance_measurement_tests --test serving_acceptance_tests --test memory_management_acceptance_tests --features performance-measurement,serving-acceptance,memory-management-acceptance model_ssd_streaming:: -- --nocapture --test-threads 1
            ;;
        test-persistent-prompt-cache-performance-support)
            lane_name="persistent-prompt-cache-performance-support"
            set -- cargo test --no-fail-fast -p astronomical-inference-worker --test performance_measurement_tests --features performance-measurement performance_measurement:: -- --nocapture --test-threads 1
            ;;
        measure-model-ssd-streaming-summary)
            lane_name="model-ssd-streaming-summary"
            set -- cargo test --release -p astronomical-inference-worker --test performance_measurement_tests --features performance-measurement should_measure_model_ssd_streaming_summarization_throughput_and_peak_memory -- --ignored --nocapture
            ;;
        measure-persistent-prompt-cache-warmup)
            lane_name="prompt-cache-warmup"
            set -- cargo test --release -p astronomical-inference-worker --features performance-measurement --test performance_measurement_tests performance_measurement::model_persistent_prompt_cache_warmup::should_measure_model_persistent_prompt_cache_warmup_acceleration -- --ignored --nocapture --exact
            ;;
        measure-persistent-prompt-cache-warmup-50k)
            lane_name="prompt-cache-warmup-50k"
            set -- cargo test --release -p astronomical-inference-worker --features performance-measurement --test performance_measurement_tests performance_measurement::model_persistent_prompt_cache_warmup::should_measure_model_persistent_prompt_cache_warmup_scaling_at_fifty_thousand_words -- --ignored --nocapture --exact
            ;;
        measure-persistent-prompt-cache-warmup-100k)
            lane_name="prompt-cache-warmup-100k"
            set -- cargo test --release -p astronomical-inference-worker --features performance-measurement --test performance_measurement_tests performance_measurement::model_persistent_prompt_cache_warmup::should_measure_model_persistent_prompt_cache_warmup_scaling_at_hundred_thousand_words -- --ignored --nocapture --exact
            ;;
        measure-model-ssd-streaming-cold-prefill-50k)
            lane_name="model-ssd-streaming-cold-prefill-50k"
            set -- cargo test --release -p astronomical-inference-worker --features performance-measurement --test performance_measurement_tests model_ssd_streaming::cold_prefill_measurements::should_measure_model_ssd_streaming_cold_prefill_at_fifty_thousand_words -- --ignored --nocapture --exact
            ;;
        measure-model-ssd-streaming-prefill-1024|measure-model-ssd-streaming-prefill-2048|measure-model-ssd-streaming-prefill-4096|measure-model-ssd-streaming-prefill-8192)
            lane_name="$journey_name"
            prefill_chunk_tokens="${journey_name##*-}"
            set -- cargo test -p astronomical-inference-worker --test performance_measurement_tests --features performance-measurement "should_measure_model_ssd_streaming_with_${prefill_chunk_tokens}_token_prefill_chunks" -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-read-concurrency)
            lane_name="model-ssd-streaming-read-concurrency"
            set -- cargo test --release -p astronomical-runtime-integration --test direct_mlx_tests --features mlx should_preserve_large_bounded_model_weight_intervals_during_parallel_positional_reads -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-complete-expert-residency)
            lane_name="model-ssd-streaming-complete-expert-residency"
            export TEST_TIMEOUT_SECONDS=120
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_keep_all_experts_resident_and_avoid_ssd_reads_when_the_model_fits_memory -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-leftover-complete-layer-seating)
            lane_name="model-ssd-streaming-leftover-complete-layer-seating"
            export TEST_TIMEOUT_SECONDS=120
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_keep_leftover_complete_expert_layers_in_ram_during_squeezed_generation -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-live-memory-ceiling-round-trip)
            lane_name="model-ssd-streaming-live-memory-ceiling-round-trip"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_serve_one_conversation_across_streaming_resident_and_streaming_memory_limits -- --ignored --nocapture --test-threads 1
            ;;
        measure-model-ssd-streaming-decode-expert-retention)
            lane_name="model-ssd-streaming-decode-expert-retention"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_reuse_retained_decode_experts_while_staying_within_the_mlx_memory_ceiling -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-cached-suffix-streaming-prefill)
            lane_name="model-ssd-streaming-cached-suffix-streaming-prefill"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_complete_cold_and_cached_append_requests_with_consistent_prefill_decode_residency -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-high-ram-cached-suffix-prefill)
            lane_name="model-ssd-streaming-high-ram-cached-suffix-prefill"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_keep_high_ram_tool_prefixed_cached_suffix_prefill_responsive -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-large-sparse-moe-tight-ceiling-prefill)
            lane_name="model-ssd-streaming-large-sparse-moe-tight-ceiling-prefill"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_recover_from_prefill_oom_without_stalling -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-prefill-memory-progress)
            lane_name="model-ssd-streaming-prefill-memory-progress"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_report_changing_bounded_mlx_memory_during_prefill -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-laguna-paging)
            lane_name="model-ssd-streaming-laguna-paging"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance model_ssd_streaming::laguna_paging_journey:: -- --ignored --nocapture --test-threads 1
            ;;
        measure-experimental-aligned-expert-packs-large-sparse-moe-generation)
            lane_name="aligned-expert-generation"
            set -- cargo test --release -p astronomical-experimental-aligned-expert-packs --test data_plane_measurement_tests should_measure_one_layer_generation_expert_data_plane -- --ignored --nocapture --test-threads 1
            ;;
        measure-experimental-aligned-expert-packs-large-sparse-moe-prompt-processing)
            lane_name="aligned-expert-prompt-processing"
            set -- cargo test --release -p astronomical-experimental-aligned-expert-packs --test data_plane_measurement_tests should_measure_one_layer_prompt_processing_expert_data_plane -- --ignored --nocapture --test-threads 1
            ;;
        measure-experimental-aligned-expert-packs-large-sparse-moe-data-plane)
            lane_name="aligned-expert-large-sparse-moe"
            set -- cargo test --release -p astronomical-experimental-aligned-expert-packs --test data_plane_measurement_tests should_measure_the_large_sparse_moe_expert_data_plane_in_both_orders -- --ignored --nocapture --test-threads 1
            ;;
        *)
            printf '%s\n' "Error: unknown disposable Cargo journey: ${journey_name}" >&2
            print_usage >&2
            exit 2
            ;;
    esac

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    if [ -n "$ignored_suite_name" ]; then
        exec "${repository_root}/scripts/run-in-disposable-cargo-target.sh" \
            --lane "$lane_name" -- \
            "${repository_root}/scripts/run-ignored-serving-acceptance.sh" "$ignored_suite_name"
    fi
    exec "${repository_root}/scripts/run-in-disposable-cargo-target.sh" \
        --lane "$lane_name" -- \
        "${repository_root}/scripts/run-bounded-cargo-test.sh" "$@"
}

main "$@"
