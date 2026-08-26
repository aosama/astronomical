#!/usr/bin/env sh

# Maps named release and qualification journeys to immutable Cargo commands.
# Their artifact ownership remains centralized in the disposable-target runner.

set -eu

print_usage() {
    printf '%s\n' "Usage: scripts/run-disposable-cargo-journey.sh JOURNEY"
    printf '%s\n' "       scripts/run-disposable-cargo-journey.sh --list"
}

print_journeys() {
    printf '%s\n' \
        accept-model-ssd-streaming \
        qualify-model-artifacts \
        qualify-cache-disabled-generation \
        qualify-cached-reverse-model-swap \
        qualify-deployed-model-rest-liveness \
        qualify-laguna-family-model-swap \
        qualify-laguna-attribution \
        qualify-qwen3-5-thinking-seed-rest \
        qualify-smallest-qwen3-5-hard-thinking-budget-rest \
        qualify-speculative-prefill-rest \
        qualify-persistent-prompt-cache \
        test-model-ssd-streaming-support \
        test-model-ssd-streaming-attribution-support \
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
        measure-model-ssd-streaming-attribution \
        measure-model-ssd-streaming-opencode-long-context-reuse \
        measure-model-ssd-streaming-complete-expert-residency \
        measure-model-ssd-streaming-live-memory-ceiling-round-trip \
        measure-model-ssd-streaming-decode-expert-retention \
        measure-model-ssd-streaming-cached-suffix-streaming-prefill \
        measure-model-ssd-streaming-high-ram-cached-suffix-prefill \
        measure-model-ssd-streaming-near-disk-oq6e-cached-suffix-prefill \
        measure-model-ssd-streaming-oq6e-tight-ceiling-prefill \
        measure-model-ssd-streaming-oq6e-long-conversation \
        measure-model-ssd-streaming-prefill-memory-progress \
        measure-model-ssd-streaming-laguna-paging \
        measure-experimental-aligned-expert-packs-ornith-generation \
        measure-experimental-aligned-expert-packs-ornith-prompt-processing \
        measure-experimental-aligned-expert-packs-oq6e-data-plane
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
        qualify-model-artifacts)
            lane_name="model-artifact-qualification"
            ignored_suite_name="model-artifacts"
            ;;
        qualify-cache-disabled-generation)
            lane_name="cache-disabled-generation"
            set -- cargo test --release -p astronomical-model-serving --test persistent_prompt_cache_qualification_tests --features direct-mlx should_generate_without_prompt_cache_storage_contract_work_when_cache_is_disabled -- --ignored --nocapture
            ;;
        qualify-cached-reverse-model-swap)
            lane_name="cached-reverse-model-swap"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_complete_cached_reverse_swap_without_hidden_model_page_ins -- --ignored --nocapture
            ;;
        qualify-deployed-model-rest-liveness)
            lane_name="deployed-rest-liveness"
            set -- cargo test --release -p astronomical-inference-worker --test model_artifact_qualification_tests --features model-artifact-qualification should_keep_the_deployed_rest_surface_healthy_across_model_artifact_prompt_reuse -- --ignored --nocapture
            ;;
        qualify-laguna-family-model-swap)
            lane_name="laguna-family-model-swap"
            set -- cargo test --release -p astronomical-inference-worker --test model_artifact_qualification_tests --features model-artifact-qualification should_swap_qwen_then_laguna_xs_then_qwen_on_one_worker -- --ignored --nocapture
            ;;
        qualify-laguna-attribution)
            lane_name="laguna-attribution"
            set -- cargo test --release -p astronomical-inference-worker --test model_artifact_qualification_tests --features model-artifact-qualification should_serve_a_laguna_xs_long_prompt_with_switchable_attribution -- --ignored --nocapture
            ;;
        qualify-qwen3-5-thinking-seed-rest)
            lane_name="qwen3-5-thinking-seed-rest"
            set -- cargo test --release -p astronomical-inference-worker --test model_artifact_qualification_tests --features model-artifact-qualification should_seed_the_first_reasoning_output_across_both_streaming_rest_apis_with_a_real_qwen3_5_model -- --ignored --nocapture
            ;;
        qualify-smallest-qwen3-5-hard-thinking-budget-rest)
            lane_name="smallest-qwen3-5-hard-thinking-budget-rest"
            set -- cargo test --release -p astronomical-inference-worker --test model_artifact_qualification_tests --features model-artifact-qualification should_use_the_smallest_configured_qwen3_5_model_to_commit_the_complete_hard_thinking_budget_transition_before_streaming_visible_answer_content_through_the_openai_chat_completions_rest_api -- --ignored --nocapture
            ;;
        qualify-speculative-prefill-rest)
            lane_name="speculative-prefill-rest"
            set -- cargo test --release -p astronomical-inference-worker --test model_artifact_qualification_tests --features model-artifact-qualification should_complete_the_cold_tool_journey_through_real_config_worker_and_rest_boundaries -- --ignored --nocapture
            ;;
        qualify-persistent-prompt-cache)
            lane_name="persistent-prompt-cache-qualification"
            ignored_suite_name="persistent-prompt-cache"
            ;;
        test-model-ssd-streaming-support)
            lane_name="model-ssd-streaming-support"
            set -- cargo test --no-fail-fast -p astronomical-inference-worker --test performance_measurement_tests --test model_artifact_qualification_tests --test memory_management_acceptance_tests --features performance-measurement,model-artifact-qualification,memory-management-acceptance model_ssd_streaming:: -- --nocapture --test-threads 1
            ;;
        test-model-ssd-streaming-attribution-support)
            lane_name="model-ssd-streaming-attribution-support"
            set -- cargo test --no-fail-fast -p astronomical-model-serving --test model_artifact_qualification_tests --features direct-mlx model_ssd_streaming:: -- --nocapture --test-threads 1
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
        measure-model-ssd-streaming-attribution)
            lane_name="model-ssd-streaming-attribution"
            set -- cargo test --release -p astronomical-model-serving --test model_artifact_qualification_tests --features direct-mlx should_measure_model_ssd_streaming_attribution_across_automatic_cold_and_warm_runs -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-opencode-long-context-reuse)
            lane_name="model-ssd-streaming-opencode-long-context-reuse"
            set -- cargo test --release -p astronomical-inference-worker --test model_artifact_qualification_tests --features model-artifact-qualification should_keep_the_worker_available_and_reuse_experts_across_repeated_opencode_long_context_requests -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-complete-expert-residency)
            lane_name="model-ssd-streaming-complete-expert-residency"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_keep_all_experts_resident_and_avoid_ssd_reads_when_the_model_fits_memory -- --ignored --nocapture
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
        measure-model-ssd-streaming-near-disk-oq6e-cached-suffix-prefill)
            lane_name="model-ssd-streaming-near-disk-oq6e-cached-suffix-prefill"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_keep_near_disk_oq6e_cached_suffix_prefill_responsive -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-oq6e-tight-ceiling-prefill)
            lane_name="model-ssd-streaming-oq6e-tight-ceiling-prefill"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_recover_from_prefill_oom_without_stalling_on_oq6e -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-oq6e-long-conversation)
            lane_name="model-ssd-streaming-oq6e-long-conversation"
            export TEST_TIMEOUT_SECONDS=1200
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_measure_oq6e_long_conversation_at_half_disk_ram -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-prefill-memory-progress)
            lane_name="model-ssd-streaming-prefill-memory-progress"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_report_changing_bounded_mlx_memory_during_prefill -- --ignored --nocapture
            ;;
        measure-model-ssd-streaming-laguna-paging)
            lane_name="model-ssd-streaming-laguna-paging"
            set -- cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance model_ssd_streaming::laguna_paging_journey:: -- --ignored --nocapture --test-threads 1
            ;;
        measure-experimental-aligned-expert-packs-ornith-generation)
            lane_name="aligned-expert-generation"
            set -- cargo test --release -p astronomical-experimental-aligned-expert-packs --test model_artifact_qualification_tests should_measure_one_layer_generation_expert_data_plane -- --ignored --nocapture --test-threads 1
            ;;
        measure-experimental-aligned-expert-packs-ornith-prompt-processing)
            lane_name="aligned-expert-prompt-processing"
            set -- cargo test --release -p astronomical-experimental-aligned-expert-packs --test model_artifact_qualification_tests should_measure_one_layer_prompt_processing_expert_data_plane -- --ignored --nocapture --test-threads 1
            ;;
        measure-experimental-aligned-expert-packs-oq6e-data-plane)
            lane_name="aligned-expert-oq6e"
            set -- cargo test --release -p astronomical-experimental-aligned-expert-packs --test model_artifact_qualification_tests should_measure_oq6e_expert_data_plane_in_both_orders -- --ignored --nocapture --test-threads 1
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
            "${repository_root}/scripts/run-ignored-qualification-suite.sh" "$ignored_suite_name"
    fi
    exec "${repository_root}/scripts/run-in-disposable-cargo-target.sh" \
        --lane "$lane_name" -- \
        "${repository_root}/scripts/run-bounded-cargo-test.sh" "$@"
}

main "$@"
