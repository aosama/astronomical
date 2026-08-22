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
        accept-memory-management \
        qualify-model-artifacts \
        qualify-deployed-model-rest-liveness \
        qualify-persistent-prompt-cache \
        test-performance-measurement-support \
        measure-model-artifact-summary \
        measure-persistent-prompt-cache-warmup \
        measure-persistent-prompt-cache-warmup-50k \
        measure-persistent-prompt-cache-warmup-100k \
        measure-model-artifact-cold-prefill-50k \
        measure-model-artifact-prefill-1024 \
        measure-model-artifact-prefill-2048 \
        measure-model-artifact-prefill-4096 \
        measure-model-artifact-prefill-8192 \
        measure-model-artifact-performance-attribution \
        measure-experimental-aligned-expert-packs-ornith-generation \
        measure-experimental-aligned-expert-packs-ornith-prompt-processing \
        measure-experimental-aligned-expert-packs-oq4e-data-plane
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
    case "$journey_name" in
        accept-memory-management)
            lane_name="memory-management-acceptance"
            set -- cargo test -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance -- --ignored --nocapture --test-threads 1
            ;;
        qualify-model-artifacts)
            lane_name="model-artifact-qualification"
            set -- cargo test --no-fail-fast -p astronomical-model-serving -p astronomical-inference-worker --test model_artifact_qualification_tests --features astronomical-model-serving/direct-mlx,astronomical-inference-worker/model-artifact-qualification -- --ignored --nocapture --test-threads 1
            ;;
        qualify-deployed-model-rest-liveness)
            lane_name="deployed-rest-liveness"
            set -- cargo test --release -p astronomical-inference-worker --test model_artifact_qualification_tests --features model-artifact-qualification should_keep_the_deployed_rest_surface_healthy_across_model_artifact_prompt_reuse -- --ignored --nocapture
            ;;
        qualify-persistent-prompt-cache)
            lane_name="persistent-prompt-cache-qualification"
            set -- cargo test --no-fail-fast -p astronomical-model-serving -p astronomical-inference-worker --test persistent_prompt_cache_qualification_tests --features astronomical-model-serving/direct-mlx,astronomical-inference-worker/model-artifact-qualification -- --ignored --skip persistent_prompt_cache_qualification::cache_interaction_matrix::should_qualify_selected_pinned_ornith_cache_interaction_matrix_cell --nocapture --test-threads 1
            ;;
        test-performance-measurement-support)
            lane_name="performance-measurement-support"
            set -- cargo test --no-fail-fast -p astronomical-inference-worker --test performance_measurement_tests --features performance-measurement -- --nocapture --test-threads 1
            ;;
        measure-model-artifact-summary)
            lane_name="model-artifact-summary"
            set -- cargo test --release -p astronomical-inference-worker --test performance_measurement_tests --features performance-measurement should_measure_model_artifact_summarization_throughput_and_peak_memory -- --ignored --nocapture
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
        measure-model-artifact-cold-prefill-50k)
            lane_name="model-artifact-cold-prefill-50k"
            set -- cargo test --release -p astronomical-inference-worker --features performance-measurement --test performance_measurement_tests performance_measurement::model_artifact_metrics_measurement::should_measure_model_artifact_cold_prefill_at_fifty_thousand_words -- --ignored --nocapture --exact
            ;;
        measure-model-artifact-prefill-1024|measure-model-artifact-prefill-2048|measure-model-artifact-prefill-4096|measure-model-artifact-prefill-8192)
            lane_name="$journey_name"
            prefill_chunk_tokens="${journey_name##*-}"
            set -- cargo test -p astronomical-inference-worker --test performance_measurement_tests --features performance-measurement "should_measure_model_throughput_with_prefill_chunck_tokens_${prefill_chunk_tokens}" -- --ignored --nocapture
            ;;
        measure-model-artifact-performance-attribution)
            lane_name="model-artifact-attribution"
            set -- cargo test --release -p astronomical-model-serving --test model_artifact_qualification_tests --features direct-mlx should_measure_qwen3_6_35b_a3b_optiq_4bit_attribution_across_automatic_cold_and_warm_runs -- --ignored --nocapture
            ;;
        measure-experimental-aligned-expert-packs-ornith-generation)
            lane_name="aligned-expert-generation"
            set -- cargo test --release -p astronomical-experimental-aligned-expert-packs --test model_artifact_qualification_tests should_measure_one_layer_generation_expert_data_plane -- --ignored --nocapture --test-threads 1
            ;;
        measure-experimental-aligned-expert-packs-ornith-prompt-processing)
            lane_name="aligned-expert-prompt-processing"
            set -- cargo test --release -p astronomical-experimental-aligned-expert-packs --test model_artifact_qualification_tests should_measure_one_layer_prompt_processing_expert_data_plane -- --ignored --nocapture --test-threads 1
            ;;
        measure-experimental-aligned-expert-packs-oq4e-data-plane)
            lane_name="aligned-expert-oq4e"
            set -- cargo test --release -p astronomical-experimental-aligned-expert-packs --test model_artifact_qualification_tests should_measure_oq4e_expert_data_plane_in_both_orders -- --ignored --nocapture --test-threads 1
            ;;
        *)
            printf '%s\n' "Error: unknown disposable Cargo journey: ${journey_name}" >&2
            print_usage >&2
            exit 2
            ;;
    esac

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    exec "${repository_root}/scripts/run-in-disposable-cargo-target.sh" \
        --lane "$lane_name" -- \
        "${repository_root}/scripts/run-bounded-cargo-test.sh" "$@"
}

main "$@"
