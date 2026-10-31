//! The accepted-debt registry for the function-length budget (issue #1637).
//!
//! Two lists live here, and both are data on purpose: they cannot be created
//! by editing the function they cover, and every entry must cite the decision
//! that accepted the debt. The production exemptions name the Family D
//! declarative tables the campaign accepted as permanent; the test-scope
//! freeze names every over-budget test function with the code-line count
//! captured when the list was written, forming a ratchet that fails when a
//! site grows or a new site appears.
//!
//! The lookup functions are pure data access so the gate mechanics (lane
//! running, classification, reporting) stay in the gate module.

/// One accepted-debt site: a production function whose code length exceeds
/// the budget by explicit decision. Keyed by source path plus function name
/// (not line number, which drifts with every edit above the site).
pub(crate) struct ProductionScopeExemption {
    pub(crate) path: &'static str,
    pub(crate) function_name: &'static str,
}

/// The complete set of production functions allowed over the 100-code-line
/// budget. The campaign tracked by issue #1637 classified all remaining
/// over-budget production functions as Family D — declarative tables where
/// serialized data is spelled as code; splitting them would obscure the data
/// they encode. The identifier tables enumerate one arm per catalog entry and
/// the tensor-profile builders enumerate one statement per physical tensor.
/// Adding an entry requires an accepting decision on the tracking issue;
/// anything else over budget in production scope fails the gate.
pub(crate) const PRODUCTION_SCOPE_EXEMPTIONS: &[ProductionScopeExemption] = &[
    ProductionScopeExemption {
        path: "crates/model-serving/src/performance_attribution/catalog.rs",
        function_name: "identifier",
    },
    ProductionScopeExemption {
        path: "crates/model-serving/src/performance_attribution/counter_catalog.rs",
        function_name: "identifier",
    },
    ProductionScopeExemption {
        path: "crates/model-serving/src/qwen_image_21/tensor_profiles/text_encoder.rs",
        function_name: "text_encoder_tensor_profiles",
    },
    ProductionScopeExemption {
        path: "crates/model-serving/src/qwen_image_21/tensor_profiles/vae.rs",
        function_name: "vae_tensor_profiles",
    },
];

/// Matches a finding site against the exemption list. Path plus name must
/// both match: two `identifier` tables exist, and the path is what tells
/// them apart. An unresolved name ("unknown") matches nothing, so a moved or
/// mangled source file fails the gate instead of silently keeping its
/// exemption.
pub(crate) fn is_exempted_production_site(source_path: &str, function_name: &str) -> bool {
    for exemption in PRODUCTION_SCOPE_EXEMPTIONS {
        if exemption.path == source_path && exemption.function_name == function_name {
            return true;
        }
    }
    false
}

/// One frozen test-scope site: a test function allowed over the 100-code-line
/// budget until it is paid down. Keyed by source path plus function name, with
/// the code-line count frozen at the moment the list was written. The freeze is
/// a ratchet, not an exemption: a known site must not grow past
/// `frozen_code_lines`, and any test-scope site not listed here is new debt and
/// fails the gate. Paying a site down removes its entry; a site that shrinks
/// below the budget stops reporting and its entry becomes inert.
///
/// The initial list captured the 83 over-budget test-scope sites inventoried on
/// issue #1637 after the first paydown batch; the first monolith batch paid
/// three of them down (the prefill gathered-dequant measurement, the
/// throughput journey driver, and the vision padding probe) and a second
/// batch paid down the nine assertion-table sites, so 71 remain frozen. Adding an entry requires an accepting decision on the tracking
/// issue, for the same reason a production exemption does.
pub(crate) struct TestScopeFreezeEntry {
    pub(crate) path: &'static str,
    pub(crate) function_name: &'static str,
    pub(crate) frozen_code_lines: u64,
}

/// The complete frozen set of test-scope functions allowed over the budget.
/// Each count is frozen at the global maximum across every feature world, so a
/// lane subset (for example the pre-commit hook's default and MLX worlds) always
/// reports at or below a site's frozen count and never reads a frozen site as
/// grown.
pub(crate) const TEST_SCOPE_FREEZE: &[TestScopeFreezeEntry] = &[
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/expert_paging/journeys/ssd_paging_decode_expert_reuse_rest_journey/support.rs",
        function_name: "assert_decode_expert_reuse_attribution",
        frozen_code_lines: 128,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/../chat_serving/installed_journeys/expert_paging/paged_prefill_chunk_attribution.rs",
        function_name: "attribute_paged_prefill_chunks",
        frozen_code_lines: 182,
    },
    TestScopeFreezeEntry {
        path: "apps/daemon/tests/fixtures/scripted_worker.rs",
        function_name: "dispatch_unstructured_chat",
        frozen_code_lines: 298,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/installed_model_contracts/resident_forward_probe.rs",
        function_name: "drive_raw_resident_model_forwards",
        frozen_code_lines: 126,
    },
    TestScopeFreezeEntry {
        path: "apps/daemon/tests/fixtures/scripted_worker_image.rs",
        function_name: "handle_image_command",
        frozen_code_lines: 127,
    },
    TestScopeFreezeEntry {
        path: "xtask/tests/app_builder_support/sandbox.rs",
        function_name: "install",
        frozen_code_lines: 103,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5/dflash_speculative_proof.rs",
        function_name: "load_proof_journey",
        frozen_code_lines: 133,
    },
    TestScopeFreezeEntry {
        path: "apps/daemon/tests/fixtures/replacement_ready_worker.rs",
        function_name: "main",
        frozen_code_lines: 130,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5/convolution_prefill_performance.rs",
        function_name: "measure_convolution_chain_costs",
        frozen_code_lines: 345,
    },
    TestScopeFreezeEntry {
        path: "crates/astronomical-jangtq2-qwen35/tests/direct_mlx/decode_throughput.rs",
        function_name: "measure_decode_throughput",
        frozen_code_lines: 111,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5/dflash_speculative_proof.rs",
        function_name: "measure_dflash_speculative_acceptance_and_decode_throughput",
        frozen_code_lines: 105,
    },
    TestScopeFreezeEntry {
        path: "crates/astronomical-jangtq2-qwen35/tests/direct_mlx/decode_throughput.rs",
        function_name: "measure_fused_gathered_projection_throughput",
        frozen_code_lines: 124,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/prompt_cache/engine/cache_miss_memory.rs",
        function_name: "measure_initial_request_expert_reclamation",
        frozen_code_lines: 132,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/performance/measurement/model_persistent_prompt_cache_warmup_worker.rs",
        function_name: "measure_worker_summarization",
        frozen_code_lines: 123,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/performance/paged_prefill/cold_prefill_measurements.rs",
        function_name: "measure_worker_summarization",
        frozen_code_lines: 131,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5/elementwise_fusion_probe.rs",
        function_name: "probe_elementwise_chains",
        frozen_code_lines: 195,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5/prompt_cache_restore_reconstruction_probe.rs",
        function_name: "probe_restore_reconstruction",
        frozen_code_lines: 107,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5/vision_rope_fusion_probe.rs",
        function_name: "probe_vision_rope_fusion",
        frozen_code_lines: 145,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/performance/throughput/completion.rs",
        function_name: "read_decode_cache_diagnostic_evidence",
        frozen_code_lines: 134,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/performance/throughput/decode_cache_experiment.rs",
        function_name: "report_attributed_policy_comparison",
        frozen_code_lines: 116,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/prompt_cache/append_only_rest.rs",
        function_name: "run_append_only_rest_journey",
        frozen_code_lines: 115,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/../chat_serving/installed_journeys/support/performance_attribution/generation.rs",
        function_name: "run_attributed_generation",
        frozen_code_lines: 107,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/prompt_cache/engine/cache_restore_peak.rs",
        function_name: "run_cache_restore_peak_acceptance",
        frozen_code_lines: 172,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/prompt_cache/persistent_prompt_cache_stats_e2e/scenario.rs",
        function_name: "run_cache_stats_e2e",
        frozen_code_lines: 238,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/performance/paged_prefill/cold_prefill_measurements.rs",
        function_name: "run_cold_prefill_measurement",
        frozen_code_lines: 117,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/chat_serving/chat/embeddings_rest.rs",
        function_name: "run_embeddings_gpu_journey",
        frozen_code_lines: 158,
    },
    TestScopeFreezeEntry {
        path: "apps/daemon/tests/fixtures/idle_worker.rs",
        function_name: "run_fixture",
        frozen_code_lines: 261,
    },
    TestScopeFreezeEntry {
        path: "apps/daemon/tests/fixtures/scripted_worker.rs",
        function_name: "run_fixture",
        frozen_code_lines: 203,
    },
    TestScopeFreezeEntry {
        path: "apps/daemon/tests/fixtures/memory_demotion_worker.rs",
        function_name: "run_fixture_async",
        frozen_code_lines: 143,
    },
    TestScopeFreezeEntry {
        path: "verification/e2e/tests/model_journeys/decode_full_fit_status_journey.rs",
        function_name: "run_full_fit_status_journey",
        frozen_code_lines: 120,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/expert_paging/journeys/cached_suffix_streaming_prefill_rest_journey.rs",
        function_name: "run_interaction_journey",
        frozen_code_lines: 168,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/expert_paging/journeys/live_memory_ceiling_expert_residency_round_trip_rest_journey.rs",
        function_name: "run_mid_streaming_raise_settlement",
        frozen_code_lines: 161,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/memory_contracts/model_swap_allocator.rs",
        function_name: "run_model_swap_allocator_contract",
        frozen_code_lines: 133,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/performance/paged_prefill/prefill_chunk_sweep.rs",
        function_name: "run_model_with_prefill_chunk_tokens",
        frozen_code_lines: 191,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/memory_contracts/mlx_c_memory.rs",
        function_name: "run_native_memory_transitions_contract",
        frozen_code_lines: 122,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/expert_paging/journeys/large_sparse_moe_streaming_paging_memory_shape_rest_journey.rs",
        function_name: "run_paging_memory_shape_experiment",
        frozen_code_lines: 208,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/performance/throughput/decode_cache_experiment.rs",
        function_name: "run_paired_decode_cache_trials",
        frozen_code_lines: 109,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/prompt_cache/restore_rest.rs",
        function_name: "run_persistent_prompt_cache_memory_rest_journey",
        frozen_code_lines: 159,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/performance/measurement/model_persistent_prompt_cache_warmup.rs",
        function_name: "run_persistent_prompt_cache_warmup_e2e",
        frozen_code_lines: 114,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/chat_serving/ornith_35b/prefill_attribution.rs",
        function_name: "run_prefill_attribution_journey",
        frozen_code_lines: 199,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/chat_serving/qwen3_5/queued_live_ceiling_demotion_rest.rs",
        function_name: "run_queued_demotion_journey",
        frozen_code_lines: 121,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/chat_serving/chat/modernbert_reference_similarity.rs",
        function_name: "run_reference_similarity_journey",
        frozen_code_lines: 118,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/expert_paging/journeys/live_memory_ceiling_expert_residency_round_trip_rest_journey.rs",
        function_name: "run_residency_round_trip",
        frozen_code_lines: 123,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/performance/measurement/model_load_to_prefill.rs",
        function_name: "run_single_measured_request",
        frozen_code_lines: 103,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/expert_paging/journeys/ssd_paging_decode_expert_reuse_rest_journey.rs",
        function_name: "run_ssd_paging_decode_expert_reuse_journey",
        frozen_code_lines: 246,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/chat_serving/chat/thinking_controls_rest.rs",
        function_name: "run_thinking_controls_gpu_journey",
        frozen_code_lines: 118,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/performance/throughput/qwen3_5_moe.rs",
        function_name: "run_two_turn_streaming_journey",
        frozen_code_lines: 112,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5_moe/expert_paging/resident_owner_page_parity.rs",
        function_name: "should_build_identical_resident_layers_from_shard_and_bounded_reads",
        frozen_code_lines: 131,
    },
    TestScopeFreezeEntry {
        path: "crates/model-serving/src/engine_backed_worker/tests/chat/prefill_progress.rs",
        function_name: "should_exclude_the_complete_restored_prompt_prefix_from_progress_without_inflating_cached_usage",
        frozen_code_lines: 186,
    },
    TestScopeFreezeEntry {
        path: "apps/daemon/tests/rest_api/config_reload/configuration_status.rs",
        function_name: "should_expose_configured_and_worker_effective_generation_with_path_free_model_summary",
        frozen_code_lines: 105,
    },
    TestScopeFreezeEntry {
        path: "crates/model-serving/src/qwen3_5_core/artifacts/tensor_spec/tests.rs",
        function_name: "should_generate_the_complete_mixed_precision_optiq_language_tensor_profile",
        frozen_code_lines: 150,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/sparse_expert_projection.rs",
        function_name: "should_match_dense_operations_reference_across_generic_assignment_rows_and_dtypes",
        frozen_code_lines: 140,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5/gated_delta.rs",
        function_name: "should_match_ops_loop_when_dispatching_one_complete_prefill_sequence",
        frozen_code_lines: 116,
    },
    TestScopeFreezeEntry {
        path: "crates/astronomical-memory-policy/src/memory_equation_tests.rs",
        function_name: "should_match_request_and_resident_guard_verdicts_when_the_analytic_floor_is_the_same_workspace",
        frozen_code_lines: 123,
    },
    TestScopeFreezeEntry {
        path: "apps/daemon/tests/integration/download_payload_concurrency.rs",
        function_name: "should_pause_across_every_in_flight_file_and_resume_without_progress_mismatch",
        frozen_code_lines: 103,
    },
    TestScopeFreezeEntry {
        path: "apps/daemon/tests/rest_api/library_catalog.rs",
        function_name: "should_publish_every_shipped_catalog_entry_to_the_library_catalog_endpoint",
        frozen_code_lines: 114,
    },
    TestScopeFreezeEntry {
        path: "crates/model-serving/src/engine_backed_worker/tests/chat/ready_and_model_lifecycle.rs",
        function_name: "should_publish_the_unused_ceiling_split_on_model_loaded_and_limit_adjusted_samples",
        frozen_code_lines: 113,
    },
    TestScopeFreezeEntry {
        path: "crates/ipc-protocol/src/image_generation_tests/protocol_contracts_tests.rs",
        function_name: "should_reject_invalid_image_generation_commands_and_settings",
        frozen_code_lines: 109,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5_moe/cached_plus_streamed_page_route.rs",
        function_name: "should_restore_single_page_assignment_order_across_partial_routes",
        frozen_code_lines: 135,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/persistent_prompt_cache_direct_writer.rs",
        function_name: "should_retry_the_same_captured_arrays_after_releasing_artificial_active_pressure",
        frozen_code_lines: 144,
    },
    TestScopeFreezeEntry {
        path: "crates/ipc-protocol/src/daemon_protocol_tests.rs",
        function_name: "should_round_trip_daemon_responses_through_the_wire_codec",
        frozen_code_lines: 114,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/persistent_prompt_cache_disk_store.rs",
        function_name: "should_save_and_load_kv_block_and_recurrent_snapshot_as_separate_files",
        frozen_code_lines: 128,
    },
    TestScopeFreezeEntry {
        path: "crates/model-serving/src/performance_attribution/performance_attribution_tests/flux2_image_tests.rs",
        function_name: "should_serialize_every_image_operation_with_graph_and_synchronization_separate",
        frozen_code_lines: 164,
    },
    TestScopeFreezeEntry {
        path: "crates/ipc-protocol/tests/integration/daemon_transport.rs",
        function_name: "should_stream_multiple_responses_for_one_request",
        frozen_code_lines: 103,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/persistent_prompt_cache_partial_tail_disk_store.rs",
        function_name: "should_supersede_only_strictly_shorter_tail_siblings",
        frozen_code_lines: 123,
    },
    TestScopeFreezeEntry {
        path: "apps/astronomical-cli/tests/integration/support/stub_daemon.rs",
        function_name: "spawn_stub_daemon",
        frozen_code_lines: 243,
    },
    TestScopeFreezeEntry {
        path: "xtask/tests/workflow_verdict_contract.rs",
        function_name: "verdict_script_decides_every_scenario_correctly",
        frozen_code_lines: 157,
    },
    TestScopeFreezeEntry {
        path: "verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5_moe/cached_plus_streamed_page_route.rs",
        function_name: "verify_compact_cached_plus_streamed_page_route_parity",
        frozen_code_lines: 126,
    },
    TestScopeFreezeEntry {
        path: "apps/inference-worker/tests/integration/flux2_klein_fixture.rs",
        function_name: "write_executable_artifact",
        frozen_code_lines: 111,
    },
    TestScopeFreezeEntry {
        path: "crates/config/tests/integration/model_discovery/flux2_klein.rs",
        function_name: "write_executable_flux2_klein_artifact",
        frozen_code_lines: 117,
    },
    TestScopeFreezeEntry {
        path: "xtask/tests/app_builder_support/sandbox.rs",
        function_name: "write_tool_shims",
        frozen_code_lines: 134,
    },
];

/// Looks up a frozen code-line count for a test-scope site by path plus
/// function name. An unresolved name ("unknown") matches nothing, so a moved or
/// mangled test file is treated as a new site and fails the gate instead of
/// silently riding a stale entry.
pub(crate) fn frozen_test_scope_code_lines(source_path: &str, function_name: &str) -> Option<u64> {
    for entry in TEST_SCOPE_FREEZE {
        if entry.path == source_path && entry.function_name == function_name {
            return Some(entry.frozen_code_lines);
        }
    }
    None
}
