use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};

use super::speculative_prefill::{
    SPECULATIVE_PREFILL_KEEP_PERCENTAGE, run_representative_generation,
};
use super::speculative_prefill_tool_control::{
    REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT, assert_schema_valid_literary_analysis_tool_call,
    literary_analysis_tools, parse_one_tool_call, prepare_representative_tool_prompt,
};

#[tokio::test]
#[ignore = "qualifies sampled target-only and protected SpecPrefill tool correctness with fixed seed 17"]
async fn should_preserve_a_schema_valid_tool_call_with_fixed_sampled_seed_17() {
    qualify_fixed_sampled_seed(17, 95_310).await;
}

#[tokio::test]
#[ignore = "qualifies sampled target-only and protected SpecPrefill tool correctness with fixed seed 50"]
async fn should_preserve_a_schema_valid_tool_call_with_fixed_sampled_seed_50() {
    qualify_fixed_sampled_seed(50, 95_312).await;
}

async fn qualify_fixed_sampled_seed(sampling_seed: u64, request_identifier_base: u64) {
    tokio::time::timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_ornith_model_artifact_directory();
        let (draft_model_directory, draft_model_id) =
            super::configured_speculative_prefill_draft_model_artifact(&target_model_directory);
        let validated_target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT as u32)
            .expect("the target artifact should validate for sampled tool qualification");
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
            .expect("the target tokenizer should load for sampled tool qualification");
        let declared_tools = literary_analysis_tools();
        let representative_tool_prompt = prepare_representative_tool_prompt(
            &tokenizer,
            validated_target_artifact.model_id(),
            &declared_tools,
            Some(sampling_seed),
        );
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;

        eprintln!(
            "[sampled-speculative-prefill-tool-control] status=progress seed={sampling_seed} phase=target_only"
        );
        let target_only_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &representative_tool_prompt,
            false,
            REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(request_identifier_base),
            None,
            mlx_memory_limits,
        )
        .await;
        eprintln!(
            "[sampled-speculative-prefill-tool-control] status=progress seed={sampling_seed} phase=protected_speculative_prefill"
        );
        let speculative_prefill_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &representative_tool_prompt,
            true,
            REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(request_identifier_base + 1),
            None,
            mlx_memory_limits,
        )
        .await;
        let target_only_tool_call = parse_one_tool_call(
            &tokenizer,
            &declared_tools,
            &target_only_measurement.generated_token_ids,
        );
        let speculative_prefill_tool_call = parse_one_tool_call(
            &tokenizer,
            &declared_tools,
            &speculative_prefill_measurement.generated_token_ids,
        );

        assert_schema_valid_literary_analysis_tool_call(&target_only_tool_call);
        assert_schema_valid_literary_analysis_tool_call(&speculative_prefill_tool_call);
        assert_eq!(
            speculative_prefill_tool_call.function_name,
            target_only_tool_call.function_name,
        );
        eprintln!(
            "[sampled-speculative-prefill-tool-control] status=success seed={sampling_seed} malformed_call_count=0"
        );
    })
    .await
    .expect("one fixed-seed sampled tool qualification should finish within 115 seconds");
}
