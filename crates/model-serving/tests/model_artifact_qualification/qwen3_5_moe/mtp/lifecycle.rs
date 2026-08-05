use std::path::Path;
use std::time::Duration;

use astronomical_ipc_protocol::{MtpRuntimeState, RequestId};
use astronomical_model_serving::{
    GeneratedToken, GenerationFinalization, InferenceEngine, PerformanceAttribution, Qwen3_5Engine,
    Qwen3_5InferenceRequest,
};

use super::IMAGE_PAD_TOKEN_ID;
use super::engine_support::{
    generation_report_for_request, load_mtp_test_engine, performance_counter_amount,
};

const INJECTED_FEEDBACK_TOKEN_IDS: [u32; 3] = [248_045, 846, 198];

#[tokio::test]
#[ignore = "loads the local MTP artifact twice and qualifies injection plus terminal lifecycle boundaries"]
async fn should_preserve_active_mtp_across_injection_and_request_lifecycle_boundaries() {
    tokio::time::timeout(Duration::from_secs(115), run_mtp_lifecycle_qualification())
        .await
        .expect("the MTP lifecycle qualification should finish within 115 seconds");
}

async fn run_mtp_lifecycle_qualification() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::super::qwen3_6_35b_a3b_oq4e_mtp_model_directory();

    eprintln!("[oq4e-mtp-lifecycle] status=start phase=target_only_injection_control");
    let target_only_injected_token_ids = run_target_only_injection_control(&model_directory).await;

    eprintln!("[oq4e-mtp-lifecycle] status=progress phase=active_engine_load");
    let (mut mtp_engine, _temporary_log_directory, performance_attribution_log_path) =
        load_mtp_test_engine(&model_directory, true, false).await;
    let engine_load_result = mtp_engine
        .load()
        .await
        .expect("the MTP lifecycle engine should load");
    assert_eq!(
        engine_load_result.mtp_runtime_state(),
        MtpRuntimeState::Active
    );

    let injection_request_id = RequestId::new(37_001);
    let active_mtp_injected_token_ids =
        run_injection_sequence(&mut mtp_engine, injection_request_id, true).await;
    assert_eq!(
        active_mtp_injected_token_ids, target_only_injected_token_ids,
        "MTP reset and reseed after injected input must preserve target-only continuation"
    );
    let injection_report =
        generation_report_for_request(&performance_attribution_log_path, injection_request_id);
    assert!(
        performance_counter_amount(&injection_report, "mtp_admitted_attempt_count") >= 2,
        "MTP must resume after replacing its pre-injection history"
    );

    eprintln!("[oq4e-mtp-lifecycle] status=progress phase=maximum_output_current_token");
    let current_token_terminal =
        run_request_to_terminal(&mut mtp_engine, RequestId::new(37_002), 1).await;
    assert_eq!(current_token_terminal.generated_token_ids.len(), 1);
    assert!(
        current_token_terminal
            .generation_finalization
            .has_reportable_state()
    );

    eprintln!("[oq4e-mtp-lifecycle] status=progress phase=maximum_output_queued_draft");
    let queued_draft_terminal =
        run_request_to_terminal(&mut mtp_engine, RequestId::new(37_003), 2).await;
    assert_eq!(queued_draft_terminal.generated_token_ids.len(), 2);
    assert!(
        queued_draft_terminal
            .generation_finalization
            .has_reportable_state()
    );

    eprintln!("[oq4e-mtp-lifecycle] status=progress phase=cancellation_with_mtp_state");
    let cancellation_request_id = RequestId::new(37_004);
    start_greedy_request(&mut mtp_engine, cancellation_request_id, 8, false).await;
    let (_first_generated_token_id, first_generation_finalization) =
        decode_next_generated_token(&mut mtp_engine, cancellation_request_id).await;
    assert!(first_generation_finalization.is_none());
    let cancellation_finalization = mtp_engine
        .cancel_generation(cancellation_request_id)
        .await
        .expect("active MTP cancellation should release the request");
    assert!(cancellation_finalization.has_reportable_state());

    eprintln!("[oq4e-mtp-lifecycle] status=progress phase=eos_and_engine_reuse");
    let eos_terminal = run_request_to_terminal(&mut mtp_engine, RequestId::new(37_005), 64).await;
    let final_eos_token_id = eos_terminal
        .generated_token_ids
        .last()
        .copied()
        .expect("the EOS request should emit at least one token");
    assert!(
        [248_044, 248_046].contains(&final_eos_token_id),
        "the request should terminate on a certified Qwen EOS token"
    );
    assert!(eos_terminal.generation_finalization.has_reportable_state());

    let reused_engine_terminal =
        run_request_to_terminal(&mut mtp_engine, RequestId::new(37_006), 1).await;
    assert_eq!(reused_engine_terminal.generated_token_ids.len(), 1);
    assert!(
        reused_engine_terminal
            .generation_finalization
            .has_reportable_state()
    );
    eprintln!("[oq4e-mtp-lifecycle] status=success");
}

async fn run_target_only_injection_control(model_directory: &Path) -> Vec<u32> {
    let (mut target_only_engine, _temporary_log_directory, _performance_attribution_log_path) =
        load_mtp_test_engine(model_directory, false, false).await;
    let engine_load_result = target_only_engine
        .load()
        .await
        .expect("the target-only injection control engine should load");
    assert_eq!(
        engine_load_result.mtp_runtime_state(),
        MtpRuntimeState::Disabled
    );
    run_injection_sequence(&mut target_only_engine, RequestId::new(37_000), false).await
}

async fn run_injection_sequence(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    performance_attribution_enabled: bool,
) -> Vec<u32> {
    start_greedy_request(engine, request_id, 6, performance_attribution_enabled).await;
    let (first_generated_token_id, first_generation_finalization) =
        decode_next_generated_token(engine, request_id).await;
    assert!(first_generation_finalization.is_none());
    engine
        .inject_input_tokens(request_id, INJECTED_FEEDBACK_TOKEN_IDS.to_vec())
        .await
        .expect("same-request feedback injection should succeed");

    let mut generated_token_ids = vec![first_generated_token_id];
    while generated_token_ids.len() < 6 {
        let (generated_token_id, generation_finalization) =
            decode_next_generated_token(engine, request_id).await;
        generated_token_ids.push(generated_token_id);
        if generation_finalization.is_some() {
            break;
        }
    }
    generated_token_ids
}

async fn run_request_to_terminal(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    maximum_output_tokens: u16,
) -> TerminalGeneration {
    start_greedy_request(engine, request_id, maximum_output_tokens, false).await;
    let mut generated_token_ids = Vec::with_capacity(maximum_output_tokens as usize);
    loop {
        let (generated_token_id, generation_finalization) =
            decode_next_generated_token(engine, request_id).await;
        generated_token_ids.push(generated_token_id);
        if let Some(generation_finalization) = generation_finalization {
            return TerminalGeneration {
                generated_token_ids,
                generation_finalization,
            };
        }
    }
}

async fn start_greedy_request(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    maximum_output_tokens: u16,
    performance_attribution_enabled: bool,
) {
    let inference_request = Qwen3_5InferenceRequest::new(
        request_id,
        super::super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS.to_vec(),
        maximum_output_tokens,
    )
    .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID);
    let inference_request = if performance_attribution_enabled {
        inference_request.with_performance_attribution(PerformanceAttribution::enabled())
    } else {
        inference_request
    };
    engine
        .start_generation(inference_request)
        .await
        .expect("the reused engine should accept the greedy request");
}

async fn decode_next_generated_token(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
) -> (u32, Option<GenerationFinalization>) {
    loop {
        match engine
            .decode_next_token(request_id)
            .await
            .expect("the MTP lifecycle request should advance")
        {
            GeneratedToken::TokenId {
                token_id,
                generation_finalization,
                ..
            } => return (token_id, generation_finalization),
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::EndOfSequence => {
                panic!("the Qwen engine should attach terminal state to an emitted token")
            }
        }
    }
}

struct TerminalGeneration {
    generated_token_ids: Vec<u32>,
    generation_finalization: GenerationFinalization,
}
