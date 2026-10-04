use std::time::{Duration, Instant};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5Model, Qwen3_5Tokenizer, RequestDecoderStateStack,
};
use astronomical_runtime_integration::MlxRuntime;
use tokio::time::timeout;

use crate::serving_acceptance::support::dense_mtp_model_directory;

const PARITY_TIMEOUT: Duration = Duration::from_secs(115);
const WINDOW_ROW_COUNT: usize = 4;
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_tokens_romeo_and_juliet.txt"
);

#[tokio::test]
#[ignore = "loads the dense MTP artifact and compares the compiled verification window against the eager window"]
async fn should_match_the_eager_verification_window_on_the_dense_mtp_artifact() {
    timeout(
        PARITY_TIMEOUT,
        compare_compiled_verification_window_with_eager_window(),
    )
    .await
    .expect("the compiled-window parity acceptance must finish within 115 seconds");
}

async fn compare_compiled_verification_window_with_eager_window() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let started_at = Instant::now();
    let model_directory = dense_mtp_model_directory();
    eprintln!("[compiled-window-parity] status=start phase=artifact_validation");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the dense-MTP artifact should validate before native loading");
    eprintln!(
        "[compiled-window-parity] status=progress phase=runtime_init elapsed_seconds={}",
        started_at.elapsed().as_secs()
    );
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize for the parity acceptance");
    eprintln!(
        "[compiled-window-parity] status=progress phase=model_load elapsed_seconds={}",
        started_at.elapsed().as_secs()
    );
    let qwen3_5_config = validated_artifact.config().clone();
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the dense-MTP artifact tokenizer should load");
    let prepared_prompt = tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id: RequestId::new(71_024),
                model: validated_artifact.model_id().to_owned(),
                messages: vec![ChatMessage::User {
                    content: format!(
                        "Continue the supplied Romeo and Juliet story.\n\n{ROMEO_AND_JULIET_SOURCE}"
                    ),
                    images: Vec::new(),
                }],
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                settings: ChatGenerationSettings {
                    max_output_tokens: 16,
                    temperature_thousandths: None,
                    top_p_thousandths: None,
                    seed: None,
                    thinking_budget: None,
                },
                qwen_thinking_channel_seed: None,
                structured_generation: None,
            },
            false,
        )
        .expect("the Romeo and Juliet prompt should prepare with artifact vocabulary");
    let prepared_token_ids = prepared_prompt.input_token_ids();
    assert!(
        prepared_token_ids.len() > WINDOW_ROW_COUNT,
        "the Romeo and Juliet prompt should contain window rows and a continuation token"
    );
    let window_start = prepared_token_ids.len() - WINDOW_ROW_COUNT - 1;
    let prompt_token_ids = &prepared_token_ids[..window_start];
    let window_token_ids =
        prepared_token_ids[window_start..window_start + WINDOW_ROW_COUNT].to_vec();
    let continuation_token_id = prepared_token_ids[window_start + WINDOW_ROW_COUNT];
    let model = Qwen3_5Model::load(
        runtime,
        validated_artifact,
        &model_directory,
        true,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the dense-MTP model should load for the parity acceptance");
    eprintln!(
        "[compiled-window-parity] status=progress phase=prompt_forward elapsed_seconds={}",
        started_at.elapsed().as_secs()
    );

    let mut request_decoder_state = crate::common::standard_request_decoder_state(&qwen3_5_config);
    model
        .forward_chunk_with_pre_final_normalization_hidden_states(
            &prompt_token_ids,
            0,
            &mut request_decoder_state,
        )
        .expect("the seeding prompt forward should seat every decoder state");
    let starting_position_tokens = prompt_token_ids.len() as u32;

    eprintln!(
        "[compiled-window-parity] status=progress phase=eager_window elapsed_seconds={}",
        started_at.elapsed().as_secs()
    );
    let state_checkpoint = request_decoder_state
        .checkpoint()
        .expect("the pre-window state checkpoint should retain every state owner");
    let eager_logits = model
        .eager_verification_window_logits_for_tests(
            &window_token_ids,
            starting_position_tokens,
            &mut request_decoder_state,
        )
        .expect("the eager verification window should produce all-position logits");
    let eager_continuation = continuation_logits(
        &model,
        &mut request_decoder_state,
        starting_position_tokens + WINDOW_ROW_COUNT as u32,
        continuation_token_id,
    );

    request_decoder_state
        .restore_checkpoint(state_checkpoint)
        .expect("the state checkpoint should restore for the compiled run");
    eprintln!(
        "[compiled-window-parity] status=progress phase=compiled_window elapsed_seconds={}",
        started_at.elapsed().as_secs()
    );
    let compile_started_at = Instant::now();
    let compiled_logits = model
        .compiled_verification_window_logits_for_tests(
            &window_token_ids,
            starting_position_tokens,
            &mut request_decoder_state,
        )
        .expect("the compiled verification window should produce all-position logits");
    eprintln!(
        "[compiled-window-parity] status=progress phase=compiled_window_done compile_seconds={:.2}",
        compile_started_at.elapsed().as_secs_f32()
    );
    let compiled_continuation = continuation_logits(
        &model,
        &mut request_decoder_state,
        starting_position_tokens + WINDOW_ROW_COUNT as u32,
        continuation_token_id,
    );

    assert_window_parity(&eager_logits, &compiled_logits, "window");
    assert_continuation_parity(&eager_continuation, &compiled_continuation);
    eprintln!(
        "[compiled-window-parity] status=success total_seconds={:.2}",
        started_at.elapsed().as_secs_f32()
    );
}

fn continuation_logits(
    model: &Qwen3_5Model,
    request_decoder_state: &mut RequestDecoderStateStack,
    continuation_position: u32,
    continuation_token_id: u32,
) -> Vec<f32> {
    let continuation_logits = model
        .forward_chunk(
            &[continuation_token_id],
            continuation_position,
            request_decoder_state,
        )
        .expect("the continuation decode forward should build on the installed window state");
    continuation_logits
        .to_vec_f32()
        .expect("the continuation logits should read back as float32")
}

fn assert_window_parity(
    eager_logits: &astronomical_runtime_integration::MlxArray,
    compiled_logits: &astronomical_runtime_integration::MlxArray,
    label: &str,
) {
    assert_eq!(
        eager_logits.shape(),
        compiled_logits.shape(),
        "{label}: the two windows must return identically shaped logits"
    );
    let eager_values = eager_logits
        .to_vec_f32()
        .expect("the eager logits should read back as float32");
    let compiled_values = compiled_logits
        .to_vec_f32()
        .expect("the compiled logits should read back as float32");
    assert!(
        eager_values.iter().all(|logit| logit.is_finite()),
        "{label}: eager logits must remain finite"
    );
    assert!(
        compiled_values.iter().all(|logit| logit.is_finite()),
        "{label}: compiled logits must remain finite"
    );
    let row_length = eager_values.len() / WINDOW_ROW_COUNT;
    let mut maximum_absolute_difference = 0.0_f32;
    for (row_index, (eager_row, compiled_row)) in eager_values
        .chunks(row_length)
        .zip(compiled_values.chunks(row_length))
        .enumerate()
    {
        let eager_best = row_index_of_maximum(eager_row);
        let compiled_best = row_index_of_maximum(compiled_row);
        assert_eq!(
            eager_best, compiled_best,
            "{label}: row {row_index} argmax token diverged between the windows"
        );
        for (eager_value, compiled_value) in eager_row.iter().zip(compiled_row) {
            maximum_absolute_difference =
                maximum_absolute_difference.max((eager_value - compiled_value).abs());
        }
    }
    eprintln!(
        "[compiled-window-parity] status=progress phase=window_parity max_abs_logit_difference={maximum_absolute_difference:.6}"
    );
}

fn assert_continuation_parity(eager_continuation: &[f32], compiled_continuation: &[f32]) {
    assert_eq!(
        eager_continuation.len(),
        compiled_continuation.len(),
        "the continuation forwards must return identically shaped logits"
    );
    assert!(
        eager_continuation.iter().all(|logit| logit.is_finite()),
        "the eager continuation logits must remain finite"
    );
    assert!(
        compiled_continuation.iter().all(|logit| logit.is_finite()),
        "the compiled continuation logits must remain finite"
    );
    let eager_best = row_index_of_maximum(eager_continuation);
    let compiled_best = row_index_of_maximum(compiled_continuation);
    assert_eq!(
        eager_best, compiled_best,
        "the continuation token diverged: the installed window states disagree"
    );
    let maximum_absolute_difference = eager_continuation
        .iter()
        .zip(compiled_continuation)
        .map(|(eager_value, compiled_value)| (eager_value - compiled_value).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "[compiled-window-parity] status=progress phase=continuation_parity max_abs_logit_difference={maximum_absolute_difference:.6}"
    );
}

fn row_index_of_maximum(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .expect("logit rows are never empty")
}
