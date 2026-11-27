//! Where one reply token of a folded ternary pack goes on the shared core,
//! node by node, as the MLX performance profiler measures it (#2653).
//!
//! The core decodes a folded pack slower than the folded engine it replaces,
//! and #2653 lists the candidates: a rotation for every rotated module where
//! the folded engine rotates once for siblings that read the same input,
//! the composed inverse rotation of the embedding row, the composed DeltaNet
//! prework, and MLX's 2-bit matrix-vector rate. The profiler's page and JSON
//! for one reply token built as one graph say which of these the time is in.
//!
//! `cargo measure-prism-shared-core-reply-token --configuration PATH`, where
//! the configuration names a folded pack. About a minute.

use mlx_bridge::MlxArray;
use mlx_performance_profiler::{MeasuredJob, ReportLocation, measure_job};
use model_serving::{PerformanceAttribution, Qwen3_5MoEPagedExecutionMode, Qwen3_5TextModel};
use serial_test::serial;

use crate::stored_prompt_state::StoredPromptState;

use super::shared_core_parity::{empty_shared_core_state, loaded_shared_core, picked_on_the_gpu};
use super::support::{
    PREFILL_CHUNK_TOKENS, WARMUP_INPUT_TOKENS, pack_directory, pack_tokenizer,
    romeo_and_juliet_prompt_token_ids,
};

const REPORT_SUBDIRECTORY: &str = "prism-shared-core-reply-token";

/// One reply token after the prompt, from the same stored state every pass.
struct ReplyTokenJob {
    shared_core: Qwen3_5TextModel,
    prompt_state: StoredPromptState,
    reply_token_id: u32,
    position: u32,
}

impl MeasuredJob for ReplyTokenJob {
    fn name(&self) -> &str {
        "shared_core_reply_token"
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        self.prompt_state.restore()?;
        // Built, not evaluated: the profiler evaluates it, timing every node.
        let logits = self
            .shared_core
            .build_forward_chunk_with_paged_execution_mode_and_performance_attribution(
                &[self.reply_token_id],
                self.position,
                &mut self.prompt_state.decoder_state.borrow_mut(),
                Qwen3_5MoEPagedExecutionMode::ProductionDefault,
                &mut PerformanceAttribution::disabled(),
            )
            .map_err(|forward_error| forward_error.to_string())?;
        Ok(vec![logits])
    }
}

#[test]
#[serial]
#[ignore = "loads the real folded pack on the shared core and profiles one reply token on the graphics processor"]
fn should_profile_one_reply_token_of_a_folded_pack_on_the_shared_core() {
    let tokenizer = pack_tokenizer();
    let prompt_token_ids: Vec<u32> =
        romeo_and_juliet_prompt_token_ids(&tokenizer, WARMUP_INPUT_TOKENS)
            .iter()
            .map(|token_id| *token_id as u32)
            .collect();
    let shared_core = loaded_shared_core(&pack_directory());
    let mut decoder_state = empty_shared_core_state(&shared_core);
    let mut last_logits = None;
    for (chunk_index, prompt_chunk) in prompt_token_ids.chunks(PREFILL_CHUNK_TOKENS).enumerate() {
        last_logits = Some(
            shared_core
                .forward_chunk(
                    prompt_chunk,
                    (chunk_index * PREFILL_CHUNK_TOKENS) as u32,
                    &mut decoder_state,
                )
                .expect("the prompt is read"),
        );
    }
    let reply_token_id =
        picked_on_the_gpu(&shared_core, &last_logits.expect("the prompt has tokens"));
    let mut job = ReplyTokenJob {
        shared_core,
        prompt_state: StoredPromptState::after_the_prompt(decoder_state),
        reply_token_id,
        position: prompt_token_ids.len() as u32,
    };
    let report_directory = std::env::temp_dir().join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report folder is writable");
    let summary = measure_job(
        &mut job,
        ReportLocation {
            title: "One reply token of a folded pack on the shared core",
            directory: &report_directory,
            file_base: "shared_core_reply_token",
        },
    )
    .expect("the profiler measures the reply token");
    println!(
        "[prism-shared-core-reply-token] wall_median_us={:.0} gpu_execution_us={:.0} executed_nodes={} report={}",
        summary.job_wall_median_microseconds,
        summary.gpu_execution_microseconds,
        summary.executed_node_count,
        report_directory
            .join("shared_core_reply_token.json")
            .display(),
    );
}
