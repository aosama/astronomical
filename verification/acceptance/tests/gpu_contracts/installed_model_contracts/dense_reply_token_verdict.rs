//! What two choices the text engine makes for a dense model held whole cost
//! a reply token, as verdicts of the MLX performance profiler (#2585).
//!
//! The retired in-memory engine built a reply token as one graph and looked
//! its embeddings up in a table it held. The text engine submits every
//! `experimental_ssd_paging_generation_graph_submission_layer_interval`
//! layers (3 by default) and reads each token's embedding row from the SSD,
//! even when the limit holds the whole model. Each comparison changes one of
//! the two and nothing else, on one loaded model, from the same stored state
//! after a Shakespeare prompt, eight greedy reply tokens a run. The greedy
//! tokens of every variant must match the served ones exactly.
//!
//! `cargo measure-dense-reply-token-verdict --configuration PATH`, where the
//! configuration names a dense model. About a minute on a 9B, two on a 27B.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use mlx_performance_profiler::{
    MeasuredProcedure, ProcedureComparison, ReportLocation, compare_procedures,
};
use model_serving::{Qwen3_5TextModel, RequestDecoderStateStack};
use serial_test::file_serial;

use super::greedy_answer::{load_materialized_streaming_model, shakespeare_prompt_token_ids};

const REPLY_TOKENS_PER_RUN: usize = 8;
const PROMPT_CHUNK_TOKENS: usize = 256;
const REPORT_SUBDIRECTORY: &str = "dense-reply-token-verdict";

/// One thing a reply token can do differently from how it is served.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReplyTokenVariant {
    AsServed,
    OneGraphPerToken,
    EmbeddingTableInMemory,
}

impl ReplyTokenVariant {
    const fn name(self) -> &'static str {
        match self {
            Self::AsServed => "as_served",
            Self::OneGraphPerToken => "one_graph_per_token",
            Self::EmbeddingTableInMemory => "embedding_table_in_memory",
        }
    }
}

/// What both procedures of a comparison share: the model, the decoder state
/// after the prompt, and the on-disk embedding rows a variant sets aside.
struct SharedReplyBench {
    model: RefCell<Qwen3_5TextModel>,
    decoder_state: RefCell<RequestDecoderStateStack>,
    stored_state: RefCell<Option<model_serving::RequestDecoderStateStackCheckpoint>>,
    set_aside_disk_embedding_rows: RefCell<
        Option<model_serving::qwen3_5_core::model::disk_embedding_rows::Qwen3_5DiskEmbeddingRows>,
    >,
    prompt_token_count: u32,
    first_reply_token_id: u32,
    /// The reply submission interval the model loaded with: what "as served" restores.
    served_reply_submission_interval_layers: u32,
}

impl SharedReplyBench {
    /// Puts the model in `variant`'s form; untimed.
    fn apply(&self, variant: ReplyTokenVariant) {
        let mut model = self.model.borrow_mut();
        model
            .chunking
            .experimental_ssd_paging_generation_graph_submission_layer_interval = match variant {
            ReplyTokenVariant::OneGraphPerToken => 0,
            _ => self.served_reply_submission_interval_layers,
        };
        let mut set_aside = self.set_aside_disk_embedding_rows.borrow_mut();
        if variant == ReplyTokenVariant::EmbeddingTableInMemory {
            if let Some(disk_embedding_rows) = model.disk_embedding_rows.take() {
                *set_aside = Some(disk_embedding_rows);
            }
        } else if let Some(disk_embedding_rows) = set_aside.take() {
            model.disk_embedding_rows = Some(disk_embedding_rows);
        }
    }

    /// Restores the decoder state stored after the prompt and stores it again.
    fn restore_the_prompt_state(&self) -> Result<(), String> {
        let mut decoder_state = self.decoder_state.borrow_mut();
        let mut stored_state = self.stored_state.borrow_mut();
        if let Some(checkpoint) = stored_state.take() {
            decoder_state
                .restore_checkpoint(checkpoint)
                .map_err(|restore_error| restore_error.to_string())?;
        }
        *stored_state = Some(
            decoder_state
                .checkpoint()
                .map_err(|checkpoint_error| checkpoint_error.to_string())?,
        );
        Ok(())
    }

    /// Eight greedy reply tokens, each picked on the GPU and read back as
    /// serving reads its sampled token.
    fn reply_tokens(&self) -> Result<Vec<u32>, String> {
        let model = self.model.borrow();
        let mut decoder_state = self.decoder_state.borrow_mut();
        let mut next_token_id = self.first_reply_token_id;
        let mut reply_token_ids = Vec::with_capacity(REPLY_TOKENS_PER_RUN);
        for reply_index in 0..REPLY_TOKENS_PER_RUN {
            let position = self.prompt_token_count + reply_index as u32;
            let logits = model
                .forward_chunk(&[next_token_id], position, &mut decoder_state)
                .map_err(|forward_error| forward_error.to_string())?;
            let picked = model
                .runtime()
                .argmax_axis(&logits, -1)
                .and_then(|token_ids| token_ids.to_vec_u32())
                .map_err(|pick_error| pick_error.to_string())?;
            next_token_id = *picked.last().ok_or("the forward picked no token")?;
            reply_token_ids.push(next_token_id);
        }
        Ok(reply_token_ids)
    }
}

/// One variant's reply tokens as a procedure the profiler times.
struct ReplyTokens {
    bench: Rc<SharedReplyBench>,
    variant: ReplyTokenVariant,
}

impl MeasuredProcedure for ReplyTokens {
    fn name(&self) -> &str {
        self.variant.name()
    }

    fn prepare(&mut self) -> Result<(), String> {
        self.bench.apply(self.variant);
        self.bench.restore_the_prompt_state()
    }

    fn run(&mut self) -> Result<(), String> {
        self.bench.reply_tokens().map(drop)
    }
}

#[tokio::test]
#[ignore = "loads the configured dense model whole and compares reply token variants"]
#[file_serial]
async fn should_give_verdicts_on_how_a_dense_reply_token_is_built() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory =
        test_harness::test_run::TestRunContext::current().required_model_directory();
    let bench = Rc::new(prepared_bench(&model_directory).await);
    let served_tokens = tokens_of(&bench, ReplyTokenVariant::AsServed);
    for variant in [
        ReplyTokenVariant::OneGraphPerToken,
        ReplyTokenVariant::EmbeddingTableInMemory,
    ] {
        assert_eq!(
            tokens_of(&bench, variant),
            served_tokens,
            "{} must choose the served reply tokens",
            variant.name()
        );
        let comparison = compare_with_served(&bench, variant);
        report(variant, &comparison);
    }
}

/// Loads the model whole, reads the prompt and keeps the state after it.
async fn prepared_bench(model_directory: &Path) -> SharedReplyBench {
    let (model, qwen3_5_config) = load_materialized_streaming_model(model_directory).await;
    model
        .materialize_and_keep_every_dense_layer()
        .expect("every dense layer is kept");
    assert!(
        model.disk_embedding_rows.is_some(),
        "the configured model must be dense, so its embedding rows come from the SSD"
    );
    let prompt_token_ids = shakespeare_prompt_token_ids(model_directory);
    let mut decoder_state = crate::common::standard_request_decoder_state(&qwen3_5_config);
    let mut last_logits = None;
    for (chunk_index, prompt_chunk) in prompt_token_ids.chunks(PROMPT_CHUNK_TOKENS).enumerate() {
        let chunk_start = u32::try_from(chunk_index * PROMPT_CHUNK_TOKENS).expect("fits u32");
        last_logits = Some(
            model
                .forward_chunk(prompt_chunk, chunk_start, &mut decoder_state)
                .expect("the prompt is read"),
        );
    }
    let first_reply_token_id = *model
        .runtime()
        .argmax_axis(&last_logits.expect("the prompt has tokens"), -1)
        .and_then(|token_ids| token_ids.to_vec_u32())
        .expect("the first reply token is picked")
        .last()
        .expect("the prompt's last position has a token");
    let served_reply_submission_interval_layers = model
        .chunking
        .experimental_ssd_paging_generation_graph_submission_layer_interval;
    SharedReplyBench {
        served_reply_submission_interval_layers,
        model: RefCell::new(model),
        decoder_state: RefCell::new(decoder_state),
        stored_state: RefCell::new(None),
        set_aside_disk_embedding_rows: RefCell::new(None),
        prompt_token_count: u32::try_from(prompt_token_ids.len()).expect("fits u32"),
        first_reply_token_id,
    }
}

fn tokens_of(bench: &SharedReplyBench, variant: ReplyTokenVariant) -> Vec<u32> {
    bench.apply(variant);
    bench
        .restore_the_prompt_state()
        .expect("the prompt state is restored");
    bench.reply_tokens().expect("the reply tokens are written")
}

fn compare_with_served(
    bench: &Rc<SharedReplyBench>,
    variant: ReplyTokenVariant,
) -> ProcedureComparison {
    let report_directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report directory is made");
    compare_procedures(
        &mut ReplyTokens {
            bench: Rc::clone(bench),
            variant: ReplyTokenVariant::AsServed,
        },
        &mut ReplyTokens {
            bench: Rc::clone(bench),
            variant,
        },
        ReportLocation {
            title: "A dense model's reply tokens",
            directory: &report_directory,
            file_base: variant.name(),
        },
    )
    .expect("the reply tokens are compared")
}

fn report(variant: ReplyTokenVariant, comparison: &ProcedureComparison) {
    eprintln!(
        "[dense-reply-token-verdict] candidate={} verdict={:?} served_ms={:.3} candidate_ms={:.3} \
         difference_ms={:.3} interval_ms=[{:.3}, {:.3}] gpu_busy_ms={:.3}->{:.3} \
         gpu_idle_ms={:.3}->{:.3} command_buffers={:.1}->{:.1} blocking_waits={:.1}->{:.1} \
         rounds_candidate_faster={}/{}",
        variant.name(),
        comparison.duration.verdict,
        comparison.duration.baseline_median_microseconds / 1_000.0,
        comparison.duration.candidate_median_microseconds / 1_000.0,
        comparison.duration.median_difference_microseconds / 1_000.0,
        comparison.duration.difference_interval_low_microseconds / 1_000.0,
        comparison.duration.difference_interval_high_microseconds / 1_000.0,
        comparison.gpu_busy.baseline_median_microseconds / 1_000.0,
        comparison.gpu_busy.candidate_median_microseconds / 1_000.0,
        comparison.gpu_idle.baseline_median_microseconds / 1_000.0,
        comparison.gpu_idle.candidate_median_microseconds / 1_000.0,
        comparison.baseline.median_command_buffer_count,
        comparison.candidate.median_command_buffer_count,
        comparison.baseline.median_blocking_wait_count,
        comparison.candidate.median_blocking_wait_count,
        comparison.duration.rounds_candidate_faster,
        comparison.duration.round_count,
    );
}
