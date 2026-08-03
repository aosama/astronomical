mod benchmark;
mod contract;
mod engine;
mod generation;
mod reporting;
mod reports;

const MODEL_ID: &str = "Qwen3.6-35B-A3B-OptiQ-4bit";
const INPUT_TOKEN_COUNT: usize = 1_024;
const OUTPUT_TOKEN_COUNT: u16 = 24;
const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);
const IMAGE_PAD_TOKEN_ID: u32 = 248_069;
const DETERMINISTIC_PROMPT_TOKEN_ID: u32 = 198;

pub(crate) use engine::{create_attributed_engine, load_engine_with_progress};
pub(crate) use generation::run_attributed_generation;
use reporting::{
    assert_attributed_memory_within_machine_cap, print_attribution_metadata,
    print_attribution_operation_table,
};
use reports::model_loading_report;
pub(crate) use reports::{
    counter_amount, generation_report_for_request, operation_total_elapsed_nanoseconds,
    read_attribution_report_documents,
};

fn qwen3_6_35b_a3b_optiq_4bit_model_directory() -> Option<std::path::PathBuf> {
    crate::common::configured_model_directory_by_id(MODEL_ID)
}
