mod contract;
mod engine;
mod generation;
mod reporting;
mod reports;

pub(super) const INPUT_TOKEN_COUNT: usize = 1_024;
pub(super) const OUTPUT_TOKEN_COUNT: u16 = 24;
const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);
const IMAGE_PAD_TOKEN_ID: u32 = 248_069;

pub(crate) use engine::{create_attributed_engine, load_engine_with_progress};
pub(crate) use generation::run_attributed_generation;
pub(crate) use reporting::{
    assert_attributed_memory_within_machine_cap, print_attribution_metadata,
    print_attribution_operation_table, print_expert_streaming_source_summary_table,
};
pub(crate) use reports::model_loading_report;
pub(crate) use reports::{
    counter_amount, generation_report_for_request, operation_total_elapsed_nanoseconds,
    read_attribution_report_documents,
};
