mod contract;
mod engine;
mod generation;
mod reports;

const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);
const IMAGE_PAD_TOKEN_ID: u32 = 248_069;

pub(crate) use engine::{create_attributed_engine, load_engine_with_progress};
pub(crate) use generation::{
    GenerationSampling, RequestAttribution, run_attributed_generation,
    run_generation_with_request_attribution, run_generation_with_sampling,
};
pub(crate) use reports::{
    counter_amount, generation_report_for_request, read_attribution_report_documents,
};
