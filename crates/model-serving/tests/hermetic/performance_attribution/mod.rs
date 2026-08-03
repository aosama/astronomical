use astronomical_model_serving::{
    GenerationPerformanceAttributionMetadata, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionLog, PerformanceAttributionOutcome,
    PerformanceCounter, PerformanceOperation,
};
use serde_json::Value;

const ATTRIBUTED_PREFILL_TRANSIENT_HIGH_WATER_BYTES: u64 = 1_234_567;
const ATTRIBUTED_RETAINED_COMPLETE_EXPERT_LAYER_COUNT: u64 = 32;

mod catalog;
mod expert_route_reuse;
mod log;
mod measurement;
mod report_metadata;
mod support;

use support::{
    model_loading_metadata, serialize_generation_report, serialize_model_loading_report,
};
