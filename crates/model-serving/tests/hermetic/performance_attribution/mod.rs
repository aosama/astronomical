use astronomical_model_serving::{
    GenerationPerformanceAttributionMetadata, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionLog, PerformanceAttributionOutcome,
    PerformanceCounter, PerformanceOperation,
};
use serde_json::Value;

const ATTRIBUTED_PREFILL_TRANSIENT_HIGH_WATER_BYTES: u64 = 1_234_567;

mod expert_route_reuse;
mod expert_streaming_source;
mod log;
mod measurement;
mod report_metadata;
mod support;

use support::{
    model_loading_metadata, serialize_generation_report, serialize_model_loading_report,
};
