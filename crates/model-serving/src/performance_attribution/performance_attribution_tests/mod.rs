mod flux2_image_tests;

use crate::{
    GenerationPerformanceAttributionMetadata, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionOutcome, PerformanceCounter,
    PerformanceOperation,
};
use serde_json::Value;

const ATTRIBUTED_PREFILL_TRANSIENT_HIGH_WATER_BYTES: u64 = 1_234_567;

mod counter_catalog;
mod expert_route_reuse;
mod expert_streaming_source;
mod measurement;
mod report_metadata;
mod route_observation;
mod support;

use support::{serialize_generation_report, serialize_model_loading_report};
