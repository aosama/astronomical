//! The stage timings every hand-off call reports, shared by the hand-off and
//! token-split journeys. The hand-off steps themselves are serving code, in
//! `astronomical_neural_engine::mlx_hand_off`.

use std::time::Duration;

use astronomical_neural_engine::timing_summary::TimingSummary;

pub(super) struct StageDurations {
    pub(super) input_preparation: Duration,
    pub(super) surface_write: Duration,
    pub(super) engines_after_write: Duration,
    pub(super) merge: Duration,
    pub(super) total: Duration,
}

pub(super) struct StageSummaries {
    pub(super) input_preparation: TimingSummary,
    pub(super) surface_write: TimingSummary,
    pub(super) engines_after_write: TimingSummary,
    pub(super) merge: TimingSummary,
    pub(super) total: TimingSummary,
}

pub(super) fn summarize_stages(stage_samples: &[StageDurations]) -> StageSummaries {
    let mut input_preparation = Vec::with_capacity(stage_samples.len());
    let mut surface_write = Vec::with_capacity(stage_samples.len());
    let mut engines_after_write = Vec::with_capacity(stage_samples.len());
    let mut merge = Vec::with_capacity(stage_samples.len());
    let mut total = Vec::with_capacity(stage_samples.len());
    for stages in stage_samples {
        input_preparation.push(stages.input_preparation);
        surface_write.push(stages.surface_write);
        engines_after_write.push(stages.engines_after_write);
        merge.push(stages.merge);
        total.push(stages.total);
    }
    StageSummaries {
        input_preparation: TimingSummary::from_samples(&input_preparation).expect("timed"),
        surface_write: TimingSummary::from_samples(&surface_write).expect("timed"),
        engines_after_write: TimingSummary::from_samples(&engines_after_write).expect("timed"),
        merge: TimingSummary::from_samples(&merge).expect("timed"),
        total: TimingSummary::from_samples(&total).expect("timed"),
    }
}
