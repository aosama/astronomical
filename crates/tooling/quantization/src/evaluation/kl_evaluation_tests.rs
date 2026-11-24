use std::path::Path;

use super::{KlDivergenceEngine, KlEvaluationRequest, evaluate_kl_divergence};
use crate::error::QuantizationError;
use crate::progress::{ProgressReporter, SilentProgressReporter};
use crate::sensitivity::CalibrationForwardSample;

/// Returns a fixed divergence per token, or fails, without any model.
struct ScriptedEngine {
    divergence_per_token: f64,
    fails: bool,
}

impl KlDivergenceEngine for ScriptedEngine {
    fn per_token_divergences(
        &mut self,
        request: &KlEvaluationRequest<'_>,
        _progress_reporter: &mut dyn ProgressReporter,
    ) -> Result<Vec<Vec<f64>>, QuantizationError> {
        if self.fails {
            return Err(QuantizationError::ProbeModelFailure {
                description: "the candidate did not load".to_owned(),
            });
        }
        let mut per_prompt_divergences = Vec::new();
        for prompt in request.prompts {
            per_prompt_divergences.push(vec![self.divergence_per_token; prompt.token_ids.len()]);
        }
        Ok(per_prompt_divergences)
    }
}

fn prompts() -> Vec<CalibrationForwardSample> {
    vec![
        CalibrationForwardSample {
            token_ids: vec![1, 2, 3],
        },
        CalibrationForwardSample {
            token_ids: vec![4, 5],
        },
    ]
}

fn request(prompts: &[CalibrationForwardSample]) -> KlEvaluationRequest<'_> {
    KlEvaluationRequest {
        reference_directory: Path::new("reference"),
        candidate_directory: Path::new("candidate"),
        prompts,
    }
}

#[test]
fn should_summarize_the_divergences_the_engine_measured() {
    let prompts = prompts();
    let mut engine = ScriptedEngine {
        divergence_per_token: 0.02,
        fails: false,
    };

    let summary =
        evaluate_kl_divergence(&mut engine, &request(&prompts), &mut SilentProgressReporter)
            .expect("the evaluation should succeed");

    assert!((summary.mean_kl_divergence - 0.02).abs() < 1e-12);
    assert_eq!(summary.prompt_count, 2);
    assert_eq!(summary.token_count, 5);
}

#[test]
fn should_refuse_an_evaluation_without_prompts() {
    let mut engine = ScriptedEngine {
        divergence_per_token: 0.02,
        fails: false,
    };

    let refusal = evaluate_kl_divergence(&mut engine, &request(&[]), &mut SilentProgressReporter);

    assert!(matches!(
        refusal,
        Err(QuantizationError::EvaluationHasNoPrompts)
    ));
}

#[test]
fn should_pass_an_engine_failure_through() {
    let prompts = prompts();
    let mut engine = ScriptedEngine {
        divergence_per_token: 0.0,
        fails: true,
    };

    let failure =
        evaluate_kl_divergence(&mut engine, &request(&prompts), &mut SilentProgressReporter);

    assert!(matches!(
        failure,
        Err(QuantizationError::ProbeModelFailure { .. })
    ));
}
