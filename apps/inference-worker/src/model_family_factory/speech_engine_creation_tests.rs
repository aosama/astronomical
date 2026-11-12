use std::path::PathBuf;

use astronomical_config::ModelFamily;
use astronomical_ipc_protocol::{
    WorkerDecisionModelConfiguration, WorkerDecisionModelFamily, WorkerModelConfiguration,
    WorkerSpeechModelConfiguration, WorkerSpeechModelFamily,
};

use super::{SpeechMemoryGrant, unloaded_speech_engine};

fn memory_grant() -> SpeechMemoryGrant {
    SpeechMemoryGrant {
        effective_mlx_memory_ceiling_bytes: 1 << 30,
        allocator_cache_memory_limit_bytes: 1 << 28,
    }
}

fn speech_configuration(model_family: WorkerSpeechModelFamily) -> WorkerModelConfiguration {
    WorkerModelConfiguration::Speech(WorkerSpeechModelConfiguration {
        model_id: "Kokoro-82M-bf16".to_owned(),
        model_family,
        artifact_revision: "a71e4d38".to_owned(),
    })
}

// Positive: both speech families build an unloaded engine without touching MLX.
#[test]
fn should_build_an_unloaded_engine_for_each_speech_family() {
    for (model_family, worker_family) in [
        (ModelFamily::Kokoro, WorkerSpeechModelFamily::Kokoro),
        (ModelFamily::Phonon, WorkerSpeechModelFamily::Phonon),
    ] {
        assert!(
            unloaded_speech_engine(
                Some(model_family),
                PathBuf::from("speech-model"),
                speech_configuration(worker_family),
                &memory_grant(),
            )
            .is_ok()
        );
    }
}

// Negative: a speech configuration for a folder of another family is refused.
#[test]
fn should_refuse_a_speech_configuration_for_another_family() {
    let Err(mismatch_reason) = unloaded_speech_engine(
        Some(ModelFamily::Laya),
        PathBuf::from("decision-model"),
        speech_configuration(WorkerSpeechModelFamily::Kokoro),
        &memory_grant(),
    ) else {
        panic!("a Laya folder is not a speech model");
    };
    assert!(mismatch_reason.contains("does not match"));
}

// Edge: a speech folder paired with a decision configuration is refused too.
#[test]
fn should_refuse_a_decision_configuration_for_a_speech_folder() {
    let decision_configuration =
        WorkerModelConfiguration::Decisions(WorkerDecisionModelConfiguration {
            model_id: "laya".to_owned(),
            model_family: WorkerDecisionModelFamily::Laya,
            artifact_revision: "abc".to_owned(),
            maximum_tokens_per_question: 512,
        });
    assert!(
        unloaded_speech_engine(
            Some(ModelFamily::Kokoro),
            PathBuf::from("speech-model"),
            decision_configuration,
            &memory_grant(),
        )
        .is_err()
    );
}
