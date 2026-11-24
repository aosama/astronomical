//! What the factory's unit tests share.

use ipc_protocol::{WorkerModelConfiguration, WorkerSpeechModelConfiguration};

/// A speech policy naming `model_family`, as the daemon would send it.
pub(super) fn speech_configuration_naming(model_family: &str) -> WorkerModelConfiguration {
    WorkerModelConfiguration::Speech(WorkerSpeechModelConfiguration {
        model_id: "speech-model".to_owned(),
        model_family: model_family.to_owned(),
        artifact_revision: "speech-revision".to_owned(),
    })
}
