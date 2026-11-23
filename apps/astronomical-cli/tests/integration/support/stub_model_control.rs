//! The stub daemon's answers to the requests that change what it holds:
//! unloading the model and setting the default model.

use std::sync::Mutex;

use ipc_protocol::{DaemonRequest, DaemonResponse};

use super::stub_daemon::StubDaemonConfig;

/// The stub's answer to unloading or setting the default model. An unload
/// names the resident model only when it reports one unloaded.
pub fn stub_model_control_answer(
    model_control_request: DaemonRequest,
    config: &StubDaemonConfig,
    default_model_id: &Mutex<Option<String>>,
    resident_model_id: Option<&String>,
) -> DaemonResponse {
    let DaemonRequest::DefaultModelSet { model_id } = model_control_request else {
        let has_unloaded =
            config.model_unload_status == ipc_protocol::DaemonModelUnloadStatus::Unloaded;
        return DaemonResponse::ModelUnloaded {
            status: config.model_unload_status,
            model_id: if has_unloaded {
                resident_model_id.cloned()
            } else {
                None
            },
        };
    };
    *default_model_id
        .lock()
        .expect("the stub default lock should stay live") = Some(model_id.clone());
    DaemonResponse::DefaultModelSet {
        default_model_id: model_id,
    }
}
