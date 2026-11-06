//! Tells the daemon a load has started, so its status can name the model during a
//! wait that can otherwise read as nothing happening.

use astronomical_ipc_protocol::{ModelLoadInFlight, ProtocolWriter, WorkerEvent};
use tokio::io::AsyncWrite;

use super::support::WorkerRuntimeError;

/// Only the initial load and a requested swap announce themselves. The promotion and
/// streaming-retry loads run inside an active generation, where the daemon's
/// cancellation wait treats any unexpected event as a failure.
pub(super) async fn announce_load_started<WriteTransport>(
    model_id: Option<&str>,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), WorkerRuntimeError>
where
    WriteTransport: AsyncWrite + Unpin,
{
    let opening_report = match model_id {
        Some(model_id) => ModelLoadInFlight::for_model(model_id),
        None => ModelLoadInFlight::unnamed(),
    };
    event_writer
        .send_event(&WorkerEvent::ModelLoadInFlight {
            model_load_in_flight: opening_report,
        })
        .await?;
    Ok(())
}

#[cfg(test)]
#[path = "load_announcement_tests.rs"]
mod tests;
