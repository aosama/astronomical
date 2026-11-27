//! The daemon's end of a worker under test, shared by the scripted
//! embeddings, decisions and speech journeys: send it commands, read its
//! events without the periodic memory samples, swap a model in, and close it
//! and check it stopped cleanly.

use ipc_protocol::{WorkerCommand, WorkerEvent, WorkerModelConfiguration};
use tokio::task::JoinHandle;

use crate::WorkerRuntimeError;

pub(super) use super::chat::ipc_worker_transport::{DaemonReader, DaemonWriter};

/// A running worker task and the daemon's two ends of its transport.
pub(super) struct WorkerUnderTest {
    pub(super) events: DaemonReader,
    pub(super) commands: DaemonWriter,
    worker_task: JoinHandle<Result<(), WorkerRuntimeError>>,
}

impl WorkerUnderTest {
    /// Takes over a worker the caller spawned and waits for it to announce
    /// that it is idle. The caller spawns it because only the concrete worker
    /// type proves its future can move to another thread.
    pub(super) async fn announced_idle(
        events: DaemonReader,
        commands: DaemonWriter,
        worker_task: JoinHandle<Result<(), WorkerRuntimeError>>,
    ) -> Self {
        let mut worker_under_test = Self {
            events,
            commands,
            worker_task,
        };
        assert!(matches!(
            worker_under_test.next_event().await,
            WorkerEvent::Idle { .. }
        ));
        worker_under_test
    }

    pub(super) async fn send(&mut self, worker_command: WorkerCommand) {
        self.commands
            .send_command(&worker_command)
            .await
            .expect("the command should reach the worker");
    }

    /// The next event that is not a periodic memory sample.
    pub(super) async fn next_event(&mut self) -> WorkerEvent {
        next_event_without_samples(&mut self.events).await
    }

    /// Asks the worker to swap a model in; the caller reads the answer.
    pub(super) async fn swap_in(
        &mut self,
        model_directory: &str,
        model_configuration: WorkerModelConfiguration,
    ) {
        self.send(WorkerCommand::SwapModel {
            model_directory: model_directory.to_owned(),
            model_configuration,
        })
        .await;
    }

    pub(super) async fn close(self) {
        self.commands
            .close()
            .await
            .expect("the command transport should close");
        self.worker_task
            .await
            .expect("the worker task should not panic")
            .expect("the worker should stop cleanly when its transport closes");
    }
}

/// The next event that is not a periodic memory sample.
pub(super) async fn next_event_without_samples(events: &mut DaemonReader) -> WorkerEvent {
    loop {
        let worker_event = super::next_event_skipping_load_progress(events).await;
        if !matches!(worker_event, WorkerEvent::MlxMemorySample { .. }) {
            return worker_event;
        }
    }
}

/// Spawns a scripted worker and returns its daemon end once it is idle. A
/// macro rather than a function because only the concrete worker type proves
/// its future can move to another thread, and every caller names its own.
macro_rules! start_worker_under_test {
    ($worker:expr) => {{
        let (events, commands, worker_read, worker_write) =
            super::chat::ipc_worker_transport::open_ipc_worker_transport();
        let worker_task = tokio::spawn($worker.run(worker_read, worker_write));
        super::daemon_side::WorkerUnderTest::announced_idle(events, commands, worker_task).await
    }};
}
pub(super) use start_worker_under_test;
