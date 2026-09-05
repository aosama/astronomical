//! Shared embeddings result types for the supervisor HTTP boundary.
//!
//! Live dispatch lives on `ImageGenerationExecutor` because `ApplicationState`
//! holds one executor trait object for chat, image, and embeddings. A second
//! embeddings trait would force every test executor to grow another impl.

use astronomical_ipc_protocol::EmbeddingsFailureReason;
use tokio::sync::mpsc;

/// One completed native embeddings computation.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingsOutput {
    pub embeddings: Vec<Vec<f32>>,
    pub input_token_counts: Vec<u32>,
    pub elapsed_millis: u64,
}

/// Failure delivered after an embeddings request was admitted to the worker.
#[derive(Clone, Debug, PartialEq)]
pub enum EmbeddingsExecutionError {
    WorkerFailure(EmbeddingsFailureReason),
    WorkerUnavailable,
}

/// Resolves when the caller disconnects from an in-flight embeddings request.
pub(crate) async fn wait_for_embeddings_disconnect(
    embeddings_result_sender: Option<
        mpsc::Sender<Result<EmbeddingsOutput, EmbeddingsExecutionError>>,
    >,
) {
    let Some(embeddings_result_sender) = embeddings_result_sender else {
        std::future::pending::<()>().await;
        return;
    };
    embeddings_result_sender.closed().await;
}
