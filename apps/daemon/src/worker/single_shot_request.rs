//! Admits one request that runs as a single forward pass (decisions,
//! embeddings) through the shared model-swap and active-request owner.
//!
//! Every such request takes the same steps: refuse while the worker is
//! unavailable, let a pending memory-limit change settle, refuse while
//! another request is active, make the requested model resident, check the
//! resident model can answer this kind, dispatch, then own the active-request
//! slot, or contain the worker when admission fails. A kind supplies only what
//! differs, through [`SingleShotRequestKind`]. Decisions and embeddings each
//! carried their own copy of these steps before; a new kind of request, such
//! as a transcription or reranking endpoint, writes none of them.

use std::future::Future;
use std::sync::{Arc, RwLock};

use ipc_protocol::RequestId;
use tokio::sync::{OwnedSemaphorePermit, mpsc, oneshot};
use tokio::time::Instant;

use crate::worker::command_arms::reject_unavailable_worker;
use crate::worker::health::{clear_active_request_progress, publish_activity};
use crate::worker::loop_types::{ActiveWorkerRequest, WorkerLoopRuntime};
use crate::worker::memory::limit::take_generation_start_after_pending_memory_limit;
use crate::worker::process::containment::{cancel_active_generation, contain_worker_failure};
use crate::worker::request_admission::{
    RequestAdmissionContext, RequestAdmissionOutcome, SINGLE_SHOT_EXECUTION_TIMEOUT,
    classify_dispatch_result, make_requested_model_resident,
};
use crate::{
    GenerationStartError, WorkerActivity, WorkerControlError, WorkerHealthSnapshot, WorkerProcess,
};

/// What a kind of single-shot request adds to the shared admission.
pub(in crate::worker) trait SingleShotRequestKind {
    /// What the request asks the worker to run.
    type Work: Send;
    /// What the request's result channel carries.
    type ResultMessage: Send;
    /// The worker-loop command's name, as the busy log reports it.
    const COMMAND_NAME: &'static str;

    fn requested_model(work: &Self::Work) -> &str;

    fn request_id(work: &Self::Work) -> RequestId;

    /// Whether the resident model advertises a surface that can run `work`.
    fn resident_model_accepts(
        health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
        work: &Self::Work,
    ) -> bool;

    /// Sends the work to the worker process.
    fn dispatch(
        worker_process: &mut WorkerProcess,
        work: Self::Work,
    ) -> impl Future<Output = Result<(), WorkerControlError>> + Send;

    /// The active-request slot's entry for a started request.
    fn active_request(
        admitted_request: AdmittedSingleShot<Self::ResultMessage>,
    ) -> ActiveWorkerRequest;
}

/// One single-shot command as the worker loop routes it.
pub(in crate::worker) struct SingleShotCommand<Kind: SingleShotRequestKind> {
    pub(in crate::worker) active_generation_permit: OwnedSemaphorePermit,
    pub(in crate::worker) work: Kind::Work,
    pub(in crate::worker) start_sender: oneshot::Sender<Result<(), GenerationStartError>>,
    pub(in crate::worker) result_sender: mpsc::Sender<Kind::ResultMessage>,
}

/// What a started request owns while it runs.
pub(in crate::worker) struct AdmittedSingleShot<ResultMessage> {
    pub(in crate::worker) active_generation_permit: OwnedSemaphorePermit,
    pub(in crate::worker) request_id: RequestId,
    pub(in crate::worker) execution_deadline: Instant,
    pub(in crate::worker) result_sender: mpsc::Sender<ResultMessage>,
}

/// Handles one single-shot command end to end: refuse while the worker is
/// unavailable, let a pending memory limit settle, admit, and contain the
/// worker when admission fails.
pub(in crate::worker) async fn admit_single_shot_command<Kind: SingleShotRequestKind>(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    mut single_shot_command: SingleShotCommand<Kind>,
) {
    if reject_unavailable_worker(runtime.is_ready, is_worker_running) {
        let _send_outcome = single_shot_command
            .start_sender
            .send(Err(GenerationStartError::WorkerUnavailable));
        return;
    }
    let Some(start_sender) = take_generation_start_after_pending_memory_limit(
        &mut runtime.pending_mlx_memory_limit_update,
        &mut runtime.worker_process,
        runtime.model_load_timeout,
        &runtime.health_snapshot,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.active_generation,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
        single_shot_command.start_sender,
    )
    .await
    else {
        return;
    };
    single_shot_command.start_sender = start_sender;
    if let Err(control_error) = handle_single_shot_command(runtime, single_shot_command).await {
        contain_worker_failure(
            &mut runtime.worker_process,
            &runtime.health_snapshot,
            &mut runtime.active_generation,
            control_error,
        )
        .await;
        runtime.is_ready = false;
    }
}

/// Admits the command, then answers its client: started, refused, or the
/// worker contained. A client that left before the start was confirmed has
/// its request cancelled.
async fn handle_single_shot_command<Kind: SingleShotRequestKind>(
    runtime: &mut WorkerLoopRuntime,
    single_shot_command: SingleShotCommand<Kind>,
) -> Result<(), WorkerControlError> {
    let SingleShotCommand {
        active_generation_permit,
        work,
        start_sender,
        result_sender,
    } = single_shot_command;
    if runtime.active_generation.is_some() {
        let _send_outcome = start_sender.send(Err(GenerationStartError::CapacityUnavailable));
        tracing::error!(
            "received {} while another request is active",
            Kind::COMMAND_NAME
        );
        return Ok(());
    }
    let admission_outcome = admit_single_shot_work::<Kind>(runtime, work, &result_sender).await?;
    match admission_outcome {
        RequestAdmissionOutcome::Started {
            request_id,
            execution_started_at,
        } => {
            runtime.active_generation = Some(Kind::active_request(AdmittedSingleShot {
                active_generation_permit,
                request_id,
                execution_deadline: execution_started_at + SINGLE_SHOT_EXECUTION_TIMEOUT,
                result_sender,
            }));
            clear_active_request_progress(&runtime.health_snapshot);
            publish_activity(&runtime.health_snapshot, WorkerActivity::Idle);
        }
        RequestAdmissionOutcome::ClientDisconnected => return Ok(()),
        RequestAdmissionOutcome::StartRejected(start_error) => {
            let _send_outcome = start_sender.send(Err(start_error));
            return Ok(());
        }
        RequestAdmissionOutcome::WorkerFailed(start_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_worker_failure(
                &mut runtime.worker_process,
                &runtime.health_snapshot,
                &mut runtime.active_generation,
                start_error,
            )
            .await;
            runtime.is_ready = false;
            return Ok(());
        }
    }
    if start_sender.send(Ok(())).is_err() {
        cancel_active_generation(
            &mut runtime.worker_process,
            &runtime.health_snapshot,
            &mut runtime.active_generation,
            runtime.cancellation_acknowledgement_timeout,
            runtime.model_load_timeout,
            &mut runtime.is_ready,
        )
        .await;
    }
    Ok(())
}

/// Runs the pre-execution admission pipeline: model residency through the
/// shared steps, then the kind's capability gate and the worker dispatch.
/// Failures come back as outcomes so the caller answers the client and
/// contains the worker.
async fn admit_single_shot_work<Kind: SingleShotRequestKind>(
    runtime: &mut WorkerLoopRuntime,
    work: Kind::Work,
    result_sender: &mpsc::Sender<Kind::ResultMessage>,
) -> Result<RequestAdmissionOutcome, WorkerControlError> {
    let mut admission_context = RequestAdmissionContext {
        worker_process: &mut runtime.worker_process,
        health_snapshot: &runtime.health_snapshot,
        is_ready: &mut runtime.is_ready,
        model_load_deadline: &mut runtime.model_load_deadline,
        active_request: &mut runtime.active_generation,
        performance_log: &mut runtime.performance_log,
        completion_log: &mut runtime.completion_log,
        model_policy_catalog: &runtime.model_policy_catalog,
        model_load_timeout: runtime.model_load_timeout,
    };
    if let Some(admission_ending_outcome) = make_requested_model_resident(
        &mut admission_context,
        Kind::requested_model(&work),
        result_sender,
    )
    .await?
    {
        return Ok(admission_ending_outcome);
    }
    if !Kind::resident_model_accepts(admission_context.health_snapshot, &work) {
        return Ok(RequestAdmissionOutcome::StartRejected(
            GenerationStartError::WorkerUnavailable,
        ));
    }
    let request_id = Kind::request_id(&work);
    let execution_started_at = Instant::now();
    let dispatch_result = Kind::dispatch(admission_context.worker_process, work).await;
    Ok(classify_dispatch_result(
        request_id,
        execution_started_at,
        dispatch_result,
    ))
}
