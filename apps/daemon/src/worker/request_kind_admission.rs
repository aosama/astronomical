//! Admits one request of a kind that needs no admission of its own
//! (decisions, embeddings, speech) through the shared model-swap and
//! active-request owner.
//!
//! Every such request takes the same steps: refuse while the worker is
//! unavailable or busy, let a pending memory-limit change settle, make the
//! requested model resident, check the resident model can answer this kind,
//! dispatch, then own the active-request slot, or contain the worker when
//! admission fails. A kind supplies only what differs, through
//! [`RequestKind`]: its checks, its dispatch, and the active-request entry
//! that bounds it while it runs. Decisions, embeddings and speech each carried
//! their own copy of these steps before; a new kind of request, such as a
//! transcription or reranking endpoint, writes none of them.
//!
//! Chat and images keep their own admission, because each records what this
//! path does not: chat its prompt-processing progress, an image its queue
//! wait and swap time.

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

type StartSender = oneshot::Sender<Result<(), GenerationStartError>>;

/// What a kind of request adds to the shared admission.
pub(in crate::worker) trait RequestKind {
    /// What the request asks the worker to run.
    type Work: Send;
    /// What the request's result channel carries.
    type ResultMessage: Send;
    /// What the kind's active-request entry needs beyond what every admitted
    /// request owns. It is read before dispatch takes the work.
    type RequestBounds: Send;
    /// The worker-loop command's name, as the busy log reports it.
    const COMMAND_NAME: &'static str;
    /// What health reports once the request has started.
    const STARTED_ACTIVITY: WorkerActivity;

    fn requested_model(work: &Self::Work) -> &str;

    fn request_id(work: &Self::Work) -> RequestId;

    fn request_bounds(work: &Self::Work, runtime: &WorkerLoopRuntime) -> Self::RequestBounds;

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
        admitted_request: AdmittedRequest<Self::ResultMessage>,
        request_bounds: Self::RequestBounds,
    ) -> ActiveWorkerRequest;
}

/// One command of a request kind as the worker loop routes it.
pub(in crate::worker) struct RequestKindCommand<Kind: RequestKind> {
    pub(in crate::worker) active_generation_permit: OwnedSemaphorePermit,
    pub(in crate::worker) work: Kind::Work,
    pub(in crate::worker) start_sender: StartSender,
    pub(in crate::worker) result_sender: mpsc::Sender<Kind::ResultMessage>,
}

/// What every started request owns while it runs.
pub(in crate::worker) struct AdmittedRequest<ResultMessage> {
    pub(in crate::worker) active_generation_permit: OwnedSemaphorePermit,
    pub(in crate::worker) request_id: RequestId,
    pub(in crate::worker) execution_started_at: Instant,
    pub(in crate::worker) result_sender: mpsc::Sender<ResultMessage>,
}

impl<ResultMessage> AdmittedRequest<ResultMessage> {
    /// The whole-request deadline of a kind that runs as one forward pass.
    pub(in crate::worker) fn single_shot_execution_deadline(&self) -> Instant {
        self.execution_started_at + SINGLE_SHOT_EXECUTION_TIMEOUT
    }
}

/// Handles one command end to end: the refusals that need no model, then
/// admission, then the answer to the client, containing the worker when
/// admission fails.
pub(in crate::worker) async fn admit_request_kind_command<Kind: RequestKind>(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    request_kind_command: RequestKindCommand<Kind>,
) {
    let RequestKindCommand {
        active_generation_permit,
        work,
        start_sender,
        result_sender,
    } = request_kind_command;
    let Some(start_sender) =
        refuse_or_settle_before_admission::<Kind>(runtime, is_worker_running, start_sender).await
    else {
        return;
    };
    let request_bounds = Kind::request_bounds(&work, runtime);
    let admission_outcome = match admit_request_work::<Kind>(runtime, work, &result_sender).await {
        Ok(admission_outcome) => admission_outcome,
        Err(control_error) => RequestAdmissionOutcome::WorkerFailed(control_error),
    };
    match admission_outcome {
        RequestAdmissionOutcome::Started {
            request_id,
            execution_started_at,
        } => {
            let admitted_request = AdmittedRequest {
                active_generation_permit,
                request_id,
                execution_started_at,
                result_sender,
            };
            let active_request = Kind::active_request(admitted_request, request_bounds);
            start_admitted_request::<Kind>(runtime, active_request, start_sender).await;
        }
        RequestAdmissionOutcome::ClientDisconnected => {}
        RequestAdmissionOutcome::StartRejected(start_error) => {
            let _send_outcome = start_sender.send(Err(start_error));
        }
        RequestAdmissionOutcome::WorkerFailed(control_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
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
}

/// Refuses while the worker is unavailable or busy, then applies a pending
/// memory-limit change. `None` means the client has had its answer.
///
/// The busy refusal comes before the memory limit, so a ceiling queued behind
/// the active request is only ever applied to an idle worker, by the idle
/// drain once that request ends. Only one request can be active at a time,
/// so a busy worker refuses here only if that slot was bypassed.
async fn refuse_or_settle_before_admission<Kind: RequestKind>(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    start_sender: StartSender,
) -> Option<StartSender> {
    if reject_unavailable_worker(runtime.is_ready, is_worker_running) {
        let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
        return None;
    }
    if runtime.active_generation.is_some() {
        let _send_outcome = start_sender.send(Err(GenerationStartError::CapacityUnavailable));
        tracing::error!(
            "received {} while another request is active",
            Kind::COMMAND_NAME
        );
        return None;
    }
    take_generation_start_after_pending_memory_limit(
        &mut runtime.pending_mlx_memory_limit_update,
        &mut runtime.worker_process,
        runtime.model_load_timeout,
        &runtime.health_snapshot,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.active_generation,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
        start_sender,
    )
    .await
}

/// Runs the pre-execution admission pipeline: model residency through the
/// shared steps, then the kind's capability gate and the worker dispatch.
/// Failures come back as outcomes so the caller answers the client and
/// contains the worker.
async fn admit_request_work<Kind: RequestKind>(
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

/// Takes the active-request slot, confirms the start, and cancels the
/// request at once when the client already hung up.
async fn start_admitted_request<Kind: RequestKind>(
    runtime: &mut WorkerLoopRuntime,
    active_request: ActiveWorkerRequest,
    start_sender: StartSender,
) {
    runtime.active_generation = Some(active_request);
    clear_active_request_progress(&runtime.health_snapshot);
    publish_activity(&runtime.health_snapshot, Kind::STARTED_ACTIVITY);
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
}
