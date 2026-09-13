//! Fail-open background trainer for the expert-route predictor.
//!
//! Generation never waits on training: observations are offered through a
//! bounded try-send, and a full or disconnected queue drops the example.
//! The trainer thread owns the weights, so they never enter MLX accounting.
//! Spawn failure, poison, and disconnect all degrade to no prediction.

use astronomical_ipc_protocol::PredictorProgramStatus;
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, TryLockError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::network::{ExpertRoutePredictor, ExpertRoutePredictorConfig};
use super::trainer::evaluate_then_train;
use crate::memory::select_top_expert_ids_from_logits;
use crate::qwen3_5_moe::expert_paging::route_observation::RouteObservationRecord;

const PREDICTOR_TRAINING_QUEUE_CAPACITY: usize = 64;
const PREDICTOR_SLICE_BUDGET: Duration = Duration::from_millis(4);
const PREDICTOR_EMBEDDING_DIM: usize = 32;
const PREDICTOR_HIDDEN_DIM: usize = 64;
const PREDICTOR_LEARNING_RATE: f32 = 0.05;
const PREDICTOR_SEED: u64 = 1;

/// Background owner: a bounded queue into one trainer thread plus counters
/// the decode path snapshots into attribution.
pub struct ExpertRoutePredictorOwner {
    observation_sender: Option<SyncSender<RouteObservationRecord>>,
    trainer_thread: Option<JoinHandle<()>>,
    predictor: Arc<Mutex<ExpertRoutePredictor>>,
    trained_record_count: Arc<AtomicU64>,
    top_k_hit_count: Arc<AtomicU64>,
    evaluated_expert_count: Arc<AtomicU64>,
    last_slice_nanoseconds: Arc<AtomicU64>,
    training_active: Arc<AtomicBool>,
    last_cpu_predict_nanoseconds: AtomicU64,
    last_ane_predict_nanoseconds: AtomicU64,
    #[cfg(target_os = "macos")]
    ane_engine: Option<astronomical_runtime_integration::PredictorAneEngine>,
}

impl Debug for ExpertRoutePredictorOwner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExpertRoutePredictorOwner")
            .field(
                "trained_record_count",
                &self.trained_record_count.load(Ordering::Relaxed),
            )
            .field(
                "top_k_hit_count",
                &self.top_k_hit_count.load(Ordering::Relaxed),
            )
            .field(
                "evaluated_expert_count",
                &self.evaluated_expert_count.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl ExpertRoutePredictorOwner {
    /// Starts the trainer for a sparse model. Dense models and spawn failures
    /// return `None` so serving continues without a predictor.
    #[must_use]
    pub fn try_start(
        layer_count: usize,
        expert_count: usize,
        vocabulary_size: u32,
        experts_per_token: usize,
    ) -> Option<Self> {
        if layer_count == 0 || expert_count == 0 || vocabulary_size == 0 {
            return None;
        }
        let config = ExpertRoutePredictorConfig {
            layer_count,
            expert_count,
            vocabulary_size,
            embedding_dim: PREDICTOR_EMBEDDING_DIM,
            hidden_dim: PREDICTOR_HIDDEN_DIM,
            learning_rate: PREDICTOR_LEARNING_RATE,
            seed: PREDICTOR_SEED,
        };
        let (observation_sender, observation_receiver) =
            sync_channel(PREDICTOR_TRAINING_QUEUE_CAPACITY);
        let trained_record_count = Arc::new(AtomicU64::new(0));
        let top_k_hit_count = Arc::new(AtomicU64::new(0));
        let evaluated_expert_count = Arc::new(AtomicU64::new(0));
        let last_slice_nanoseconds = Arc::new(AtomicU64::new(0));
        let training_active = Arc::new(AtomicBool::new(false));
        let trained_record_count_for_thread = Arc::clone(&trained_record_count);
        let top_k_hit_count_for_thread = Arc::clone(&top_k_hit_count);
        let evaluated_expert_count_for_thread = Arc::clone(&evaluated_expert_count);
        let last_slice_nanoseconds_for_thread = Arc::clone(&last_slice_nanoseconds);
        let training_active_for_thread = Arc::clone(&training_active);
        let top_k = experts_per_token.max(1);
        let predictor = Arc::new(Mutex::new(ExpertRoutePredictor::new(config)));
        let predictor_for_thread = Arc::clone(&predictor);
        let trainer_thread = thread::Builder::new()
            .name("expert-route-predictor".to_owned())
            .spawn(move || {
                while let Ok(first_record) = observation_receiver.recv() {
                    training_active_for_thread.store(true, Ordering::Relaxed);
                    let slice_started_at = Instant::now();
                    let mut record = first_record;
                    loop {
                        let (hit_count, evaluated_count) = {
                            let Ok(mut predictor) = predictor_for_thread.try_lock() else {
                                break;
                            };
                            evaluate_then_train(&mut predictor, &record, top_k)
                        };
                        top_k_hit_count_for_thread.fetch_add(hit_count, Ordering::Relaxed);
                        evaluated_expert_count_for_thread
                            .fetch_add(evaluated_count, Ordering::Relaxed);
                        trained_record_count_for_thread.fetch_add(1, Ordering::Relaxed);
                        if slice_started_at.elapsed() >= PREDICTOR_SLICE_BUDGET {
                            break;
                        }
                        match observation_receiver.try_recv() {
                            Ok(next_record) => record = next_record,
                            Err(_) => break,
                        }
                    }
                    last_slice_nanoseconds_for_thread.store(
                        u64::try_from(slice_started_at.elapsed().as_nanos()).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                    training_active_for_thread.store(false, Ordering::Relaxed);
                }
            })
            .ok()?;
        Some(Self {
            observation_sender: Some(observation_sender),
            trainer_thread: Some(trainer_thread),
            predictor,
            trained_record_count,
            top_k_hit_count,
            evaluated_expert_count,
            last_slice_nanoseconds,
            training_active,
            last_cpu_predict_nanoseconds: AtomicU64::new(0),
            last_ane_predict_nanoseconds: AtomicU64::new(0),
            #[cfg(target_os = "macos")]
            ane_engine: std::env::var_os("ASTRONOMICAL_EXPERT_ROUTE_PREDICTOR_COREML").and_then(
                |model_path| {
                    astronomical_runtime_integration::PredictorAneEngine::try_load(
                        std::path::Path::new(&model_path),
                    )
                },
            ),
        })
    }

    /// Offers one observation. A full queue drops it rather than blocking decode.
    pub fn try_submit(&self, observation: RouteObservationRecord) {
        let Some(observation_sender) = self.observation_sender.as_ref() else {
            return;
        };
        match observation_sender.try_send(observation) {
            Ok(()) | Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {}
        }
    }

    #[must_use]
    pub fn trained_record_count(&self) -> u64 {
        self.trained_record_count.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn top_k_hit_count(&self) -> u64 {
        self.top_k_hit_count.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn evaluated_expert_count(&self) -> u64 {
        self.evaluated_expert_count.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn last_slice_nanoseconds(&self) -> u64 {
        self.last_slice_nanoseconds.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn last_cpu_predict_nanoseconds(&self) -> u64 {
        self.last_cpu_predict_nanoseconds.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn last_ane_predict_nanoseconds(&self) -> u64 {
        self.last_ane_predict_nanoseconds.load(Ordering::Relaxed)
    }

    /// Snapshot for `/v1/status`. Pages avoided stay 0 until extra SSD prefetch exists.
    #[must_use]
    pub fn program_status(&self) -> PredictorProgramStatus {
        PredictorProgramStatus::from_cpu_counts(
            self.training_active.load(Ordering::Relaxed),
            self.top_k_hit_count(),
            self.evaluated_expert_count(),
            0,
            0,
        )
    }

    /// Ranks leftover experts the next token is predicted to need.
    ///
    /// `try_lock` so decode never waits on a training step. Contention or
    /// poison returns `None` and the warm table keeps its existing policy.
    #[must_use]
    pub fn try_predict_top_experts_per_layer(
        &self,
        token_id: u32,
        previous_token_route: Option<&[Option<Vec<u16>>]>,
        top_k: usize,
    ) -> Option<Vec<Vec<usize>>> {
        let (cpu_logits, packed_head_inputs, layer_count, input_dim, expert_count) = {
            let predictor = match self.predictor.try_lock() {
                Ok(predictor) => predictor,
                Err(TryLockError::WouldBlock | TryLockError::Poisoned(_)) => return None,
            };
            let cpu_started_at = Instant::now();
            let cpu_logits = predictor.forward_logits(token_id, previous_token_route);
            self.last_cpu_predict_nanoseconds.store(
                u64::try_from(cpu_started_at.elapsed().as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            let config = predictor.config();
            (
                cpu_logits,
                predictor.packed_head_inputs(token_id, previous_token_route),
                config.layer_count,
                config.head_input_dim(),
                config.expert_count,
            )
        };
        #[cfg(target_os = "macos")]
        if let Some(ane_engine) = self.ane_engine.as_ref() {
            let ane_started_at = Instant::now();
            if let Some(flat_logits) =
                ane_engine.predict(&packed_head_inputs, layer_count, input_dim, expert_count)
            {
                self.last_ane_predict_nanoseconds.store(
                    u64::try_from(ane_started_at.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                return Some(flat_logits_to_top_k(
                    &flat_logits,
                    layer_count,
                    expert_count,
                    top_k,
                ));
            }
        }
        let _ = (packed_head_inputs, layer_count, input_dim, expert_count);
        Some(
            cpu_logits
                .iter()
                .map(|layer_logits| select_top_expert_ids_from_logits(layer_logits, top_k))
                .collect(),
        )
    }
}

fn flat_logits_to_top_k(
    flat_logits: &[f32],
    layer_count: usize,
    expert_count: usize,
    top_k: usize,
) -> Vec<Vec<usize>> {
    (0..layer_count)
        .map(|layer_index| {
            let start = layer_index * expert_count;
            let layer_logits = &flat_logits[start..start + expert_count];
            select_top_expert_ids_from_logits(layer_logits, top_k)
        })
        .collect()
}

impl Drop for ExpertRoutePredictorOwner {
    fn drop(&mut self) {
        // Disconnect the queue so the trainer's recv returns. Do not join:
        // a trainer stuck on a write lock must not stall model teardown.
        self.observation_sender.take();
        let _ = self.trainer_thread.take();
    }
}
