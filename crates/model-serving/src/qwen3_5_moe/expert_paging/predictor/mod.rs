//! On-device expert-route predictor (issue #538).
//!
//! A tiny pure-Rust network predicts, for every sparse decoder layer, which
//! experts the current token will route to. Inputs are the token identifier
//! and the previous token's routed-expert bitmap; outputs are per-layer expert
//! logits trained against the true routes the native router already selected
//! (#536's observation history). Training is hand-rolled stochastic gradient
//! descent on host `f32` arrays with a strictly bounded per-slice wall-time
//! budget, so the trainer can never slow generation.
//!
//! Everything here is CPU-only and dependency-free by design: the Apple Neural
//! Engine cannot host an online trainer (compiled programs bake weights, and
//! every update forces a recompile), and Core ML's update task is batch
//! personalization with model-file replacement. In-place Rust weight updates
//! are the only architecture that fits a per-token feedback loop.
//!
//! Layout: `network.rs` owns weights, forward, backward, and the SGD update;
//! `trainer.rs` owns the budget-bounded drain of the observation ring;
//! `evaluation.rs` owns accuracy measurement. Predictor weights never enter
//! any MLX memory report.

mod evaluation;
mod gradients;
mod network;
mod ops;
mod serving;
mod trainer;

pub use evaluation::{PredictorLayerAccuracy, evaluate_predictor_accuracy};
pub use network::{ExpertRoutePredictor, ExpertRoutePredictorConfig};
pub use serving::ExpertRoutePredictorOwner;
pub use trainer::{TrainerSliceOutcome, evaluate_then_train, train_predictor_slice};
