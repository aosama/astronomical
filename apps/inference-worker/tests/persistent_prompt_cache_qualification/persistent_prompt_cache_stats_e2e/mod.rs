use std::{net::SocketAddr, time::Duration};

use astronomical_supervisor::{
    GenerationPerformanceLog, WorkerHandle, build_application_with_discovered_models,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    time::{Instant, MissedTickBehavior, interval, sleep},
};

use super::cache_stats_worker_launcher::create_cache_stats_worker_configuration;
use crate::common::discovered_model_artifact;

pub(super) const MODEL_ID: &str = crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID;
const READY_ATTEMPT_LIMIT: u8 = 70;
const PROGRESS_INTERVAL_SECONDS: u64 = 5;
const STATUS_POLL_INTERVAL_MILLIS: u64 = 2_000;

const FIVE_THOUSAND_WORD_ROMEO_AND_JULIET_PROMPT: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

mod http_transport;
mod live_progress;
mod scenario;

use scenario::{five_thousand_word_case, run_cache_stats_e2e_with_timeout};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads Ornith and exercises the persistent prompt cache end-to-end with a 5K-word Romeo and Juliet prompt"]
async fn should_observe_a_cache_miss_then_a_cache_hit_with_5k_romeo_and_juliet_words() {
    run_cache_stats_e2e_with_timeout(five_thousand_word_case()).await;
}
