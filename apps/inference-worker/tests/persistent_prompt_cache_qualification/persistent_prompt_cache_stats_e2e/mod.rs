use std::{net::SocketAddr, time::Duration};

use astronomical_supervisor::{
    GenerationPerformanceLog, WorkerHandle,
    build_application_with_config_warning_and_discovered_models,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    time::{Instant, MissedTickBehavior, interval, sleep},
};

use super::cache_stats_worker_launcher::create_cache_stats_worker_configuration;
use crate::common::{discovered_model_artifact, single_model_directories};

pub(super) const MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
const READY_ATTEMPT_LIMIT: u8 = 70;
const PROGRESS_INTERVAL_SECONDS: u64 = 5;
const STATUS_POLL_INTERVAL_MILLIS: u64 = 2_000;

const TWO_THOUSAND_WORD_PROMPT: &str =
    include_str!("../persistent_prompt_cache_stats_e2e_prompt.txt");

mod http_transport;
mod live_progress;
mod scenario;

use scenario::{run_cache_stats_e2e_with_timeout, two_thousand_word_case};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the complete local 22 GB model and exercises the persistent prompt cache end-to-end with a 2K-word prompt"]
async fn should_observe_a_cache_miss_then_a_cache_hit_with_2k_words() {
    run_cache_stats_e2e_with_timeout(two_thousand_word_case()).await;
}
