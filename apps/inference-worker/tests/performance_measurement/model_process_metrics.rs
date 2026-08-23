use std::time::Duration;

use tokio::{process::Command, time::timeout};

const COMMAND_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Copy)]
pub(crate) struct WorkerPhysicalFootprint {
    pub(crate) current_bytes: u64,
    pub(crate) peak_bytes: u64,
}

pub(crate) async fn find_worker_process_id() -> u32 {
    for _process_lookup_attempt in 1..=20 {
        let parent_process_id = std::process::id().to_string();
        let process_lookup_output = timeout(
            COMMAND_TIMEOUT,
            Command::new("/usr/bin/pgrep")
                .args([
                    "-P",
                    &parent_process_id,
                    "-f",
                    "astronomical-inference-worker",
                ])
                .output(),
        )
        .await
        .expect("the worker process lookup should not time out")
        .expect("pgrep should start");
        if process_lookup_output.status.success() {
            let process_id_text = String::from_utf8(process_lookup_output.stdout)
                .expect("pgrep output should contain UTF-8");
            let process_ids = process_id_text
                .split_whitespace()
                .map(|process_id| {
                    process_id
                        .parse::<u32>()
                        .expect("pgrep should report numeric process identifiers")
                })
                .collect::<Vec<_>>();
            if let [worker_process_id] = process_ids.as_slice() {
                return *worker_process_id;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("the metrics harness could not identify the inference-worker child process");
}

pub(crate) async fn measure_worker_physical_footprint(
    worker_process_id: u32,
) -> Result<WorkerPhysicalFootprint, String> {
    let footprint_output = timeout(
        COMMAND_TIMEOUT,
        Command::new("/usr/bin/footprint")
            .args([
                "--pid",
                &worker_process_id.to_string(),
                "--format",
                "bytes",
                "--noCategories",
            ])
            .output(),
    )
    .await
    .map_err(|_| "footprint timed out".to_owned())?
    .map_err(|command_error| format!("footprint failed to start: {command_error}"))?;
    if !footprint_output.status.success() {
        return Err(format!(
            "footprint exited unsuccessfully: {}",
            String::from_utf8_lossy(&footprint_output.stderr)
        ));
    }
    let footprint_text = String::from_utf8(footprint_output.stdout)
        .map_err(|encoding_error| format!("footprint output is not UTF-8: {encoding_error}"))?;
    Ok(WorkerPhysicalFootprint {
        current_bytes: parse_footprint_bytes(&footprint_text, "phys_footprint:")?,
        peak_bytes: parse_footprint_bytes(&footprint_text, "phys_footprint_peak:")?,
    })
}

fn parse_footprint_bytes(footprint_text: &str, metric_prefix: &str) -> Result<u64, String> {
    footprint_text
        .lines()
        .find_map(|footprint_line| {
            footprint_line
                .trim()
                .strip_prefix(metric_prefix)
                .and_then(|byte_text| byte_text.trim().strip_suffix(" B"))
                .map(|byte_text| byte_text.replace(',', ""))
        })
        .ok_or_else(|| format!("footprint output did not report {metric_prefix}"))?
        .parse::<u64>()
        .map_err(|parse_error| {
            format!("footprint bytes are not an unsigned integer: {parse_error}")
        })
}
