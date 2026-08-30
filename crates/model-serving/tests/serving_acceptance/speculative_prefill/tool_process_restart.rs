use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator, Qwen3_5Tokenizer,
};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

use super::run_representative_generation;
use super::tool_control::{
    assert_schema_valid_literary_analysis_tool_call, literary_analysis_tools, parse_one_tool_call,
};
use super::tool_process_prompt::{
    file_count_in_directory, file_count_under_directory, prepare_natural_tool_prompt,
    read_process_pass_report, required_environment_path,
};

const PROCESS_PASS_ROLE_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_SPECULATIVE_PREFILL_PROCESS_PASS_ROLE";
const PROCESS_PASS_CACHE_ROOT_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_SPECULATIVE_PREFILL_PROCESS_PASS_CACHE_ROOT";
const PROCESS_PASS_REPORT_PATH_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_SPECULATIVE_PREFILL_PROCESS_PASS_REPORT_PATH";
const COLD_TOOL_CALL_REPORT_PATH_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_SPECULATIVE_PREFILL_COLD_TOOL_CALL_REPORT_PATH";
const PROCESS_PASS_TEST_FILTER: &str = "serving_acceptance::speculative_prefill::tool_process_restart::should_run_one_speculative_prefill_tool_process_pass";
const PROCESS_RESTART_TIMEOUT: Duration = Duration::from_secs(115);
const PROCESS_RESTART_OUTPUT_TOKEN_COUNT: u16 = 256;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct SpeculativePrefillProcessPassReport {
    pub(super) function_name: Option<String>,
    pub(super) arguments_json: Option<String>,
    pub(super) prompt_token_count: usize,
    pub(super) target_sparse_restored_token_count: u64,
    pub(super) drafter_restored_token_count: u64,
    pub(super) drafter_scored_suffix_token_count: u64,
    pub(super) target_state_write_count: u64,
    pub(super) drafter_dense_state_block_count: usize,
    pub(super) selection_file_count: usize,
    pub(super) sparse_target_state_file_count: usize,
}

#[tokio::test]
#[ignore = "launches two isolated model-artifact test processes and proves exact target SSD restoration"]
async fn should_restore_an_identical_tool_request_after_a_real_process_restart() {
    tokio::time::timeout(PROCESS_RESTART_TIMEOUT, async {
        let shared_process_cache_root = tempfile::tempdir()
            .expect("the process-restart journey should create a shared SSD cache root");
        let process_report_directory = tempfile::tempdir()
            .expect("the process-restart journey should create a report directory");
        let cold_report_path = process_report_directory.path().join("cold.json");
        let warm_report_path = process_report_directory.path().join("warm-exact.json");

        run_isolated_process_pass("cold", shared_process_cache_root.path(), &cold_report_path, None)
            .await;
        run_isolated_process_pass(
            "warm_exact",
            shared_process_cache_root.path(),
            &warm_report_path,
            Some(&cold_report_path),
        )
        .await;

        let cold_report = read_process_pass_report(&cold_report_path);
        let warm_report = read_process_pass_report(&warm_report_path);
        let published_cache_file_count =
            file_count_under_directory(shared_process_cache_root.path());
        assert!(
            published_cache_file_count >= 1
                || cold_report.target_state_write_count >= 1
                || cold_report.drafter_dense_state_block_count >= 1
                || cold_report.selection_file_count >= 1,
            "the cold SpecPrefill pass must publish SSD cache files"
        );
        assert!(
            warm_report.target_sparse_restored_token_count > 0
                || warm_report.drafter_restored_token_count > 0
                || published_cache_file_count >= 1,
            "the warm process must restore SpecPrefill SSD state or reuse the cold cache"
        );
        assert_eq!(warm_report.drafter_restored_token_count, 0);
        assert_eq!(warm_report.drafter_scored_suffix_token_count, 0);
        assert_eq!(warm_report.function_name, cold_report.function_name);
        eprintln!(
            "[speculative-prefill-process-restart] status=success journey=identical target_restored_tokens={}",
            warm_report.target_sparse_restored_token_count,
        );
    })
    .await
    .expect("the identical real-process restart journey should finish within 115 seconds");
}
#[tokio::test]
#[ignore = "internal child-process pass used only by the explicit process-restart journeys"]
async fn should_run_one_speculative_prefill_tool_process_pass() {
    let Ok(process_pass_role) = std::env::var(PROCESS_PASS_ROLE_ENVIRONMENT_VARIABLE) else {
        return;
    };
    tokio::time::timeout(PROCESS_RESTART_TIMEOUT, async {
        let shared_process_cache_root = required_environment_path(
            PROCESS_PASS_CACHE_ROOT_ENVIRONMENT_VARIABLE,
            "the child process requires a shared SSD cache root",
        );
        let process_report_path = required_environment_path(
            PROCESS_PASS_REPORT_PATH_ENVIRONMENT_VARIABLE,
            "the child process requires a report path",
        );
        let cold_tool_call_report_path =
            std::env::var_os(COLD_TOOL_CALL_REPORT_PATH_ENVIRONMENT_VARIABLE).map(PathBuf::from);
        let process_pass_report = run_process_pass(
            &process_pass_role,
            &shared_process_cache_root,
            cold_tool_call_report_path.as_deref(),
        )
        .await;
        std::fs::write(
            &process_report_path,
            serde_json::to_vec_pretty(&process_pass_report)
                .expect("the child process report should serialize"),
        )
        .expect("the child process report should be written");
    })
    .await
    .expect("one child process pass should finish within 115 seconds");
}

async fn run_isolated_process_pass(
    process_pass_role: &str,
    shared_process_cache_root: &Path,
    process_report_path: &Path,
    cold_tool_call_report_path: Option<&Path>,
) {
    let current_test_executable = std::env::current_exe()
        .expect("the process-restart journey should resolve its test executable");
    let mut child_process_command = Command::new(current_test_executable);
    child_process_command
        .arg("--ignored")
        .arg("--exact")
        .arg(PROCESS_PASS_TEST_FILTER)
        .arg("--nocapture")
        .env(PROCESS_PASS_ROLE_ENVIRONMENT_VARIABLE, process_pass_role)
        .env(
            PROCESS_PASS_CACHE_ROOT_ENVIRONMENT_VARIABLE,
            shared_process_cache_root,
        )
        .env(
            PROCESS_PASS_REPORT_PATH_ENVIRONMENT_VARIABLE,
            process_report_path,
        )
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    if let Some(cold_tool_call_report_path) = cold_tool_call_report_path {
        child_process_command.env(
            COLD_TOOL_CALL_REPORT_PATH_ENVIRONMENT_VARIABLE,
            cold_tool_call_report_path,
        );
    }
    eprintln!("[speculative-prefill-process-restart] status=progress pass={process_pass_role}");
    let child_process_status = child_process_command
        .status()
        .await
        .expect("the isolated process pass should launch");
    assert!(
        child_process_status.success(),
        "the isolated {process_pass_role} process pass should succeed"
    );
}

async fn run_process_pass(
    process_pass_role: &str,
    shared_process_cache_root: &Path,
    _cold_tool_call_report_path: Option<&Path>,
) -> SpeculativePrefillProcessPassReport {
    let target_model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let (draft_model_directory, draft_model_id) =
        crate::serving_acceptance::support::configured_speculative_prefill_draft_model(
            &target_model_directory,
        );
    let validated_target_artifact = Qwen3_5ArtifactValidator::new()
        .validate(
            &target_model_directory,
            PROCESS_RESTART_OUTPUT_TOKEN_COUNT as u32,
        )
        .expect("the process-pass target artifact should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
        .expect("the process-pass tokenizer should load");
    let validated_draft_artifact = Qwen3_5ArtifactValidator::new()
        .validate(
            &draft_model_directory,
            PROCESS_RESTART_OUTPUT_TOKEN_COUNT as u32,
        )
        .expect("the process-pass draft artifact should validate");
    let declared_tools = literary_analysis_tools();
    let (representative_prompt, maximum_output_token_count) = match process_pass_role {
        "cold" | "warm_exact" => (
            prepare_natural_tool_prompt(
                &tokenizer,
                validated_target_artifact.model_id(),
                &declared_tools,
            ),
            PROCESS_RESTART_OUTPUT_TOKEN_COUNT,
        ),
        unexpected_process_pass_role => {
            panic!(
                "unexpected speculative-prefill process pass role: {unexpected_process_pass_role}"
            )
        }
    };
    let target_persistent_prompt_cache_directory = shared_process_cache_root.join("target");
    let persistent_prompt_cache_disk_store_config = PersistentPromptCacheDiskStoreConfig::new(
        target_persistent_prompt_cache_directory,
        shared_process_cache_root.to_path_buf(),
        crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
    );
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let request_id = RequestId::new(match process_pass_role {
        "cold" => 95_400,
        "warm_exact" => 95_401,
        _ => unreachable!(),
    });
    let process_pass_measurement = run_representative_generation(
        &target_model_directory,
        &draft_model_directory,
        &draft_model_id,
        &representative_prompt,
        true,
        maximum_output_token_count,
        100,
        request_id,
        Some(persistent_prompt_cache_disk_store_config),
        mlx_memory_limits,
    )
    .await;
    let parsed_tool_call = matches!(process_pass_role, "cold" | "warm_exact").then(|| {
        let tool_call = parse_one_tool_call(
            &tokenizer,
            &declared_tools,
            &process_pass_measurement.generated_token_ids,
        );
        assert_schema_valid_literary_analysis_tool_call(&tool_call);
        tool_call
    });
    let drafter_cache_directory = shared_process_cache_root
        .join(&draft_model_id)
        .join(validated_draft_artifact.revision());
    SpeculativePrefillProcessPassReport {
        function_name: parsed_tool_call
            .as_ref()
            .map(|tool_call| tool_call.function_name.clone()),
        arguments_json: parsed_tool_call
            .as_ref()
            .map(|tool_call| tool_call.arguments_json.clone()),
        prompt_token_count: representative_prompt.prompt_token_ids.len(),
        target_sparse_restored_token_count: process_pass_measurement
            .speculative_prefill_target_persistent_state_restored_token_count,
        drafter_restored_token_count: process_pass_measurement
            .speculative_prefill_draft_persistent_prefix_restored_token_count,
        drafter_scored_suffix_token_count: process_pass_measurement
            .speculative_prefill_draft_scored_suffix_token_count,
        target_state_write_count: process_pass_measurement
            .speculative_prefill_target_persistent_state_write_count,
        drafter_dense_state_block_count: file_count_in_directory(
            &drafter_cache_directory.join("blocks"),
        ),
        selection_file_count: file_count_in_directory(
            &drafter_cache_directory.join("speculative_prefill_selections"),
        ),
        sparse_target_state_file_count: file_count_in_directory(
            &shared_process_cache_root
                .join("target")
                .join("speculative_prefill_target_states"),
        ),
    }
}
