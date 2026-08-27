use std::fs;
use std::path::Path;

use astronomical_model_serving::{
    Qwen3_5MtpSidecarDeclaration, Qwen3_5MtpSidecarValidationError, TensorDtype, TensorProfile,
    validate_qwen3_5_mtp_sidecar_for_tests, validate_qwen3_5_mtp_sidecar_result_for_tests,
};
use tempfile::TempDir;

fn mtp_profiles() -> Vec<TensorProfile> {
    vec![
        TensorProfile {
            name: "language_model.mtp.proj.weight".to_owned(),
            dtype: TensorDtype::UInt32,
            shape: vec![1],
        },
        TensorProfile {
            name: "language_model.mtp.proj.scales".to_owned(),
            dtype: TensorDtype::AffineQuantizationFloat,
            shape: vec![1],
        },
        TensorProfile {
            name: "language_model.mtp.proj.biases".to_owned(),
            dtype: TensorDtype::AffineQuantizationFloat,
            shape: vec![1],
        },
    ]
}

fn write_sidecar(model_directory: &Path, relative_path: &str, tensors: &[(&str, &str, &[u8])]) {
    let sidecar_path = model_directory.join(relative_path);
    if let Some(parent_directory) = sidecar_path.parent() {
        fs::create_dir_all(parent_directory).expect("the sidecar parent should be created");
    }
    let mut payload = Vec::new();
    let mut entries = Vec::new();
    for (stored_name, dtype, tensor_payload) in tensors {
        let start_offset = payload.len();
        payload.extend_from_slice(tensor_payload);
        entries.push(format!(
            r#""{stored_name}":{{"dtype":"{dtype}","shape":[1],"data_offsets":[{start_offset},{}]}}"#,
            payload.len()
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(header.as_bytes());
    file_bytes.extend_from_slice(&payload);
    fs::write(sidecar_path, file_bytes).expect("the generated sidecar should be written");
}

fn complete_sidecar_tensors() -> Vec<(&'static str, &'static str, &'static [u8])> {
    vec![
        ("mtp.proj.weight", "U32", &[0, 0, 0, 0]),
        ("mtp.proj.scales", "BF16", &[0, 0]),
        ("mtp.proj.biases", "BF16", &[0, 0]),
    ]
}

fn write_raw_sidecar(model_directory: &Path, header: &str, payload: &[u8]) {
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(header.as_bytes());
    file_bytes.extend_from_slice(payload);
    fs::write(model_directory.join("mtp.safetensors"), file_bytes)
        .expect("the raw sidecar should be written");
}

#[test]
fn should_accept_root_and_nested_qwen_mtp_sidecars_with_exact_accounting() {
    for relative_path in ["mtp.safetensors", "optiq/mtp.safetensors"] {
        let model_directory = TempDir::new().expect("the model directory should be created");
        write_sidecar(
            model_directory.path(),
            relative_path,
            &complete_sidecar_tensors(),
        );
        let declaration = Qwen3_5MtpSidecarDeclaration::parse(relative_path)
            .expect("the safe Qwen sidecar declaration should parse");

        let outcome = validate_qwen3_5_mtp_sidecar_for_tests(
            model_directory.path(),
            &declaration,
            &mtp_profiles(),
            &[],
        );

        assert!(outcome.is_available());
        assert_eq!(outcome.source_count(), 1);
        assert_eq!(outcome.tensor_count(), 3);
        assert_eq!(outcome.payload_bytes(), 8);
        assert_eq!(
            outcome.stored_name("language_model.mtp.proj.weight"),
            Some("mtp.proj.weight")
        );
    }
}

#[test]
fn should_preserve_target_only_for_missing_malformed_partial_or_wrong_dtype_sidecars() {
    let scenarios: Vec<(&str, Vec<(&str, &str, &[u8])>)> = vec![
        ("malformed", Vec::new()),
        ("partial", vec![("mtp.proj.weight", "U32", &[0, 0, 0, 0])]),
        (
            "wrong-dtype",
            vec![
                ("mtp.proj.weight", "BF16", &[0, 0]),
                ("mtp.proj.scales", "BF16", &[0, 0]),
                ("mtp.proj.biases", "BF16", &[0, 0]),
            ],
        ),
    ];
    for (scenario_name, tensors) in scenarios {
        let model_directory = TempDir::new().expect("the model directory should be created");
        if scenario_name == "malformed" {
            fs::write(model_directory.path().join("mtp.safetensors"), b"invalid")
                .expect("the malformed sidecar should be written");
        } else {
            write_sidecar(model_directory.path(), "mtp.safetensors", &tensors);
        }
        let declaration = Qwen3_5MtpSidecarDeclaration::parse("mtp.safetensors")
            .expect("the safe declaration should parse");
        let outcome = validate_qwen3_5_mtp_sidecar_for_tests(
            model_directory.path(),
            &declaration,
            &mtp_profiles(),
            &[],
        );
        assert!(!outcome.is_available(), "scenario {scenario_name}");
        assert_eq!(outcome.source_count(), 0, "scenario {scenario_name}");
        assert_eq!(outcome.payload_bytes(), 0, "scenario {scenario_name}");
    }

    let missing_directory = TempDir::new().expect("the model directory should be created");
    let declaration = Qwen3_5MtpSidecarDeclaration::parse("missing.safetensors")
        .expect("the safe missing declaration should parse");
    let missing_outcome = validate_qwen3_5_mtp_sidecar_for_tests(
        missing_directory.path(),
        &declaration,
        &mtp_profiles(),
        &[],
    );
    assert!(!missing_outcome.is_available());
}

#[test]
fn should_accept_sidecar_with_extra_undeclared_tensors_as_future_extensibility() {
    let model_directory = TempDir::new().expect("the model directory should be created");
    write_sidecar(
        model_directory.path(),
        "mtp.safetensors",
        &[
            ("mtp.proj.weight", "U32", &[0, 0, 0, 0]),
            ("mtp.proj.scales", "BF16", &[0, 0]),
            ("mtp.proj.biases", "BF16", &[0, 0]),
            ("mtp.unexpected", "BF16", &[0, 0]),
        ],
    );
    let declaration = Qwen3_5MtpSidecarDeclaration::parse("mtp.safetensors")
        .expect("the safe declaration should parse");
    let outcome = validate_qwen3_5_mtp_sidecar_for_tests(
        model_directory.path(),
        &declaration,
        &mtp_profiles(),
        &[],
    );
    assert!(
        outcome.is_available(),
        "an extra undeclared tensor must not reject the sidecar"
    );
    assert_eq!(
        outcome.tensor_count(),
        3,
        "only profiled tensors enter the inventory"
    );
}

#[test]
fn should_report_structured_sidecar_validation_errors() {
    // A dtype mismatch yields a ProfileValidationFailed diagnostic with a human-readable cause.
    let wrong_dtype_directory = TempDir::new().expect("the model directory should be created");
    write_sidecar(
        wrong_dtype_directory.path(),
        "mtp.safetensors",
        &[
            ("mtp.proj.weight", "BF16", &[0, 0]),
            ("mtp.proj.scales", "BF16", &[0, 0]),
            ("mtp.proj.biases", "BF16", &[0, 0]),
        ],
    );
    let declaration = Qwen3_5MtpSidecarDeclaration::parse("mtp.safetensors")
        .expect("the safe declaration should parse");
    let wrong_dtype_error = validate_qwen3_5_mtp_sidecar_result_for_tests(
        wrong_dtype_directory.path(),
        &declaration,
        &mtp_profiles(),
        &[],
    )
    .expect_err("a dtype mismatch must fail validation");
    assert!(
        matches!(
            wrong_dtype_error,
            Qwen3_5MtpSidecarValidationError::ProfileValidationFailed { ref tensor_name, .. }
                if tensor_name == "language_model.mtp.proj.weight"
        ),
        "unexpected error: {wrong_dtype_error:?}"
    );
    assert!(
        wrong_dtype_error.to_string().contains("dtype mismatch"),
        "diagnostic was: {}",
        wrong_dtype_error
    );

    // A partial sidecar (missing profile tensors) yields a MissingProfileTensor diagnostic.
    let partial_directory = TempDir::new().expect("the model directory should be created");
    write_sidecar(
        partial_directory.path(),
        "mtp.safetensors",
        &[("mtp.proj.weight", "U32", &[0, 0, 0, 0])],
    );
    let declaration = Qwen3_5MtpSidecarDeclaration::parse("mtp.safetensors")
        .expect("the safe declaration should parse");
    let partial_error = validate_qwen3_5_mtp_sidecar_result_for_tests(
        partial_directory.path(),
        &declaration,
        &mtp_profiles(),
        &[],
    )
    .expect_err("a partial sidecar must fail validation");
    assert!(
        matches!(
            partial_error,
            Qwen3_5MtpSidecarValidationError::MissingProfileTensor { ref tensor_name }
                if tensor_name == "language_model.mtp.proj.scales"
        ),
        "unexpected error: {partial_error:?}"
    );
    assert!(
        partial_error.to_string().contains("not found in sidecar"),
        "diagnostic was: {}",
        partial_error
    );
}

#[test]
fn should_reject_unsafe_qwen_sidecar_paths_before_filesystem_access() {
    for unsafe_path in [
        "",
        "/mtp.safetensors",
        "../mtp.safetensors",
        "weights/../mtp.safetensors",
        "./mtp.safetensors",
        "weights//mtp.safetensors",
        "C:\\mtp.safetensors",
        "mtp.bin",
    ] {
        assert!(
            Qwen3_5MtpSidecarDeclaration::parse(unsafe_path).is_err(),
            "unsafe path {unsafe_path:?}"
        );
    }
}

#[test]
fn should_disable_optional_mtp_when_embedded_and_sidecar_canonical_names_collide() {
    let model_directory = TempDir::new().expect("the model directory should be created");
    write_sidecar(
        model_directory.path(),
        "mtp.safetensors",
        &complete_sidecar_tensors(),
    );
    let declaration = Qwen3_5MtpSidecarDeclaration::parse("mtp.safetensors")
        .expect("the safe declaration should parse");
    let outcome = validate_qwen3_5_mtp_sidecar_for_tests(
        model_directory.path(),
        &declaration,
        &mtp_profiles(),
        &["language_model.mtp.proj.weight".to_owned()],
    );

    assert!(!outcome.is_available());
    assert_eq!(outcome.source_count(), 0);
    assert_eq!(outcome.tensor_count(), 0);
}

#[test]
fn should_preserve_target_only_for_wrong_shape_and_invalid_offsets() {
    let wrong_shape_directory = TempDir::new().expect("the model directory should be created");
    write_raw_sidecar(
        wrong_shape_directory.path(),
        r#"{"mtp.proj.weight":{"dtype":"U32","shape":[2],"data_offsets":[0,8]},"mtp.proj.scales":{"dtype":"BF16","shape":[1],"data_offsets":[8,10]},"mtp.proj.biases":{"dtype":"BF16","shape":[1],"data_offsets":[10,12]}}"#,
        &[0; 12],
    );
    let invalid_offset_directory =
        TempDir::new().expect("the invalid-offset model directory should be created");
    write_raw_sidecar(
        invalid_offset_directory.path(),
        r#"{"mtp.proj.weight":{"dtype":"U32","shape":[1],"data_offsets":[1,5]},"mtp.proj.scales":{"dtype":"BF16","shape":[1],"data_offsets":[5,7]},"mtp.proj.biases":{"dtype":"BF16","shape":[1],"data_offsets":[7,9]}}"#,
        &[0; 9],
    );
    let declaration = Qwen3_5MtpSidecarDeclaration::parse("mtp.safetensors")
        .expect("the safe declaration should parse");

    for model_directory in [
        wrong_shape_directory.path(),
        invalid_offset_directory.path(),
    ] {
        let outcome = validate_qwen3_5_mtp_sidecar_for_tests(
            model_directory,
            &declaration,
            &mtp_profiles(),
            &[],
        );
        assert!(!outcome.is_available());
        assert_eq!(outcome.source_count(), 0);
    }
}

#[test]
fn should_preserve_target_only_when_an_ordinary_sidecar_symlink_escapes_the_model_directory() {
    use std::os::unix::fs::symlink;

    let model_directory = TempDir::new().expect("the model directory should be created");
    let outside_directory = TempDir::new().expect("the outside directory should be created");
    write_sidecar(
        outside_directory.path(),
        "outside.safetensors",
        &complete_sidecar_tensors(),
    );
    symlink(
        outside_directory.path().join("outside.safetensors"),
        model_directory.path().join("mtp.safetensors"),
    )
    .expect("the escape symlink should be created");
    let declaration = Qwen3_5MtpSidecarDeclaration::parse("mtp.safetensors")
        .expect("the safe declaration should parse");

    let outcome = validate_qwen3_5_mtp_sidecar_for_tests(
        model_directory.path(),
        &declaration,
        &mtp_profiles(),
        &[],
    );

    assert!(!outcome.is_available());
    assert_eq!(outcome.source_count(), 0);
}
