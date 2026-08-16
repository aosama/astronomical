use astronomical_model_serving::{
    MtpDraftDepth, Qwen3_5MtpArtifactCapability, Qwen3_5MtpContract, Qwen3_5MtpContractError,
    qwen3_5_mtp_tensor_names,
};

use crate::common::qwen3_5_moe::certified_ornith_config;

const SUPPORTED_CONTRACT: &str = r#"{
  "base_hidden_variant": "post_norm",
  "hidden_variant": "post_norm",
  "concat_order": "embedding_hidden",
  "mtp_position_mode": "local"
}"#;

fn config_with_contract(contract: &str) -> Vec<u8> {
    format!(r#"{{"mtplx_mtp_contract":{contract},"ignored_path":"/fictional-publisher/source"}}"#)
        .into_bytes()
}

#[test]
fn should_parse_agreeing_config_and_runtime_contract_metadata() {
    let config_bytes = config_with_contract(SUPPORTED_CONTRACT);
    let runtime_bytes = format!(
        r#"{{"arch_id":"qwen3-next-mtp","mtplx_version":"2.3.0","mtp_contract":{SUPPORTED_CONTRACT},"mtp_depth_default":2,"mtp_depth_max":3,"source_repo":"/fictional-publisher/repository","speed_evidence":{{"tok_s":99}}}}"#,
    );

    let contract = Qwen3_5MtpContract::parse(&config_bytes, Some(runtime_bytes.as_bytes()))
        .expect("the supported contract should parse");

    assert_eq!(
        contract.artifact_maximum_depth().map(MtpDraftDepth::get),
        Some(3)
    );
    assert_eq!(
        contract.artifact_default_depth().map(MtpDraftDepth::get),
        Some(2)
    );
    assert_eq!(contract.architecture_id(), Some("qwen3-next-mtp"));
    assert_eq!(contract.runtime_version(), Some("2.3.0"));
    assert!(!format!("{contract:?}").contains("fictional-publisher"));
}

#[test]
fn should_reject_disagreeing_duplicate_contract_fields_without_retaining_foreign_metadata() {
    let config_bytes = config_with_contract(SUPPORTED_CONTRACT);
    let runtime_bytes = br#"{
      "mtp_contract": {
        "base_hidden_variant": "pre_norm",
        "hidden_variant": "post_norm",
        "concat_order": "embedding_hidden",
        "mtp_position_mode": "local"
      },
      "source_repo": "/fictional-publisher/repository"
    }"#;

    let error = Qwen3_5MtpContract::parse(&config_bytes, Some(runtime_bytes))
        .expect_err("disagreeing selected fields must reject optional MTP");

    assert_eq!(error, Qwen3_5MtpContractError::FieldDisagreement);
    assert!(!error.to_string().contains("fictional-publisher"));
}

#[test]
fn should_reject_malformed_incompatible_and_oversized_optional_runtime_contracts() {
    let config_bytes = config_with_contract(SUPPORTED_CONTRACT);
    let incompatible_runtime = br#"{"mtp_contract":{"base_hidden_variant":"pre_norm","hidden_variant":"post_norm","concat_order":"embedding_hidden","mtp_position_mode":"local"}}"#;
    let oversized_runtime = vec![b' '; 64 * 1024 + 1];

    assert_eq!(
        Qwen3_5MtpContract::parse(&config_bytes, Some(b"{"))
            .expect_err("malformed runtime JSON must reject optional MTP"),
        Qwen3_5MtpContractError::Malformed
    );
    assert_eq!(
        Qwen3_5MtpContract::parse(b"{}", Some(incompatible_runtime))
            .expect_err("an unsupported hidden contract must reject optional MTP"),
        Qwen3_5MtpContractError::Incompatible
    );
    assert_eq!(
        Qwen3_5MtpContract::parse(&config_bytes, Some(&oversized_runtime))
            .expect_err("an oversized runtime contract must reject optional MTP"),
        Qwen3_5MtpContractError::RuntimeDocumentTooLarge
    );
}

#[test]
fn should_reject_a_declared_foreign_mtp_architecture_identity() {
    let config_bytes = config_with_contract(SUPPORTED_CONTRACT);
    let foreign_runtime =
        format!(r#"{{"arch_id":"fictional-foreign-mtp","mtp_contract":{SUPPORTED_CONTRACT}}}"#,);

    let error = Qwen3_5MtpContract::parse(&config_bytes, Some(foreign_runtime.as_bytes()))
        .expect_err("a foreign architecture identity must not authorize Qwen MTP arithmetic");

    assert_eq!(error, Qwen3_5MtpContractError::Incompatible);
}

#[test]
fn should_allow_missing_contract_metadata_for_known_compatible_physical_weights() {
    let contract = Qwen3_5MtpContract::parse(b"{}", None)
        .expect("missing optional metadata must not reject target or MTP weights");

    assert_eq!(contract.artifact_maximum_depth(), None);
    assert_eq!(contract.artifact_default_depth(), None);
}

#[test]
fn should_validate_mtp_draft_depth_boundaries() {
    assert!(MtpDraftDepth::new(0).is_err());
    assert_eq!(MtpDraftDepth::new(1).expect("depth one is valid").get(), 1);
    assert_eq!(MtpDraftDepth::new(2).expect("depth two is valid").get(), 2);
    assert_eq!(
        MtpDraftDepth::new(3).expect("depth three is valid").get(),
        3
    );
    assert!(MtpDraftDepth::new(4).is_err());
}

#[test]
fn should_resolve_complete_known_qwen_inventory_with_named_contract_depth_safely() {
    let config = certified_ornith_config();
    let config_bytes = config_with_contract(SUPPORTED_CONTRACT);
    let runtime_bytes = format!(
        r#"{{"arch_id":"qwen3-next-mtp","mtplx_version":"2.3.0","mtp_contract":{SUPPORTED_CONTRACT},"mtp_depth_max":9}}"#,
    );
    let contract = Qwen3_5MtpContract::parse(&config_bytes, Some(runtime_bytes.as_bytes()))
        .expect("the named contract should parse");

    let capability = Qwen3_5MtpArtifactCapability::from_canonical_tensor_names(
        &config,
        qwen3_5_mtp_tensor_names(&config),
        Some(&contract),
    );

    assert!(matches!(
        capability,
        Qwen3_5MtpArtifactCapability::MtpCapable {
            stored_mtp_layer_count: 1,
            artifact_maximum_draft_depth,
            artifact_default_draft_depth: None,
            ..
        } if artifact_maximum_draft_depth.get() == 3
    ));
}
