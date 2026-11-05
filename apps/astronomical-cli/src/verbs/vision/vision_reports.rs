//! Plain-language reports for `astronomical vision`. They say what a finding means
//! for someone deciding what to do, because tensor counts alone do not.

use std::io::Write;

use astronomical_vision_tower::{
    BaseMatchReport, BaseMatchVerdict, TensorAgreement, TowerAdditionReport, TowerInspection,
    TowerStatus, TowerSummary,
};

const BYTES_PER_GIBIBYTE: f64 = 1_073_741_824.0;
const PARAMETERS_PER_MILLION: f64 = 1_000_000.0;
const SHORT_REVISION_LENGTH: usize = 7;

pub fn render_inspection_text(
    inspection: &TowerInspection,
    output: &mut dyn Write,
) -> std::io::Result<()> {
    writeln!(
        output,
        "{}",
        artifact_heading(&inspection.artifact, inspection.pinned_revision.as_deref())
    )?;
    writeln!(
        output,
        "Vision tower: {}",
        status_sentence(inspection.status)
    )?;
    if let Some(tower) = &inspection.tower {
        render_tower_details(inspection, tower, output)?;
    }
    writeln!(
        output,
        "Weights: {} tensors, {:.2} GiB, in {} {}",
        inspection.total_tensor_count,
        inspection.total_weight_bytes as f64 / BYTES_PER_GIBIBYTE,
        inspection.weight_file_count,
        plural_files(inspection.weight_file_count),
    )?;
    if let Some(model_type) = &inspection.config.model_type {
        writeln!(output, "Model type: {model_type}")?;
    }
    if let Some(license) = &inspection.license {
        writeln!(output, "License: {license}")?;
    }
    Ok(())
}

fn render_tower_details(
    inspection: &TowerInspection,
    tower: &TowerSummary,
    output: &mut dyn Write,
) -> std::io::Result<()> {
    writeln!(
        output,
        "  Size: {:.1}M parameters, {:.2} GiB",
        tower.parameter_count as f64 / PARAMETERS_PER_MILLION,
        tower.byte_count as f64 / BYTES_PER_GIBIBYTE,
    )?;
    writeln!(
        output,
        "  Blocks: {}, {} tensors",
        tower.block_count, tower.tensor_count
    )?;
    let mut precision_names = Vec::with_capacity(tower.data_type_counts.len());
    for data_type in tower.data_type_counts.keys() {
        precision_names.push(data_type.as_str());
    }
    writeln!(output, "  Precision: {}", precision_names.join(", "))?;
    if let Some(vision_config) = &inspection.config.vision
        && let (Some(tower_width), Some(text_width)) = (
            vision_config.output_hidden_size,
            inspection.config.text_hidden_size,
        )
    {
        writeln!(
            output,
            "  Output width: {tower_width}, text width: {text_width}"
        )?;
    }
    writeln!(output, "  Stored in: {}", tower.weight_files.join(", "))?;
    writeln!(
        output,
        "  Tensor names start with: {}",
        tower.name_prefixes.join(", ")
    )?;
    if tower.is_quantized {
        writeln!(
            output,
            "  Note: this tower is quantized, and Astronomical only serves unquantized towers."
        )?;
    }
    if tower.is_in_weight_index == Some(false) {
        writeln!(
            output,
            "  Note: model.safetensors.index.json does not list this tower, so loaders that go by the index will miss it."
        )?;
    }
    Ok(())
}

const fn status_sentence(status: TowerStatus) -> &'static str {
    match status {
        TowerStatus::Present => "yes",
        TowerStatus::Partial => {
            "partial. Tower weights exist, but the config is missing or disagrees with them."
        }
        TowerStatus::AbsentWithVisionConfig => {
            "none. The config describes one, but no weight file holds it, so this model reads text only."
        }
        TowerStatus::AbsentWithImageTokens => {
            "none. The config lists image tokens, but no weight file holds a tower, so this model reads text only."
        }
        TowerStatus::Absent => "none. Nothing in this model refers to images.",
    }
}

pub fn render_match_text(report: &BaseMatchReport, output: &mut dyn Write) -> std::io::Result<()> {
    writeln!(
        output,
        "Text:  {}",
        artifact_heading(&report.text_artifact, report.text_revision.as_deref())
    )?;
    writeln!(
        output,
        "Tower: {}",
        artifact_heading(&report.tower_artifact, report.tower_revision.as_deref())
    )?;
    writeln!(output, "Verdict: {}", verdict_sentence(report.verdict))?;
    for incompatibility in &report.incompatibilities {
        writeln!(output, "  - {incompatibility}")?;
    }
    if !report.compared_tensors.is_empty() {
        writeln!(output, "Evidence: {}", evidence_sentence(report))?;
    }
    if !report.unchecked.is_empty() {
        writeln!(
            output,
            "Not stated in the configs, so not checked: {}",
            report.unchecked.join(", ")
        )?;
    }
    if report.text_already_has_tower {
        writeln!(
            output,
            "Note: the text model already has a vision tower of its own."
        )?;
    }
    Ok(())
}

pub fn render_addition_text(
    report: &TowerAdditionReport,
    output: &mut dyn Write,
) -> std::io::Result<()> {
    writeln!(output, "Built: {}", report.output_directory.display())?;
    writeln!(
        output,
        "Tower: {} tensors, {:.2} GiB, from {}",
        report.tower_tensor_count,
        report.tower_byte_count as f64 / BYTES_PER_GIBIBYTE,
        artifact_heading(
            &report.base_match.tower_artifact,
            report.base_match.tower_revision.as_deref()
        ),
    )?;
    writeln!(
        output,
        "Text weights: unchanged, from {}",
        artifact_heading(
            &report.base_match.text_artifact,
            report.base_match.text_revision.as_deref()
        ),
    )?;
    writeln!(
        output,
        "Verdict: {}",
        verdict_sentence(report.base_match.verdict)
    )?;
    writeln!(
        output,
        "Evidence: {}",
        evidence_sentence(&report.base_match)
    )?;
    writeln!(
        output,
        "Total: {} files, {:.2} GiB, hashes in astronomical-build.json",
        report.file_count,
        report.total_byte_count as f64 / BYTES_PER_GIBIBYTE,
    )?;
    if !report.missing_image_settings.is_empty() {
        writeln!(
            output,
            "Note: neither source had {}, which some loaders need for images.",
            report.missing_image_settings.join(", ")
        )?;
    }
    writeln!(
        output,
        "Next: test it with real images, then add a model card before publishing."
    )
}

const fn verdict_sentence(verdict: BaseMatchVerdict) -> &'static str {
    match verdict {
        BaseMatchVerdict::SameBase => {
            "same base. These came from the same weights, so this tower belongs with this text model."
        }
        BaseMatchVerdict::DerivedFromBase => {
            "derived. The text weights were edited from this base, so the tower probably fits, but test it with real images."
        }
        BaseMatchVerdict::NoSharedWeights => {
            "no shared weights. The shapes fit, but these were trained separately, so do not assume this tower works here."
        }
        BaseMatchVerdict::Incompatible => "incompatible. These cannot work together:",
        BaseMatchVerdict::Inconclusive => {
            "inconclusive. Too few tensors could be compared to say either way."
        }
    }
}

fn evidence_sentence(report: &BaseMatchReport) -> String {
    let mut identical_count = 0_usize;
    let mut rounded_count = 0_usize;
    let mut offset_count = 0_usize;
    let mut different_count = 0_usize;
    let mut constant_count = 0_usize;
    for compared_tensor in &report.compared_tensors {
        match compared_tensor.agreement {
            TensorAgreement::Identical => identical_count += 1,
            TensorAgreement::EqualAfterRounding => rounded_count += 1,
            TensorAgreement::OffsetByOne => offset_count += 1,
            TensorAgreement::Different => different_count += 1,
            TensorAgreement::Uninformative => constant_count += 1,
        }
    }
    format!(
        "compared {} of {} tensors both sides keep unquantized: {identical_count} identical, {rounded_count} equal after rounding, {offset_count} matching apart from the norm storage offset, {different_count} different, {constant_count} constant and not counted",
        report.compared_tensors.len(),
        report.comparable_tensor_count,
    )
}

fn artifact_heading(artifact: &str, pinned_revision: Option<&str>) -> String {
    match pinned_revision {
        Some(pinned_revision) => {
            let short_revision_end = pinned_revision.len().min(SHORT_REVISION_LENGTH);
            format!("{artifact} at {}", &pinned_revision[..short_revision_end])
        }
        None => artifact.to_owned(),
    }
}

const fn plural_files(file_count: u64) -> &'static str {
    if file_count == 1 { "file" } else { "files" }
}
