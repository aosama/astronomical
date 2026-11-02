//! Rendering for `quantize plan` and `quantize inspect`: plain text for a
//! person, one JSON document with `--json` for a script.

use std::collections::BTreeMap;
use std::io::Write;

use astronomical_quantization::allocation::AllocationResult;
use astronomical_quantization::artifact::OptiQArtifactMetadata;
use astronomical_quantization::artifact::metadata::PerLayerEntry;

/// The `plan` table: one row per module sorted by name, then the summary
/// and any overshoot note.
pub fn render_plan_text(
    allocation: &AllocationResult,
    floor_components_per_block: usize,
    output: &mut dyn Write,
) -> std::io::Result<()> {
    let mut sorted_decisions = allocation.decisions.clone();
    sorted_decisions.sort_by(compare_decisions_by_module_name);
    writeln!(
        output,
        "{:<64} {:>5} {:>14}",
        "Module", "Bits", "Parameters"
    )?;
    for decision in &sorted_decisions {
        writeln!(
            output,
            "{:<64} {:>5} {:>14}",
            decision.module_name,
            decision.bit_width.bits(),
            decision.logical_parameter_count
        )?;
    }
    writeln!(output)?;
    writeln!(
        output,
        "Target bits per weight: {:.2}",
        allocation.target_bits_per_weight
    )?;
    writeln!(
        output,
        "Achieved bits per weight: {:.2}",
        allocation.achieved_bits_per_weight
    )?;
    for (bit_width, module_count) in allocation.module_count_by_bit_width() {
        writeln!(
            output,
            "Modules at {} bits: {module_count}",
            bit_width.bits()
        )?;
    }
    if let Some(overshoot_note) = allocation.overshoot_note(floor_components_per_block) {
        writeln!(output, "Note: {overshoot_note}")?;
    }
    Ok(())
}

/// The `inspect` summary of an artifact's metadata.
pub fn render_inspect_text(
    metadata: &OptiQArtifactMetadata,
    output: &mut dyn Write,
) -> std::io::Result<()> {
    match metadata {
        OptiQArtifactMetadata::MixedPrecision(mixed_metadata) => {
            writeln!(output, "Method: {}", mixed_metadata.method)?;
            writeln!(output, "Base model: {}", mixed_metadata.base_model)?;
            writeln!(output, "Default bits: {}", mixed_metadata.default_bits)?;
            writeln!(output, "Group size: {}", mixed_metadata.group_size)?;
        }
        OptiQArtifactMetadata::StaticAllocation(static_metadata) => {
            writeln!(output, "Method: {}", static_metadata.method)?;
            writeln!(output, "Base model: {}", static_metadata.base_model)?;
            writeln!(output, "Reference: {}", static_metadata.reference)?;
            writeln!(
                output,
                "Target bits per weight: {:.2}",
                static_metadata.target_bpw
            )?;
            writeln!(
                output,
                "Achieved bits per weight: {:.2}",
                static_metadata.achieved_bpw
            )?;
        }
    }
    writeln!(output, "Quantized modules: {}", metadata.per_layer().len())?;
    for (bits, module_count) in module_count_by_bits(metadata.per_layer()) {
        writeln!(output, "Modules at {bits} bits: {module_count}")?;
    }
    Ok(())
}

/// How many modules were written at each width.
#[must_use]
pub fn module_count_by_bits(per_layer: &BTreeMap<String, PerLayerEntry>) -> BTreeMap<u32, usize> {
    let mut module_count_by_bits = BTreeMap::new();
    for per_layer_entry in per_layer.values() {
        *module_count_by_bits
            .entry(per_layer_entry.bits)
            .or_insert(0) += 1;
    }
    module_count_by_bits
}

fn compare_decisions_by_module_name(
    left: &astronomical_quantization::allocation::ModuleQuantizationDecision,
    right: &astronomical_quantization::allocation::ModuleQuantizationDecision,
) -> std::cmp::Ordering {
    left.module_name.cmp(&right.module_name)
}
