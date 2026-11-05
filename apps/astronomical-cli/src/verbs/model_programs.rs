//! `astronomical models programs`: what compiled Neural Engine programs are on this
//! Mac, where, and how much space they take. It reads only the config and the disk,
//! so it can answer with no daemon running.

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::Path;

use astronomical_config::ListedNeuralEngineProgramSet;

use crate::formatting::format_gigabytes;

const SHORT_REVISION_LENGTH: usize = 12;

/// What the report needs, gathered by the caller.
pub struct ModelProgramsReport<'a> {
    pub program_directory: &'a Path,
    pub program_sets: &'a [ListedNeuralEngineProgramSet],
    /// `None` when discovery failed, so nothing is wrongly called missing.
    pub installed_model_ids: Option<&'a BTreeSet<String>>,
}

pub fn render_model_programs(
    report: &ModelProgramsReport<'_>,
    output: &mut dyn Write,
) -> io::Result<()> {
    writeln!(
        output,
        "Neural Engine programs: {}",
        report.program_directory.display()
    )?;
    if report.program_sets.is_empty() {
        return writeln!(
            output,
            "None yet. They are built when Neural Engine prefill is on and a model first loads."
        );
    }
    let mut total_bytes = 0_u64;
    for program_set in report.program_sets {
        total_bytes = total_bytes.saturating_add(program_set.byte_count);
        writeln!(
            output,
            "{}",
            set_line(program_set, report.installed_model_ids)
        )?;
    }
    writeln!(
        output,
        "Total: {} GB in {} sets",
        format_gigabytes(total_bytes),
        report.program_sets.len()
    )
}

fn set_line(
    program_set: &ListedNeuralEngineProgramSet,
    installed_model_ids: Option<&BTreeSet<String>>,
) -> String {
    let revision_end = program_set.model_revision.len().min(SHORT_REVISION_LENGTH);
    let mut line = format!(
        "  {} ({}) {}: {} programs, {} GB",
        program_set.model_id,
        program_set
            .model_revision
            .get(..revision_end)
            .unwrap_or(&program_set.model_revision),
        program_set.set_name,
        program_set.program_count,
        format_gigabytes(program_set.byte_count),
    );
    if !program_set.is_complete {
        line.push_str(" [incomplete]");
    }
    if let Some(installed_model_ids) = installed_model_ids
        && !installed_model_ids.contains(&program_set.model_id)
    {
        line.push_str(" [model not found]");
    }
    line
}
