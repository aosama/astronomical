//! `cargo xtask small-mac-matrix [--base-url URL] [--model ID]... [--mac-gb N]...`:
//! finds out how catalog models behave on Macs smaller than this one, before
//! a user with such a Mac does (issue #2413).
//!
//! It drives the Development instance over its local API: for each model and
//! each simulated Mac it unloads the model, sets the memory limit that Mac
//! would get, asks for a few tokens, and records whether a reply came and how
//! long it took. It refuses any instance that is not Development, and it puts
//! the memory limit back when it ends.
//!
//! The simulation is the memory limit only. A real small Mac also has less
//! memory for everything else and usually a slower chip, so a pass here is
//! necessary, not sufficient; a failure here is real.

mod matrix_cells;
mod matrix_run;

use std::path::Path;

#[cfg(test)]
pub(crate) use matrix_cells::{
    CellOutcome, MatrixCell, parse_chat_outcome, parse_matrix_options, render_matrix,
    simulated_memory_limit_gb,
};

pub fn run(_repository_root: &Path, command_arguments: &[String]) -> Result<(), String> {
    let matrix_options = matrix_cells::parse_matrix_options(command_arguments)?;
    let matrix_cells = matrix_run::run_matrix(&matrix_options)?;
    println!("{}", matrix_cells::render_matrix(&matrix_cells));
    Ok(())
}
