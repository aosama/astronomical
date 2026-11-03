//! The patch pipeline the native build applies to MLX and MLX-C, parsed out
//! of the native CMakeLists so provisioning and the CMake build share one
//! definition. Parsing instead of duplicating the list also keeps the native
//! build identity untouched by this tool. A unit test pins the parser against
//! the real CMakeLists, so a restructure fails loudly instead of silently
//! extracting unpatched headers.

use std::path::{Path, PathBuf};

const PATCH_FILE_PREFIX: &str = "PATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PatchStep {
    /// `mlx` or `mlx_c`, the FetchContent target the patch belongs to.
    pub(crate) dependency_name: String,
    /// Repository-relative from the CMakeLists; absolute when a tool or test
    /// supplies its own `--patch-list`.
    pub(crate) patch_path: String,
}

/// Reads `FetchContent_Declare(` blocks whose first argument is `mlx` or
/// `mlx_c` and collects their `PATCH_FILE=` arguments in order.
pub(crate) fn parse_cmake_patch_pipeline(cmake_lists_text: &str) -> Vec<PatchStep> {
    let mut patch_steps = Vec::new();
    let mut is_inside_declaration = false;
    let mut declaration_target = String::new();
    for cmake_line in cmake_lists_text.lines() {
        if is_inside_declaration && cmake_line.starts_with(')') {
            is_inside_declaration = false;
            declaration_target.clear();
        } else if !is_inside_declaration && cmake_line.starts_with("FetchContent_Declare(") {
            is_inside_declaration = true;
            declaration_target.clear();
        } else if is_inside_declaration && declaration_target.is_empty() {
            let candidate_target = cmake_line.trim();
            if candidate_target == "mlx" || candidate_target == "mlx_c" {
                declaration_target = candidate_target.to_owned();
            }
        } else if is_inside_declaration && let Some(patch_path) = patch_path_on_line(cmake_line) {
            patch_steps.push(PatchStep {
                dependency_name: declaration_target.clone(),
                patch_path,
            });
        }
    }
    patch_steps
}

fn patch_path_on_line(cmake_line: &str) -> Option<String> {
    let prefix_position = cmake_line.find(PATCH_FILE_PREFIX)?;
    let path_and_rest = &cmake_line[prefix_position + PATCH_FILE_PREFIX.len()..];
    let patch_path = match path_and_rest.find('"') {
        Some(quote_position) => &path_and_rest[..quote_position],
        None => path_and_rest,
    };
    Some(patch_path.to_owned())
}

/// Reads an explicit patch list: one `dependency<TAB>patch-path` per line.
pub(crate) fn parse_patch_list(patch_list_text: &str) -> Result<Vec<PatchStep>, String> {
    let mut patch_steps = Vec::new();
    for patch_list_line in patch_list_text.lines() {
        if patch_list_line.is_empty() {
            continue;
        }
        let Some((dependency_name, patch_path)) = patch_list_line.split_once('\t') else {
            return Err(format!(
                "patch list line is not dependency<TAB>patch-path: {patch_list_line}"
            ));
        };
        patch_steps.push(PatchStep {
            dependency_name: dependency_name.to_owned(),
            patch_path: patch_path.to_owned(),
        });
    }
    Ok(patch_steps)
}

pub(crate) fn render_patch_list(patch_steps: &[PatchStep]) -> String {
    let mut patch_list_text = String::new();
    for patch_step in patch_steps {
        patch_list_text.push_str(&format!(
            "{}\t{}\n",
            patch_step.dependency_name, patch_step.patch_path
        ));
    }
    patch_list_text
}

pub(crate) fn resolve_patch_path(repository_root: &Path, patch_path: &str) -> PathBuf {
    let candidate_path = Path::new(patch_path);
    if candidate_path.is_absolute() {
        candidate_path.to_path_buf()
    } else {
        repository_root.join(candidate_path)
    }
}

/// Every patch must exist before extraction starts, so a renamed patch
/// fails here instead of producing an unpatched tree.
pub(crate) fn require_patch_files(
    repository_root: &Path,
    patch_steps: &[PatchStep],
) -> Result<(), String> {
    for patch_step in patch_steps {
        if !resolve_patch_path(repository_root, &patch_step.patch_path).is_file() {
            return Err(format!(
                "patch pipeline references a missing patch file: {}",
                patch_step.patch_path
            ));
        }
    }
    Ok(())
}
