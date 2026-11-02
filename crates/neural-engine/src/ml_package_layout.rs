//! Writes an ML Program as the `.mlpackage` directory Core ML compiles from.
//!
//! An ML Program cannot be a single `.mlmodel` file once its weights live in
//! a blob file, so it ships as a package: `Manifest.json` naming the root
//! model item, the model document under `Data/com.apple.CoreML/`, and the
//! weights beside it in `weights/weight.bin`, the path the program's
//! `@model_path/weights/weight.bin` values resolve against.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::ml_program_projection_document::MlProgramPackageContents;

// Item identifiers only need to be unique within one manifest, so fixed
// values keep the package byte-for-byte reproducible.
const MODEL_ITEM_IDENTIFIER: &str = "A1B2C3D4-0161-4000-8000-000000000001";
const WEIGHTS_ITEM_IDENTIFIER: &str = "A1B2C3D4-0161-4000-8000-000000000002";
const CORE_ML_DATA_DIRECTORY: &str = "Data/com.apple.CoreML";

/// The package's `Manifest.json`.
#[must_use]
pub fn package_manifest_json() -> String {
    format!(
        r#"{{
  "fileFormatVersion": "1.0.0",
  "itemInfoEntries": {{
    "{MODEL_ITEM_IDENTIFIER}": {{
      "author": "com.apple.CoreML",
      "description": "CoreML Model Specification",
      "name": "model.mlmodel",
      "path": "com.apple.CoreML/model.mlmodel"
    }},
    "{WEIGHTS_ITEM_IDENTIFIER}": {{
      "author": "com.apple.CoreML",
      "description": "CoreML Model Weights",
      "name": "weights",
      "path": "com.apple.CoreML/weights"
    }}
  }},
  "rootModelIdentifier": "{MODEL_ITEM_IDENTIFIER}"
}}
"#
    )
}

/// Writes `projection.mlpackage` inside `parent_directory` and returns its
/// path.
pub fn write_ml_package(
    parent_directory: &Path,
    contents: &MlProgramPackageContents,
) -> io::Result<PathBuf> {
    let package_path = parent_directory.join("projection.mlpackage");
    let core_ml_data_path = package_path.join(CORE_ML_DATA_DIRECTORY);
    let weights_path = core_ml_data_path.join("weights");
    fs::create_dir_all(&weights_path)?;
    fs::write(package_path.join("Manifest.json"), package_manifest_json())?;
    fs::write(
        core_ml_data_path.join("model.mlmodel"),
        &contents.model_document,
    )?;
    fs::write(weights_path.join("weight.bin"), &contents.weight_blob_file)?;
    Ok(package_path)
}

#[cfg(test)]
#[path = "ml_package_layout_tests.rs"]
mod tests;
