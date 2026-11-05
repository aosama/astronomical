//! Writes a model document and loads or compiles it, the two steps every journey
//! repeats. Journeys only ever expect these to succeed, so one boxed error type is
//! enough for their failure messages.

use std::error::Error;
use std::path::Path;

use astronomical_neural_engine::core_ml::{CoreMlComputeUnits, CoreMlProgram, compile_model_to};
use astronomical_neural_engine::documents::neural_network::{
    encode_projection_document, write_model_document,
};
use astronomical_neural_engine::documents::package_layout::write_ml_package;
use astronomical_neural_engine::documents::projection::{
    MlProgramWeights, encode_ml_program_projection,
};
use astronomical_neural_engine::program_shape::ProgramShape;

pub(crate) fn load_neural_network(
    shape: ProgramShape,
    fp16_weights: &[u16],
    compute_units: CoreMlComputeUnits,
    contraction_part_count: usize,
    model_directory: &Path,
) -> Result<CoreMlProgram, Box<dyn Error>> {
    let model_document = encode_projection_document(shape, fp16_weights, contraction_part_count)?;
    let model_path = write_model_document(model_directory, &model_document)?;
    Ok(CoreMlProgram::load_from_path(
        &model_path,
        shape,
        compute_units,
    )?)
}

pub(crate) fn load_ml_program(
    shape: ProgramShape,
    weights: MlProgramWeights<'_>,
    compute_units: CoreMlComputeUnits,
    contraction_part_count: usize,
    model_directory: &Path,
) -> Result<CoreMlProgram, Box<dyn Error>> {
    let package_contents = encode_ml_program_projection(shape, weights, contraction_part_count)?;
    let package_path = write_ml_package(model_directory, &package_contents)?;
    Ok(CoreMlProgram::load_from_path(
        &package_path,
        shape,
        compute_units,
    )?)
}

pub(crate) fn compile_ml_program_to(
    shape: ProgramShape,
    weights: MlProgramWeights<'_>,
    contraction_part_count: usize,
    model_directory: &Path,
    destination: &Path,
) -> Result<(), Box<dyn Error>> {
    let package_contents = encode_ml_program_projection(shape, weights, contraction_part_count)?;
    let package_path = write_ml_package(model_directory, &package_contents)?;
    Ok(compile_model_to(&package_path, destination)?)
}

pub(crate) fn load_ml_program_mlp(
    shape: ProgramShape,
    intermediate_channels: usize,
    weights: astronomical_neural_engine::documents::mlp::MlpFp16Weights<'_>,
    contraction_part_width: usize,
    compute_units: CoreMlComputeUnits,
    model_directory: &Path,
) -> Result<CoreMlProgram, Box<dyn Error>> {
    let package_contents = astronomical_neural_engine::documents::mlp::encode_ml_program_mlp(
        shape,
        intermediate_channels,
        weights,
        contraction_part_width,
    )?;
    let package_path = write_ml_package(model_directory, &package_contents)?;
    Ok(CoreMlProgram::load_from_path(
        &package_path,
        shape,
        compute_units,
    )?)
}

pub(crate) fn load_ml_program_rotated_mlp(
    layout: astronomical_neural_engine::documents::rotated_mlp::RotatedMlpLayout,
    weights: astronomical_neural_engine::documents::rotated_mlp::RotatedMlpWeights<'_>,
    compute_units: CoreMlComputeUnits,
    model_directory: &Path,
) -> Result<CoreMlProgram, Box<dyn Error>> {
    let package_contents =
        astronomical_neural_engine::documents::rotated_mlp::encode_ml_program_rotated_mlp(
            layout, weights,
        )?;
    let package_path = write_ml_package(model_directory, &package_contents)?;
    Ok(CoreMlProgram::load_from_path(
        &package_path,
        layout.shape,
        compute_units,
    )?)
}
