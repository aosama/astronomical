//! The compiled form of a model on disk: compiling without loading, loading what
//! was compiled earlier, and keeping or discarding what Core ML left behind.

use std::ffi::{CStr, CString, OsStr, c_char};
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use super::error::{CoreMlProgramError, NATIVE_ERROR_CAPACITY, native_error_text};
use super::program::CoreMlProgram;
use crate::core_ml::ffi::{
    astronomical_core_ml_compile_model_to, astronomical_core_ml_program_compiled_model_path,
    astronomical_core_ml_program_discard_compiled_model,
};

// macOS's PATH_MAX.
const COMPILED_MODEL_PATH_CAPACITY: usize = 1024;

impl CoreMlProgram {
    /// Moves the compiled model out of the temporary directory to `destination` for
    /// later loads. The loaded program stays usable and no longer owns the files.
    pub fn keep_compiled_model_files_at(
        &mut self,
        destination: &Path,
    ) -> Result<(), CoreMlProgramError> {
        let mut path_buffer = [0 as c_char; COMPILED_MODEL_PATH_CAPACITY];
        // SAFETY: the handle is valid for this borrow and the buffer is writable for
        // its full stated capacity.
        let path_status = unsafe {
            astronomical_core_ml_program_compiled_model_path(
                self.native_program.as_ptr(),
                path_buffer.as_mut_ptr(),
                COMPILED_MODEL_PATH_CAPACITY as u32,
            )
        };
        if path_status != 0 {
            return Err(CoreMlProgramError::NoCompiledModelFiles);
        }
        // SAFETY: the bridge wrote a NUL-terminated path into the buffer on success.
        let compiled_path = unsafe { CStr::from_ptr(path_buffer.as_ptr()) };
        if let Some(parent_directory) = destination.parent() {
            fs::create_dir_all(parent_directory)?;
        }
        fs::rename(
            Path::new(OsStr::from_bytes(compiled_path.to_bytes())),
            destination,
        )?;
        self.discard_compiled_model_files();
        Ok(())
    }

    /// Removes the compiled model from the temporary directory while the loaded
    /// program stays usable. Release also removes it, but a process that exits
    /// without releasing would otherwise leave one copy of the weights behind per
    /// program, which is the leak this exists to close.
    pub fn discard_compiled_model_files(&mut self) {
        // SAFETY: the handle is valid for this borrow, and the bridge clears its
        // stored path so release cannot remove the directory a second time.
        unsafe {
            astronomical_core_ml_program_discard_compiled_model(self.native_program.as_ptr());
        }
    }
}

/// Compiles a written model and leaves the compiled model at `destination` without
/// loading it, for a program that will be kept and loaded from there. See the
/// bridge's `astronomical_core_ml_compile_model_to` for why the first load must
/// happen at the final path.
pub fn compile_model_to(package_path: &Path, destination: &Path) -> Result<(), CoreMlProgramError> {
    if let Some(parent_directory) = destination.parent() {
        fs::create_dir_all(parent_directory)?;
    }
    let package_path_text = CString::new(package_path.to_string_lossy().as_bytes())
        .map_err(CoreMlProgramError::from_interior_nul)?;
    let destination_text = CString::new(destination.to_string_lossy().as_bytes())
        .map_err(CoreMlProgramError::from_interior_nul)?;
    let mut native_error = [0 as c_char; NATIVE_ERROR_CAPACITY];
    // SAFETY: both paths are valid NUL-terminated strings that outlive the call,
    // and the error buffer is writable for its full stated capacity.
    let status = unsafe {
        astronomical_core_ml_compile_model_to(
            package_path_text.as_ptr(),
            destination_text.as_ptr(),
            native_error.as_mut_ptr(),
            NATIVE_ERROR_CAPACITY as u32,
        )
    };
    if status != 0 {
        return Err(CoreMlProgramError::Load(native_error_text(&native_error)));
    }
    Ok(())
}

#[cfg(test)]
#[path = "compiled_model_files_tests.rs"]
mod tests;
