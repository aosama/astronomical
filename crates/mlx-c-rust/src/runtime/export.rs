//! The MLX export and graph-utilities domain: function export/import and
//! graph rendering through the node namer.

use std::os::raw::c_void;

use crate::error::check_status;
use crate::raw;
use crate::types::closures::MlxClosure;
use crate::types::closures::MlxClosureKwargs;
use crate::types::maps::MlxMapStringToArray;
use crate::{MlxArray, MlxArrayVector, MlxCError};

/// A host path argument translated into a C string.
fn path_argument(path: &str, operation: &'static str) -> Result<std::ffi::CString, MlxCError> {
    std::ffi::CString::new(path).map_err(|_| MlxCError {
        operation,
        description: "path contains an interior null byte".to_owned(),
    })
}

/// Exports a closure and example arguments to an MLX function file.
///
/// # Errors
/// Returns the captured MLX-C description when export fails.
pub fn export_function(
    path: &str,
    function: &MlxClosure,
    example_arguments: &MlxArrayVector,
    shapeless: bool,
) -> Result<(), MlxCError> {
    let operation: &'static str = "export an MLX function";
    let path_value = path_argument(path, operation)?;
    // SAFETY: The path remains valid for the call and both handles are live.
    let status = unsafe {
        raw::mlx_export_function(
            path_value.as_ptr(),
            function.raw(),
            example_arguments.raw(),
            shapeless,
        )
    };
    check_status(status, operation)
}

/// Exports a kwargs closure and example arguments to an MLX function file.
///
/// # Errors
/// Returns the captured MLX-C description when export fails.
pub fn export_function_with_keywords(
    path: &str,
    function: &MlxClosureKwargs,
    example_arguments: &MlxArrayVector,
    keyword_arguments: &MlxMapStringToArray,
    shapeless: bool,
) -> Result<(), MlxCError> {
    let operation: &'static str = "export an MLX function with keywords";
    let path_value = path_argument(path, operation)?;
    // SAFETY: The path remains valid for the call and all handles are live.
    let status = unsafe {
        raw::mlx_export_function_kwargs(
            path_value.as_ptr(),
            function.raw(),
            example_arguments.raw(),
            keyword_arguments.raw(),
            shapeless,
        )
    };
    check_status(status, operation)
}

/// Owned MLX function exporter released exactly once.
#[derive(Debug)]
pub struct MlxFunctionExporter(raw::mlx_function_exporter);

impl MlxFunctionExporter {
    /// Builds an exporter that writes new function versions into the file.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when creation fails.
    pub fn new(path: &str, function: &MlxClosure, shapeless: bool) -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX function exporter";
        let path_value = path_argument(path, operation)?;
        // SAFETY: The path remains valid for the call, the closure handle is
        // live, and the returned handle enters RAII ownership.
        let raw_exporter = unsafe {
            raw::mlx_function_exporter_new(path_value.as_ptr(), function.raw(), shapeless)
        };
        if raw_exporter.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty function exporter handle".to_owned(),
            });
        }
        Ok(Self(raw_exporter))
    }

    /// Writes one compiled specialization for the arguments.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the specialization fails.
    pub fn apply(&self, arguments: &MlxArrayVector) -> Result<(), MlxCError> {
        let operation: &'static str = "specialize an MLX function exporter";
        // SAFETY: Both handles are live for the duration of the call.
        let status = unsafe { raw::mlx_function_exporter_apply(self.0, arguments.raw()) };
        check_status(status, operation)
    }

    /// Writes one compiled specialization for the arguments and keywords.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the specialization fails.
    pub fn apply_with_keywords(
        &self,
        arguments: &MlxArrayVector,
        keyword_arguments: &MlxMapStringToArray,
    ) -> Result<(), MlxCError> {
        let operation: &'static str = "specialize an MLX function exporter with keywords";
        // SAFETY: All handles are live for the duration of the call.
        let status = unsafe {
            raw::mlx_function_exporter_apply_kwargs(
                self.0,
                arguments.raw(),
                keyword_arguments.raw(),
            )
        };
        check_status(status, operation)
    }
}

impl Drop for MlxFunctionExporter {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live exporter exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_function_exporter_free(self.0);
        }
    }
}

/// Owned MLX imported function released exactly once.
#[derive(Debug)]
pub struct MlxImportedFunction(raw::mlx_imported_function);

impl MlxImportedFunction {
    /// Loads a previously exported MLX function file.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when loading fails.
    pub fn load(path: &str) -> Result<Self, MlxCError> {
        let operation: &'static str = "import an MLX function";
        let path_value = path_argument(path, operation)?;
        // SAFETY: The path remains valid for the call and the returned
        // handle enters RAII ownership.
        let raw_function = unsafe { raw::mlx_imported_function_new(path_value.as_ptr()) };
        if raw_function.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty imported function handle".to_owned(),
            });
        }
        Ok(Self(raw_function))
    }

    /// Applies the imported function to the arguments.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when application fails.
    pub fn apply(&self, arguments: &MlxArrayVector) -> Result<MlxArrayVector, MlxCError> {
        let operation: &'static str = "apply an imported MLX function";
        let mut output = MlxArrayVector::empty(operation)?;
        // SAFETY: Both handles are live and the output pointer is valid
        // writable storage.
        let status =
            unsafe { raw::mlx_imported_function_apply(output.raw_mut(), self.0, arguments.raw()) };
        check_status(status, operation)?;
        Ok(output)
    }

    /// Applies the imported function to the arguments and keywords.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when application fails.
    pub fn apply_with_keywords(
        &self,
        arguments: &MlxArrayVector,
        keyword_arguments: &MlxMapStringToArray,
    ) -> Result<MlxArrayVector, MlxCError> {
        let operation: &'static str = "apply an imported MLX function with keywords";
        let mut output = MlxArrayVector::empty(operation)?;
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_imported_function_apply_kwargs(
                output.raw_mut(),
                self.0,
                arguments.raw(),
                keyword_arguments.raw(),
            )
        };
        check_status(status, operation)?;
        Ok(output)
    }
}

impl Drop for MlxImportedFunction {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live imported function exactly
        // once and never accesses the handle afterward.
        unsafe {
            raw::mlx_imported_function_free(self.0);
        }
    }
}

/// Owned MLX node namer released exactly once.
#[derive(Debug)]
pub struct MlxNodeNamer(raw::mlx_node_namer);

impl MlxNodeNamer {
    /// Creates an empty namer.
    pub fn new() -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX node namer";
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_namer = unsafe { raw::mlx_node_namer_new() };
        if raw_namer.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty node namer handle".to_owned(),
            });
        }
        Ok(Self(raw_namer))
    }

    /// Names one array node.
    pub fn set_name(&mut self, array: &MlxArray, name: &str) -> Result<(), MlxCError> {
        let operation: &'static str = "name an MLX graph node";
        let name_argument = path_argument(name, operation)?;
        // SAFETY: The namer and array handles are live and the name outlives
        // the call.
        let status =
            unsafe { raw::mlx_node_namer_set_name(self.0, array.raw(), name_argument.as_ptr()) };
        check_status(status, operation)
    }

    pub(crate) const fn raw(&self) -> raw::mlx_node_namer {
        self.0
    }

    /// Copies the name recorded for the array node.
    pub fn name_of(&self, array: &MlxArray) -> Result<String, MlxCError> {
        let operation: &'static str = "read an MLX graph node name";
        let mut name_pointer: *const std::os::raw::c_char = std::ptr::null();
        // SAFETY: The namer and array handles are live and the output
        // pointer is valid writable storage.
        let status =
            unsafe { raw::mlx_node_namer_get_name(&mut name_pointer, self.0, array.raw()) };
        check_status(status, operation)?;
        if name_pointer.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned a null node name".to_owned(),
            });
        }
        // SAFETY: MLX loans the null-terminated name from the live namer;
        // the text is copied before this call returns.
        let borrowed = unsafe { std::ffi::CStr::from_ptr(name_pointer) };
        Ok(borrowed.to_string_lossy().into_owned())
    }
}

impl Drop for MlxNodeNamer {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live namer exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_node_namer_free(self.0);
        }
    }
}

/// A rendered-graph destination backed by a C `FILE*` the caller owns for
/// the duration of one rendering call.
///
/// # Safety
/// The pointer must be a valid open `FILE` for the duration of the render
/// call and must not be closed while rendering.
pub struct GraphOutputFile(pub *mut c_void);

/// Writes the graph behind the outputs to a Graphviz DOT file.
///
/// # Safety
/// The output file must be a valid open `FILE` for the duration of the call.
///
/// # Errors
/// Returns the captured MLX-C description when rendering fails.
pub unsafe fn export_graph_to_dot(
    output_file: GraphOutputFile,
    namer: &MlxNodeNamer,
    outputs: &MlxArrayVector,
) -> Result<(), MlxCError> {
    let operation: &'static str = "export an MLX graph to DOT";
    // SAFETY: The caller upholds the FILE lifetime contract for the call.
    let status =
        unsafe { raw::mlx_export_to_dot(output_file.0.cast(), namer.raw(), outputs.raw()) };
    check_status(status, operation)
}

/// Prints the graph behind the outputs as text.
///
/// # Safety
/// The output file must be a valid open `FILE` for the duration of the call.
///
/// # Errors
/// Returns the captured MLX-C description when printing fails.
pub unsafe fn print_graph(
    output_file: GraphOutputFile,
    namer: &MlxNodeNamer,
    outputs: &MlxArrayVector,
) -> Result<(), MlxCError> {
    let operation: &'static str = "print an MLX graph";
    // SAFETY: The caller upholds the FILE lifetime contract for the call.
    let status = unsafe { raw::mlx_print_graph(output_file.0.cast(), namer.raw(), outputs.raw()) };
    check_status(status, operation)
}
