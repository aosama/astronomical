//! Protocol-buffer encoders for the parts of Core ML's `MIL.proto` an ML
//! Program projection needs: tensor types, immediate and blob-file values,
//! argument bindings, and operations.
//!
//! Field numbers follow `CoreML.Specification.MILSpec`. Protobuf `map` fields
//! are repeated entry messages with the key in field 1 and the value in
//! field 2, which is how [`append_map_entry`] writes them.

use crate::protobuf_wire::{
    append_bytes_field, append_string_field, append_varint, append_varint_field,
};

/// The element types the projection uses, each holding its `MILSpec.DataType`
/// code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum MilDataType {
    Bool = 1,
    String = 2,
    Float16 = 10,
    Int32 = 23,
    UnsignedInt8 = 31,
    UnsignedInt4 = 35,
}

impl MilDataType {
    fn proto_code(self) -> u64 {
        self as u64
    }
}

/// A `ValueType` holding a tensor of `data_type` with constant `dimensions`;
/// an empty `dimensions` slice is a rank-0 scalar.
#[must_use]
pub fn tensor_value_type(data_type: MilDataType, dimensions: &[usize]) -> Vec<u8> {
    let mut tensor_type = Vec::new();
    append_varint_field(&mut tensor_type, 1, data_type.proto_code());
    append_varint_field(&mut tensor_type, 2, dimensions.len() as u64);
    for dimension_size in dimensions {
        let mut constant_dimension = Vec::new();
        append_varint_field(&mut constant_dimension, 1, *dimension_size as u64);
        let mut dimension = Vec::new();
        append_bytes_field(&mut dimension, 1, &constant_dimension);
        append_bytes_field(&mut tensor_type, 3, &dimension);
    }
    let mut value_type = Vec::new();
    append_bytes_field(&mut value_type, 1, &tensor_type);
    value_type
}

/// A `NamedValueType`: a function input or an operation output.
#[must_use]
pub fn named_value_type(value_name: &str, value_type: &[u8]) -> Vec<u8> {
    let mut named_type = Vec::new();
    append_string_field(&mut named_type, 1, value_name);
    append_bytes_field(&mut named_type, 2, value_type);
    named_type
}

/// An immediate rank-1 (or, with no `dimensions`, scalar) int32 tensor.
#[must_use]
pub fn int32_tensor_value(int32_values: &[i32], dimensions: &[usize]) -> Vec<u8> {
    let mut packed_varints = Vec::new();
    for int32_value in int32_values {
        // Protobuf encodes a negative int32 as its 64-bit sign extension.
        append_varint(&mut packed_varints, i64::from(*int32_value) as u64);
    }
    let mut repeated_ints = Vec::new();
    append_bytes_field(&mut repeated_ints, 1, &packed_varints);
    let mut tensor_value = Vec::new();
    append_bytes_field(&mut tensor_value, 2, &repeated_ints);
    immediate_tensor_value(MilDataType::Int32, dimensions, &tensor_value)
}

/// An immediate scalar bool.
#[must_use]
pub fn bool_scalar_value(bool_value: bool) -> Vec<u8> {
    let mut packed_varints = Vec::new();
    append_varint(&mut packed_varints, u64::from(bool_value));
    let mut repeated_bools = Vec::new();
    append_bytes_field(&mut repeated_bools, 1, &packed_varints);
    let mut tensor_value = Vec::new();
    append_bytes_field(&mut tensor_value, 3, &repeated_bools);
    immediate_tensor_value(MilDataType::Bool, &[], &tensor_value)
}

/// An immediate scalar string.
#[must_use]
pub fn string_scalar_value(text: &str) -> Vec<u8> {
    let mut repeated_strings = Vec::new();
    append_string_field(&mut repeated_strings, 1, text);
    let mut tensor_value = Vec::new();
    append_bytes_field(&mut tensor_value, 4, &repeated_strings);
    immediate_tensor_value(MilDataType::String, &[], &tensor_value)
}

/// A tensor stored in the weight-blob file at the metadata `blob_offset`.
#[must_use]
pub fn blob_file_tensor_value(
    data_type: MilDataType,
    dimensions: &[usize],
    blob_file_name: &str,
    blob_offset: u64,
) -> Vec<u8> {
    let mut blob_file_value = Vec::new();
    append_string_field(&mut blob_file_value, 1, blob_file_name);
    append_varint_field(&mut blob_file_value, 2, blob_offset);
    let mut value = Vec::new();
    append_bytes_field(&mut value, 2, &tensor_value_type(data_type, dimensions));
    append_bytes_field(&mut value, 5, &blob_file_value);
    value
}

fn immediate_tensor_value(
    data_type: MilDataType,
    dimensions: &[usize],
    tensor_value: &[u8],
) -> Vec<u8> {
    let mut immediate_value = Vec::new();
    append_bytes_field(&mut immediate_value, 1, tensor_value);
    let mut value = Vec::new();
    append_bytes_field(&mut value, 2, &tensor_value_type(data_type, dimensions));
    append_bytes_field(&mut value, 3, &immediate_value);
    value
}

/// One entry of a protobuf `map<string, Message>` field.
pub fn append_map_entry(message: &mut Vec<u8>, field_number: u32, key: &str, encoded_value: &[u8]) {
    let mut map_entry = Vec::new();
    append_string_field(&mut map_entry, 1, key);
    append_bytes_field(&mut map_entry, 2, encoded_value);
    append_bytes_field(message, field_number, &map_entry);
}

/// One MIL operation, built field by field. Every operation carries its own
/// name as the `name` attribute, as Apple's serializer writes it.
pub struct MilOperation {
    encoded_operation: Vec<u8>,
}

impl MilOperation {
    #[must_use]
    pub fn new(operation_type: &str, operation_name: &str) -> Self {
        let mut encoded_operation = Vec::new();
        append_string_field(&mut encoded_operation, 1, operation_type);
        append_map_entry(
            &mut encoded_operation,
            5,
            "name",
            &string_scalar_value(operation_name),
        );
        Self { encoded_operation }
    }

    /// Binds `parameter_name` to a value produced earlier in the block.
    #[must_use]
    pub fn input_named(mut self, parameter_name: &str, bound_value_name: &str) -> Self {
        let mut binding = Vec::new();
        append_string_field(&mut binding, 1, bound_value_name);
        self.append_argument(parameter_name, &binding);
        self
    }

    /// Binds `parameter_name` to several values produced earlier, which is how
    /// variadic operations such as `concat` take their inputs.
    #[must_use]
    pub fn input_named_list(mut self, parameter_name: &str, bound_value_names: &[String]) -> Self {
        let mut argument = Vec::new();
        for bound_value_name in bound_value_names {
            let mut binding = Vec::new();
            append_string_field(&mut binding, 1, bound_value_name);
            append_bytes_field(&mut argument, 1, &binding);
        }
        append_map_entry(&mut self.encoded_operation, 2, parameter_name, &argument);
        self
    }

    /// Binds `parameter_name` directly to a value, which is how iOS 18
    /// `constexpr_*` operations take their blob-file inputs.
    #[must_use]
    pub fn input_value(mut self, parameter_name: &str, encoded_value: &[u8]) -> Self {
        let mut binding = Vec::new();
        append_bytes_field(&mut binding, 2, encoded_value);
        self.append_argument(parameter_name, &binding);
        self
    }

    #[must_use]
    pub fn attribute(mut self, attribute_name: &str, encoded_value: &[u8]) -> Self {
        append_map_entry(
            &mut self.encoded_operation,
            5,
            attribute_name,
            encoded_value,
        );
        self
    }

    #[must_use]
    pub fn output(mut self, output_name: &str, value_type: &[u8]) -> Self {
        append_bytes_field(
            &mut self.encoded_operation,
            3,
            &named_value_type(output_name, value_type),
        );
        self
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.encoded_operation
    }

    fn append_argument(&mut self, parameter_name: &str, binding: &[u8]) {
        let mut argument = Vec::new();
        append_bytes_field(&mut argument, 1, binding);
        append_map_entry(&mut self.encoded_operation, 2, parameter_name, &argument);
    }
}

#[cfg(test)]
#[path = "mil_program_wire_tests.rs"]
mod tests;
