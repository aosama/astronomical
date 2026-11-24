//! Re-encodes one contraction part of fp16 weights as 8-bit codes into lookup
//! tables that each span several output rows.
//!
//! The Neural Engine only runs lookup tables at its full rate when one table
//! spans several rows; one table per row is about 14 times slower (issue #161).
//! A 2-bit ternary checkpoint cannot keep its own per-block tables for that
//! reason, but it holds so few distinct values that 16 rows of a 1,024-channel
//! part fit one 256-entry table with nothing rounded: three values per block,
//! with many blocks sharing them. Where a table would overflow, the two closest
//! values merge, so the error lands where it is smallest.

use crate::documents::model_interface::ModelDocumentError;
use crate::weights::half_precision::{f16_bits_to_f32, f32_to_f16_bits};

/// Entries in every table, the only size an 8-bit code can address.
pub const LOOKUP_TABLE_ENTRY_COUNT: usize = 256;
const FP16_BIT_PATTERN_COUNT: usize = 1 << 16;
const FP16_SIGN_BIT: u16 = 0x8000;

/// Row-major `[output_channels][input_channels]` codes, with one padded table
/// per `rows_per_table` rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedLookupTables {
    pub output_channels: usize,
    pub input_channels: usize,
    pub rows_per_table: usize,
    pub codes: Vec<u8>,
    pub fp16_tables: Vec<u16>,
    /// Weights whose value moved because their table overflowed.
    pub merged_weight_count: usize,
}

/// A run of neighbouring values, in the sortable order, sharing one entry.
#[derive(Clone, Copy)]
struct ValueGroup {
    first_sortable: u16,
    last_sortable: u16,
}

/// Reused across tables so a part with thousands of tables allocates once.
struct TableScratch {
    is_present: Vec<bool>,
    occurrence_counts: Vec<u32>,
    sortable_to_code: Vec<u8>,
    distinct_sortable: Vec<u16>,
}

impl SharedLookupTables {
    pub fn from_fp16_weights(
        fp16_weights: &[u16],
        output_channels: usize,
        input_channels: usize,
        rows_per_table: usize,
    ) -> Result<Self, ModelDocumentError> {
        validate_shape(
            fp16_weights.len(),
            output_channels,
            input_channels,
            rows_per_table,
        )?;
        let mut encoded = Self {
            output_channels,
            input_channels,
            rows_per_table,
            codes: vec![0; fp16_weights.len()],
            fp16_tables: Vec::with_capacity(
                output_channels / rows_per_table * LOOKUP_TABLE_ENTRY_COUNT,
            ),
            merged_weight_count: 0,
        };
        let mut scratch = TableScratch {
            is_present: vec![false; FP16_BIT_PATTERN_COUNT],
            occurrence_counts: vec![0; FP16_BIT_PATTERN_COUNT],
            sortable_to_code: vec![0; FP16_BIT_PATTERN_COUNT],
            distinct_sortable: Vec::with_capacity(LOOKUP_TABLE_ENTRY_COUNT * 4),
        };
        let table_weight_count = rows_per_table * input_channels;
        for (table_index, table_weights) in
            fp16_weights.chunks_exact(table_weight_count).enumerate()
        {
            let table_codes =
                &mut encoded.codes[table_index * table_weight_count..][..table_weight_count];
            encoded.merged_weight_count += encode_one_table(
                table_weights,
                table_codes,
                &mut encoded.fp16_tables,
                &mut scratch,
            );
        }
        Ok(encoded)
    }

    /// The fp16 weights the tables reconstruct, row-major.
    #[must_use]
    pub fn reconstructed_fp16(&self) -> Vec<u16> {
        let table_weight_count = self.rows_per_table * self.input_channels;
        let mut reconstructed = Vec::with_capacity(self.codes.len());
        for (weight_index, code) in self.codes.iter().enumerate() {
            let table_index = weight_index / table_weight_count;
            reconstructed.push(
                self.fp16_tables[table_index * LOOKUP_TABLE_ENTRY_COUNT + usize::from(*code)],
            );
        }
        reconstructed
    }
}

fn validate_shape(
    weight_count: usize,
    output_channels: usize,
    input_channels: usize,
    rows_per_table: usize,
) -> Result<(), ModelDocumentError> {
    if rows_per_table == 0 || !output_channels.is_multiple_of(rows_per_table) {
        return Err(ModelDocumentError::UnevenLookupTableRows {
            output_channels,
            rows_per_table,
        });
    }
    let expected_count = output_channels * input_channels;
    if weight_count != expected_count || expected_count == 0 {
        return Err(ModelDocumentError::WeightCountMismatch {
            expected: expected_count,
            actual: weight_count,
        });
    }
    Ok(())
}

/// Returns how many weights moved because this table overflowed.
fn encode_one_table(
    table_weights: &[u16],
    table_codes: &mut [u8],
    fp16_tables: &mut Vec<u16>,
    scratch: &mut TableScratch,
) -> usize {
    collect_distinct_values(table_weights, scratch);
    let groups = groups_within_table_capacity(&scratch.distinct_sortable);
    let mut merged_weight_count = 0;
    let first_entry = fp16_tables.len();
    for (code, group) in groups.iter().enumerate() {
        fp16_tables.push(group_entry_fp16(*group));
        let is_merged_group = group.first_sortable != group.last_sortable;
        for sortable_value in group.first_sortable..=group.last_sortable {
            let pattern_index = usize::from(sortable_value);
            scratch.sortable_to_code[pattern_index] = code as u8;
            if is_merged_group {
                merged_weight_count += scratch.occurrence_counts[pattern_index] as usize;
            }
        }
    }
    // Core ML requires every table to hold exactly 2^bits entries; the padding
    // is never indexed because no code reaches it.
    fp16_tables.resize(first_entry + LOOKUP_TABLE_ENTRY_COUNT, 0);
    for (weight_code, fp16_weight) in table_codes.iter_mut().zip(table_weights) {
        *weight_code = scratch.sortable_to_code[usize::from(sortable_fp16(*fp16_weight))];
    }
    clear_scratch(scratch);
    merged_weight_count
}

fn collect_distinct_values(table_weights: &[u16], scratch: &mut TableScratch) {
    for fp16_weight in table_weights {
        let sortable_value = sortable_fp16(*fp16_weight);
        let pattern_index = usize::from(sortable_value);
        if !scratch.is_present[pattern_index] {
            scratch.is_present[pattern_index] = true;
            scratch.distinct_sortable.push(sortable_value);
        }
        scratch.occurrence_counts[pattern_index] += 1;
    }
    scratch.distinct_sortable.sort_unstable();
}

fn clear_scratch(scratch: &mut TableScratch) {
    for sortable_value in &scratch.distinct_sortable {
        scratch.is_present[usize::from(*sortable_value)] = false;
        scratch.occurrence_counts[usize::from(*sortable_value)] = 0;
    }
    scratch.distinct_sortable.clear();
}

/// Merges the two closest neighbours until the table has room. Tables overflow
/// by a few dozen values at most, so a rescan per merge stays cheap and keeps
/// this simpler than a priority queue.
fn groups_within_table_capacity(distinct_sortable: &[u16]) -> Vec<ValueGroup> {
    let mut groups = Vec::with_capacity(distinct_sortable.len());
    for sortable_value in distinct_sortable {
        groups.push(ValueGroup {
            first_sortable: *sortable_value,
            last_sortable: *sortable_value,
        });
    }
    while groups.len() > LOOKUP_TABLE_ENTRY_COUNT {
        let merge_index = closest_neighbour_index(&groups);
        groups[merge_index].last_sortable = groups[merge_index + 1].last_sortable;
        groups.remove(merge_index + 1);
    }
    groups
}

fn closest_neighbour_index(groups: &[ValueGroup]) -> usize {
    let mut closest_index = 0;
    let mut closest_gap = f32::INFINITY;
    for (group_index, neighbours) in groups.windows(2).enumerate() {
        let gap = value_of_sortable(neighbours[1].first_sortable)
            - value_of_sortable(neighbours[0].last_sortable);
        if gap < closest_gap {
            closest_gap = gap;
            closest_index = group_index;
        }
    }
    closest_index
}

fn group_entry_fp16(group: ValueGroup) -> u16 {
    if group.first_sortable == group.last_sortable {
        return unsortable_fp16(group.first_sortable);
    }
    let midpoint =
        (value_of_sortable(group.first_sortable) + value_of_sortable(group.last_sortable)) / 2.0;
    f32_to_f16_bits(midpoint)
}

fn value_of_sortable(sortable_value: u16) -> f32 {
    f16_bits_to_f32(unsortable_fp16(sortable_value))
}

/// Maps an fp16 bit pattern to an integer that orders like its value, so the
/// values sort without a comparison callback and neighbours stay adjacent.
fn sortable_fp16(fp16_bits: u16) -> u16 {
    if fp16_bits & FP16_SIGN_BIT == 0 {
        fp16_bits | FP16_SIGN_BIT
    } else {
        !fp16_bits
    }
}

fn unsortable_fp16(sortable_value: u16) -> u16 {
    if sortable_value & FP16_SIGN_BIT != 0 {
        sortable_value & !FP16_SIGN_BIT
    } else {
        !sortable_value
    }
}

#[cfg(test)]
#[path = "shared_lookup_tables_tests.rs"]
mod tests;
