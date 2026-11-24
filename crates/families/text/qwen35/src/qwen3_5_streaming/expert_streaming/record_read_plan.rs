//! Turns the slices a batch of experts needs into the fewest reads.
//!
//! Each expert is nine slices of a file (a weight, scales and biases for each
//! of three projections), and each slice belongs in a different table. Where
//! the file stores an expert's slices back to back, as an Expert Atlas layer
//! file does, nine slices are one read that scatters into nine destinations;
//! where it stores them gigabytes apart, as a stock checkpoint does, they
//! stay nine reads. This module only decides the grouping, from offsets and
//! lengths, so it can be tested without a file or a graphics processor.

/// One slice of a file that one destination wants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageSlice {
    /// Which open file the slice is in; slices of different files never join.
    pub file_index: usize,
    pub file_offset: u64,
    pub byte_count: usize,
}

/// The least a part of a cut read holds. A record longer than twice this is
/// read as several overlapping requests instead of one, so storage works on
/// it from more than one place at once.
const READ_PART_MINIMUM_BYTES: usize = 512 * 1024;

/// Groups `slices` into reads: each group is a run of slice indexes whose
/// slices follow one another in the same file with no gap, in file order.
/// A long run is cut into parts of at least `READ_PART_MINIMUM_BYTES`.
pub fn plan_record_reads(slices: &[StorageSlice]) -> Vec<Vec<usize>> {
    let mut slice_indexes_in_file_order: Vec<usize> = (0..slices.len()).collect();
    slice_indexes_in_file_order.sort_by_key(|&slice_index| {
        (
            slices[slice_index].file_index,
            slices[slice_index].file_offset,
            slice_index,
        )
    });
    let mut contiguous_runs: Vec<Vec<usize>> = Vec::new();
    for slice_index in slice_indexes_in_file_order {
        let slice = slices[slice_index];
        if slice.byte_count == 0 {
            continue;
        }
        let continues_last_run = contiguous_runs.last().is_some_and(|last_run| {
            let last_slice = slices[last_run[last_run.len() - 1]];
            last_slice.file_index == slice.file_index
                && last_slice.file_offset + last_slice.byte_count as u64 == slice.file_offset
        });
        if continues_last_run {
            let last_run_index = contiguous_runs.len() - 1;
            contiguous_runs[last_run_index].push(slice_index);
        } else {
            contiguous_runs.push(vec![slice_index]);
        }
    }
    let mut planned_reads = Vec::with_capacity(contiguous_runs.len() * 3);
    for contiguous_run in contiguous_runs {
        cut_a_long_run_into_parts(slices, contiguous_run, &mut planned_reads);
    }
    planned_reads
}

/// Appends `run` to `planned_reads`, cut wherever a part has reached
/// `READ_PART_MINIMUM_BYTES` and at least that much is still to come, so no
/// part is a sliver. A cut falls only between slices: a destination cannot
/// be split.
fn cut_a_long_run_into_parts(
    slices: &[StorageSlice],
    run: Vec<usize>,
    planned_reads: &mut Vec<Vec<usize>>,
) {
    let mut bytes_still_to_come = 0_usize;
    for &slice_index in &run {
        bytes_still_to_come += slices[slice_index].byte_count;
    }
    let mut part = Vec::new();
    let mut part_byte_count = 0_usize;
    for slice_index in run {
        let slice_byte_count = slices[slice_index].byte_count;
        part.push(slice_index);
        part_byte_count += slice_byte_count;
        bytes_still_to_come -= slice_byte_count;
        if part_byte_count >= READ_PART_MINIMUM_BYTES
            && bytes_still_to_come >= READ_PART_MINIMUM_BYTES
        {
            planned_reads.push(std::mem::take(&mut part));
            part_byte_count = 0;
        }
    }
    if !part.is_empty() {
        planned_reads.push(part);
    }
}

#[cfg(test)]
#[path = "record_read_plan_tests.rs"]
mod tests;
